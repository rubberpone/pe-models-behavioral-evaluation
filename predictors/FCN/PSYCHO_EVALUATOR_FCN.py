#!/usr/bin/env python3
"""
PSYCHO_EVALUATOR_FCN.py — FCN F0 estimation for all five psychophysical experiments.

Processes each HDF5 stimulus file through the FCN pipeline:
  1. Load audio from HDF5
  2. Preprocess: 32kHz→8kHz resample, symmetric pad, normalize
  3. Run FCN inference → 486-bin sigmoid activation
  4. Decode F0 via local weighted average
  5. Convert 486→700 via Gaussian interpolation
  6. Save predictions as JSON (mirroring reference format exactly)

Usage:
    conda activate FCN_psychophysics
    python PSYCHO_EVALUATOR_FCN.py \
        --data_root /data/zyf/MERT_psychophysics/data/dataset_psychophysics \
        --output_dir ./FCN_psy_predictions \
        --model_type fcn_993 \
        --device cuda

Changes:
  v3 (2026-08-05):
    - Expanded the .npz output with f0_hz (n_trials,) float32: per-trial
      decoded F0 in Hz from continuous_f0_from_activation() (same values
      as the '_f0_hz' key in the JSON). Error trials are stored as NaN.
    - JSON output format and interpolation sigma=1.5 UNCHANGED.
  v2 (2026-08-05):
    - Added raw activation storage: the per-segment 486-bin sigmoid
      activation is collected (before the Gaussian 486->700 interpolation)
      and saved as a compressed .npz file alongside each JSON prediction.
      .npz keys: raw_activations (float32), freqs_source (float32),
      model_name (str), experiment_name (str), sigma_used (float).
      Only successful trials are stored (error trials are excluded),
      matching the non-None entries of 'f0_label:labels_pred' in the JSON.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import h5py
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fcn_predict import FCNPredictor
from utils.audio_utils import preprocess_single
from utils.probability_utils import fcn_to_700, continuous_f0_from_activation


# ============================================================================
# Configuration
# ============================================================================

# Experiment definitions: (hdf5_filename, audio_key, experiment_name)
EXPERIMENTS = [
    ("stim_ERB_B1.hdf5", "tone_in_noise",           "ERB_B1"),
    ("stim_ERB_B1_fine.hdf5", "tone_in_noise",     "ERB_B1_fine"),
    ("stim_ERB_B2.hdf5", "tone_in_noise",           "ERB_B2"),
    ("stim_ERB_B2_v2.hdf5", "tone_in_noise",           "ERB_B2_v2"),
    ("stim_ERB_B3.hdf5", "stimuli/signal",          "ERB_B3"),
    ("stim_ERB_B4.hdf5", "stimuli/signal",          "ERB_B4"),
    ("stim_ERB_B5.hdf5", "stimuli/signal_in_noise", "ERB_B5"),
    ("stim_ERB_B6.hdf5", "tone_in_noise",           "ERB_B6"),
]

# Reference directory for JSON structure templates
REFERENCE_DIR = "/data/zyf/MERT_psychophysics/mert_psy_predictions_example"

# Default paths（策略 B 副本：指向本项目）
DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), '..', 'data')
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(PROJECT_ROOT), '..', 'predictions', 'strategy_B', 'FCN')
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")

# Batch size for inference
INFERENCE_BATCH_SIZE = 64


# ============================================================================
# Helper Functions
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def load_reference_template(experiment_name):
    """
    Load reference prediction JSON as a structural template.

    The MERT reference files contain the exact key structure we need to match.

    Args:
        experiment_name: e.g., "BernsteinOxenham2005"

    Returns:
        dict or None
    """
    ref_path = os.path.join(REFERENCE_DIR, f"predictions_mert_{experiment_name}.json")
    if os.path.exists(ref_path):
        with open(ref_path, 'r') as f:
            return json.load(f)
    return None


def load_audio_and_metadata(hdf5_path, audio_key):
    """
    Load audio waveform and metadata from HDF5 file.

    Args:
        hdf5_path: path to HDF5 file
        audio_key: key path for audio data

    Returns:
        tuple of (audio, metadata_dict) or (None, None) on error
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            audio = f[audio_key][:]
            n_segments = audio.shape[0]

            # Collect all metadata datasets as a flat dict
            metadata = {}
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Keep as numpy array for now
                    metadata[name] = obj[:]

            f.visititems(collect_datasets)

            return audio, metadata, n_segments

    except Exception as e:
        print(f"  [ERROR] Failed to load {hdf5_path}: {e}")
        return None, None, 0


def get_audio_key_for_experiment(experiment_name):
    """Get the audio key path for a given experiment."""
    for hdf5_name, audio_key, exp_name in EXPERIMENTS:
        if exp_name == experiment_name:
            return audio_key
    return None


def process_experiment(predictor, hdf5_filename, audio_key, experiment_name,
                       data_root, device):
    """
    Process one psychophysics experiment and return prediction results dict.

    The output dict structure mirrors the reference MERT template:
      - Top-level keys from the reference template
      - 'f0_label:labels_pred': list of int (predicted class labels)
      - 'f0_label:probs_out': list of lists (700-dim probability vectors)
      - Updated metadata fields

    Args:
        predictor: FCNPredictor instance
        hdf5_filename: name of the HDF5 file
        audio_key: key path inside HDF5 for audio data
        experiment_name: display name of the experiment
        data_root: root directory containing HDF5 files
        device: torch device

    Returns:
        dict or None
    """
    hdf5_path = os.path.join(data_root, hdf5_filename)

    if not os.path.exists(hdf5_path):
        print(f"  [ERROR] HDF5 file not found: {hdf5_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Processing: {experiment_name}")
    print(f"  HDF5: {hdf5_filename}")
    print(f"  Audio key: {audio_key}")

    # Load audio data
    audio, metadata, n_segments = load_audio_and_metadata(hdf5_path, audio_key)
    if audio is None:
        return None

    print(f"  Segments: {n_segments}")
    print(f"  Audio shape: {audio.shape}")

    # Load reference template
    template = load_reference_template(experiment_name)
    if template is None:
        print(f"  [WARNING] No reference template found for {experiment_name}")
    else:
        print(f"  Reference template loaded with {len(template)} keys")

    min_input_length = predictor.min_input_length
    print(f"  Model: {predictor.model_type}, min_input_length={min_input_length}")

    # ------------------------------------------------------------------
    # Preprocess and run inference
    # ------------------------------------------------------------------
    all_pred_labels = []  # list of int
    all_pred_probs = []   # list of lists (700 floats)
    all_f0_hz = []        # list of float (optional, for reference)
    errors = []
    raw_activations_list = []  # (486,) per-bin sigmoid activations (successful trials only)

    for i in tqdm(range(n_segments), desc=f"  Predicting {experiment_name}"):
        try:
            # Preprocess single audio segment
            # audio[i] shape: (4800,) = 150ms @ 32kHz
            # Take first 50ms (1600 samples) → resample to 8kHz → pad → normalize
            audio_norm, pad_left = preprocess_single(audio[i], min_input_length)

            # Run inference
            f0_hz, activation_486 = predictor.predict(audio_norm)

            # Collect raw activation before Gaussian interpolation
            raw_activations_list.append(activation_486.copy())

            # Convert 486 → 700
            labels_pred, probs_700 = fcn_to_700(activation_486, sigma=1.5)

            all_pred_labels.append(int(labels_pred))
            all_pred_probs.append(probs_700.tolist())
            all_f0_hz.append(float(f0_hz))

        except Exception as e:
            all_pred_labels.append(None)
            all_pred_probs.append(None)
            all_f0_hz.append(None)
            errors.append(i)
            if len(errors) <= 5:
                print(f"  [ERROR] Segment {i}: {e}")

    if errors:
        print(f"  ⚠️  {len(errors)}/{n_segments} segments had errors and were set to None")

    # ------------------------------------------------------------------
    # Build output dict matching reference template structure
    # ------------------------------------------------------------------
    if template is not None:
        result = OrderedDict()
        # Preserve template key order
        for key in template:
            result[key] = template[key]
    else:
        result = OrderedDict()

    # Update predictions
    result["f0_label:labels_pred"] = all_pred_labels
    result["f0_label:probs_out"] = all_pred_probs

    # Update metadata
    result["experiment_name"] = experiment_name
    result["model_name"] = f"FCN_{predictor.model_type}"
    result["timestamp"] = datetime.now().isoformat()

    # Add FCN-specific fields
    result["_f0_hz"] = all_f0_hz  # optional continuous estimate

    return result, raw_activations_list


def generate_evaluation_report(generated_files, experiment_results, device_str,
                                model_type, min_input_length, data_root,
                                output_dir):
    """
    Generate evaluation_report.json with comprehensive metadata.

    Args:
        generated_files: list of output filenames
        experiment_results: list of (exp_name, n_trials, n_errors) tuples
        device_str: device used for inference
        model_type: FCN variant used
        min_input_length: minimum input length
        data_root: path to HDF5 data
        output_dir: path to output directory

    Returns:
        dict: report data
    """
    total_attempted = len(EXPERIMENTS)
    successful = len(generated_files)

    # Count total trials and errors
    total_trials = sum(r[1] for r in experiment_results)
    total_errors = sum(r[2] for r in experiment_results)

    # Architecture details based on model_type
    arch_details = {
        'fcn_1953': {
            'filters': [256, 32, 32, 128, 256, 512],
            'widths': [32, 64, 64, 64, 64, 64],
            'num_conv_layers': 6,
            'maxpool_layers': 3,
        },
        'fcn_993': {
            'filters': [256, 32, 32, 128, 256, 512],
            'widths': [32, 32, 32, 32, 32, 32],
            'num_conv_layers': 6,
            'maxpool_layers': 3,
        },
        'fcn_929': {
            'filters': [256, 32, 128, 256, 512],
            'widths': [32, 64, 64, 64, 64],
            'num_conv_layers': 5,
            'maxpool_layers': 2,
        },
    }

    arch = arch_details.get(model_type, {})

    experiment_details = []
    for exp_name, n_trials, n_errors in experiment_results:
        experiment_details.append({
            'experiment': exp_name,
            'trials': n_trials,
            'errors': n_errors,
            'success_rate': round((n_trials - n_errors) / n_trials * 100, 2) if n_trials > 0 else 0,
        })

    report = OrderedDict([
        ("evaluation_date", datetime.now().isoformat()),
        ("model_name", "FCN"),
        ("implementation", "Fully-Convolutional Network (Ardaillon & Roebel 2019), "
                           "PyTorch port from official Keras/TensorFlow"),
        ("model_type", model_type),
        ("architecture", {
            "filters": arch.get('filters', []),
            "widths": arch.get('widths', []),
            "num_conv_layers": arch.get('num_conv_layers', 0),
            "num_classifier_layers": 1,
            "total_conv_layers": arch.get('num_conv_layers', 0) + 1,
            "maxpool_layers": arch.get('maxpool_layers', 0),
            "convolution_type": "Conv1d (equivalent to official Keras Conv2D [width, 1])",
            "padding": "valid (no zero-padding in convolutions)",
            "activation": "ReLU (conv layers), Sigmoid (classifier)",
            "normalization": "BatchNorm1d after each conv block",
            "output_bins": 486,
            "output_activation": "per-bin sigmoid (binary cross-entropy training)",
            "frequency_range": "30–1000 Hz",
            "bin_spacing": "12.5 cents (96 bins/octave)",
        }),
        ("input_sr", 8000),
        ("min_input_length", min_input_length),
        ("padding_strategy", "symmetric zero-padding (differs from official one-sided padding; "
                             "symmetric is more appropriate for isolated short stimuli)"),
        ("normalization", "global mean/variance (approximation of official sliding-window "
                          "normalization with wrap padding)"),
        ("decoding_method", "local weighted average (±4 bins, cents domain) following "
                            "official to_local_average_cents"),
        ("aggregation", "none (single-pass fully-convolutional)"),
        ("output_activation_type", "per-bin sigmoid (binary cross-entropy training)"),
        ("interpolation_sigma", 1.5),
        ("interpolation_method", "Gaussian interpolation (486→700)"),
        ("interpolation_sigma_cents", 9.375),
        ("reference_frequency_cents", 10.0),
        ("data_path", data_root),
        ("output_directory", output_dir),
        ("timestamp", datetime.now().isoformat()),
        ("total_experiments_attempted", total_attempted),
        ("successful_experiments", successful),
        ("total_trials", total_trials),
        ("total_errors", total_errors),
        ("overall_success_rate", round(successful / total_attempted * 100, 1) if total_attempted > 0 else 0),
        ("experiments", [exp[2] for exp in EXPERIMENTS]),
        ("experiment_details", experiment_details),
        ("device_used", device_str),
        ("generated_files", generated_files),
        ("total_generated_files", len(generated_files)),
    ])

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FCN Psychophysics Evaluator — F0 estimation for behavioral experiments"
    )
    parser.add_argument(
        '--data_root', type=str, default=DEFAULT_DATA_ROOT,
        help=f'Root directory containing HDF5 stimulus files (default: {DEFAULT_DATA_ROOT})'
    )
    parser.add_argument(
        '--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for JSON predictions (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--model_type', type=str, default='fcn_993',
        choices=['fcn_1953', 'fcn_993', 'fcn_929'],
        help='FCN model variant (default: fcn_993)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use: cuda, cpu, or auto-detect (default)'
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
        help=f'Directory containing .pt checkpoint files (default: {DEFAULT_CHECKPOINT_DIR})'
    )
    parser.add_argument(
        '--experiments', type=str, default=None,
        help='逗号分隔的实验名过滤（如 ERB_B6）；默认 None = 全部 6 个数据集'
    )

    args = parser.parse_args()

    # 策略 B 副本新增: 按数据集过滤（支持 B6 单独试运行/按数据集恢复）
    experiment_filter = None
    if args.experiments:
        experiment_filter = set(
            s.strip() for s in args.experiments.split(',') if s.strip())

    print("=" * 60)
    print("FCN Psychophysics Evaluator")
    print("=" * 60)

    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    device_str = str(device)
    print(f"Device: {device_str}")

    # Data root
    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)
    print(f"Data root: {data_root}")

    # Output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\n[1/3] Loading FCN model ({args.model_type})...")

    predictor = FCNPredictor(
        model_type=args.model_type,
        checkpoint_dir=args.checkpoint_dir,
        device=device_str,
    )
    info = predictor.get_model_info()
    print(f"  Model loaded: {info['model_type']}")
    print(f"  Min input length: {info['min_input_length']} samples @ 8kHz")
    print(f"  Filters: {info['filters']}")
    print(f"  Widths: {info['widths']}")

    # ------------------------------------------------------------------
    # Process each experiment
    # ------------------------------------------------------------------
    print(f"\n[2/3] Processing experiments...")

    generated_files = []
    experiment_results = []  # (exp_name, n_trials, n_errors)

    for hdf5_filename, audio_key, experiment_name in EXPERIMENTS:
        if experiment_filter is not None and experiment_name not in experiment_filter:
            print(f"  [SKIP] {experiment_name} (not in --experiments filter)")
            continue
        try:
            ret = process_experiment(
                predictor=predictor,
                hdf5_filename=hdf5_filename,
                audio_key=audio_key,
                experiment_name=experiment_name,
                data_root=data_root,
                device=device_str,
            )

            if ret is None:
                print(f"  ❌ Failed to process {experiment_name}")
                experiment_results.append((experiment_name, 0, 0))
                continue

            result, raw_activations_list = ret

            # Count trials and errors
            all_labels = result.get("f0_label:labels_pred", [])
            n_trials = len(all_labels)
            n_errors = sum(1 for l in all_labels if l is None)

            # Save predictions
            output_filename = f"predictions_fcn_{experiment_name}.json"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w') as f:
                json.dump(result, f, cls=NumpyEncoder)

            # Save raw activations (collected before Gaussian interpolation) as .npz
            npz_path = output_path.replace('.json', '_raw.npz')
            np.savez_compressed(
                npz_path,
                raw_activations=np.asarray(raw_activations_list, dtype=np.float32),
                freqs_source=(30.0 * (2 ** (np.arange(486) * 12.5 / 1200))).astype(np.float32),
                model_name=f"FCN_{predictor.model_type}",
                experiment_name=experiment_name,
                sigma_used=1.5,
                # v3: per-trial decoded F0 (Hz); NaN for error trials
                f0_hz=np.array([np.nan if v is None else float(v)
                                for v in result.get("_f0_hz", [])],
                               dtype=np.float32),
            )

            generated_files.append(output_filename)
            experiment_results.append((experiment_name, n_trials, n_errors))

            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✅ Saved: {output_path} ({file_size_mb:.1f} MB)")

        except Exception as e:
            print(f"  ❌ Error processing {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            experiment_results.append((experiment_name, 0, 0))

    # ------------------------------------------------------------------
    # Generate evaluation report
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating evaluation report...")

    report = generate_evaluation_report(
        generated_files=generated_files,
        experiment_results=experiment_results,
        device_str=device_str,
        model_type=args.model_type,
        min_input_length=predictor.min_input_length,
        data_root=data_root,
        output_dir=output_dir,
    )

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)

    print(f"  ✅ Report saved: {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"  Generated {len(generated_files)}/{len(EXPERIMENTS)} prediction files")
    print(f"  Output directory: {output_dir}")
    for fname in generated_files:
        fpath = os.path.join(output_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    - {fname} ({size_mb:.1f} MB)")
    print(f"  Report: evaluation_report.json")


if __name__ == "__main__":
    main()
