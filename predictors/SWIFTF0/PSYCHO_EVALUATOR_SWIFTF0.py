"""
PSYCHO_EVALUATOR_SWIFTF0.py — Main execution script for SwiftF0 psychophysics inference.

Processes all five psychophysical experiment HDF5 files through SwiftF0,
performs 200→700 Gaussian interpolation, and generates JSON prediction files
matching the MERT reference format (column-oriented).

Usage:
    python PSYCHO_EVALUATOR_SWIFTF0.py \
        --data_root /data/zyf/MERT_psychophysics/data/dataset_psychophysics \
        --output_dir ./SWIFTF0_psy_predictions
"""

# ============================================================================
# Version history
# ============================================================================
# v2 (2026-08-05): Added raw activation storage — per-trial 200-dim softmax
#   activations are collected BEFORE Gaussian interpolation (200→700,
#   sigma=2.0) and saved to <json_basename>_raw.npz alongside each JSON file
#   (keys: raw_activations, freqs_source, model_name, experiment_name,
#   sigma_used). JSON output format unchanged.
# v2.1 (2026-08-05): Expanded .npz — added per-trial model raw outputs
#   f0_hz (float32, ONNX 'pitch_hz' output), confidence (float32, mean of
#   ONNX 'confidence' output), voicing_ratio (float32, fraction of frames
#   with confidence > 0.9, SwiftF0's V/UV threshold). Failed trials are
#   stored as NaN rows, aligned 1:1 with raw_activations.
# ============================================================================

import os
import sys
import json
import time
import argparse
import traceback
import warnings
from datetime import datetime

import numpy as np
import h5py
from tqdm import tqdm

from swiftf0_predict import SwiftF0Predictor
from utils.audio_utils import load_audio_from_hdf5
from utils.probability_utils import swiftf0_to_700


# =============================================================================
# Experiment definitions
# =============================================================================

# Each tuple: (hdf5_filename, audio_key_path, experiment_display_name)
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

# Trial-level keys to extract from HDF5 (varies per experiment)
# We extract all 1D datasets with length == n_trials
# Config-level keys: extracted but kept as-is


def load_hdf5_metadata(hdf5_path: str, n_trials: int) -> dict:
    """
    Extract trial-level metadata and config from an HDF5 file.

    Returns tuple of (trial_data, config_data).
    """
    # Known audio dataset names (must skip to avoid bloating output)
    AUDIO_KEYS = {'tone_in_noise', 'signal', 'signal_in_noise'}

    trial_data = {}
    config_data = {}

    with h5py.File(hdf5_path, 'r') as f:
        # Collect top-level datasets
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                # Skip audio datasets entirely
                if key in AUDIO_KEYS:
                    continue
                val = item[()]
                if item.ndim == 1 and item.shape[0] == n_trials:
                    # Convert to list; handle numpy types
                    if np.issubdtype(item.dtype, np.integer):
                        trial_data[key] = [int(x) for x in val]
                    elif np.issubdtype(item.dtype, np.floating):
                        trial_data[key] = [float(x) for x in val]
                    else:
                        trial_data[key] = val.tolist()
                else:
                    # Config-level (small datasets only)
                    config_data[key] = _safe_convert(val)
            elif isinstance(item, h5py.Group):
                for sub_key in item.keys():
                    sub_item = item[sub_key]
                    if isinstance(sub_item, h5py.Dataset):
                        # Skip audio datasets
                        if sub_key in AUDIO_KEYS:
                            continue
                        val = sub_item[()]
                        if sub_item.ndim == 1 and sub_item.shape[0] == n_trials:
                            if np.issubdtype(sub_item.dtype, np.integer):
                                trial_data[sub_key] = [int(x) for x in val]
                            else:
                                trial_data[sub_key] = [float(x) for x in val]
                        else:
                            full_key = f"config_{item.name.replace('/', '_')}/{sub_key}"
                            config_data[full_key] = _safe_convert(val)

    return trial_data, config_data


def _safe_convert(val):
    """Safely convert numpy scalar/array to Python native types.

    策略 B 副本修正（2026-08-14）：B1/B6 HDF5 含字符串 config 数据集
    （base_f0_list 等），h5py 读回为 object 数组包 bytes → val.tolist()
    产出含 bytes 的列表 → json.dump 抛 TypeError（JSON 截断失效）。
    此处对 bytes 元素解码为 str。Josh 原 HDF5 无字符串 config，原版未踩到。
    """
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return [val.item()]
        elif val.ndim == 1 and len(val) <= 100:
            out = val.tolist()
            return [x.decode('utf-8') if isinstance(x, bytes) else x
                    for x in out]
        else:
            return [val.tolist()] if val.ndim <= 2 else [str(val.shape)]
    elif isinstance(val, (np.integer,)):
        return [int(val)]
    elif isinstance(val, (np.floating,)):
        return [float(val)]
    elif isinstance(val, (bytes, np.bytes_)):
        return [val.decode('utf-8')]
    else:
        return [val]


def process_experiment(
    predictor: SwiftF0Predictor,
    hdf5_path: str,
    audio_key: str,
    experiment_name: str,
    output_dir: str,
    orig_sr: int = 32000,
):
    """
    Process a single experiment: load audio, run SwiftF0, interpolate, save JSON.

    Returns:
        dict with n_trials, n_failures, failure_reasons
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  HDF5: {hdf5_path}")
    print(f"  Audio key: {audio_key}")
    print(f"{'='*60}")

    # Load audio (all trials)
    print("  Loading audio...")
    try:
        audio_all = load_audio_from_hdf5(hdf5_path, audio_key)
    except Exception as e:
        print(f"  [ERROR] Failed to load audio: {e}")
        return {'n_trials': 0, 'n_failures': 0, 'failure_reasons': [str(e)]}

    n_trials = audio_all.shape[0]
    print(f"  Loaded {n_trials} trials, shape={audio_all.shape}")

    # Pre-resample all audio to 16kHz (bypass librosa per trial)
    print("  Resampling all audio to 16kHz...")
    import librosa
    t0 = time.time()
    new_n_samples = int(audio_all.shape[1] * 16000 / orig_sr)
    audio_16k = np.zeros((n_trials, new_n_samples), dtype=np.float32)
    for i in range(n_trials):
        audio_16k[i] = librosa.resample(
            audio_all[i].astype(np.float64), orig_sr=orig_sr, target_sr=16000
        ).astype(np.float32)
    print(f"  Resampled in {time.time()-t0:.1f}s, shape={audio_16k.shape}")

    # Get ONNX session directly for batch inference
    session = predictor.detector.pitch_session
    input_name = predictor.detector.pitch_input_name

    # Load metadata
    print("  Loading metadata...")
    trial_metadata, config_metadata = load_hdf5_metadata(hdf5_path, n_trials)

    # Initialize output lists
    labels_pred_list = []
    probs_out_list = []
    activation_list = []  # raw 200-dim softmax activations (pre-interpolation)
    pitch_hz_list = []
    confidence_list = []
    voicing_ratio_list = []
    failures = []

    # Process trials using batched ONNX inference
    print(f"  Processing {n_trials} trials...")
    start_time = time.time()

    for i in tqdm(range(n_trials), desc=f"  {experiment_name}", unit="trial"):
        try:
            # Run ONNX model directly on 16kHz audio (skips resampling)
            ort_inputs = {input_name: audio_16k[i][None, :].astype(np.float32)}
            outputs = session.run(
                ['pitch_hz', 'confidence', 'activation_200'], ort_inputs
            )
            activation_200 = outputs[2][0]  # (T, 200)

            # Aggregate across time frames (mean)
            if activation_200.ndim == 1:
                act = activation_200
            else:
                act = activation_200.mean(axis=0)
                if act.sum() > 0:
                    act = act / act.sum()

            # Collect raw activation before interpolation
            activation_list.append(act.copy())

            # Collect model raw outputs (ONNX outputs[0]=pitch_hz, [1]=confidence)
            pitch_arr = np.asarray(outputs[0]).flatten()
            conf_arr = np.asarray(outputs[1]).flatten()
            pitch_hz_list.append(
                float(pitch_arr[0]) if pitch_arr.size > 0 else float('nan')
            )
            confidence_list.append(
                float(np.mean(conf_arr)) if conf_arr.size > 0 else float('nan')
            )
            # SwiftF0 thresholds confidence at 0.9 for V/UV (swiftf0_predict.py)
            voicing_ratio_list.append(
                float(np.mean(conf_arr > 0.9)) if conf_arr.size > 0 else float('nan')
            )

            # 200→700 Gaussian interpolation
            labels_pred, probs_700 = swiftf0_to_700(act, sigma=2.0)

            labels_pred_list.append(int(labels_pred))
            probs_out_list.append(probs_700.tolist())

        except Exception as e:
            failures.append({
                'trial_index': i,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })
            # Use NaN/zeros as placeholder for failed trials
            labels_pred_list.append(-1)
            probs_out_list.append([0.0] * 700)
            activation_list.append(None)
            pitch_hz_list.append(float('nan'))
            confidence_list.append(float('nan'))
            voicing_ratio_list.append(float('nan'))

    elapsed = time.time() - start_time
    n_success = n_trials - len(failures)
    print(f"  Completed: {n_success}/{n_trials} successful ({elapsed:.1f}s)")

    if failures:
        print(f"  [WARNING] {len(failures)} trials failed")
        for f in failures[:3]:
            print(f"    Trial {f['trial_index']}: {f['error']}")

    # Build output dict (column-oriented, matching MERT reference format)
    output = {}

    # Add SwiftF0 predictions
    output['f0_label:labels_pred'] = labels_pred_list
    output['f0_label:probs_out'] = probs_out_list

    # Add trial-level metadata from HDF5
    for key, values in trial_metadata.items():
        # Skip audio data and f0_label: prefixed keys
        if key not in ['tone_in_noise', 'signal', 'signal_in_noise']:
            output[key] = values

    # Add config-level metadata
    for key, value in config_metadata.items():
        output[key] = value

    # Add experiment metadata
    output['experiment_name'] = experiment_name
    output['model_name'] = 'swiftf0'
    output['timestamp'] = datetime.now().isoformat()

    # Save JSON
    output_filename = f"predictions_swiftf0_{experiment_name}.json"
    output_path = os.path.join(output_dir, output_filename)

    print(f"  Saving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output, f)

    # Save raw activations alongside the JSON (same basename + _raw.npz)
    f_min_swift, f_max_swift, n_bins_swift = 46.875, 2093.75, 200
    delta_swift = np.log2(f_max_swift / f_min_swift) / (n_bins_swift - 1)
    freqs_source = f_min_swift * (2 ** (np.arange(n_bins_swift) * delta_swift))
    raw_acts = np.array([
        a if a is not None else np.full(n_bins_swift, np.nan, dtype=np.float32)
        for a in activation_list], dtype=np.float32)
    npz_path = output_path.replace('.json', '_raw.npz')
    np.savez_compressed(
        npz_path,
        raw_activations=raw_acts.astype(np.float32),
        freqs_source=freqs_source.astype(np.float32),
        f0_hz=np.asarray(pitch_hz_list, dtype=np.float32),
        confidence=np.asarray(confidence_list, dtype=np.float32),
        voicing_ratio=np.asarray(voicing_ratio_list, dtype=np.float32),
        model_name='swiftf0',
        experiment_name=experiment_name,
        sigma_used=2.0,
    )
    print(f"  Raw activations saved to: {npz_path}")

    # Report file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    return {
        'n_trials': n_trials,
        'n_failures': len(failures),
        'failure_reasons': [f['error'] for f in failures],
        'output_file': output_filename,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SwiftF0 Psychophysics Evaluator — F0 estimation pipeline"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='/data/zyf/MERT_psychophysics/data/dataset_psychophysics',
        help='Path to directory containing HDF5 stimulus files.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./SWIFTF0_psy_predictions',
        help='Directory to save JSON prediction files.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to modified ONNX model (default: ./model_with_activation.onnx).'
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


    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  SwiftF0 Psychophysics Evaluator")
    print("=" * 60)
    print(f"  Data root:   {args.data_root}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Model path:  {args.model_path or './model_with_activation.onnx'}")
    print()

    # Initialize predictor
    print("Initializing SwiftF0Predictor...")
    predictor = SwiftF0Predictor(model_path=args.model_path)
    print("  Predictor initialized successfully.")
    print(f"  Framework: ONNX Runtime (onnxruntime)")
    print(f"  Input SR: 16000 Hz")
    print()

    # Process each experiment
    results = {}
    total_trials = 0
    total_failures = 0

    for hdf5_filename, audio_key, experiment_name in EXPERIMENTS:
        if experiment_filter is not None and experiment_name not in experiment_filter:
            print(f"  [SKIP] {experiment_name} (not in --experiments filter)")
            continue
        hdf5_path = os.path.join(args.data_root, hdf5_filename)

        if not os.path.exists(hdf5_path):
            print(f"\n[SKIP] HDF5 file not found: {hdf5_path}")
            results[experiment_name] = {
                'n_trials': 0,
                'n_failures': 0,
                'error': 'HDF5 file not found',
            }
            continue

        result = process_experiment(
            predictor=predictor,
            hdf5_path=hdf5_path,
            audio_key=audio_key,
            experiment_name=experiment_name,
            output_dir=args.output_dir,
        )

        results[experiment_name] = result
        total_trials += result.get('n_trials', 0)
        total_failures += result.get('n_failures', 0)

    # Generate evaluation report
    print(f"\n{'='*60}")
    print("  Generating evaluation report...")
    print(f"{'='*60}")

    report = {
        'evaluation_date': datetime.now().isoformat(),
        'model_name': 'SwiftF0',
        'implementation': 'swift-f0 PyPI package (ONNX Runtime, modified model)',
        'framework': 'ONNX Runtime (onnxruntime)',
        'input_sr': 16000,
        'stft_params': 'Hann window, n_fft=1024, hop_length=256 (inside ONNX model)',
        'frequency_band': '46.875–2093.75 Hz (132 bins, inside ONNX model)',
        'log_compression': 'log(X + 1e-8) (inside ONNX model)',
        'output_bins': 200,
        'decoding': 'Local expected value (19-bin window, inside ONNX model)',
        'interpolation_sigma': 2.0,
        'interpolation_method': 'Gaussian interpolation (200→700)',
        'cent_reference': '46.875 Hz',
        'data_path': args.data_root,
        'output_directory': os.path.abspath(args.output_dir),
        'total_predictions_attempted': total_trials,
        'successful_predictions': total_trials - total_failures,
        'success_rate': round((total_trials - total_failures) / max(total_trials, 1) * 100, 2),
        'experiments': [exp[2] for exp in EXPERIMENTS],
        'device_used': 'CPU (ONNX Runtime CPUExecutionProvider)',
        'model_file': 'model_with_activation.onnx',
        'generated_files': [
            f"predictions_swiftf0_{exp[2]}.json" for exp in EXPERIMENTS
        ],
        'experiment_details': results,
    }

    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Report saved to: {report_path}")
    print()
    print("=" * 60)
    print(f"  ALL DONE")
    print(f"  Total trials:     {total_trials}")
    print(f"  Successful:       {total_trials - total_failures}")
    print(f"  Failed:           {total_failures}")
    print(f"  Success rate:     {report['success_rate']}%")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
