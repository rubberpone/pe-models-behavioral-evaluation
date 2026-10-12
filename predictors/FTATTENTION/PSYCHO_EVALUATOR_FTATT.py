#!/usr/bin/env python3
"""
PSYCHO_EVALUATOR_FTATT.py — Generate FT Attention predictions for all five
psychophysical experiments.

Loads the pre-trained FT Attention model (TF/Keras), processes each stimulus
through the CFP → model → aggregation → 320→700 mapping pipeline, and saves
prediction JSON files.

Usage:
    conda activate FTATTENTION_psychophysics
    python PSYCHO_EVALUATOR_FTATT.py \
        --data_root /data/zyf/MERT_psychophysics/data/dataset_psychophysics \
        --output_dir ./FTATTENTION_psy_predictions \
        --device cuda

Changes:
  v2 (2026-08-05):
    - Added raw activation storage: each segment's 321-dim activation
      (1 unvoiced + 320 voiced bins) is collected before the Gaussian
      321->700 interpolation and saved as a compressed .npz file alongside
      each JSON prediction.
      .npz keys: raw_activations (float32), freqs_source (float32),
      model_name (str), experiment_name (str), sigma_used (float).
      Only successful segments are stored (error segments are excluded),
      matching the non-None entries of 'f0_label:labels_pred' in the JSON.
      Both the batched and the per-sample fallback paths collect activations.

  v2.1 (2026-08-05):
    - Expanded .npz output: added per-segment model raw outputs
      f0_hz (float32, argmax of voiced bins 1..320 → bin-center Hz on the
      FT-Attention CFP frequency grid) and unvoiced_prob (float32,
      activation_321[0]), aligned 1:1 with raw_activations rows
      (successful segments only, from both batch and fallback paths).
      JSON output and interpolation sigma unchanged.
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime

import numpy as np
import h5py
from tqdm import tqdm

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from ft_attention_predict import FTAttentionPredictor
from utils.audio_utils import (
    load_audio_from_hdf5,
    resample_audio,
    ensure_mono,
    compute_cfp,
    pad_audio_centered,
)
from utils.probability_utils import ft_attention_to_700


# ============================================================================
# Configuration
# ============================================================================

# Experiment definitions: (hdf5_filename, audio_key_path, experiment_display_name)
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

# Audio parameters
ORIG_SR = 32000
TARGET_SR = 8000
DURATION_S = 0.05
TARGET_DURATION_S = 1.28   # ~128 CFP frames at hop=80
TARGET_SAMPLES = int(1.29 * TARGET_SR)  # 10320 samples → exactly 128 CFP frames at hop=80

# Batch size for model inference (CFP computation done per-segment)
INFERENCE_BATCH_SIZE = 64

# CFP parameters (must match training config)
CFP_HOP = 80
CFP_WINDOW = 768
CFP_FMIN = 31.0
CFP_FMAX = 1250.0
CFP_BINS_PER_OCTAVE = 60


# ============================================================================
# Helper: Numpy JSON encoder
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


# ============================================================================
# HDF5 Data Extraction
# ============================================================================

def extract_hdf5_datasets(hdf5_path):
    """
    Extract all datasets from an HDF5 file as lists (for JSON serialization).

    Args:
        hdf5_path: path to the HDF5 file

    Returns:
        dict mapping dataset paths → lists
    """
    # Audio dataset keys to skip (would bloat JSON to GBs)
    AUDIO_KEYS = {'tone_in_noise', 'signal', 'signal_in_noise', 'stimuli/signal',
                  'stimuli/signal_in_noise'}
    result = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Skip audio waveform datasets
            if name in AUDIO_KEYS or name.endswith('/signal') or name.endswith('/signal_in_noise') or name.endswith('/tone_in_noise'):
                return
            data = obj[:]
            # Convert scalars to single-element lists for JSON consistency
            # 策略 B 副本修正（2026-08-14）：B1/B6 的字符串 config 数据集
            # h5py 读回为 object 数组包 bytes → json.dump 抛 TypeError；
            # 此处对 bytes 解码为 str（Josh 原 HDF5 无字符串 config）。
            if data.ndim == 0 or data.size == 1:
                val = data.item() if hasattr(data, 'item') else float(data.flat[0])
                if isinstance(val, (bytes, np.bytes_)):
                    val = val.decode('utf-8')
                result[name] = [val]
            else:
                out = data.tolist()
                result[name] = [
                    x.decode('utf-8') if isinstance(x, bytes) else x
                    for x in out]

    with h5py.File(hdf5_path, 'r') as f:
        f.visititems(visitor)

    return result


# ============================================================================
# Core Processing
# ============================================================================

def decode_ftatt_f0_hz(activation_321):
    """Decode scalar f0_hz (Hz) from a 321-dim FT-Attention activation.

    Argmax over voiced bins (indices 1..320); the bin-center frequency of
    the winning bin is returned using FT-Attention's own CFP frequency
    mapping: freqs = concat([31.0], 31.0 * 2**(arange(320)/60)).
    """
    freqs = np.concatenate([[31.0], 31.0 * (2 ** (np.arange(320) / 60))])
    return float(freqs[1:][np.argmax(activation_321[1:])])


def process_experiment(predictor, data_dir, hdf5_filename, audio_key, experiment_name):
    """
    Process one psychophysics experiment.

    Args:
        predictor: FTAttentionPredictor instance
        data_dir: path to directory containing HDF5 files
        hdf5_filename: name of the HDF5 file
        audio_key: HDF5 key for audio data (e.g., 'tone_in_noise')
        experiment_name: display name (e.g., 'BernsteinOxenham2005')

    Returns:
        dict with prediction results, or None on failure
    """
    hdf5_path = os.path.join(data_dir, hdf5_filename)

    if not os.path.exists(hdf5_path):
        print(f"  [ERROR] HDF5 file not found: {hdf5_path}")
        return None, None, None, None

    print(f"\n{'='*60}")
    print(f"Processing: {experiment_name}")
    print(f"  HDF5 file: {hdf5_path}")
    print(f"  Audio key: {audio_key}")

    # Extract all HDF5 datasets for the output JSON
    print("  Extracting HDF5 datasets...")
    result = extract_hdf5_datasets(hdf5_path)

    # Load audio data
    try:
        audio, _ = load_audio_from_hdf5(hdf5_path, audio_key, idx=None)
    except Exception as e:
        print(f"  [ERROR] Failed to load audio: {e}")
        return None, None, None, None

    n_segments = audio.shape[0]
    print(f"  Segments: {n_segments}")
    print(f"  Audio shape: {audio.shape}")

    # Initialize prediction lists
    all_pred_labels = []
    all_pred_probs = []
    errors = []
    raw_activations_list = []  # (321,) raw activations (successful segments only)
    f0_hz_list = []            # scalar f0_hz per successful segment (Hz)
    unvoiced_prob_list = []    # unvoiced probability = activation_321[0]

    # ------------------------------------------------------------------
    # Process segments in batches
    # ------------------------------------------------------------------
    pbar = tqdm(total=n_segments, desc=f"  {experiment_name}", unit="seg")

    for batch_start in range(0, n_segments, INFERENCE_BATCH_SIZE):
        batch_end = min(batch_start + INFERENCE_BATCH_SIZE, n_segments)
        batch_size = batch_end - batch_start

        # Collect CFP features for the batch
        batch_cfp_list = []
        batch_indices = []

        for i in range(batch_start, batch_end):
            idx = i
            try:
                # Load single segment
                seg = audio[idx]  # (4800,) at 32 kHz

                # Preprocess
                seg_mono = ensure_mono(seg)
                seg_8k = resample_audio(seg_mono, orig_sr=ORIG_SR, target_sr=TARGET_SR)
                seg_padded = pad_audio_centered(seg_8k, TARGET_SAMPLES)

                # Compute CFP
                cfp, cen_freq, time = compute_cfp(
                    seg_padded, sr=TARGET_SR, hop=CFP_HOP, window_size=CFP_WINDOW,
                    fmin=CFP_FMIN, fmax=CFP_FMAX, n_bins_per_octave=CFP_BINS_PER_OCTAVE
                )
                # cfp shape: (3, 320, T) → transpose to (320, T, 3) for Keras
                cfp_input = np.transpose(cfp, (1, 2, 0))  # (320, T, 3)

                # Pad/trim time dimension to exactly 128 frames for fixed model input
                T_actual = cfp_input.shape[1]
                if T_actual < 128:
                    pad_w = 128 - T_actual
                    cfp_input = np.pad(cfp_input, ((0, 0), (0, pad_w), (0, 0)),
                                       mode='constant', constant_values=0)
                elif T_actual > 128:
                    cfp_input = cfp_input[:, :128, :]

                batch_cfp_list.append(cfp_input)
                batch_indices.append(idx)

            except Exception as e:
                print(f"\n  [ERROR] CFP computation failed for segment {idx}: {e}")
                all_pred_labels.append(None)
                all_pred_probs.append(None)
                errors.append(idx)
                pbar.update(1)

        if len(batch_cfp_list) == 0:
            continue

        # Stack CFP features into a batch
        # All should have same T (≈128 frames) due to identical padding
        try:
            batch_cfp = np.stack(batch_cfp_list, axis=0)  # (B, 320, T, 3)
        except ValueError as e:
            # If T differs (shouldn't happen with fixed padding), fall back to per-sample
            print(f"\n  [WARNING] CFPs have different T, processing individually: {e}")
            for cfp_input, idx in zip(batch_cfp_list, batch_indices):
                try:
                    cfp_batch = cfp_input[np.newaxis, ...]  # (1, 320, T, 3)
                    probs = predictor.model.predict(cfp_batch, verbose=0)
                    probs = np.squeeze(probs, axis=0)  # (321, T)

                    # Aggregate
                    activation_321 = probs.mean(axis=1)  # (321,)
                    raw_activations_list.append(activation_321.copy())
                    f0_hz_list.append(decode_ftatt_f0_hz(activation_321))
                    unvoiced_prob_list.append(float(activation_321[0]))
                    label_pred, probs_700 = ft_attention_to_700(activation_321)

                    all_pred_labels.append(label_pred)
                    all_pred_probs.append(probs_700.tolist())
                except Exception as e2:
                    print(f"\n  [ERROR] Inference failed for segment {idx}: {e2}")
                    all_pred_labels.append(None)
                    all_pred_probs.append(None)
                    errors.append(idx)
                pbar.update(1)
            continue

        # Batch model inference
        try:
            probs_batch = predictor.model.predict(batch_cfp, verbose=0)  # (B, 321, 128)
        except Exception as e:
            print(f"\n  [ERROR] Batch inference failed [{batch_start}:{batch_end}]: {e}")
            for idx in batch_indices:
                all_pred_labels.append(None)
                all_pred_probs.append(None)
                errors.append(idx)
            pbar.update(len(batch_indices))
            continue

        # Process each output in the batch
        actual_batch_size = probs_batch.shape[0]
        for b in range(actual_batch_size):
            try:
                probs = probs_batch[b]  # (321, T) or (321,) for T=1

                if probs.ndim == 1:
                    activation_321 = probs  # (321,), single frame
                else:
                    activation_321 = probs.mean(axis=1)  # (321,) mean over frames

                # Collect raw activation before Gaussian interpolation
                raw_activations_list.append(activation_321.copy())
                f0_hz_list.append(decode_ftatt_f0_hz(activation_321))
                unvoiced_prob_list.append(float(activation_321[0]))

                # Map 321 → 700
                label_pred, probs_700 = ft_attention_to_700(activation_321)

                all_pred_labels.append(label_pred)
                all_pred_probs.append(probs_700.tolist())

            except Exception as e:
                print(f"\n  [ERROR] Post-processing failed for batch item {b}: {e}")
                all_pred_labels.append(None)
                all_pred_probs.append(None)
                errors.append(batch_indices[b])

        pbar.update(len(batch_indices))

    pbar.close()

    # ------------------------------------------------------------------
    # Build output dict
    # ------------------------------------------------------------------
    # Preserve original HDF5 data and add predictions + metadata
    result["f0_label:labels_pred"] = all_pred_labels
    result["f0_label:probs_out"] = all_pred_probs

    # Metadata
    result["experiment_name"] = experiment_name
    result["model_name"] = "FT-Attention"
    result["timestamp"] = datetime.now().isoformat()

    # Add FT Attention-specific metadata
    result["_ftatt_metadata"] = {
        "implementation": "Pre-trained FT Attention (ICASSP 2021)",
        "input_sr": TARGET_SR,
        "cfp_channels": "3 (power-scaled spectrogram, GC, GCoS)",
        "stft_params": {
            "sr": TARGET_SR,
            "window": "blackmanharris",
            "n_fft": CFP_WINDOW,
            "hop_length": CFP_HOP,
            "fr": 2.0,
        },
        "cfp_gamma": [0.24, 0.6, 1.0],
        "cfp_normalization": "log(1+x) + min-max per channel",
        "n_freq_bins": "320 (voiced) + 1 (unvoiced) = 321 total",
        "n_output_classes": 321,
        "bins_per_octave": CFP_BINS_PER_OCTAVE,
        "freq_range": "31–1250 Hz (voiced bins) + unvoiced bin (0 Hz)",
        "frame_mode": "padded to 1.28s, 128-frame CFP segments",
        "aggregation": "mean (321-dim activation), unvoiced bin dropped for 700 mapping",
        "output_activation": "softmax (built into model)",
        "interpolation_sigma": 2.0,
        "interpolation_method": "Gaussian interpolation (320→700)",
        "interpolation_ref_freq": 31.0,
    }

    if errors:
        print(f"  ⚠️  {len(errors)}/{n_segments} segments had errors")
    else:
        print(f"  ✅ All {n_segments} segments processed successfully")

    return result, raw_activations_list, f0_hz_list, unvoiced_prob_list


# ============================================================================
# Evaluation Report
# ============================================================================

def generate_evaluation_report(generated_files, failed_experiments, data_dir, output_dir, device_str):
    """
    Generate evaluation_report.json summarizing the entire run.

    Args:
        generated_files: list of successfully generated filenames
        failed_experiments: list of (experiment_name, error_message) tuples
        data_dir: data root path
        output_dir: output directory path
        device_str: device used

    Returns:
        dict: report data
    """
    report = {
        "evaluation_date": datetime.now().isoformat(),
        "model_name": "FT-Attention",
        "implementation": "Pre-trained FT Attention (ICASSP 2021)",
        "input_sr": TARGET_SR,
        "cfp_channels": "3 (power-scaled spectrogram, GC, GCoS)",
        "stft_params": {
            "sr": TARGET_SR,
            "window": "blackmanharris",
            "n_fft": CFP_WINDOW,
            "hop_length": CFP_HOP,
            "fr": 2.0,
        },
        "cfp_gamma": [0.24, 0.6, 1.0],
        "cfp_normalization": "log(1+x) + min-max per channel",
        "n_freq_bins": "320 (voiced) + 1 (unvoiced) = 321 total",
        "n_output_classes": 321,
        "bins_per_octave": CFP_BINS_PER_OCTAVE,
        "freq_range": "31–1250 Hz (voiced bins) + unvoiced bin (0 Hz)",
        "frame_mode": "padded to 1.28s, 128-frame CFP segments",
        "aggregation": "median (voiced Hz) + mean (321-dim activation), unvoiced bin dropped for 700 mapping",
        "output_activation": "softmax",
        "interpolation_sigma": 2.0,
        "interpolation_method": "Gaussian interpolation (320→700)",
        "interpolation_ref_freq": 31.0,
        "data_path": data_dir,
        "output_directory": output_dir,
        "timestamp": datetime.now().isoformat(),
        "device_used": device_str,
        "total_experiments": len(EXPERIMENTS),
        "successful_experiments": len(generated_files),
        "failed_experiments": [
            {"name": name, "error": err} for name, err in failed_experiments
        ],
        "generated_files": generated_files,
        "total_generated_files": len(generated_files),
    }

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FT Attention Psychophysics Evaluator — Generate F0 predictions"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/zyf/MERT_psychophysics/data/dataset_psychophysics",
        help="Path to directory containing HDF5 psychophysics stimulus files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./FTATTENTION_psy_predictions",
        help="Directory for output JSON prediction files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (cuda or cpu). Note: TF uses GPU if available.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pre-trained .h5 weights (default: FTANet-melodic-main/model/ftanet.h5)",
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


    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("=" * 70)
    print("FT Attention Psychophysics Evaluator")
    print("=" * 70)

    # Configure TF to use GPU if available
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Device: CPU (forced)")
    else:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Device: CUDA ({len(gpus)} GPU(s) available)")
        else:
            print("Device: CPU (no GPU detected)")

    # Absolute paths
    data_dir = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Data root:    {data_dir}")
    print(f"Output dir:   {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n[1/2] Loading FT Attention model...")
    try:
        predictor = FTAttentionPredictor(model_path=args.model_path)
        print("  Model loaded successfully.")
    except Exception as e:
        print(f"  [FATAL] Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Process experiments
    # ------------------------------------------------------------------
    print("\n[2/2] Processing experiments...")

    generated_files = []
    failed_experiments = []

    for hdf5_filename, audio_key, experiment_name in EXPERIMENTS:
        if experiment_filter is not None and experiment_name not in experiment_filter:
            print(f"  [SKIP] {experiment_name} (not in --experiments filter)")
            continue
        try:
            ret = process_experiment(
                predictor=predictor,
                data_dir=data_dir,
                hdf5_filename=hdf5_filename,
                audio_key=audio_key,
                experiment_name=experiment_name,
            )

            if ret is None:
                failed_experiments.append((experiment_name, "process_experiment returned None"))
                print(f"  ❌ Failed to process {experiment_name}")
                continue

            result, raw_activations_list, f0_hz_list, unvoiced_prob_list = ret

            # Save predictions
            output_filename = f"predictions_ftatt_{experiment_name}.json"
            output_path = os.path.join(output_dir, output_filename)

            print(f"  Writing {output_filename} ...")
            with open(output_path, 'w') as f:
                json.dump(result, f, cls=NumpyEncoder)

            # Save raw activations (collected before Gaussian interpolation) as .npz
            npz_path = output_path.replace('.json', '_raw.npz')
            np.savez_compressed(
                npz_path,
                raw_activations=np.asarray(raw_activations_list, dtype=np.float32),
                freqs_source=np.concatenate(
                    [[31.0], 31.0 * (2 ** (np.arange(320) / 60))]
                ).astype(np.float32),
                f0_hz=np.asarray(f0_hz_list, dtype=np.float32),
                unvoiced_prob=np.asarray(unvoiced_prob_list, dtype=np.float32),
                model_name='FT-Attention',
                experiment_name=experiment_name,
                sigma_used=2.0,
            )

            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            generated_files.append(output_filename)
            print(f"  ✅ Saved: {output_path} ({file_size_mb:.1f} MB)")

        except Exception as e:
            failed_experiments.append((experiment_name, str(e)))
            print(f"  ❌ Error processing {experiment_name}: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Generate evaluation report
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Generating evaluation report...")

    device_str = "cpu" if args.device == "cpu" else (
        "cuda" if any(generated_files) else "cpu"
    )
    report = generate_evaluation_report(
        generated_files=generated_files,
        failed_experiments=failed_experiments,
        data_dir=data_dir,
        output_dir=output_dir,
        device_str=device_str,
    )

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)
    print(f"  ✅ Report saved: {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    print(f"  Successful: {len(generated_files)}/{len(EXPERIMENTS)} experiments")
    for fname in generated_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"    ✅ {fname} ({size_mb:.1f} MB)")
    for name, err in failed_experiments:
        print(f"    ❌ {name}: {err}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
