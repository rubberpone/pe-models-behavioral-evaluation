#!/usr/bin/env python3
"""
PSYCHO_EVALUATOR_HARMOF0.py — Batched HarmoF0 predictions for all five
psychophysical experiments.

Uses batch processing on GPU for maximum throughput.

Usage:
    conda activate HARMOF0_psychophysics
    python PSYCHO_EVALUATOR_HARMOF0.py \\
        --data_root /data/zyf/MERT_psychophysics/data/dataset_psychophysics \\
        --output_dir ./HARMOF0_psy_predictions \\
        --device cuda

=============================================================================
VERSION HISTORY
=============================================================================
v3 (2026-08-05): EXPAND .npz keys with per-trial model outputs
  - Added f0_hz (n_segments,) float32: per-segment decoded F0 in Hz
    (freqs output of predict_multi_frame_batch, argmax decode).
  - Error segments (labels=None in JSON) are stored as NaN to keep array
    alignment with the JSON.
  - Existing JSON output format and interpolation sigma=1.5 UNCHANGED.
  - Backup: backup/PSYCHO_EVALUATOR_HARMOF0.py.bak

v2 (2026-08-05): ADD raw activation storage for sigma sensitivity analysis
  - Added np.savez_compressed() call to store raw 352-dim activations
    alongside each prediction JSON (as *_raw.npz files).
  - Raw activations are saved BEFORE Gaussian interpolation (sigma=1.5),
    enabling subsequent sigma scanning without re-running model inference.
  - .npz keys: raw_activations (n_segments, 352) float32, freqs_source (352,)
    float32, model_name, experiment_name, sigma_used.
  - Error segments (labels=None in JSON) are stored as NaN rows in
    raw_activations to keep array alignment with the JSON.
  - Existing JSON output format is UNCHANGED.
  - Backup: backup/PSYCHO_EVALUATOR_HARMOF0.py.bak
=============================================================================
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.audio_utils import (
    load_all_audio_from_hdf5,
    load_all_metadata_from_hdf5,
    resample_audio,
    extract_overlapping_frames,
    ensure_mono,
)
from utils.probability_utils import harmof0_to_700_batch
from harmof0_predict import HarmoF0BatchPredictor


# ============================================================================
# Configuration
# ============================================================================

# Experiment definitions: (hdf5_filename, audio_key, experiment_display_name)
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
TARGET_SR = 16000
DURATION_S = 0.05   # 50 ms
EXPECTED_LEN_32K = int(ORIG_SR * DURATION_S)     # 1600 samples
EXPECTED_LEN_16K = int(TARGET_SR * DURATION_S)   # 800 samples

# Multi-frame parameters
NUM_FRAMES = 5
WINDOW_MS = 64
HOP_MS = 10

# Gaussian interpolation parameter
SIGMA = 1.5

# GPU batch size (frames per batch)
GPU_BATCH_FRAMES = 10000

# CPU batch size (audio segments processed at once in pre-processing)
CPU_PREPROC_CHUNK = 5000


# ============================================================================
# JSON Encoding
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
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
# Experiment Processing (batched)
# ============================================================================

def preprocess_audio_chunk(audio_chunk):
    """
    Preprocess a chunk of audio segments for HarmoF0 inference.

    For each 50ms audio segment (1600 samples @ 32kHz):
    1. Resample to 16kHz → 800 samples
    2. Extract 5 overlapping 64ms windows (10ms hop) → 5 × 1024 samples

    Args:
        audio_chunk: numpy array of shape (C, 4800)

    Returns:
        frames_chunk: numpy array of shape (C * 5, 1024)
    """
    c = audio_chunk.shape[0]
    all_frames = np.zeros((c * NUM_FRAMES, 1024), dtype=np.float32)

    for i in range(c):
        # Take first 50ms (1600 samples at 32kHz)
        audio_seg = audio_chunk[i].astype(np.float64)

        if len(audio_seg) > EXPECTED_LEN_32K:
            audio_seg = audio_seg[:EXPECTED_LEN_32K]

        # Resample from 32kHz to 16kHz
        audio_16k = resample_audio(audio_seg, ORIG_SR, TARGET_SR)
        if len(audio_16k) > EXPECTED_LEN_16K:
            audio_16k = audio_16k[:EXPECTED_LEN_16K]

        # Extract 5 overlapping frames
        frames = extract_overlapping_frames(
            audio_16k, sr=TARGET_SR,
            window_ms=WINDOW_MS, hop_ms=HOP_MS, num_frames=NUM_FRAMES,
        )

        for f_idx, frame in enumerate(frames):
            all_frames[i * NUM_FRAMES + f_idx] = frame

    return all_frames


def process_experiment(predictor, data_dir, hdf5_filename, audio_key, experiment_name):
    """
    Process one psychophysics experiment using batched GPU inference.

    Args:
        predictor: HarmoF0BatchPredictor instance
        data_dir: path to directory containing HDF5 files
        hdf5_filename: name of the HDF5 file
        audio_key: key path within HDF5
        experiment_name: display name

    Returns:
        dict with prediction results, or None on failure
    """
    hdf5_path = os.path.join(data_dir, hdf5_filename)

    if not os.path.exists(hdf5_path):
        print(f"  [ERROR] HDF5 file not found: {hdf5_path}")
        return None, 0, [], []

    print(f"\n{'=' * 50}")
    print(f"Processing: {experiment_name}")
    print(f"  HDF5: {hdf5_path}")
    print(f"  Audio key: {audio_key}")

    # ------------------------------------------------------------------
    # Load audio data
    # ------------------------------------------------------------------
    try:
        audio = load_all_audio_from_hdf5(hdf5_path, audio_key)
    except Exception as e:
        print(f"  [ERROR] Failed to load audio from {hdf5_path}: {e}")
        return None, 0, [], []

    n_segments = audio.shape[0]
    print(f"  Segments: {n_segments}")

    # ------------------------------------------------------------------
    # Load metadata from HDF5
    # ------------------------------------------------------------------
    try:
        metadata = load_all_metadata_from_hdf5(hdf5_path)
    except Exception as e:
        print(f"  [WARNING] Failed to load metadata: {e}")
        metadata = {}

    # ------------------------------------------------------------------
    # Run batched GPU inference
    # ------------------------------------------------------------------
    all_pred_labels = []
    all_pred_probs = []
    all_f0_hz = []  # v3: per-segment decoded F0 (Hz)
    all_raw_activations = []  # v2: store raw 352-dim activations before interpolation
    total_errors = 0

    pbar = tqdm(total=n_segments, desc=f"  Predicting {experiment_name}",
                unit="seg")

    for chunk_start in range(0, n_segments, CPU_PREPROC_CHUNK):
        chunk_end = min(chunk_start + CPU_PREPROC_CHUNK, n_segments)
        chunk_audio = audio[chunk_start:chunk_end]
        chunk_size = chunk_end - chunk_start

        try:
            # ------------------------------------------------------------------
            # Step 1: CPU preprocess — extract frames
            # ------------------------------------------------------------------
            chunk_frames = preprocess_audio_chunk(chunk_audio)
            # shape: (chunk_size * 5, 1024)

            # ------------------------------------------------------------------
            # Step 2: GPU inference — batch all frames
            # ------------------------------------------------------------------
            freqs, activations = predictor.predict_multi_frame_batch(
                chunk_frames,
                n_frames_per_sample=NUM_FRAMES,
                batch_size=GPU_BATCH_FRAMES,
            )
            # freqs: (chunk_size,), activations: (chunk_size, 352)

            # v3: collect per-segment decoded F0 (Hz)
            all_f0_hz.extend(freqs.tolist())

            # ------------------------------------------------------------------
            # Step 3: 352 → 700 mapping — batched
            # ------------------------------------------------------------------
            # v2: collect raw activations BEFORE interpolation
            for i in range(chunk_size):
                all_raw_activations.append(activations[i].copy())

            labels_pred, probs_700 = harmof0_to_700_batch(activations, sigma=SIGMA)
            # labels_pred: (chunk_size,), probs_700: (chunk_size, 700)

            # Collect results
            all_pred_labels.extend(labels_pred.tolist())
            all_pred_probs.extend(probs_700.tolist())

            pbar.update(chunk_size)

        except Exception as e:
            print(f"\n  [ERROR] Chunk [{chunk_start}:{chunk_end}]: {e}")
            for _ in range(chunk_size):
                all_pred_labels.append(None)
                all_pred_probs.append(None)
                # v2: error trials not stored — fill with NaN to keep alignment
                all_raw_activations.append(np.full(352, np.nan, dtype=np.float32))
            all_f0_hz.extend([np.nan] * chunk_size)  # v3: NaN keeps alignment with JSON
            total_errors += chunk_size
            pbar.update(chunk_size)

    pbar.close()

    # ------------------------------------------------------------------
    # Build output dict
    # ------------------------------------------------------------------
    result = dict(metadata)
    result["f0_label:labels_pred"] = all_pred_labels
    result["f0_label:probs_out"] = all_pred_probs
    result["experiment_name"] = experiment_name
    result["model_name"] = "HarmoF0"
    result["timestamp"] = datetime.now().isoformat()

    if total_errors:
        print(f"  ⚠️  {total_errors}/{n_segments} segments had errors")

    return result, total_errors, all_raw_activations, all_f0_hz


# ============================================================================
# Evaluation Report
# ============================================================================

def generate_evaluation_report(
    generated_files,
    failed_experiments,
    data_path,
    output_dir,
    device_str,
    total_errors,
):
    total_attempted = len(EXPERIMENTS)
    successful = len(generated_files)

    report = {
        "model_name": "HarmoF0",
        "implementation": "harmof0.PitchTracker (official)",
        "input_frames": "5 frames, 10ms hop (multi-frame mode, default)",
        "interpolation_sigma": SIGMA,
        "interpolation_method": "Gaussian interpolation (352→700)",
        "data_path": str(data_path),
        "output_directory": str(output_dir),
        "evaluation_date": datetime.now().isoformat(),
        "timestamp": datetime.now().isoformat(),
        "total_experiments_attempted": total_attempted,
        "successful_experiments": successful,
        "success_rate": round(successful / total_attempted * 100, 1) if total_attempted > 0 else 0,
        "total_segment_errors": total_errors,
        "experiments": [exp[2] for exp in EXPERIMENTS],
        "device_used": device_str,
        "generated_files": generated_files,
        "failed_experiments": [
            {"name": name, "error": err} for name, err in failed_experiments
        ],
        "total_generated_files": len(generated_files),
    }

    return report


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="HarmoF0 Psychophysics Evaluator")
    parser.add_argument("--data_root", type=str,
                        default="/data/zyf/MERT_psychophysics/data/dataset_psychophysics",
                        help="Path to directory with HDF5 stimulus files")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "HARMOF0_psy_predictions"),
                        help="Path to output directory for prediction JSONs")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device for HarmoF0 inference")
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
    print("HarmoF0 Psychophysics Evaluator (Batched)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialize batched predictor
    # ------------------------------------------------------------------
    print("\n[1/3] Initializing HarmoF0 (batched mode)...")
    predictor = HarmoF0BatchPredictor(device=args.device)
    print("  HarmoF0 model loaded successfully.")

    # ------------------------------------------------------------------
    # Process each experiment
    # ------------------------------------------------------------------
    print("\n[2/3] Processing experiments...")

    generated_files = []
    failed_experiments = []
    total_errors = 0

    for hdf5_filename, audio_key, experiment_name in EXPERIMENTS:
        if experiment_filter is not None and experiment_name not in experiment_filter:
            print(f"  [SKIP] {experiment_name} (not in --experiments filter)")
            continue
        try:
            result, n_errors, all_raw_activations, all_f0_hz = process_experiment(
                predictor=predictor,
                data_dir=args.data_root,
                hdf5_filename=hdf5_filename,
                audio_key=audio_key,
                experiment_name=experiment_name,
            )
            total_errors += n_errors

            if result is None:
                failed_experiments.append((experiment_name, "Processing returned None"))
                print(f"  ❌ Failed to process {experiment_name}")
                continue

            # Save predictions
            output_filename = f"predictions_harmof0_{experiment_name}.json"
            output_path = os.path.join(args.output_dir, output_filename)

            with open(output_path, 'w') as f:
                json.dump(result, f, cls=NumpyEncoder, indent=2)

            # v2: save raw 352-dim activations as .npz (before interpolation)
            raw_path = output_path.replace('.json', '_raw.npz')
            freqs_source = 27.5 * (2 ** (np.arange(352) / 48))  # 25¢/bin, base 27.5 Hz
            np.savez_compressed(
                raw_path,
                raw_activations=np.array(all_raw_activations, dtype=np.float32),
                freqs_source=freqs_source.astype(np.float32),
                model_name="HarmoF0",
                experiment_name=experiment_name,
                sigma_used=SIGMA,
                # v3: per-segment decoded F0 (Hz); NaN for error segments
                f0_hz=np.array(all_f0_hz, dtype=np.float32),
            )

            generated_files.append(output_filename)
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✅ Saved: {output_filename} ({size_mb:.1f} MB)")

        except Exception as e:
            failed_experiments.append((experiment_name, str(e)))
            print(f"  ❌ Error processing {experiment_name}: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Generate evaluation report
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating evaluation report...")

    report = generate_evaluation_report(
        generated_files=generated_files,
        failed_experiments=failed_experiments,
        data_path=args.data_root,
        output_dir=args.output_dir,
        device_str=args.device,
        total_errors=total_errors,
    )

    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)

    print(f"  ✅ Report saved: evaluation_report.json")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("✅ Evaluation complete!")
    print(f"   Generated {len(generated_files)}/{len(EXPERIMENTS)} prediction files")
    print(f"   Total segment errors: {total_errors}")
    print(f"   Output directory: {args.output_dir}")
    for fname in generated_files:
        fpath = os.path.join(args.output_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"     - {fname} ({size_mb:.1f} MB)")

    if failed_experiments:
        print(f"\n   Failed experiments:")
        for name, err in failed_experiments:
            print(f"     - {name}: {err}")


if __name__ == "__main__":
    main()
