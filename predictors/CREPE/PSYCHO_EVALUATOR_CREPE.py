#!/usr/bin/env python3
"""
PSYCHO_EVALUATOR_CREPE.py - Generate CREPE predictions for all six
Strategy B psychophysical experiments (B1..B6).

=====================================================================
策略 B 副本（2026-08-14）: 复制自 /data/zyf/CREPE_psychophysics/，
原文件未做任何修改。本副本的适配改动（仅三处，其余逻辑与原版逐行一致）:
  ① EXPERIMENTS 表 → 6 个 B 数据集（B1/B2/B6=tone_in_noise、
    B3/B4=stimuli/signal、B5=stimuli/signal_in_noise）
  ② 默认路径 → PE_model_ERB_estimation 项目内
  ③ 新增 --experiments 过滤参数（按数据集单独运行，如 B6 试运行）
原版 VERSION HISTORY（v1→v3）见下方保留块。
=====================================================================

Loads pre-trained CREPE model (via torchcrepe), processes each stimulus
through the pipeline, and saves prediction JSON files matching the
reference format from MERT.

Pipeline per stimulus:
  1. Load 32kHz audio from HDF5
  2. Resample to 16kHz (CREPE input rate)
  3. Run CREPE inference via crepe_predict.predict_batch()
  4. Aggregate frames: median(pitch), mean(activation), mean(periodicity)
  5. Map 360-dim activation → 700-dim probability via Gaussian interpolation
  6. Store in JSON matching reference key structure

Usage:
    conda run -n CREPE_psychophysics python PSYCHO_EVALUATOR_CREPE.py \
        --data_root /data/zyf/PE_model_ERB_estimation/data \
        --output_dir /data/zyf/PE_model_ERB_estimation/predictions/strategy_B/CREPE/B6 \
        --experiments ERB_B6 --device cuda:1

=============================================================================
VERSION HISTORY（原版保留）
=============================================================================
v3 (2026-08-05): EXPAND .npz keys with per-trial model outputs
  - Added f0_hz (n_trials,) float32: per-trial decoded F0 in Hz (median of
    frame-level weighted_argmax pitches from predict_batch).
  - Added confidence (n_trials,) float32: per-trial mean frame-level
    periodicity (predict_batch 'periodicities'; CREPE confidence).
  - Failed/error trials are stored as NaN to keep 1:1 alignment with
    raw_activations rows and JSON trial indices.
  - Existing JSON output format and interpolation sigma=1.5 UNCHANGED.

v2 (2026-08-05): ADD raw activation storage for sigma sensitivity analysis
  - Added np.savez_compressed() call to store raw 360-dim sigmoid activations
    alongside each prediction JSON (as *_raw.npz files).
  - Raw activations are saved BEFORE Gaussian interpolation (sigma=1.5),
    enabling subsequent sigma scanning without re-running model inference.
  - .npz keys: raw_activations (n_trials, 360) float32, freqs_source (360,)
    float32, model_name, experiment_name, sigma_used.

v1 (2026-06-30): Initial version — 360-dim sigmoid activation → 700-dim
  Gaussian interpolation (sigma=1.5) → JSON output.
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from scipy.signal import resample_poly

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.audio_utils import load_audio_from_hdf5
from utils.probability_utils import crepe_to_700
from crepe_predict import predict_batch


# ============================================================================
# Configuration
# ============================================================================

# Audio parameters
ORIG_SR = 32000        # Original sampling rate in HDF5 files
TARGET_SR = 16000      # CREPE input sampling rate

# Default paths（策略 B 副本：指向本项目）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, '..', '..', 'data')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, '..', '..', 'predictions', 'strategy_B', 'CREPE')
# B 实验无 MERT 参考模板（load_reference_template 返回 None，
# JSON 由 result dict 直接构建，含分析所需全部键）
REFERENCE_DIR = "/data/zyf/MERT_psychophysics/mert_psy_predictions_example"

# Experiment definitions: (hdf5_filename, audio_key, experiment_name)
# 音频键与 Josh 实验对齐: B1/B2/B6=tone_in_noise, B3/B4=stimuli/signal,
# B5=stimuli/signal_in_noise
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
    Load the MERT reference prediction JSON as a structural template.

    Args:
        experiment_name: e.g., 'BernsteinOxenham2005'

    Returns:
        dict or None
    """
    ref_path = os.path.join(REFERENCE_DIR,
                            f"predictions_mert_{experiment_name}.json")
    if os.path.exists(ref_path):
        with open(ref_path, 'r') as f:
            return json.load(f)
    return None


def preprocess_audio(audio, orig_sr=ORIG_SR, target_sr=TARGET_SR):
    """
    Resample audio from original to target sampling rate.

    Uses scipy's resample_poly for efficient rational resampling.

    Args:
        audio: numpy.ndarray of shape (n_trials, n_samples) or (n_samples,).
        orig_sr: Original sampling rate (default: 32000).
        target_sr: Target sampling rate (default: 16000).

    Returns:
        numpy.ndarray resampled to target_sr.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    # Rational resampling: 32k → 16k is exact downsampling by 2
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd    # 1
    down = orig_sr // gcd    # 2

    if audio.ndim == 1:
        resampled = resample_poly(audio.astype(np.float64), up=up, down=down)
        return resampled.astype(np.float32)

    # Batch resampling along axis=1
    resampled = resample_poly(audio.astype(np.float64), up=up, down=down,
                              axis=1)
    return resampled.astype(np.float32)


# ============================================================================
# Main Evaluator
# ============================================================================

def process_experiment(hdf5_filename, audio_key, experiment_name,
                       data_root, step_size, device,
                       batch_size=256, output_dir=None):
    """
    Process one psychophysics experiment and return prediction results.

    Args:
        hdf5_filename: Name of the HDF5 file.
        audio_key: Key path within the HDF5 for audio data.
        experiment_name: Display name of the experiment.
        data_root: Root directory containing HDF5 files.
        step_size: Frame shift in milliseconds.
        device: 'cuda' or 'cpu'.
        batch_size: Number of trials to process in one GPU batch.

    Returns:
        tuple (result_dict, n_errors):
          - result_dict: dict with prediction results, or None on failure.
          - n_errors: int, number of failed trials, or error string on failure.
    """
    hdf5_path = os.path.join(data_root, hdf5_filename)

    if not os.path.exists(hdf5_path):
        print(f"  [ERROR] HDF5 file not found: {hdf5_path}")
        return None, "HDF5 file not found"

    print(f"\n{'='*50}")
    print(f"Processing: {experiment_name}")
    print(f"  HDF5: {hdf5_path}")
    print(f"  Audio key: {audio_key}")

    # Load audio data
    audio = load_audio_from_hdf5(hdf5_path, audio_key)
    if audio is None:
        return None, "Failed to load audio from HDF5"

    n_trials = audio.shape[0]
    print(f"  Trials: {n_trials}")
    print(f"  Audio shape: {audio.shape}")

    # Load reference template for key structure
    template = load_reference_template(experiment_name)
    if template is None:
        print(f"  [WARNING] No reference template found for {experiment_name}")

    # ------------------------------------------------------------------
    # Preprocess: resample all audio to 16kHz
    # ------------------------------------------------------------------
    print(f"  Resampling {ORIG_SR}Hz → {TARGET_SR}Hz...")
    audio_16k = preprocess_audio(audio)
    print(f"  Resampled shape: {audio_16k.shape}")

    # Compute hop length from step_size
    hop_length = int(TARGET_SR * step_size / 1000)
    print(f"  Step size: {step_size} ms → hop_length: {hop_length} samples")

    # ------------------------------------------------------------------
    # Run CREPE predictions in batches
    # ------------------------------------------------------------------
    all_pred_labels = []
    all_pred_probs = []
    all_pitch_hz = []
    all_periodicity = []  # v3: per-trial confidence (mean frame-level periodicity)
    all_raw_activations = []  # v2: store raw 360-dim activations for sigma analysis
    error_indices = []

    n_batches = (n_trials + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc=f"  Predicting {experiment_name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_trials)
        batch_audio = audio_16k[start_idx:end_idx]

        try:
            # Run CREPE batch inference
            # audio is already at 16kHz, so predict_batch skips resampling
            pitches, activations, periodicities = predict_batch(
                batch_audio, TARGET_SR, step_size=step_size, device=device
            )

            if pitches is None:
                # Entire batch failed
                for i in range(end_idx - start_idx):
                    all_pred_labels.append(None)
                    all_pred_probs.append(None)
                    all_pitch_hz.append(None)
                    all_periodicity.append(np.nan)
                error_indices.extend(range(start_idx, end_idx))
                continue

            # Process each trial in the batch
            for i, (pitch_hz, act_360, period) in enumerate(zip(pitches, activations, periodicities)):
                try:
                    # v2: save raw activation BEFORE interpolation
                    all_raw_activations.append(act_360.copy())
                    # v3: per-trial confidence (mean frame-level periodicity)
                    all_periodicity.append(float(period))
                    # Map 360→700 via Gaussian interpolation
                    label_pred, probs_700 = crepe_to_700(act_360)

                    all_pred_labels.append(label_pred)
                    all_pred_probs.append(probs_700.tolist())
                    all_pitch_hz.append(pitch_hz)

                except Exception as e:
                    print(f"  [WARN] Trial {start_idx + i}: 360→700 mapping "
                          f"failed: {e}")
                    all_pred_labels.append(None)
                    all_pred_probs.append(None)
                    all_pitch_hz.append(None)
                    all_periodicity.append(np.nan)
                    error_indices.append(start_idx + i)

        except Exception as e:
            print(f"  [ERROR] Batch {batch_idx} [{start_idx}:{end_idx}] failed: {e}")
            import traceback
            traceback.print_exc()
            for i in range(end_idx - start_idx):
                all_pred_labels.append(None)
                all_pred_probs.append(None)
                all_pitch_hz.append(None)
                all_periodicity.append(np.nan)
            error_indices.extend(range(start_idx, end_idx))

    # ------------------------------------------------------------------
    # Build output dict matching reference template structure
    # ------------------------------------------------------------------
    if template is not None:
        # Start from template to preserve exact keys and structure
        result = dict(template)
    else:
        result = {}

    # Update predictions
    result["f0_label:labels_pred"] = all_pred_labels
    result["f0_label:probs_out"] = all_pred_probs

    # Update metadata
    result["experiment_name"] = experiment_name
    result["model_name"] = "CREPE"
    result["timestamp"] = datetime.now().isoformat()

    # Report errors
    n_errors = len(error_indices)
    if n_errors > 0:
        print(f"  ⚠️  {n_errors}/{n_trials} trials had errors")

    # ------------------------------------------------------------------
    # v2: Save raw 360-dim activations as .npz for sigma sensitivity analysis
    # ------------------------------------------------------------------
    if output_dir is not None and len(all_raw_activations) > 0:
        # CREPE bin center frequencies: 360 bins, 20 cents, starting at 32.70 Hz
        freqs_source = 32.70 * (2.0 ** (np.arange(360) * 20.0 / 1200.0))
        raw_path = os.path.join(output_dir,
                                f"predictions_crepe_{experiment_name}_raw.npz")
        np.savez_compressed(raw_path,
            raw_activations=np.array(all_raw_activations, dtype=np.float32),
            freqs_source=freqs_source.astype(np.float32),
            model_name="CREPE",
            experiment_name=experiment_name,
            sigma_used=1.5,
            # v3: per-trial decoded F0 (Hz) and confidence; error trials are NaN
            f0_hz=np.array([np.nan if v is None else float(v)
                            for v in all_pitch_hz], dtype=np.float32),
            confidence=np.array(all_periodicity, dtype=np.float32))
        print(f"  💾 Raw activations saved: {raw_path} "
              f"({os.path.getsize(raw_path)/1024/1024:.1f} MB)")

    return result, n_errors


def generate_evaluation_report(generated_files, experiments_processed,
                               failed_experiments, device_str, data_root,
                               output_dir, step_size):
    """
    Generate evaluation_report.json summarizing the CREPE inference run.

    Args:
        generated_files: list of output filenames.
        experiments_processed: list of experiment names successfully processed.
        failed_experiments: list of (exp_name, reason) tuples.
        device_str: Device used for inference.
        data_root: Path to HDF5 data directory.
        output_dir: Path to output directory.
        step_size: Frame shift in ms.

    Returns:
        dict: report data.
    """
    total_attempted = len(EXPERIMENTS)
    successful = len(generated_files)

    report = {
        "evaluation_date": datetime.now().isoformat(),
        "model_name": "CREPE",
        "backend": "torchcrepe",
        "decoder": "weighted_argmax",
        "step_size_ms": step_size,
        "hop_length_samples": int(TARGET_SR * step_size / 1000),
        "interpolation_sigma": 1.5,
        "data_root": str(data_root),
        "output_directory": str(output_dir),
        "total_experiments_attempted": total_attempted,
        "successful_experiments": successful,
        "success_rate": round(successful / total_attempted * 100, 1)
                        if total_attempted > 0 else 0,
        "experiments": [exp[2] for exp in EXPERIMENTS],
        "experiments_processed": experiments_processed,
        "device_used": device_str,
        "generated_files": generated_files,
        "total_generated_files": len(generated_files),
    }

    if failed_experiments:
        report["failed_experiments"] = [
            {"experiment": name, "reason": reason}
            for name, reason in failed_experiments
        ]

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description='CREPE Psychophysics Evaluator'
    )
    parser.add_argument(
        '--data_root', type=str, default=DEFAULT_DATA_ROOT,
        help='Directory containing psychophysics HDF5 files'
    )
    parser.add_argument(
        '--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help='Directory for output JSON prediction files'
    )
    parser.add_argument(
        '--step_size', type=int, default=10,
        help='Frame shift in milliseconds (default: 10)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for inference: cuda or cpu'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Number of trials per GPU batch (default: 256)'
    )
    parser.add_argument(
        '--experiments', type=str, default=None,
        help='逗号分隔的实验名过滤（如 ERB_B6 或 ERB_B1,ERB_B2）;'
             '默认 None = 全部 6 个数据集'
    )

    args = parser.parse_args()

    # 策略 B 副本新增: 按数据集过滤（支持 B6 单独试运行/按数据集恢复）
    experiment_filter = None
    if args.experiments:
        experiment_filter = set(
            s.strip() for s in args.experiments.split(',') if s.strip())

    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 60)
    print("CREPE Psychophysics Evaluator")
    print("=" * 60)
    print(f"Data root:    {args.data_root}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Step size:    {args.step_size} ms")
    print(f"Device:       {args.device}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Model:        CREPE (full) via torchcrepe")
    print(f"Decoder:      weighted_argmax")
    print(f"Hop length:   {int(TARGET_SR * args.step_size / 1000)} samples")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Process each experiment
    # ------------------------------------------------------------------
    print("\n[1/2] Processing experiments...")

    generated_files = []
    experiments_processed = []
    failed_experiments = []
    total_errors = 0

    for hdf5_filename, audio_key, experiment_name in EXPERIMENTS:
        if experiment_filter is not None and experiment_name not in experiment_filter:
            print(f"  [SKIP] {experiment_name} (not in --experiments filter)")
            continue
        try:
            result, n_errors = process_experiment(
                hdf5_filename=hdf5_filename,
                audio_key=audio_key,
                experiment_name=experiment_name,
                data_root=args.data_root,
                step_size=args.step_size,
                device=args.device,
                batch_size=args.batch_size,
                output_dir=args.output_dir,  # v2: pass output_dir for .npz save
            )

            if result is None:
                reason = n_errors if isinstance(n_errors, str) else "Unknown error"
                failed_experiments.append((experiment_name, reason))
                print(f"  ❌ Failed to process {experiment_name}: {reason}")
                continue

            total_errors += n_errors

            # Save predictions
            output_filename = f"predictions_crepe_{experiment_name}.json"
            output_path = os.path.join(args.output_dir, output_filename)

            with open(output_path, 'w') as f:
                json.dump(result, f, cls=NumpyEncoder)

            generated_files.append(output_filename)
            experiments_processed.append(experiment_name)

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✅ Saved: {output_path} ({file_size:.1f} MB)")

        except Exception as e:
            print(f"  ❌ Error processing {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_experiments.append((experiment_name, str(e)))

    # ------------------------------------------------------------------
    # Generate evaluation report
    # ------------------------------------------------------------------
    print(f"\n[2/2] Generating evaluation report...")

    report = generate_evaluation_report(
        generated_files=generated_files,
        experiments_processed=experiments_processed,
        failed_experiments=failed_experiments,
        device_str=args.device,
        data_root=args.data_root,
        output_dir=args.output_dir,
        step_size=args.step_size,
    )

    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)

    print(f"  ✅ Report saved: {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("✅ Evaluation complete!")
    print(f"   Experiments processed: {len(experiments_processed)}/{len(EXPERIMENTS)}")
    if failed_experiments:
        print(f"   Failed experiments: {len(failed_experiments)}")
        for name, reason in failed_experiments:
            print(f"     - {name}: {reason}")
    print(f"   Total trial errors: {total_errors}")
    print(f"   Output directory: {args.output_dir}")
    for fname in generated_files:
        fpath = os.path.join(args.output_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"     - {fname} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
