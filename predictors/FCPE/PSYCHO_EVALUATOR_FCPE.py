#!/usr/bin/env python3
"""
PSYCHO_EVALUATOR_FCPE.py — Generate FCPE predictions for all five
psychophysical experiments.

Loads the official FCPE model (torchfcpe), processes each stimulus through
the end-to-end pipeline (raw waveform → internal log-mel → CFNaiveMelPE →
local argmax decoding → frame aggregation → 360→700 Gaussian interpolation),
and saves prediction JSON files matching the reference format.

Usage:
    conda activate FCPE_psychophysics
    python PSYCHO_EVALUATOR_FCPE.py \
        --data_root /data/zyf/MERT_psychophysics/data/dataset_psychophysics \
        --output_dir ./FCPE_psy_predictions \
        --device cuda

=============================================================================
VERSION HISTORY
=============================================================================
v2 (2026-08-05): ADD raw activation storage for sigma sensitivity analysis
  - Added np.savez_compressed() call to store raw 360-dim sigmoid activations
    alongside each prediction JSON (as *_raw.npz files).
  - Raw activations are saved BEFORE Gaussian interpolation (sigma=1.5),
    enabling subsequent sigma scanning without re-running model inference.
  - .npz keys: raw_activations (n_segments, 360) float32, freqs_source (360,)
    float32, model_name, experiment_name, sigma_used.
  - Error segments (labels=None in JSON) are stored as NaN rows in
    raw_activations to keep array alignment with the JSON.
  - Existing JSON output format is UNCHANGED.
  - Backup: backup/PSYCHO_EVALUATOR_FCPE.py.bak
v3 (2026-08-05): ADD model raw outputs to .npz
  - .npz now also stores: f0_hz (n_segments,) float32 (from all_f0_hz,
    already collected in the trial loop as a numpy array).
  - Error segments now set f0_hz to NaN (previously left at 0.0) so the
    .npz aligns with the JSON (labels=null). JSON still reports None for
    error entries — output format UNCHANGED.
  - Per-frame unvoiced info is not recoverable from the aggregated 360-dim
    activations, so no unvoiced_ratio key is stored.
  - All numeric arrays are float32.
  - JSON output format and interpolation sigma values are UNCHANGED.
  - Backup: backup/PSYCHO_EVALUATOR_FCPE.py.bak
=============================================================================
"""

import os
import sys
import json
import argparse
import traceback
import numpy as np
import torch
import h5py
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fcpe_predict import FCPEPredictor
from utils.audio_utils import load_audio_from_hdf5, ensure_mono
from utils.probability_utils import fcpe_to_700


# ============================================================================
# Configuration
# ============================================================================

# Experiment definitions: (hdf5_filename, audio_key, display_name)
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

# Reference template directory (for structural key compatibility)
REFERENCE_DIR = "/data/zyf/MERT_psychophysics/mert_psy_predictions_example"

# HDF5 datasets to copy into the output JSON (flattened)
# These are copied from the HDF5 file to provide ground truth and metadata
HDF5_SKIP_KEYS = {'tone_in_noise', 'stimuli', 'config_tone'}  # Skip large audio data and config groups


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
# Helper Functions
# ============================================================================

def load_reference_template(experiment_name: str, ref_dir: str) -> dict:
    """
    Load the MERT reference prediction JSON as a structural template.

    Args:
        experiment_name: e.g., "BernsteinOxenham2005"
        ref_dir: Path to the directory containing reference prediction files.

    Returns:
        dict or empty dict if not found.
    """
    ref_path = os.path.join(ref_dir, f"predictions_mert_{experiment_name}.json")
    if os.path.exists(ref_path):
        with open(ref_path, 'r') as f:
            return json.load(f)
    print(f"  [WARNING] No reference template found at {ref_path}")
    return {}


def get_hdf5_datasets(hdf5_path: str) -> dict:
    """
    Extract all datasets from an HDF5 file into a flattened dict.

    Nested groups are flattened with '/' separator (e.g., 'config_tone/fs').
    Large audio datasets and config groups are skipped.

    Args:
        hdf5_path: Path to the HDF5 file.

    Returns:
        dict mapping key → list (converted from numpy arrays).
    """
    result = {}

    def _extract(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Skip audio data (large) and config groups
            parts = name.split('/')
            if parts[0] in HDF5_SKIP_KEYS:
                return
            try:
                data = obj[:]
                # Handle scalar datasets
                if data.ndim == 0:
                    result[name] = [data.item()]
                else:
                    result[name] = data.tolist()
            except Exception:
                pass  # Skip unreadable datasets silently

    with h5py.File(hdf5_path, 'r') as f:
        f.visititems(_extract)

    return result


def process_experiment(predictor: FCPEPredictor, hdf5_path: str,
                       audio_key: str, experiment_name: str,
                       ref_dir: str, batch_size: int = 64) -> dict:
    """
    Process one psychophysics experiment: load audio, run FCPE inference,
    and return prediction results in the reference-compatible format.

    Args:
        predictor: FCPEPredictor instance.
        hdf5_path: Path to the HDF5 file.
        audio_key: Key for audio data within the HDF5 (e.g., 'tone_in_noise').
        experiment_name: Display name (e.g., 'BernsteinOxenham2005').
        batch_size: Number of segments to process in a CPU batch (for progress).

    Returns:
        dict with predictions in reference format, or None on error.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {experiment_name}")
    print(f"  HDF5: {hdf5_path}")
    print(f"  Audio key: {audio_key}")

    # ------------------------------------------------------------------
    # Load audio data
    # ------------------------------------------------------------------
    audio = load_audio_from_hdf5(hdf5_path, audio_key)
    if audio is None:
        return None

    n_segments = audio.shape[0]
    orig_sr = 32000  # All stimuli are 32 kHz
    print(f"  Segments: {n_segments} | Shape: {audio.shape} | SR: {orig_sr} Hz")

    # ------------------------------------------------------------------
    # Load reference template for structural compatibility
    # ------------------------------------------------------------------
    template = load_reference_template(experiment_name, ref_dir)

    # ------------------------------------------------------------------
    # Run FCPE inference (using numpy arrays for memory efficiency)
    # ------------------------------------------------------------------
    # Pre-allocate numpy arrays to avoid Python list-of-lists overhead
    all_pred_labels = np.zeros(n_segments, dtype=np.int32)
    all_pred_probs = np.zeros((n_segments, 700), dtype=np.float32)
    all_f0_hz = np.zeros(n_segments, dtype=np.float32)
    error_mask = np.zeros(n_segments, dtype=bool)
    errors = []
    raw_activations = []  # v2: store raw 360-dim activations before interpolation

    cent_table = predictor.get_cent_table()

    # Process each audio segment individually
    for idx in tqdm(range(n_segments), desc=f"  FCPE inference ({experiment_name})"):
        try:
            # Extract single segment
            segment = audio[idx]  # shape: (4800,) at 32 kHz
            segment = ensure_mono(segment)

            # FCPE predictor handles everything
            f0_hz, activation_360 = predictor.predict(segment, sr=orig_sr)

            # v2: collect raw activation BEFORE interpolation
            raw_activations.append(activation_360.copy())

            # 360 → 700 Gaussian interpolation
            labels_pred, probs_700 = fcpe_to_700(activation_360, sigma=1.5,
                                                 cent_table=cent_table)

            all_pred_labels[idx] = labels_pred
            all_pred_probs[idx] = probs_700.astype(np.float32)
            all_f0_hz[idx] = float(f0_hz)

        except Exception as e:
            errors.append(idx)
            error_mask[idx] = True
            # v2: error trials not stored — fill with NaN to keep alignment
            raw_activations.append(np.full(360, np.nan, dtype=np.float32))
            # v3: NaN for failed trials in .npz f0_hz (JSON still uses None)
            all_f0_hz[idx] = np.nan
            if len(errors) <= 5:
                print(f"\n  [ERROR] Segment {idx}: {e}")

    # ------------------------------------------------------------------
    # Build output dict
    # ------------------------------------------------------------------
    if template:
        # Start from template to preserve exact key structure
        result = dict(template)
    else:
        # Build from scratch using HDF5 datasets
        result = get_hdf5_datasets(hdf5_path)

    # Convert numpy arrays to lists for JSON serialization
    # Mark error entries as null
    labels_pred_list = all_pred_labels.tolist()
    f0_hz_list = all_f0_hz.tolist()
    probs_out_list = all_pred_probs.tolist()

    for idx in errors:
        labels_pred_list[idx] = None
        f0_hz_list[idx] = None
        probs_out_list[idx] = None

    # Update prediction arrays
    result["f0_label:labels_pred"] = labels_pred_list
    result["f0_label:probs_out"] = probs_out_list

    # Add FCPE-specific outputs
    result["f0_hz"] = f0_hz_list

    # Update metadata
    result["experiment_name"] = experiment_name
    result["model_name"] = "FCPE"
    result["timestamp"] = datetime.now().isoformat()

    # Remove MERT-specific keys that don't apply to FCPE
    # (these were from the nervegram pipeline, not applicable here)
    mert_specific_keys = [
        'nervegram_spike_tensor_fs', 'pin_dBSPL', 'pin_fs',
    ]
    for k in mert_specific_keys:
        result.pop(k, None)

    # Remove cochlear model config keys from template (not applicable to FCPE)
    keys_to_remove = [k for k in result if k.startswith('config_bez2018model/')]
    for k in keys_to_remove:
        result.pop(k, None)

    # Report errors
    if errors:
        print(f"  ⚠️  {len(errors)}/{n_segments} segments had errors and were set to null")

    return result, len(errors), raw_activations, all_f0_hz  # v3: + all_f0_hz


def generate_evaluation_report(generated_files: list, experiment_results: list,
                               device_str: str, data_root: str,
                               output_dir: str) -> dict:
    """
    Generate evaluation_report.json.

    Args:
        generated_files: List of output filenames.
        experiment_results: List of (exp_name, n_total, n_errors) tuples.
        device_str: Device used for inference.
        data_root: Path to data directory.
        output_dir: Path to output directory.

    Returns:
        dict: Report data.
    """
    total_attempted = len(EXPERIMENTS)
    successful = len(generated_files)

    # Count total stimuli and errors
    total_stimuli = sum(n for _, n, _ in experiment_results)
    total_errors = sum(e for _, _, e in experiment_results)

    report = {
        "model_name": "FCPE",
        "implementation": (
            "FCPE (CFNaiveMelPE, Conformer-based) from official torchfcpe package "
            "(github.com/CNChTu/FCPE)"
        ),
        "input_type": (
            "raw waveform (model internally extracts log-mel/STFT features)"
        ),
        "feature_params": {
            "sr": 16000,
            "n_fft": 1024,
            "win_size": 1024,
            "hop_size": 160,
            "note": "exact bins depend on bundled checkpoint config (mel or STFT)"
        },
        "decoding": "local weighted average (local argmax, ±4 bins, in cent space)",
        "aggregation": (
            "median of frequencies, mean of sigmoid activations "
            "(custom — not part of official FCPE)"
        ),
        "interpolation_sigma": 1.5,
        "interpolation_method": (
            "Gaussian interpolation (360→700), no forced normalization (sigmoid source)"
        ),
        "activation_type": "sigmoid (per-bin binary classification, NOT softmax)",
        "cent_reference": "10 Hz",
        "evaluation_date": datetime.now().isoformat(),
        "data_path": data_root,
        "output_directory": output_dir,
        "device_used": device_str,
        "total_experiments_attempted": total_attempted,
        "successful_experiments": successful,
        "total_stimuli_processed": total_stimuli,
        "total_stimulus_errors": total_errors,
        "success_rate": round(successful / total_attempted * 100, 1) if total_attempted > 0 else 0,
        "experiments": [exp[2] for exp in EXPERIMENTS],
        "experiment_details": [
            {
                "name": name,
                "n_stimuli": n_total,
                "n_errors": n_errors,
                "status": "success" if n_errors == 0 else f"{n_errors} errors"
            }
            for name, n_total, n_errors in experiment_results
        ],
        "generated_files": generated_files,
        "total_generated_files": len(generated_files),
    }

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="FCPE Psychophysics Evaluator — Generate F0 predictions for all experiments"
    )
    parser.add_argument(
        '--data_root', type=str,
        default='/data/zyf/MERT_psychophysics/data/dataset_psychophysics',
        help='Path to directory containing HDF5 stimulus files'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='./FCPE_psy_predictions',
        help='Output directory for prediction JSON files'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for inference: cuda or cpu'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='(Reserved) Batch size for inference progress display'
    )
    parser.add_argument(
        '--reference_dir', type=str,
        default=None,
        help='Path to reference prediction examples for structural templates '
             '(default: MERT example directory)'
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


    # Use custom reference dir if provided
    ref_dir = args.reference_dir if args.reference_dir else REFERENCE_DIR

    print("=" * 60)
    print("FCPE Psychophysics Evaluator")
    print("=" * 60)
    print(f"Data root:    {args.data_root}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Device:       {args.device}")
    print(f"Reference:    {ref_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load FCPE model
    # ------------------------------------------------------------------
    print("\n[1/3] Loading FCPE model...")
    predictor = FCPEPredictor(device=args.device)

    # ------------------------------------------------------------------
    # Process each experiment
    # ------------------------------------------------------------------
    print("\n[2/3] Processing experiments...")

    generated_files = []
    experiment_results = []  # (name, n_total, n_errors)

    for hdf5_filename, audio_key, experiment_name in EXPERIMENTS:
        if experiment_filter is not None and experiment_name not in experiment_filter:
            print(f"  [SKIP] {experiment_name} (not in --experiments filter)")
            continue
        hdf5_path = os.path.join(args.data_root, hdf5_filename)

        if not os.path.exists(hdf5_path):
            print(f"\n  [ERROR] HDF5 file not found: {hdf5_path}")
            experiment_results.append((experiment_name, 0, -1))
            continue

        try:
            result, n_errors, raw_activations, all_f0_hz = process_experiment(
                predictor=predictor,
                hdf5_path=hdf5_path,
                audio_key=audio_key,
                experiment_name=experiment_name,
                ref_dir=ref_dir,
                batch_size=args.batch_size
            )

            if result is None:
                print(f"  ❌ Failed to process {experiment_name}")
                experiment_results.append((experiment_name, 0, -1))
                continue

            # Count total stimuli
            pred_labels = result.get("f0_label:labels_pred", [])
            n_total = len(pred_labels)
            experiment_results.append((experiment_name, n_total, n_errors))

            # Save predictions
            output_filename = f"predictions_fcpe_{experiment_name}.json"
            output_path = output_dir / output_filename

            print(f"  Saving to: {output_path} ({n_total} entries)...")
            with open(output_path, 'w') as f:
                json.dump(result, f, cls=NumpyEncoder)

            # v2: save raw 360-dim activations as .npz (before interpolation)
            cent_table = predictor.get_cent_table()  # (360,) cents relative to 10 Hz
            freqs_source = 10.0 * (2 ** (cent_table / 1200))
            raw_path = str(output_path).replace('.json', '_raw.npz')
            np.savez_compressed(
                raw_path,
                raw_activations=np.array(raw_activations, dtype=np.float32),
                freqs_source=freqs_source.astype(np.float32),
                f0_hz=all_f0_hz.astype(np.float32),  # v3: raw model f0 (NaN on error)
                model_name="FCPE",
                experiment_name=experiment_name,
                sigma_used=1.5,
            )

            generated_files.append(output_filename)
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Saved: {output_filename} ({file_size_mb:.1f} MB)")

        except Exception as e:
            print(f"  ❌ Error processing {experiment_name}: {e}")
            traceback.print_exc()
            experiment_results.append((experiment_name, 0, -1))

    # ------------------------------------------------------------------
    # Generate evaluation report
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating evaluation report...")

    device_str = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    report = generate_evaluation_report(
        generated_files=generated_files,
        experiment_results=experiment_results,
        device_str=device_str,
        data_root=args.data_root,
        output_dir=str(output_dir),
    )

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)

    print(f"  ✅ Report saved: {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("✅ Evaluation complete!")
    print(f"   Generated {len(generated_files)}/{len(EXPERIMENTS)} prediction files")
    print(f"   Output directory: {output_dir}")
    for fname in generated_files:
        fpath = output_dir / fname
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"     - {fname} ({size_mb:.1f} MB)")

    # Print error summary
    total_errors = sum(e for _, _, e in experiment_results)
    if total_errors > 0:
        print(f"\n   ⚠️  Total stimulus errors: {total_errors}")
    else:
        print(f"\n   ✅ All stimuli processed successfully (0 errors)")


if __name__ == "__main__":
    main()
