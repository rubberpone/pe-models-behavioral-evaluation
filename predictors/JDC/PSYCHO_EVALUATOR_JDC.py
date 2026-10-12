#!/usr/bin/env python3
"""
PSYCHO_EVALUATOR_JDC.py — Generate JDC F0 predictions for all five
psychophysical experiments.

Loads each HDF5 stimulus file, runs the pre-trained JDC model on every
stimulus waveform, extracts frame-level pitch probabilities via the
dual-output CRNN, converts 722-dim activations to 700-dim probability
distributions via Gaussian interpolation (distribution→distribution),
and saves structured JSON files matching the exact MERT reference format.

Usage:
    source /home/zyf/anaconda3/etc/profile.d/conda.sh
    conda activate JDC_psychophysics
    cd /data/zyf/JDC_psychophysics
    python PSYCHO_EVALUATOR_JDC.py \
        --data_root ../MERT_psychophysics/data/dataset_psychophysics \
        --output_dir ./JDC_psy_predictions \
        --device cuda

Changes:
  v2 (2026-08-05):
    - Added raw activation storage: each stimulus's 722-dim activation
      (1 unvoiced + 721 pitch bins) is collected before the Gaussian
      722->700 interpolation and saved as a compressed .npz file alongside
      each JSON prediction.
      .npz keys: raw_activations (float32), freqs_source (float32),
      model_name (str), experiment_name (str), sigma_used (float).
      Only successful stimuli are stored (error stimuli are excluded),
      matching the non-None entries of 'f0_label:labels_pred' in the JSON.

  v2.1 (2026-08-05):
    - Expanded .npz output: added per-stimulus model raw outputs
      f0_hz (float32, scalar Hz from JDC per-frame decoding) and
      non_voice_prob (float32, activation_722[0] before 722->700 mapping),
      aligned 1:1 with raw_activations rows (successful stimuli only).
      JSON output and interpolation sigma unchanged.
"""

import os
import sys
import json
import argparse
import traceback
import numpy as np
import torch
import h5py
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jdc_predict import JDCPredictor
from utils.probability_utils import jdc_to_700
from utils.audio_utils import (
    load_audio_from_hdf5,
    get_hdf5_num_stimuli,
    ensure_mono,
    resample_audio,
)


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(PROJECT_ROOT), '..', '..', 'data'
)

DEFAULT_REFERENCE_DIR = os.path.join(
    os.path.dirname(PROJECT_ROOT), 'MERT_psychophysics',
    'mert_psy_predictions_example'
)

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'JDC_psy_predictions')

DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, 'pretrained_models', 'jdc_weights.pt')
DEFAULT_MEAN_PATH = os.path.join(
    PROJECT_ROOT, 'melodyExtraction_JDC-master', 'x_data_mean_total_31.npy'
)
DEFAULT_STD_PATH = os.path.join(
    PROJECT_ROOT, 'melodyExtraction_JDC-master', 'x_data_std_total_31.npy'
)

N_CLASSES = 700

# ============================================================================
# Experiment definitions
# ============================================================================

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

# bez2018 peripheral model keys to strip from reference templates
BEZ2018_KEYS = [
    'config_bez2018model/IhcLowPass_cutoff',
    'config_bez2018model/IhcLowPass_order',
    'config_bez2018model/bandwidth_scale_factor',
    'config_bez2018model/buffer_end_dur',
    'config_bez2018model/buffer_start_dur',
    'config_bez2018model/cihc',
    'config_bez2018model/cohc',
    'config_bez2018model/implnt',
    'config_bez2018model/max_spikes_per_train',
    'config_bez2018model/nervegram_dur',
    'config_bez2018model/noiseType',
    'config_bez2018model/num_spike_trains',
    'config_bez2018model/pin_dBSPL_flag',
    'config_bez2018model/species',
    'config_bez2018model/spont',
    'config_bez2018model/synapseMode',
    'config_bez2018model/tabs',
    'config_bez2018model/trel',
    'nervegram_spike_tensor_fs',
    'pin_dBSPL',
    'pin_fs',
]


# ============================================================================
# JSON encoder
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ============================================================================
# Reference template loading
# ============================================================================

def load_reference_template(reference_dir, experiment_name):
    """Load the MERT reference prediction JSON as a structural template."""
    ref_path = os.path.join(reference_dir, f'predictions_mert_{experiment_name}.json')
    if os.path.exists(ref_path):
        with open(ref_path, 'r') as f:
            return json.load(f)
    print(f'  [WARN] Reference template not found: {ref_path}')
    return {}


# ============================================================================
# HDF5 metadata loading
# ============================================================================

def load_hdf5_metadata(hdf5_path):
    """
    Load all 1D metadata datasets from HDF5 file.
    Returns dict of {key: numpy_array}.
    """
    metadata = {}
    with h5py.File(hdf5_path, 'r') as f:
        for key in f.keys():
            item = f[key]
            if hasattr(item, 'shape') and len(item.shape) == 1 and item.shape[0] > 1:
                metadata[key] = item[:]
    return metadata


# ============================================================================
# Per-experiment processing
# ============================================================================

def process_experiment(
    hdf5_path,
    audio_key,
    experiment_name,
    reference_dir,
    predictor,
    device,
    batch_size=256,
):
    """
    Process one psychophysics experiment using JDC with batched inference.

    Workflow:
      1. Load all audio from HDF5
      2. Precompute spectrogram features for all stimuli
      3. Batch model inference on GPU
      4. Post-process: 722 → 700 Gaussian interpolation

    Returns:
        dict with prediction arrays and metadata, ready for JSON serialization.
    """
    if not os.path.exists(hdf5_path):
        print(f'  [ERROR] HDF5 file not found: {hdf5_path}')
        return None, None, None, None

    print(f'\n{"=" * 60}')
    print(f'Processing: {experiment_name}')
    print(f'  HDF5:     {hdf5_path}')
    print(f'  Audio key: {audio_key}')

    # ------------------------------------------------------------------
    # 1. Load audio data
    # ------------------------------------------------------------------
    try:
        with h5py.File(hdf5_path, 'r') as f:
            parts = audio_key.split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            audio_all = obj[:]  # (N, 4800) — 150 ms at 32 kHz
            n_stimuli = audio_all.shape[0]
    except Exception as e:
        print(f'  [ERROR] Failed to read "{audio_key}" from {hdf5_path}: {e}')
        return None, None, None, None

    print(f'  Stimuli:  {n_stimuli}  |  Audio shape: {audio_all.shape}')

    # ------------------------------------------------------------------
    # 2. Load reference template for structure
    # ------------------------------------------------------------------
    template = load_reference_template(reference_dir, experiment_name)

    # ------------------------------------------------------------------
    # 3. Precompute features for all stimuli
    # ------------------------------------------------------------------
    print(f'  Precomputing spectrogram features...')
    from utils.audio_utils import extract_jdc_feature, get_num_original_frames

    all_features = []       # list of (31, 513, 1) arrays
    n_original_frames_list = []

    for i in tqdm(range(n_stimuli), desc='  Feature extraction', unit='stim'):
        try:
            waveform = audio_all[i].copy()
            waveform = ensure_mono(waveform)
            waveform_8k = resample_audio(waveform, orig_sr=32000, target_sr=8000)

            n_orig = get_num_original_frames(
                len(waveform_8k), sr=8000, hop_length=80, n_fft=1024
            )

            feat = extract_jdc_feature(
                waveform_8k, sr=8000, n_frames=31,
                hop_length=80, n_fft=1024,
                mean_path=predictor.mean_path,
                std_path=predictor.std_path,
            )  # (1, 31, 513, 1)

            all_features.append(feat[0])  # (31, 513, 1)
            n_original_frames_list.append(n_orig)

        except Exception as e:
            tqdm.write(f'  [ERROR] Feature extraction stim {i}: {e}')
            all_features.append(None)
            n_original_frames_list.append(0)

    n_valid_features = sum(1 for f in all_features if f is not None)
    print(f'  Features computed: {n_valid_features}/{n_stimuli}')

    # ------------------------------------------------------------------
    # 4. Batch model inference
    # ------------------------------------------------------------------
    print(f'  Running JDC in batches (batch_size={batch_size})...')

    all_pred_labels = [None] * n_stimuli
    all_pred_probs = [None] * n_stimuli
    errors = []
    raw_activations_list = []  # (722,) raw activations (successful stimuli only)
    all_f0_hz = []             # scalar f0_hz per successful stimulus (Hz)
    all_uv_prob = []           # non-voice probability = activation_722[0]

    # Collect valid feature indices
    valid_indices = [i for i, f in enumerate(all_features) if f is not None]

    for batch_start in tqdm(range(0, len(valid_indices), batch_size),
                            desc='  JDC inference', unit='batch'):
        batch_indices = valid_indices[batch_start:batch_start + batch_size]

        # Stack features: (B, 31, 513, 1)
        batch_features = np.stack([all_features[i] for i in batch_indices], axis=0)
        batch_n_orig = [n_original_frames_list[i] for i in batch_indices]

        try:
            batch_results = predictor.predict_batch_from_features(
                batch_features, batch_n_orig
            )

            for idx, result in zip(batch_indices, batch_results):
                try:
                    # Collect raw activation before Gaussian interpolation
                    raw_activations_list.append(result['activation_722'].copy())
                    # Collect model raw outputs (aligned with raw_activations rows)
                    all_f0_hz.append(float(result.get('f0_hz', np.nan)))
                    all_uv_prob.append(float(result['activation_722'][0]))

                    # 722 → 700 Gaussian interpolation
                    labels_pred, probs_700 = jdc_to_700(
                        result['activation_722'], sigma=1.5
                    )
                    all_pred_labels[idx] = labels_pred
                    all_pred_probs[idx] = probs_700.tolist()

                except Exception as e:
                    tqdm.write(f'  [ERROR] 722→700 mapping stim {idx}: {e}')
                    all_pred_labels[idx] = None
                    all_pred_probs[idx] = None
                    errors.append(idx)

        except Exception as e:
            tqdm.write(f'  [ERROR] Batch inference [{batch_start}:{batch_start+batch_size}]: {e}')
            traceback.print_exc()
            for idx in batch_indices:
                all_pred_labels[idx] = None
                all_pred_probs[idx] = None
                errors.append(idx)

    # ------------------------------------------------------------------
    # 4. Build output dict from template
    # ------------------------------------------------------------------
    if template:
        result = dict(template)
    else:
        # Build from HDF5 metadata if no template available
        result = {}
        metadata = load_hdf5_metadata(hdf5_path)
        for key, arr in metadata.items():
            if len(arr) == n_stimuli:
                result[key] = arr.tolist()
            else:
                result[key] = [arr.tolist()]

    # Remove bez2018 peripheral-model config keys
    for key in BEZ2018_KEYS:
        result.pop(key, None)

    # Update prediction arrays
    result['f0_label:labels_pred'] = all_pred_labels
    result['f0_label:probs_out'] = all_pred_probs

    # Update metadata
    result['experiment_name'] = experiment_name
    result['model_name'] = 'JDC'
    result['timestamp'] = datetime.now().isoformat()

    # Add JDC-specific config
    result['jdc_config'] = {
        'model': 'JDC (Joint Detection and Classification)',
        'architecture': 'CRNN with Bi-LSTM (dual-branch: pitch + voice detection)',
        'implementation': 'PyTorch reimplementation',
        'parameters': '~4.2M',
        'input_sr': 8000,
        'stft_params': {
            'n_fft': 1024,
            'hop_length': 80,
            'window': 'hann',
            'win_length': 1024,
            'n_freq_bins': 513,
            'n_frames_per_chunk': 31,
        },
        'input_norm': 'Z-score (pre-computed training statistics)',
        'log_compression': 'power_to_db (10×log10, ref=np.max)',
        'aggregation': 'median (frequency) + element-wise mean (722-dim activation)',
        'output_range': '73.4–987.8 Hz (D2–B5, 722 bins = 1 non-voice + 721 pitch)',
        'interpolation': 'Gaussian (distribution→distribution, sigma=1.5 bins ≈ 9.3 cents)',
        'non_voice_class': 'Index 0 excluded from 721→700 mapping via activation_722[1:]',
        'post_processing': '5-sample median filter',
        'n_output_classes': N_CLASSES,
    }

    if errors:
        print(f'  ⚠️  {len(errors)} stimuli had errors and were set to None')

    n_valid = sum(1 for v in all_pred_labels if v is not None)
    print(f'  ✅ Done — {n_valid}/{n_stimuli} valid predictions')
    return result, raw_activations_list, all_f0_hz, all_uv_prob


# ============================================================================
# Evaluation report
# ============================================================================

def generate_evaluation_report(
    generated_files,
    experiments_processed,
    total_stimuli,
    errors_by_experiment,
    data_root,
    output_dir,
    device_str,
):
    """Generate the evaluation_report.json file."""
    return {
        'evaluation_date': datetime.now().isoformat(),
        'model_name': 'JDC',
        'implementation': 'PyTorch reimplementation of JDC (CRNN with Bi-LSTM)',
        'input_sr': 8000,
        'stft_params': {
            'n_fft': 1024,
            'hop_length': 80,
            'window': 'hann',
            'n_frames': 31,
            'n_freq_bins': 513,
        },
        'aggregation': 'median (frequency) + element-wise mean (722-dim activation)',
        'interpolation_sigma': 1.5,
        'interpolation_method': 'Gaussian interpolation (721 pitch bins → 700 bins, excluding non-voice class)',
        'non_voice_class_handling': 'Index 0 is non-voice (pitch=0); excluded from 721→700 interpolation via activation_722[1:]',
        'data_path': str(os.path.abspath(data_root)),
        'output_directory': str(os.path.abspath(output_dir)),
        'device_used': device_str,
        'total_stimuli_processed': total_stimuli,
        'experiments': experiments_processed,
        'num_experiments': len(experiments_processed),
        'errors_by_experiment': errors_by_experiment,
        'generated_files': generated_files,
        'total_generated_files': len(generated_files),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='JDC Psychophysics Evaluator — MERT-compatible output format'
    )
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--reference_dir', type=str, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--mean_path', type=str, default=DEFAULT_MEAN_PATH)
    parser.add_argument('--std_path', type=str, default=DEFAULT_STD_PATH)

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
    print('=' * 60)
    print('JDC Psychophysics Evaluator (MERT-compatible format)')
    print('=' * 60)

    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir)
    reference_dir = os.path.abspath(args.reference_dir)

    print(f'Data root:      {data_root}')
    print(f'Output dir:     {output_dir}')
    print(f'Reference dir:  {reference_dir}')
    print(f'Output classes: {N_CLASSES}')

    os.makedirs(output_dir, exist_ok=True)

    # Set up device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        device_str = 'cuda'
    else:
        if args.device == 'cuda':
            print('CUDA not available, falling back to CPU')
        device = torch.device('cpu')
        device_str = 'cpu'

    print(f'Device:         {device}')

    # ------------------------------------------------------------------
    # Initialize predictor
    # ------------------------------------------------------------------
    print(f'\nInitializing JDC predictor...')
    predictor = JDCPredictor(
        checkpoint_path=args.checkpoint,
        device=device,
        mean_path=args.mean_path,
        std_path=args.std_path,
    )
    print(f'  Model loaded from: {args.checkpoint}')
    print(f'  Mean stats from:   {args.mean_path}')
    print(f'  Std stats from:    {args.std_path}')

    # ------------------------------------------------------------------
    # Process experiments
    # ------------------------------------------------------------------
    print(f'\n[1/2] Processing experiments with JDC...')

    generated_files = []
    experiments_processed = []
    errors_by_experiment = {}
    total_stimuli = 0

    for hdf5_filename, audio_key, experiment_name in EXPERIMENTS:
        if experiment_filter is not None and experiment_name not in experiment_filter:
            print(f"  [SKIP] {experiment_name} (not in --experiments filter)")
            continue
        hdf5_path = os.path.join(data_root, hdf5_filename)

        try:
            ret = process_experiment(
                hdf5_path=hdf5_path,
                audio_key=audio_key,
                experiment_name=experiment_name,
                reference_dir=reference_dir,
                predictor=predictor,
                device=device,
            )

            if ret is None:
                print(f'  ❌ Failed to process {experiment_name}')
                errors_by_experiment[experiment_name] = 'Fatal error — no output generated'
                continue

            result, raw_activations_list, all_f0_hz, all_uv_prob = ret

            n_stim = len(result.get('f0_label:labels_pred', []))
            total_stimuli += n_stim

            # Count errors (None entries)
            n_errors = sum(1 for v in result.get('f0_label:labels_pred', []) if v is None)
            errors_by_experiment[experiment_name] = n_errors

            output_filename = f'predictions_jdc_{experiment_name}.json'
            output_path = os.path.join(output_dir, output_filename)

            print(f'  Writing {output_filename}...')
            with open(output_path, 'w') as f:
                json.dump(result, f, cls=NumpyEncoder)

            # Save raw activations (collected before Gaussian interpolation) as .npz
            npz_path = output_path.replace('.json', '_raw.npz')
            midi_values = np.concatenate([[38.0], 38.0 + np.arange(1, 722) / 16])
            np.savez_compressed(
                npz_path,
                raw_activations=np.asarray(raw_activations_list, dtype=np.float32),
                freqs_source=(440.0 * (2 ** ((midi_values - 69.0) / 12))).astype(np.float32),
                f0_hz=np.asarray(all_f0_hz, dtype=np.float32),
                non_voice_prob=np.asarray(all_uv_prob, dtype=np.float32),
                model_name='JDC',
                experiment_name=experiment_name,
                sigma_used=1.5,
            )

            generated_files.append(output_filename)
            experiments_processed.append(experiment_name)

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f'  ✅ Saved: {output_filename}  ({size_mb:.2f} MB, {n_stim} stimuli)')

        except Exception as e:
            print(f'  ❌ Fatal error processing {experiment_name}: {e}')
            traceback.print_exc()
            errors_by_experiment[experiment_name] = f'Fatal: {str(e)}'

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f'\n[2/2] Generating evaluation report...')

    report = generate_evaluation_report(
        generated_files=generated_files,
        experiments_processed=experiments_processed,
        total_stimuli=total_stimuli,
        errors_by_experiment=errors_by_experiment,
        data_root=data_root,
        output_dir=output_dir,
        device_str=device_str,
    )

    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)
    print(f'  ✅ Report saved: {report_path}')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}')
    print('JDC Evaluation Complete!')
    print(f'   Experiments:  {len(experiments_processed)}/{len(EXPERIMENTS)}')
    print(f'   Total stimuli: {total_stimuli}')
    print(f'   Output dir:   {output_dir}')
    for fname in generated_files:
        fpath = os.path.join(output_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f'     - {fname}  ({size_mb:.2f} MB)')

    if errors_by_experiment:
        total_errors = sum(
            v for v in errors_by_experiment.values()
            if isinstance(v, int)
        )
        if total_errors > 0:
            print(f'\n  ⚠️  {total_errors} total stimulus-level errors across all experiments')


if __name__ == '__main__':
    main()
