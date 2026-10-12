#!/usr/bin/env python3
"""
PSYCHO_EVALUATOR_PENN.py — Generate PENN F0 predictions for all five
psychophysical experiments.

Loads each HDF5 stimulus file, runs the PENN model (default: FCNF0++)
on every stimulus waveform, aggregates frame-level predictions, converts
the 1440-bin posteriorgram to a 700-class probability distribution via
Gaussian interpolation, and saves structured JSON files matching the
exact MERT reference format.

Usage:
    source /home/zyf/anaconda3/etc/profile.d/conda.sh
    conda activate PENN_psychophysics
    cd /data/zyf/PENN_psychophysics
    python PSYCHO_EVALUATOR_PENN.py \\
        --data_root ../MERT_psychophysics/data/dataset_psychophysics \\
        --output_dir ./PENN_FCN_psy_predictions \\
        --model fcn \\
        --device cuda
"""

# ============================================================================
# Version history
# ============================================================================
# v2 (2026-08-05): Added raw activation storage — per-trial 1440-dim softmax
#   posteriorgrams are collected BEFORE Gaussian interpolation (1440→700,
#   sigma=1.5, via fcnf0pp_to_700) and saved to <json_basename>_raw.npz
#   alongside each JSON file (keys: raw_activations, freqs_source, model_name,
#   experiment_name, sigma_used). JSON output format unchanged.
# v3 (2026-08-05): ADD model raw outputs to .npz
#   - .npz now also stores: f0_hz (n_stimuli,) float32 (from all_f0_hz, already
#     collected in the trial loop) and confidence (n_stimuli,) float32 (from
#     all_periodicity, already collected in the trial loop — periodicity is the
#     entropy-based voicing confidence, 0–1).
#   - Error stimuli (None in JSON) are stored as NaN in both arrays to keep
#     alignment with the JSON.
#   - All numeric arrays are float32.
#   - JSON output format and interpolation sigma values are UNCHANGED.
# ============================================================================

import os
import sys
import json
import argparse
import traceback
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# Local imports
from penn_predict import PENNPredictor, get_model_config, MODEL_CONFIGS
from utils.probability_utils import fcnf0pp_to_700

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

# Local FCNF0++ checkpoint — bundled with penn-master or downloaded manually.
# Falls back to None (HuggingFace auto-download) if the file doesn't exist.
# 策略 B 副本（2026-08-14）：原版硬编码同门机器路径 /data/lym/F0_SOTA/...——
# 该路径在这台服务器上恰好存在，策略 A 的默认运行可能悄然使用了它。
# 副本改用项目内自持权重（pretrained_models/fcnf0++.pt，内容同源），
# 切断对外部账户目录的运行时依赖。
LOCAL_FCNF0PP_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'pretrained_models', 'fcnf0++.pt')

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

# Keys from the bez2018 auditory periphery model that should be stripped
# from the reference template (not applicable to PENN).
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
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ============================================================================
# Helpers
# ============================================================================

def load_audio_from_hdf5(hdf5_path, audio_key):
    """Load audio waveform from HDF5 file. Returns (N, samples) or None."""
    import h5py
    try:
        with h5py.File(hdf5_path, 'r') as f:
            parts = audio_key.split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            audio = obj[:]
        return np.asarray(audio, dtype=np.float32)
    except Exception as e:
        print(f'  [ERROR] Failed to read "{audio_key}" from {hdf5_path}: {e}')
        return None


def load_reference_template(reference_dir, experiment_name):
    """Load MERT reference JSON as structural template. Returns dict or {}."""
    ref_path = os.path.join(reference_dir, f'predictions_mert_{experiment_name}.json')
    if os.path.exists(ref_path):
        with open(ref_path, 'r') as f:
            return json.load(f)
    print(f'  [WARN] Reference template not found: {ref_path}')
    return {}


# ============================================================================
# Per-experiment processing
# ============================================================================

def process_experiment(hdf5_path, audio_key, experiment_name, predictor, reference_dir):
    """Process one psychophysics experiment.

    1. Load audio from HDF5
    2. Run PENN on each stimulus
    3. Convert 1440-bin posteriorgram → 700-class via Gaussian interpolation
    4. Save output in MERT-compatible JSON format
    """
    if not os.path.exists(hdf5_path):
        print(f'  [ERROR] HDF5 file not found: {hdf5_path}')
        return None, 0, 0, None

    print(f'\n{"=" * 60}')
    print(f'Processing: {experiment_name}')
    print(f'  HDF5:      {hdf5_path}')
    print(f'  Audio key: {audio_key}')

    # ------------------------------------------------------------------
    # 1. Load audio
    # ------------------------------------------------------------------
    audio = load_audio_from_hdf5(hdf5_path, audio_key)
    if audio is None:
        return None, 0, 0, None

    n_stimuli = audio.shape[0]
    print(f'  Stimuli:   {n_stimuli}  |  Audio shape: {audio.shape}')

    # ------------------------------------------------------------------
    # 2. Load reference template (preserves ground-truth parameter lists)
    # ------------------------------------------------------------------
    template = load_reference_template(reference_dir, experiment_name)

    # ------------------------------------------------------------------
    # 3. Run PENN on each stimulus
    # ------------------------------------------------------------------
    all_pred_labels = []   # int class labels (0–699)
    all_pred_probs = []    # 700-element probability vectors
    all_f0_hz = []         # continuous pitch estimate (Hz)
    all_periodicity = []   # periodicity confidence (0–1)
    all_posteriorgrams = []  # raw 1440-dim softmax posteriorgrams (pre-interpolation)
    n_errors = 0

    pbar = tqdm(total=n_stimuli, desc=f'  PENN → {experiment_name}', unit='stim')

    for i in range(n_stimuli):
        stimulus_waveform = audio[i]  # shape: (4800,) at 32 kHz

        try:
            f0_hz, periodicity_val, posteriorgram = predictor.predict(
                stimulus_waveform, sample_rate=32000
            )

            # Collect raw posteriorgram before interpolation
            all_posteriorgrams.append(posteriorgram.copy())

            # Convert 1440-bin posteriorgram → 700-class
            labels_pred, probs_700 = fcnf0pp_to_700(posteriorgram)

            all_pred_labels.append(labels_pred)
            all_pred_probs.append(probs_700.tolist())
            all_f0_hz.append(f0_hz)
            all_periodicity.append(periodicity_val)

        except Exception as e:
            if i < 5:  # Only print first few errors
                print(f'\n  [ERROR] Stimulus {i}: {e}')
                traceback.print_exc()
            all_pred_labels.append(None)
            all_pred_probs.append(None)
            all_f0_hz.append(None)
            all_periodicity.append(None)
            all_posteriorgrams.append(None)
            n_errors += 1

        pbar.update(1)

    pbar.close()

    # ------------------------------------------------------------------
    # 4. Build output dict
    # ------------------------------------------------------------------
    if template:
        result = dict(template)
    else:
        result = {}

    # Remove bez2018 peripheral-model config keys (not applicable)
    for key in BEZ2018_KEYS:
        result.pop(key, None)

    # Update prediction arrays
    result['f0_label:labels_pred'] = all_pred_labels
    result['f0_label:probs_out'] = all_pred_probs

    # Add PENN-specific additional fields
    result['f0_hz'] = all_f0_hz
    result['periodicity'] = all_periodicity

    # Update metadata
    result['experiment_name'] = experiment_name
    result['model_name'] = predictor.model_config['name']
    result['timestamp'] = datetime.now().isoformat()

    # Report errors
    if n_errors > 0:
        print(f'  ⚠️  {n_errors} stimuli had errors and were set to None')

    n_valid = sum(1 for v in all_pred_labels if v is not None)
    print(f'  ✅ Done — {n_valid}/{n_stimuli} valid predictions')
    # v3: also return raw model outputs f0_hz and periodicity (None on error)
    return result, n_valid, n_errors, all_posteriorgrams, all_f0_hz, all_periodicity


# ============================================================================
# Evaluation report
# ============================================================================

def generate_evaluation_report(
    generated_files, experiments_processed, experiments_failed,
    total_stimuli, total_errors, model_config, data_root, output_dir,
    device_str, checkpoint_used
):
    """Generate evaluation_report.json."""
    report = {
        'evaluation_date': datetime.now().isoformat(),
        'model_name': model_config['name'],
        'framework': 'PENN (pip install penn)',
        'checkpoint': checkpoint_used,
        'pitch_bins': model_config['n_bins'],
        'cents_per_bin': model_config['cents_per_bin'],
        'fmin_source': model_config['fmin'],
        'decoder': model_config['decoder'],
        'periodicity_method': 'entropy-based (built-in)',
        'aggregation': 'median (pitch/periodicity), mean (posteriorgram)',
        'interpolation_sigma': 1.5,
        'interpolation_method': 'Gaussian interpolation (1440→700)',
        'data_path': str(os.path.abspath(data_root)),
        'output_directory': str(os.path.abspath(output_dir)),
        'reference_directory': str(os.path.abspath(DEFAULT_REFERENCE_DIR)),
        'total_stimuli_processed': total_stimuli,
        'total_errors': total_errors,
        'success_rate': round(
            (total_stimuli - total_errors) / total_stimuli * 100, 1
        ) if total_stimuli > 0 else 0,
        'experiments_processed': experiments_processed,
        'experiments_failed': experiments_failed,
        'device_used': device_str,
        'generated_files': generated_files,
        'total_generated_files': len(generated_files),
    }
    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PENN Psychophysics Evaluator — MERT-compatible output format'
    )
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Directory containing HDF5 stimulus files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-derived from --model if omitted)')
    parser.add_argument('--reference_dir', type=str, default=DEFAULT_REFERENCE_DIR,
                        help='Directory containing MERT reference JSON templates')
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn', 'crepe++', 'deepf0++'],
                        help='Model variant (default: fcn = FCNF0++)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: "cuda" (GPU 0) or "cpu"')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pt file (overrides default)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='(Reserved) PENN processes one stimulus at a time')

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
    # Resolve model configuration
    # ------------------------------------------------------------------
    model_cfg = get_model_config(args.model)

    # Determine checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        checkpoint_desc = args.checkpoint
    elif args.model == 'fcn' and os.path.exists(LOCAL_FCNF0PP_CHECKPOINT):
        checkpoint_path = LOCAL_FCNF0PP_CHECKPOINT
        checkpoint_desc = f'Local: {LOCAL_FCNF0PP_CHECKPOINT}'
    else:
        checkpoint_path = model_cfg['checkpoint']
        checkpoint_desc = 'HuggingFace (auto-download)' if model_cfg['checkpoint'] is None else model_cfg['checkpoint']

    # Determine output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(PROJECT_ROOT, model_cfg['output_dir'])

    # Determine device
    if args.device == 'cuda' and torch.cuda.is_available():
        gpu = 0
        device_str = 'cuda'
    else:
        gpu = None
        device_str = 'cpu'

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print('=' * 60)
    print('PENN Psychophysics Evaluator')
    print('=' * 60)
    print(f'Model:         {model_cfg["name"]}')
    print(f'Checkpoint:    {checkpoint_desc}')
    print(f'Decoder:       {model_cfg["decoder"]}')
    print(f'Pitch bins:    {model_cfg["n_bins"]} ({model_cfg["cents_per_bin"]}¢)')
    print(f'Freq range:    {model_cfg["fmin"]}–1984.0 Hz')
    print(f'Data root:     {os.path.abspath(args.data_root)}')
    print(f'Output dir:    {output_dir}')
    print(f'Reference dir: {os.path.abspath(args.reference_dir)}')
    print(f'Device:        {device_str}')
    print(f'JSON prefix:   {model_cfg["json_prefix"]}')

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialise predictor
    # ------------------------------------------------------------------
    print(f'\n[1/3] Initialising PENN predictor ({model_cfg["name"]})...')
    predictor = PENNPredictor(
        model_name=args.model,
        gpu=gpu,
        checkpoint_override=checkpoint_path,
    )
    print('  Predictor ready.')

    # ------------------------------------------------------------------
    # Process experiments
    # ------------------------------------------------------------------
    print(f'\n[2/3] Processing experiments...')

    generated_files = []
    experiments_processed = []
    experiments_failed = []
    total_stimuli = 0
    total_errors = 0

    for hdf5_filename, audio_key, experiment_name in EXPERIMENTS:
        if experiment_filter is not None and experiment_name not in experiment_filter:
            print(f"  [SKIP] {experiment_name} (not in --experiments filter)")
            continue
        hdf5_path = os.path.join(args.data_root, hdf5_filename)

        try:
            result, n_valid, n_errors, raw_list, f0_list, per_list = process_experiment(
                hdf5_path=hdf5_path,
                audio_key=audio_key,
                experiment_name=experiment_name,
                predictor=predictor,
                reference_dir=args.reference_dir,
            )

            if result is None:
                print(f'  ❌ Failed to process {experiment_name}')
                experiments_failed.append(experiment_name)
                continue

            n_stim = n_valid + n_errors
            total_stimuli += n_stim
            total_errors += n_errors

            output_filename = (
                f'{model_cfg["json_prefix"]}_{experiment_name}.json'
            )
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w') as f:
                json.dump(result, f, cls=NumpyEncoder)

            # Save raw posteriorgrams alongside the JSON (same basename + _raw.npz)
            raw_acts = np.array([
                pg if pg is not None else np.full(model_cfg['n_bins'], np.nan, dtype=np.float32)
                for pg in raw_list], dtype=np.float32)
            freqs_source = model_cfg['fmin'] * (
                2 ** (np.arange(model_cfg['n_bins']) * model_cfg['cents_per_bin'] / 1200)
            )
            npz_path = output_path.replace('.json', '_raw.npz')
            # v3: also store raw model outputs f0_hz and confidence (NaN on error)
            f0_arr = np.array([f if f is not None else np.nan for f in f0_list],
                              dtype=np.float32)
            conf_arr = np.array([p if p is not None else np.nan for p in per_list],
                                dtype=np.float32)
            np.savez_compressed(
                npz_path,
                raw_activations=raw_acts.astype(np.float32),
                freqs_source=freqs_source.astype(np.float32),
                f0_hz=f0_arr,
                confidence=conf_arr,
                model_name=model_cfg['name'],
                experiment_name=experiment_name,
                sigma_used=1.5,
            )

            generated_files.append(output_filename)
            experiments_processed.append(experiment_name)

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f'  ✅ Saved: {output_filename}  ({size_mb:.2f} MB, {n_stim} stimuli, {n_errors} errors)')

        except Exception as e:
            print(f'  ❌ Fatal error processing {experiment_name}: {e}')
            traceback.print_exc()
            experiments_failed.append(experiment_name)

    # ------------------------------------------------------------------
    # Generate evaluation report
    # ------------------------------------------------------------------
    print(f'\n[3/3] Generating evaluation report...')

    report = generate_evaluation_report(
        generated_files=generated_files,
        experiments_processed=experiments_processed,
        experiments_failed=experiments_failed,
        total_stimuli=total_stimuli,
        total_errors=total_errors,
        model_config=model_cfg,
        data_root=args.data_root,
        output_dir=output_dir,
        device_str=device_str,
        checkpoint_used=checkpoint_desc,
    )

    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)
    print(f'  ✅ Report saved: {report_path}')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}')
    print('✅ PENN Evaluation Complete!')
    print(f'   Model:        {model_cfg["name"]}')
    print(f'   Experiments:  {len(experiments_processed)}/{len(EXPERIMENTS)}')
    print(f'   Failed exp:   {len(experiments_failed)}')
    print(f'   Total stimuli: {total_stimuli}')
    print(f'   Total errors:  {total_errors}')
    print(f'   Success rate:  {report["success_rate"]}%')
    print(f'   Output dir:   {output_dir}')
    for fname in generated_files:
        fpath = os.path.join(output_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f'     - {fname}  ({size_mb:.2f} MB)')


if __name__ == '__main__':
    main()
