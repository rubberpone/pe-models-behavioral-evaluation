#!/usr/bin/env python3
"""
PSYCHO_EVALUATOR_RMVPE.py — Generate RMVPE F0 predictions for all five
psychophysical experiments.

Loads each HDF5 stimulus file, runs the RMVPE E2E model on every stimulus
waveform, decodes pitch via local weighted average, maps 360 sigmoid
activations to 700-class distributions via Gaussian interpolation, and
saves structured JSON files matching the exact MERT reference format.

Output format per stimulus (matching MERT reference):
  - f0_label:labels_pred : int  ∈ [0, 699]   — predicted class label
  - f0_label:probs_out   : list[float] × 700 — probability distribution

Usage:
    source /home/zyf/anaconda3/etc/profile.d/conda.sh
    conda activate RMVPE_psychophysics
    cd /data/zyf/RMVPE_psychophysics
    python PSYCHO_EVALUATOR_RMVPE.py \
        --data_root ../MERT_psychophysics/data/dataset_psychophysics \
        --output_dir ./RMVPE_psy_predictions \
        --device cuda \
        --voicing_threshold 0.0

=============================================================================
VERSION HISTORY
=============================================================================
v2 (2026-08-05): ADD raw activation storage for sigma sensitivity analysis
  - Added np.savez_compressed() call to store raw 360-dim sigmoid activations
    alongside each prediction JSON (as *_raw.npz files).
  - Raw activations are saved BEFORE Gaussian interpolation (sigma=1.5),
    enabling subsequent sigma scanning without re-running model inference.
  - .npz keys: raw_activations (n_stimuli, 360) float32, freqs_source (360,)
    float32, model_name, experiment_name, sigma_used.
  - Error stimuli (labels=None in JSON) are stored as NaN rows in
    raw_activations to keep array alignment with the JSON.
  - Existing JSON output format is UNCHANGED.
  - Backup: backup/PSYCHO_EVALUATOR_RMVPE.py.bak
v3 (2026-08-05): ADD model raw outputs to .npz
  - .npz now also stores: f0_hz (n_stimuli,) float32 (from all_f0_hz, already
    collected in the trial loop) and confidence (n_stimuli,) float32 (from
    all_confidences, already collected in the trial loop).
  - Error stimuli are stored as NaN in both arrays to keep alignment with
    the JSON (labels=None).
  - All numeric arrays are float32 (raw_activations, freqs_source, f0_hz,
    confidence).
  - JSON output format and interpolation sigma values are UNCHANGED.
  - Backup: backup/PSYCHO_EVALUATOR_RMVPE.py.bak
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
from datetime import datetime
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from rmvpe_predict import RMVPEPredictor
from utils.probability_utils import (
    rmvpe_to_700_fast as rmvpe_to_700,
    RMVPE_CONST, RMVPE_CENTS_PER_BIN, RMVPE_REF_FREQ,
)


# ============================================================================
# Paths
# ============================================================================

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(PROJECT_ROOT), '..', '..', 'data'
)

DEFAULT_REFERENCE_DIR = os.path.join(
    os.path.dirname(PROJECT_ROOT), 'MERT_psychophysics',
    'mert_psy_predictions_example'
)

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'RMVPE_psy_predictions')
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'pretrained_models', 'rmvpe.pth')

N_CLASSES = 700


# ============================================================================
# Experiment definitions
# ============================================================================

EXPERIMENTS = [
    {'hdf5_file': 'stim_ERB_B1.hdf5', 'audio_key': 'tone_in_noise',
     'experiment_name': 'ERB_B1'},
    {'hdf5_file': 'stim_ERB_B1_fine.hdf5', 'audio_key': 'tone_in_noise',
     'experiment_name': 'ERB_B1_fine'},
    {'hdf5_file': 'stim_ERB_B2.hdf5', 'audio_key': 'tone_in_noise',
     'experiment_name': 'ERB_B2'},
    {'hdf5_file': 'stim_ERB_B2_v2.hdf5', 'audio_key': 'tone_in_noise',
     'experiment_name': 'ERB_B2_v2'},
    {'hdf5_file': 'stim_ERB_B3.hdf5', 'audio_key': 'stimuli/signal',
     'experiment_name': 'ERB_B3'},
    {'hdf5_file': 'stim_ERB_B4.hdf5', 'audio_key': 'stimuli/signal',
     'experiment_name': 'ERB_B4'},
    {'hdf5_file': 'stim_ERB_B5.hdf5', 'audio_key': 'stimuli/signal_in_noise',
     'experiment_name': 'ERB_B5'},
    {'hdf5_file': 'stim_ERB_B6.hdf5', 'audio_key': 'tone_in_noise',
     'experiment_name': 'ERB_B6'},
]

# bez2018 model-specific keys to strip from reference templates
# (these are MERT-specific and not applicable to RMVPE)
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

# Keys we replace with RMVPE predictions (must overwrite not merge)
PREDICTION_KEYS = [
    'f0_label:labels_pred',
    'f0_label:probs_out',
]


# ============================================================================
# JSON encoder
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
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
# HDF5 helpers
# ============================================================================

def load_audio_from_hdf5(hdf5_path: str, audio_key: str) -> np.ndarray:
    """
    Extract audio waveform dataset from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file.
        audio_key: Internal path to the audio dataset (e.g. 'tone_in_noise').

    Returns:
        numpy array of shape (n_stimuli, n_samples) or None on failure.
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            parts = audio_key.split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            return obj[:]
    except Exception as e:
        print(f'  [ERROR] Failed to read "{audio_key}" from {hdf5_path}: {e}')
        return None


def load_reference_template(reference_dir: str, experiment_name: str) -> dict:
    """Load the MERT reference prediction JSON as a structural template."""
    ref_path = os.path.join(reference_dir, f'predictions_mert_{experiment_name}.json')
    if os.path.exists(ref_path):
        with open(ref_path, 'r') as f:
            return json.load(f)
    print(f'  [WARN] Reference template not found: {ref_path}')
    return {}


# ============================================================================
# Per-experiment processing
# ============================================================================

def process_experiment(
    hdf5_path: str,
    audio_key: str,
    experiment_name: str,
    predictor: RMVPEPredictor,
    reference_dir: str,
    batch_size: int = 256,
    voicing_threshold: float = 0.0,
    sigma: float = 1.5,
) -> dict:
    """
    Process one psychophysics experiment:

    1. Load HDF5 audio
    2. Load reference template for key structure
    3. Run RMVPE on each stimulus
    4. Decode pitch via local weighted average
    5. Map 360→700 via Gaussian interpolation
    6. Build output matching reference JSON structure
    """
    if not os.path.exists(hdf5_path):
        print(f'  [ERROR] HDF5 file not found: {hdf5_path}')
        return None

    print(f'\n{"=" * 60}')
    print(f'Processing: {experiment_name}')
    print(f'  HDF5:     {hdf5_path}')
    print(f'  Audio key: {audio_key}')

    # ------------------------------------------------------------------
    # 1. Load audio
    # ------------------------------------------------------------------
    audio = load_audio_from_hdf5(hdf5_path, audio_key)
    if audio is None:
        return None

    n_stimuli = audio.shape[0]
    n_samples = audio.shape[1]
    print(f'  Stimuli:  {n_stimuli}  |  Audio shape: {audio.shape}  |  '
          f'Duration: {n_samples / 32000 * 1000:.0f} ms @ 32 kHz')

    # ------------------------------------------------------------------
    # 2. Load reference template
    # ------------------------------------------------------------------
    template = load_reference_template(reference_dir, experiment_name)
    if not template:
        print(f'  [WARN] No reference template for {experiment_name}; '
              f'building minimal output')

    # ------------------------------------------------------------------
    # 3. Preprocess all audio to 16 kHz tensors (once)
    # ------------------------------------------------------------------
    import torchaudio
    print(f'  Preprocessing audio (32 kHz → 16 kHz, mono, normalize)...')

    # Resample all stimuli at once using torchaudio
    audio_tensor = torch.from_numpy(audio).float()  # (N, 4800)
    if audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
        # Already (N, T) — assume mono; if stereo, take mean
        pass
    resampler = torchaudio.transforms.Resample(32000, 16000)
    audio_16k = resampler(audio_tensor)  # (N, 2400)
    n_orig = audio_16k.shape[1]  # 2400

    # Normalize each stimulus to [-1, 1]
    max_vals = audio_16k.abs().max(dim=1, keepdim=True).values
    max_vals = max_vals.clamp(min=1.0)  # don't amplify quiet signals
    audio_16k = audio_16k / max_vals

    # Ensure min length (2400 >= 2048, so no padding needed)
    min_len = 2048
    if n_orig < min_len:
        pad_total = min_len - n_orig
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio_16k = torch.nn.functional.pad(audio_16k, (pad_left, pad_right),
                                            mode='constant', value=0)

    print(f'  Preprocessed: {audio_16k.shape} @ 16 kHz')

    # ------------------------------------------------------------------
    # 4. Run RMVPE inference in batches
    # ------------------------------------------------------------------
    all_pred_labels = []
    all_pred_probs = []
    all_f0_hz = []
    all_confidences = []
    errors = []
    raw_activations = []  # v2: store raw 360-dim activations before interpolation

    # Move model to GPU once
    model = predictor.model
    device = predictor.device

    pbar = tqdm(total=n_stimuli, desc=f'  RMVPE → {experiment_name}', unit='stim')

    for batch_start in range(0, n_stimuli, batch_size):
        batch_end = min(batch_start + batch_size, n_stimuli)
        batch_audio = audio_16k[batch_start:batch_end].to(device)  # (B, 2400)

        try:
            with torch.inference_mode():
                hidden_vec, output = model(batch_audio)
                # output: (B, T_frames, 360)

            # Process each item in batch
            for i in range(batch_end - batch_start):
                activation = output[i, 0, :].cpu().numpy()  # (360,) — first frame

                # Decode pitch
                f0_hz, confidence = predictor.decode_pitch(
                    activation, threshold=predictor.voicing_threshold)

                # v2: collect raw activation BEFORE interpolation
                raw_activations.append(activation.copy())

                # Map 360 → 700
                labels_pred, probs_700 = rmvpe_to_700(activation, sigma=sigma)

                all_pred_labels.append(int(labels_pred))
                all_pred_probs.append(probs_700.tolist())
                all_f0_hz.append(float(f0_hz))
                all_confidences.append(float(confidence))

        except Exception as e:
            stim_idx = batch_start
            msg = f'  [ERROR] Batch [{batch_start}:{batch_end}]: {e}'
            traceback.print_exc()
            print(msg)
            for _ in range(batch_end - batch_start):
                all_pred_labels.append(None)
                all_pred_probs.append(None)
                all_f0_hz.append(None)
                all_confidences.append(None)
                # v2: error trials not stored — fill with NaN to keep alignment
                raw_activations.append(np.full(360, np.nan, dtype=np.float32))
            errors.extend(range(batch_start, batch_end))

        pbar.update(batch_end - batch_start)

    pbar.close()

    # ------------------------------------------------------------------
    # 4. Build output dict (matching reference structure)
    # ------------------------------------------------------------------
    if template:
        result = dict(template)
    else:
        result = {}

    # Remove bez2018 peripheral-model config keys
    for key in BEZ2018_KEYS:
        result.pop(key, None)

    # Update prediction arrays
    result['f0_label:labels_pred'] = all_pred_labels
    result['f0_label:probs_out'] = all_pred_probs

    # Update metadata
    result['experiment_name'] = experiment_name
    result['model_name'] = 'RMVPE'
    result['timestamp'] = datetime.now().isoformat()

    # Add RMVPE-specific fields
    result['rmvpe_config'] = {
        'model': 'RMVPE (E2E — Deep U-Net + BiGRU)',
        'implementation': 'Dream-High/RMVPE (official)',
        'parameters': '15.4M',
        'input_sr': 16000,
        'input_type': 'raw audio waveform — model internally computes log mel-spectrogram',
        'mel_params': {
            'n_fft': 1024,
            'hop_length': 160,
            'n_mels': 128,
            'fmin': 30,
            'fmax': 8000,
            'mel_formula': 'HTK (htk=True)',
            'center_padding': False,
            'log_clamp': '1e-5',
            'note': 'Checkpoint uses N_MELS=128/WINDOW=1024/hop=160 '
                    '(matches publicly available pretrained weights from '
                    'lj1995/VoiceConversionWebUI). Official repo uses 256/2048/320 '
                    'but no matching checkpoint is publicly available.',
        },
        'pitch_decoding': 'local weighted average (peak ±4 bins), CONST=1997.3794084376191',
        'voicing_threshold': voicing_threshold,
        'voicing_note': 'Official RMVPE default=0.0 used; psychophysical stimuli '
                        'have low confidence (0.01-0.07). Threshold 0.5 would '
                        'classify all as unvoiced.',
        'output_activation': 'Sigmoid (independent per-bin confidence)',
        'interpolation': {
            'method': 'Gaussian interpolation (360 sigmoid activations → 700-bin distribution)',
            'sigma': 1.5,
            'sigma_cents': 9.375,
            'ref_freq': 10.0,
            'rmvpe_const': 1997.3794084376191,
        },
        'inference_mode': 'single-frame (zero-pad → feed whole stimulus → take first frame)',
    }

    # Optionally store continuous F0 and confidence
    result['f0_hz_rmvpe'] = all_f0_hz
    result['confidence_rmvpe'] = all_confidences

    n_valid = sum(1 for v in all_pred_labels if v is not None)
    n_failed = len(errors)

    print(f'  ✅ Done — {n_valid}/{n_stimuli} valid predictions'
          + (f', {n_failed} failed' if n_failed else ''))

    return {
        'result': result,
        'n_stimuli': n_stimuli,
        'n_valid': n_valid,
        'n_failed': n_failed,
        'failed_indices': errors,
        'raw_activations': raw_activations,
        'f0_hz': all_f0_hz,          # v3: raw model outputs (None on error)
        'confidence': all_confidences,  # v3: raw model outputs (None on error)
    }


# ============================================================================
# Evaluation report
# ============================================================================

def generate_evaluation_report(
    generated_files: list,
    experiments_processed: list,
    experiment_details: list,
    total_stimuli: int,
    total_failed: int,
    data_root: str,
    output_dir: str,
    model_path: str,
    device_str: str,
    voicing_threshold: float,
    batch_size: int,
) -> dict:
    """Generate the evaluation_report.json summary."""
    return {
        'evaluation_date': datetime.now().isoformat(),
        'model_name': 'RMVPE',
        'implementation': 'E2E (Deep U-Net + BiGRU) from official repo (Dream-High/RMVPE)',
        'input_sr': 16000,
        'input_type': 'raw audio waveform (16 kHz) — model internally computes log mel-spectrogram',
        'mel_params': {
            'n_fft': 1024,
            'hop_length': 160,
            'n_mels': 128,
            'fmin': 30,
            'fmax': 8000,
            'mel_formula': 'HTK',
            'center_padding': False,
            'log_clamp': '1e-5',
            'note': 'Parameters match publicly available checkpoint '
                    '(lj1995/VoiceConversionWebUI). Official repo uses '
                    '256/2048/320 but no matching checkpoint is available.',
        },
        'pitch_decoding': 'local weighted average (peak ±4 bins), CONST=1997.3794084376191',
        'voicing_threshold': voicing_threshold,
        'voicing_note': 'Official RMVPE default threshold=0.0 used because psychophysical '
                        'harmonic complexes produce low confidence values (0.01-0.07) '
                        'compared to vocal/music signals. Threshold 0.5 would classify '
                        'all stimuli as unvoiced, which is incorrect for these data.',
        'reference_freq': 10.0,
        'output_activation': 'Sigmoid (independent per-bin confidence, not softmax distribution)',
        'interpolation_sigma': 1.5,
        'interpolation_method': 'Gaussian interpolation (360 sigmoid activations → 700-bin distribution)',
        'model_path': str(os.path.abspath(model_path)),
        'data_path': str(os.path.abspath(data_root)),
        'output_directory': str(os.path.abspath(output_dir)),
        'total_stimuli_processed': total_stimuli,
        'failed_predictions': total_failed,
        'success_rate': round(100.0 * (total_stimuli - total_failed) / max(total_stimuli, 1), 2),
        'experiments': experiments_processed,
        'experiment_details': experiment_details,
        'device_used': device_str,
        'batch_size': batch_size,
        'generated_files': generated_files,
        'total_generated_files': len(generated_files),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RMVPE Psychophysics Evaluator — MERT-compatible output format'
    )
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--reference_dir', type=str, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--voicing_threshold', type=float, default=0.0,
                        help='Voicing threshold (default: 0.0 = official RMVPE default)')
    parser.add_argument('--batch_size', type=int, default=1,  # 1 = process one-by-one
                        help='Stimuli per batch (default: 1, process one-by-one)')
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='Gaussian interpolation sigma in target bins')

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
    print('RMVPE Psychophysics Evaluator (MERT-compatible format)')
    print('=' * 60)

    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir)
    reference_dir = os.path.abspath(args.reference_dir)
    model_path = os.path.abspath(args.model_path)

    print(f'Data root:      {data_root}')
    print(f'Output dir:     {output_dir}')
    print(f'Reference dir:  {reference_dir}')
    print(f'Model path:     {model_path}')
    print(f'Device:         {args.device}')
    print(f'Voicing thresh: {args.voicing_threshold}')
    print(f'Sigma:          {args.sigma}')
    print(f'Output classes: {N_CLASSES}')

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------------
    print(f'\n[1/3] Loading RMVPE model...')

    if args.device == 'cuda' and torch.cuda.is_available():
        device_str = 'cuda'
        print(f'  Device: CUDA (GPU available)')
    else:
        device_str = 'cpu'
        print(f'  Device: CPU')

    try:
        predictor = RMVPEPredictor(
            model_path=model_path,
            device=args.device if torch.cuda.is_available() else 'cpu',
            voicing_threshold=args.voicing_threshold,
        )
        print(f'  ✅ Model loaded successfully')
    except Exception as e:
        print(f'  ❌ Failed to load model: {e}')
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Process experiments
    # ------------------------------------------------------------------
    print(f'\n[2/3] Processing experiments with RMVPE...')

    generated_files = []
    experiments_processed = []
    experiment_details = []
    total_stimuli = 0
    total_failed = 0

    for exp_def in EXPERIMENTS:
        if experiment_filter is not None and exp_def.get('experiment_name') not in experiment_filter:
            print(f"  [SKIP] {exp_def.get('experiment_name')} (not in --experiments filter)")
            continue
        hdf5_path = os.path.join(data_root, exp_def['hdf5_file'])
        experiment_name = exp_def['experiment_name']

        try:
            result_data = process_experiment(
                hdf5_path=hdf5_path,
                audio_key=exp_def['audio_key'],
                experiment_name=experiment_name,
                predictor=predictor,
                reference_dir=reference_dir,
                batch_size=args.batch_size,
                voicing_threshold=args.voicing_threshold,
                sigma=args.sigma,
            )

            if result_data is None:
                print(f'  ❌ Failed to process {experiment_name}')
                experiment_details.append({
                    'experiment': experiment_name,
                    'status': 'FAILED',
                    'error': 'process_experiment returned None',
                })
                continue

            result = result_data['result']
            n_stim = result_data['n_stimuli']
            n_valid = result_data['n_valid']
            n_failed = result_data['n_failed']

            total_stimuli += n_stim
            total_failed += n_failed

            # Save JSON output
            output_filename = f'predictions_rmvpe_{experiment_name}.json'
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w') as f:
                json.dump(result, f, cls=NumpyEncoder)

            # v2: save raw 360-dim activations as .npz (before interpolation)
            cents_rmvpe = np.arange(360) * RMVPE_CENTS_PER_BIN + RMVPE_CONST
            freqs_source = RMVPE_REF_FREQ * (2 ** (cents_rmvpe / 1200))
            raw_path = output_path.replace('.json', '_raw.npz')
            # v3: also store raw model outputs f0_hz and confidence (NaN on error)
            np.savez_compressed(
                raw_path,
                raw_activations=np.array(result_data['raw_activations'], dtype=np.float32),
                freqs_source=freqs_source.astype(np.float32),
                f0_hz=np.array(
                    [f if f is not None else np.nan for f in result_data['f0_hz']],
                    dtype=np.float32,
                ),
                confidence=np.array(
                    [c if c is not None else np.nan for c in result_data['confidence']],
                    dtype=np.float32,
                ),
                model_name="RMVPE",
                experiment_name=experiment_name,
                sigma_used=args.sigma,
            )

            generated_files.append(output_filename)
            experiments_processed.append(experiment_name)
            experiment_details.append({
                'experiment': experiment_name,
                'n_stimuli': n_stim,
                'n_valid': n_valid,
                'n_failed': n_failed,
                'output_file': output_filename,
            })

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f'  💾 Saved: {output_filename}  ({size_mb:.2f} MB, {n_stim} stimuli)')

        except Exception as e:
            print(f'  ❌ Fatal error processing {experiment_name}: {e}')
            traceback.print_exc()
            experiment_details.append({
                'experiment': experiment_name,
                'status': 'FATAL ERROR',
                'error': str(e),
            })

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f'\n[3/3] Generating evaluation report...')

    report = generate_evaluation_report(
        generated_files=generated_files,
        experiments_processed=experiments_processed,
        experiment_details=experiment_details,
        total_stimuli=total_stimuli,
        total_failed=total_failed,
        data_root=data_root,
        output_dir=output_dir,
        model_path=model_path,
        device_str=device_str,
        voicing_threshold=args.voicing_threshold,
        batch_size=args.batch_size,
    )

    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)
    print(f'  ✅ Report saved: {report_path}')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f'\n{"=" * 60}')
    print('✅ RMVPE Evaluation Complete!')
    print(f'   Experiments:  {len(experiments_processed)}/{len(EXPERIMENTS)}')
    print(f'   Total stimuli: {total_stimuli}')
    print(f'   Failed:        {total_failed}')
    print(f'   Success rate:  {round(100.0 * (total_stimuli - total_failed) / max(total_stimuli, 1), 2)}%')
    print(f'   Output dir:    {output_dir}')
    for fname in generated_files:
        fpath = os.path.join(output_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f'     - {fname}  ({size_mb:.2f} MB)')


if __name__ == '__main__':
    main()
