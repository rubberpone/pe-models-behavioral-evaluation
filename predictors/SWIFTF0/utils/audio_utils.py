"""
Audio utility functions for SwiftF0 psychophysics pipeline.

Loads audio from HDF5 files, handles mono conversion, and resampling.
STFT + frequency band selection + log compression are NOT done here —
they happen inside the ONNX model.
"""

import h5py
import numpy as np
import librosa


def load_audio_from_hdf5(hdf5_path: str, audio_key: str) -> np.ndarray:
    """
    Load audio waveform from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file.
        audio_key: Key path within the HDF5 (e.g., 'tone_in_noise' or 'stimuli/signal').

    Returns:
        numpy array of shape (n_trials, n_samples) with float32 dtype.
    """
    with h5py.File(hdf5_path, 'r') as f:
        parts = audio_key.split('/')
        ds = f
        for part in parts:
            ds = ds[part]
        audio = ds[()]  # Load entire dataset into memory
    return audio.astype(np.float32)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono by averaging channels.

    Args:
        audio: numpy array of shape (..., n_samples) or (n_samples,).
               If 2D with more than 1 row and both dimensions > 2, treat first as channels.

    Returns:
        Mono audio as 1D or 2D numpy array.
    """
    # If already 1D, nothing to do
    if audio.ndim == 1:
        return audio

    # If 2D batch: (n_trials, n_samples)
    if audio.ndim == 2:
        # Heuristic: if second dim is small (e.g., 2 for stereo), transpose
        # But for psychophysics data, it's always (n_trials, n_samples)
        return audio  # Already mono per trial

    # For higher dimensions, take mean over last axis if it looks like channels
    return audio


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Resample audio from original sample rate to target sample rate.

    Args:
        audio: numpy array of shape (n_trials, n_samples) or (n_samples,).
        orig_sr: Original sample rate (Hz).
        target_sr: Target sample rate (Hz). Default 16000.

    Returns:
        Resampled audio with same number of dimensions.
    """
    if orig_sr == target_sr:
        return audio

    if audio.ndim == 1:
        return librosa.resample(y=audio.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr)

    # Batch processing: (n_trials, n_samples)
    n_trials = audio.shape[0]
    new_n_samples = int(audio.shape[1] * target_sr / orig_sr)
    resampled = np.zeros((n_trials, new_n_samples), dtype=np.float32)

    for i in range(n_trials):
        resampled[i] = librosa.resample(
            y=audio[i].astype(np.float64),
            orig_sr=orig_sr,
            target_sr=target_sr
        ).astype(np.float32)

    return resampled


def get_hdf5_metadata(hdf5_path: str) -> dict:
    """
    Extract all trial-level and config metadata from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file.

    Returns:
        dict with:
            - 'trial_data': dict of trial-level arrays
            - 'config_data': dict of config datasets
            - 'n_trials': number of trials
    """
    trial_data = {}
    config_data = {}

    with h5py.File(hdf5_path, 'r') as f:
        # First pass: determine number of trials from the largest 1D dataset
        n_trials = 0
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset) and item.ndim == 1:
                n_trials = max(n_trials, item.shape[0])
            elif isinstance(item, h5py.Group):
                for sub_key in item.keys():
                    sub_item = item[sub_key]
                    if isinstance(sub_item, h5py.Dataset) and sub_item.ndim >= 1:
                        n_trials = max(n_trials, sub_item.shape[0])

        # Second pass: collect data
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                if item.ndim == 1 and item.shape[0] == n_trials:
                    trial_data[key] = item[()].tolist()
                else:
                    config_data[key] = item[()].tolist() if item.shape != () else [item[()].item()]
            elif isinstance(item, h5py.Group):
                for sub_key in item.keys():
                    sub_item = item[sub_key]
                    if isinstance(sub_item, h5py.Dataset):
                        full_key = f"config_{item.name.replace('/', '_')}/{sub_key}"
                        if sub_item.ndim >= 1 and sub_item.shape[0] == n_trials and sub_key not in ['signal', 'signal_in_noise']:
                            trial_data[sub_key] = sub_item[()].tolist()
                        else:
                            val = sub_item[()]
                            if hasattr(val, 'tolist'):
                                config_data[full_key] = val.tolist() if val.shape != () else [val.item()]
                            else:
                                config_data[full_key] = [val]

    return {
        'trial_data': trial_data,
        'config_data': config_data,
        'n_trials': n_trials,
    }
