"""
Audio utility functions for CREPE psychophysics pipeline.

Provides functions for loading audio from HDF5 files, resampling,
and ensuring mono format.
"""

import h5py
import numpy as np
import torch
import librosa


def load_audio_from_hdf5(hdf5_path, audio_key):
    """
    Load audio waveform from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file.
        audio_key: Key path within the HDF5 file (e.g., 'tone_in_noise'
                   or 'stimuli/signal').

    Returns:
        numpy.ndarray of shape (n_trials, n_samples) at the original
        sampling rate, or None on error.
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            parts = audio_key.split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            audio = obj[:]
            return audio
    except Exception as e:
        print(f"  [ERROR] Failed to read audio key '{audio_key}' "
              f"from {hdf5_path}: {e}")
        return None


def resample_audio(audio, orig_sr, target_sr):
    """
    Resample audio from original to target sampling rate.

    Handles both single-trial (1D) and multi-trial (2D) arrays.

    Args:
        audio: numpy.ndarray, shape (n_samples,) or (n_trials, n_samples).
        orig_sr: Original sampling rate in Hz.
        target_sr: Target sampling rate in Hz.

    Returns:
        numpy.ndarray with the same number of dimensions, resampled.
    """
    if orig_sr == target_sr:
        return audio

    if audio.ndim == 1:
        return librosa.resample(
            audio.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr
        ).astype(np.float32)

    # Multi-trial: resample each trial
    n_trials = audio.shape[0]
    target_len = int(audio.shape[1] * target_sr / orig_sr)
    resampled = np.zeros((n_trials, target_len), dtype=np.float32)

    for i in range(n_trials):
        resampled[i] = librosa.resample(
            audio[i].astype(np.float64), orig_sr=orig_sr, target_sr=target_sr
        ).astype(np.float32)

    return resampled


def ensure_mono(audio):
    """
    Ensure audio is mono. If stereo (last dim == 2), average channels.

    Args:
        audio: numpy.ndarray of shape (..., n_samples) or (..., n_samples, 2).

    Returns:
        numpy.ndarray with mono audio.
    """
    if audio.ndim >= 2 and audio.shape[-1] == 2:
        return np.mean(audio, axis=-1).astype(np.float32)
    return audio.astype(np.float32)


def get_hdf5_structure(hdf5_path):
    """
    Inspect and return the structure of an HDF5 file for debugging.

    Args:
        hdf5_path: Path to the HDF5 file.

    Returns:
        dict mapping key paths to (shape, dtype) tuples.
    """
    structure = {}

    def _inspect(name, obj):
        if isinstance(obj, h5py.Dataset):
            structure[name] = (obj.shape, str(obj.dtype))

    with h5py.File(hdf5_path, 'r') as f:
        f.visititems(_inspect)

    return structure
