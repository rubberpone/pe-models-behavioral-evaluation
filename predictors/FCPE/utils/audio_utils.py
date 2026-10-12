"""
Audio utility functions for FCPE psychophysics pipeline.

Handles loading audio from HDF5 files, resampling, mono conversion,
and tensor preparation for FCPE inference.
"""

import numpy as np
import torch
import h5py
from typing import Optional, Tuple


def load_audio_from_hdf5(hdf5_path: str, audio_key: str) -> Optional[np.ndarray]:
    """
    Load audio waveform from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file.
        audio_key: Key path within the HDF5 file (e.g., 'tone_in_noise' or
                   'stimuli/signal').

    Returns:
        numpy array of shape (N, n_samples) or None on error.
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            parts = audio_key.split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            return obj[:]
    except Exception as e:
        print(f"  [ERROR] Failed to read '{audio_key}' from {hdf5_path}: {e}")
        return None


def resample_audio(audio: np.ndarray, orig_sr: int,
                   target_sr: int = 16000) -> np.ndarray:
    """
    Resample audio from original sampling rate to target sampling rate.

    Uses scipy's resample_poly for efficient rational-ratio resampling.

    Args:
        audio: Audio waveform of shape (..., n_samples).
        orig_sr: Original sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 16000).

    Returns:
        Resampled audio of shape (..., n_target_samples).
    """
    from scipy.signal import resample_poly

    if orig_sr == target_sr:
        return audio

    # Compute rational resampling factors
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    return resample_poly(audio.astype(np.float64), up=up, down=down, axis=-1)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is mono (single channel).

    If audio has more than one channel, take the mean across channels.
    For 1-D arrays, add a channel dimension.

    Args:
        audio: Audio waveform, shape can be (n_samples,), (n_channels, n_samples),
               or (N, n_channels, n_samples).

    Returns:
        Mono audio with shape matching input batch dimensions, squeezing
        channel dimension for 1-D inputs.
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # (n_channels, n_samples) or (N, n_samples)
        # If second dim looks like audio samples (>> 2), treat as (N, n_samples)
        if audio.shape[0] <= 16 and audio.shape[1] > 100:
            # Likely (n_channels, n_samples)
            return np.mean(audio, axis=0) if audio.shape[0] > 1 else audio[0]
        else:
            # Likely (batch, n_samples)
            return audio
    elif audio.ndim == 3:
        # (N, n_channels, n_samples)
        if audio.shape[1] <= 16:
            return np.mean(audio, axis=1)
        else:
            return audio[:, 0, :] if audio.shape[1] > 1 else audio[:, 0, :]
    return audio


def prepare_tensor(audio: np.ndarray, sr: int,
                   device: str = 'cuda') -> Tuple[torch.Tensor, int]:
    """
    Convert numpy audio array to PyTorch tensor ready for FCPE inference.

    Note: FCPE's internal Wav2Mel module handles resampling and feature
    extraction automatically. We just need to provide the waveform tensor
    with shape (1, n_samples, 1) and the correct sample rate.

    The model will internally resample to 16 kHz via torchaudio.transforms.Resample.

    Args:
        audio: 1-D numpy array of audio samples at original sample rate.
        sr: Original sample rate in Hz.
        device: Device string ('cuda' or 'cpu').

    Returns:
        tensor: PyTorch tensor of shape (1, n_samples, 1).
        sr: Sample rate (passed through for model's internal resampling).
    """
    # Ensure a clean float32 copy
    audio = np.asarray(audio, dtype=np.float32).copy()

    # Ensure 1-D
    if audio.ndim > 1:
        audio = audio.squeeze()

    # Normalize to prevent clipping if needed (peak normalization)
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak * 0.95
    elif peak < 1e-8:
        # Silent audio — return zeros
        pass

    # Shape: (1, n_samples, 1) as required by FCPE's InferCFNaiveMelPE
    tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1)
    tensor = tensor.to(device)

    return tensor, sr
