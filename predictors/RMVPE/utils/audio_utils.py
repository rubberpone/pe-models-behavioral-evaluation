#!/usr/bin/env python3
"""
Audio utility functions for RMVPE psychophysics pipeline.

Handles:
  - Loading audio from HDF5 files
  - Resampling (32 kHz → 16 kHz)
  - Mono conversion
  - Zero-padding to minimum length (≥ 2048 samples)

Note: The E2E model internally computes mel-spectrograms.
We do NOT pre-compute mel features — only audio-level preprocessing.
"""

import numpy as np
import torch
import torchaudio
import h5py
import warnings
from typing import Tuple, Optional


def load_audio_from_hdf5(
    hdf5_path: str,
    audio_key: str,
    idx: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio waveform from an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file.
        audio_key: Internal HDF5 path to the audio dataset
                   (e.g. 'tone_in_noise', 'stimuli/signal').
        idx: If provided, load only stimulus at this index.
             If None, load the entire dataset.

    Returns:
        audio: numpy array of shape (n_samples,) or (n_stimuli, n_samples).
        sr: Sampling rate in Hz (extracted from HDF5 metadata if available,
            otherwise defaults to 32000).
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Navigate to the audio dataset via the key path
        parts = audio_key.split('/')
        obj = f
        for part in parts:
            obj = obj[part]

        if idx is not None:
            audio = obj[idx]
        else:
            audio = obj[:]

        # Try to extract sampling rate from HDF5 metadata
        sr = _extract_sampling_rate(f, default=32000)

    # Ensure numpy array
    audio = np.asarray(audio, dtype=np.float32)

    return audio, sr


def _extract_sampling_rate(hdf5_file: h5py.File, default: int = 32000) -> int:
    """
    Extract sampling rate from HDF5 metadata.

    Searches for 'sr', 'fs', 'config_tone/fs' keys in the file.
    Returns the default value (32000) if not found.
    """
    # Common locations for sampling rate metadata
    candidate_keys = [
        'sr',
        'config_tone/fs',
    ]

    for key in candidate_keys:
        try:
            parts = key.split('/')
            obj = hdf5_file
            for part in parts:
                obj = obj[part]
            sr = int(obj[()])
            return sr
        except (KeyError, TypeError, ValueError):
            continue

    return default


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Ensure audio is single-channel (mono).

    If the audio has multiple channels, averages them.
    If already 1-D, returns unchanged.

    Args:
        audio: numpy array of shape (n_samples,) or (n_channels, n_samples).

    Returns:
        Mono audio as 1-D numpy array.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return audio.squeeze(0)
        # Average channels
        return audio.mean(axis=0)

    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Resample audio to target sampling rate.

    Uses torchaudio.transforms.Resample for high-quality resampling.

    Args:
        audio: 1-D numpy array.
        orig_sr: Original sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 16000 for RMVPE).

    Returns:
        Resampled audio as 1-D numpy array.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    audio_tensor = torch.from_numpy(audio.astype(np.float32))

    # Add batch and channel dims: (1, 1, T)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
    )
    resampled = resampler(audio_tensor)

    return resampled.squeeze(0).numpy().astype(np.float32)


def pad_to_min_length(
    audio: np.ndarray,
    target_len: int = 2048,
    mode: str = 'constant',
) -> np.ndarray:
    """
    Pad audio symmetrically to at least target_len samples.

    This is CRITICAL for RMVPE: the model's internal STFT uses a 2048-sample
    Hann window with NO center padding. If the audio is shorter than 2048
    samples, the STFT will produce 0 frames.

    Padding is symmetric (equal amounts on left and right) with zero-fill.

    Args:
        audio: 1-D numpy array.
        target_len: Minimum length in samples (default: 2048 = RMVPE's n_fft).
        mode: Padding mode — 'constant' for zero-padding,
              'reflect' for reflect padding (matches official code's intent).

    Returns:
        Padded audio as 1-D numpy array of length max(len(audio), target_len).
    """
    audio = np.asarray(audio, dtype=np.float32)
    current_len = len(audio)

    if current_len >= target_len:
        return audio

    pad_total = target_len - current_len
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    if mode == 'constant':
        audio = np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0)
    elif mode == 'reflect':
        audio = np.pad(audio, (pad_left, pad_right), mode='reflect')
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    return audio


def preprocess_audio(
    audio: np.ndarray,
    orig_sr: int = 32000,
    target_sr: int = 16000,
    min_len: int = 2048,
    pad_mode: str = 'constant',
) -> Tuple[np.ndarray, int]:
    """
    Complete audio preprocessing pipeline for RMVPE.

    Steps:
      1. Ensure mono
      2. Resample to 16 kHz
      3. Pad to ≥ 2048 samples (symmetric zero-padding)

    Args:
        audio: Raw audio as numpy array.
        orig_sr: Original sampling rate (default: 32000).
        target_sr: Target sampling rate (default: 16000).
        min_len: Minimum length in samples (default: 2048).
        pad_mode: Padding mode ('constant' or 'reflect').

    Returns:
        audio_processed: 1-D numpy array (float32, mono, 16 kHz, ≥ 2048 samples).
        n_original_samples: Number of samples after resampling but before padding.
    """
    # 1. Mono
    audio = ensure_mono(audio)

    # 2. Resample
    audio = resample_audio(audio, orig_sr, target_sr)
    n_original = len(audio)

    # 3. Pad
    audio = pad_to_min_length(audio, target_len=min_len, mode=pad_mode)

    return audio, n_original
