#!/usr/bin/env python3
"""
Audio utility functions for FT Attention psychophysics pipeline.

Handles: HDF5 loading, resampling, mono conversion, CFP computation.
"""

import os
import sys
import numpy as np
from scipy.signal import resample_poly

# Import official CFP implementation from FTANet-melodic-main
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FTANET_PATH = os.path.join(_PROJECT_ROOT, "FTANet-melodic-main")
if _FTANET_PATH not in sys.path:
    sys.path.insert(0, _FTANET_PATH)

from cfp import (
    CFP_filterbank,
    lognorm,
    norm,
    feature_extraction,
)


def load_audio_from_hdf5(hdf5_path, audio_key, idx=None):
    """
    Load audio waveform(s) from HDF5 file.

    Args:
        hdf5_path: path to HDF5 file
        audio_key: key path like 'tone_in_noise' or 'stimuli/signal'
        idx: if provided, return the idx-th segment only.
             Otherwise return the entire array.

    Returns:
        (audio, sr) where audio is ndarray and sr is the sampling rate (Hz).
        If idx is provided, audio is 1-D array of shape (samples,).
        Otherwise audio is 2-D array of shape (N_segments, samples).
    """
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        parts = audio_key.split('/')
        obj = f
        for part in parts:
            obj = obj[part]
        data = obj[:]  # (N_segments, samples) float32

        # Try to get the sampling rate from the HDF5 config
        sr = 32000  # default per specification
        for sr_key in ['config_tone/fs', 'fs', 'config_tone/dur']:
            try:
                sr_obj = f
                for part in sr_key.split('/'):
                    sr_obj = sr_obj[part]
                sr_val = sr_obj[()]
                if isinstance(sr_val, np.ndarray):
                    sr_val = sr_val.item()
                if sr_key.endswith('/fs'):
                    sr = int(sr_val)
                break
            except (KeyError, ValueError):
                continue

    if idx is not None:
        audio = data[idx]
    else:
        audio = data

    return audio, sr


def ensure_mono(audio):
    """
    Convert multi-channel audio to mono.

    Args:
        audio: ndarray, possibly multi-channel

    Returns:
        mono audio as 1-D ndarray
    """
    if audio.ndim > 1 and audio.shape[-1] > 1:
        # Multi-channel: take mean across channels
        return np.mean(audio, axis=-1)
    elif audio.ndim > 1 and min(audio.shape) == audio.shape[-1]:
        # Possible multi-channel, shape = (samples, channels)
        # Already handled above, but flatten if 2D with last dim=1
        pass
    return np.squeeze(audio)


def resample_audio(audio, orig_sr, target_sr=8000):
    """
    Resample audio to target sampling rate using polyphase filtering.

    Args:
        audio: 1-D ndarray
        orig_sr: original sampling rate (Hz)
        target_sr: target sampling rate (Hz), default 8000

    Returns:
        resampled 1-D ndarray
    """
    if orig_sr == target_sr:
        return audio.astype(np.float64)

    # Use scipy.signal.resample_poly for high-quality resampling
    # Compute gcd to find up/down factors
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    audio_f64 = audio.astype(np.float64)
    resampled = resample_poly(audio_f64, up=up, down=down)
    return resampled


def compute_cfp(audio, sr=8000, hop=80, window_size=768,
                fmin=31.0, fmax=1250.0, n_bins_per_octave=60):
    """
    Compute the CFP (Combined Frequency and Periodicity) 3-channel
    representation for FT Attention inference.

    This uses the OFFICIAL CFP implementation from FTANet-melodic-main/cfp.py.
    The three channels are:
      - tfrL0: Power-scaled log-frequency spectrogram
      - tfrLF: Generalized cepstrum (GC) in log-frequency domain
      - tfrLQ: Generalized cepstrum of spectrum (GCoS) in log-frequency domain

    Parameters MUST match training configuration:
      sr=8000, window=blackmanharris(768), hop=80, fr=2.0,
      gamma=[0.24, 0.6, 1.0], log(1+x)+min-max normalization per channel.

    Args:
        audio: 1-D ndarray, audio waveform at `sr` Hz
        sr: sampling rate (must be 8000 Hz)
        hop: hop size for STFT (must be 80)
        window_size: STFT window size (must be 768)
        fmin: minimum frequency for log-frequency mapping (31.0 Hz)
        fmax: maximum frequency for log-frequency mapping (1250.0 Hz)
        n_bins_per_octave: number of frequency bins per octave (60)

    Returns:
        cfp: ndarray of shape (3, n_freq_bins, T) — 3-channel CFP tensor
        cen_freq: list of center frequencies
        time: array of time stamps (seconds)
    """
    import scipy.signal

    # Use the official feature_extraction from cfp.py
    # It internally uses: fr=2.0, blackmanharris window, gamma=[0.24, 0.6, 1.0]
    Z, time, CenFreq, tfrL0, tfrLF, tfrLQ = feature_extraction(
        audio, sr, Hop=hop, Window=window_size,
        StartFreq=fmin, StopFreq=fmax, NumPerOct=n_bins_per_octave
    )

    # Apply log-norm and min-max normalization per channel
    tfrL0_n = norm(lognorm(tfrL0))[np.newaxis, :, :]  # (1, F, T)
    tfrLF_n = norm(lognorm(tfrLF))[np.newaxis, :, :]  # (1, F, T)
    tfrLQ_n = norm(lognorm(tfrLQ))[np.newaxis, :, :]  # (1, F, T)

    # Concatenate along channel axis: (3, F, T)
    W = np.concatenate((tfrL0_n, tfrLF_n, tfrLQ_n), axis=0)

    return W, CenFreq, time


def pad_audio_centered(audio, target_length):
    """
    Zero-pad audio to target_length, centering the original audio.

    Args:
        audio: 1-D ndarray
        target_length: desired total length in samples

    Returns:
        padded audio of length target_length
    """
    if len(audio) >= target_length:
        return audio[:target_length]

    pad_total = target_length - len(audio)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0)
