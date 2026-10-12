"""
Audio utility functions for FCN psychophysics pipeline.

Handles: HDF5 loading, resampling to 8 kHz, mono conversion,
symmetric zero-padding, and global mean/variance normalization.
"""

import numpy as np
from scipy.signal import resample_poly


def load_audio_from_hdf5(hdf5_path, audio_key):
    """
    Load audio waveform from HDF5 file.

    Args:
        hdf5_path: path to HDF5 file
        audio_key: key path like 'tone_in_noise' or 'stimuli/signal'

    Returns:
        tuple of (audio_array, sample_rate)
        audio_array: numpy array of shape (N_segments, N_samples) or None
        sample_rate: sampling rate in Hz
    """
    import h5py

    try:
        with h5py.File(hdf5_path, 'r') as f:
            parts = audio_key.split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            audio = obj[:]

            # Try to get sampling rate from config
            sr = 32000  # default
            try:
                # Different HDF5 files may store sr differently
                for sr_key in ['sr', 'config_tone/fs', 'pin_fs']:
                    try:
                        parts_sr = sr_key.split('/')
                        sr_obj = f
                        for part in parts_sr:
                            sr_obj = sr_obj[part]
                        sr = int(float(sr_obj[()]))
                        break
                    except (KeyError, ValueError):
                        continue
            except Exception:
                pass

            return audio, sr
    except Exception as e:
        print(f"  [ERROR] Failed to read '{audio_key}' from {hdf5_path}: {e}")
        return None, None


def resample_audio(audio, orig_sr, target_sr=8000):
    """
    Resample audio to target sampling rate using scipy's resample_poly.

    Args:
        audio: numpy array of shape (..., samples)
               Can be 1D (samples,) or 2D (N, samples)
        orig_sr: original sampling rate in Hz
        target_sr: target sampling rate in Hz (default: 8000)

    Returns:
        resampled numpy array with same number of leading dimensions
    """
    if orig_sr == target_sr:
        return audio.copy()

    # Compute resampling ratio
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    was_1d = (audio.ndim == 1)
    if was_1d:
        audio = audio.reshape(1, -1)

    resampled = resample_poly(audio.astype(np.float64), up=up, down=down, axis=1)

    if was_1d:
        resampled = resampled.squeeze(0)

    return resampled.astype(np.float32)


def ensure_mono(audio):
    """
    Ensure audio is mono (single channel).

    Args:
        audio: numpy array of shape (samples,) or (N, samples) or (N, channels, samples)

    Returns:
        numpy array with single channel dimension removed if applicable.
        For 2D (N, samples) input, assumes already mono and returns as-is.
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # Shape (N, samples) — already mono, N segments
        return audio
    elif audio.ndim == 3:
        # Shape (N, channels, samples) — take first channel
        return audio[:, 0, :]
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")


def pad_audio_to_min_length(audio, min_length):
    """
    Symmetrically zero-pad audio to meet FCN minimum input length.

    For a batch of segments (2D), pads each segment independently.

    Args:
        audio: 1D numpy array (samples,) or 2D (N, samples)
        min_length: minimum input length for the chosen FCN variant

    Returns:
        padded audio of shape (samples,)=min_length or (N, min_length)
    """
    was_1d = (audio.ndim == 1)
    if was_1d:
        audio = audio.reshape(1, -1)

    audio_len = audio.shape[1]

    if audio_len >= min_length:
        # If longer, take center portion
        start = (audio_len - min_length) // 2
        result = audio[:, start:start + min_length]
        if was_1d:
            result = result.squeeze(0)
        return result

    pad_len = min_length - audio_len
    pad_left = pad_len // 2
    pad_right = pad_len - pad_left

    result = np.pad(audio, ((0, 0), (pad_left, pad_right)), mode='constant')

    if was_1d:
        result = result.squeeze(0)

    return result


def normalize_audio(audio):
    """
    Global mean/variance normalization.

    Approximates the official FCN sliding-window normalization
    (window size = model_input_size, wrap boundary handling).

    Args:
        audio: numpy array of shape (samples,) or (N, samples)

    Returns:
        normalized audio with same shape, mean≈0, std≈1
    """
    was_1d = (audio.ndim == 1)
    if was_1d:
        audio = audio.reshape(1, -1)

    # Per-segment normalization
    mean = np.mean(audio, axis=1, keepdims=True)
    std = np.std(audio, axis=1, keepdims=True)

    normalized = (audio - mean) / (std + 1e-8)

    if was_1d:
        normalized = normalized.squeeze(0)

    return normalized


def preprocess_single(audio_32khz, min_input_length):
    """
    Full preprocessing pipeline for a single 32 kHz audio segment.

    Steps:
      1. Ensure mono
      2. Take first 50 ms (1600 samples at 32 kHz)
      3. Resample to 8 kHz → 400 samples
      4. Symmetric zero-pad to min_input_length
      5. Global mean/variance normalization

    Args:
        audio_32khz: 1D numpy array of 32 kHz audio (can be longer than 50ms)
        min_input_length: minimum input length for the FCN variant

    Returns:
        tuple of (audio_norm_1d, n_pad_left)
        audio_norm_1d: 1D numpy array of length min_input_length, normalized
        n_pad_left: number of zeros padded on the left (for reference)
    """
    # 1. Ensure mono
    audio = ensure_mono(audio_32khz)

    # 2. Take first 50 ms at 32 kHz
    samples_50ms_32k = int(32000 * 0.05)  # 1600
    if len(audio) > samples_50ms_32k:
        audio = audio[:samples_50ms_32k]

    # 3. Resample to 8 kHz
    audio_8k = resample_audio(audio, orig_sr=32000, target_sr=8000)

    # 4. Symmetric zero-pad
    orig_len = len(audio_8k)
    if orig_len < min_input_length:
        pad_len = min_input_length - orig_len
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
    else:
        pad_len = 0
        pad_left = 0
        pad_right = 0

    audio_padded = pad_audio_to_min_length(audio_8k, min_input_length)

    # 5. Global normalization
    audio_norm = normalize_audio(audio_padded)

    return audio_norm, pad_left


def preprocess_batch(audio_batch_32khz, min_input_length):
    """
    Full preprocessing pipeline for a batch of 32 kHz audio segments.

    Args:
        audio_batch_32khz: numpy array of shape (N, samples) at 32 kHz
        min_input_length: minimum input length for the FCN variant

    Returns:
        numpy array of shape (N, min_input_length), normalized
    """
    N = audio_batch_32khz.shape[0]
    results = np.zeros((N, min_input_length), dtype=np.float32)

    for i in range(N):
        results[i], _ = preprocess_single(
            audio_batch_32khz[i], min_input_length
        )

    return results


if __name__ == '__main__':
    # Quick test
    import sys

    # Test with synthetic 32 kHz audio
    test_audio = np.random.randn(4800).astype(np.float32)  # 150ms at 32kHz

    for min_len in [1953, 993, 929]:
        processed, pad_left = preprocess_single(test_audio, min_len)
        print(f"min_input={min_len}: output_len={len(processed)}, "
              f"pad_left={pad_left}, "
              f"mean={np.mean(processed):.4f}, std={np.std(processed):.4f}")
