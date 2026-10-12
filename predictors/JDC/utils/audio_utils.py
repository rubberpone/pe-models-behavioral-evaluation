"""
Audio utility functions for JDC pitch estimation.

Provides HDF5 loading, resampling, mono conversion, and the critical
dB-scale log-magnitude spectrogram feature extraction following the
official JDC preprocessing pipeline (featureExtraction.py).
"""

import os
import numpy as np
import librosa
import h5py
import warnings


def load_audio_from_hdf5(hdf5_path, audio_key, idx=None):
    """
    Load audio waveform from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        audio_key: Dataset key (e.g., 'tone_in_noise', 'stimuli/signal')
        idx: Optional index to load a single stimulus (int or slice)

    Returns:
        (audio, sr) — numpy array and sampling rate
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Navigate to dataset
        parts = audio_key.split('/')
        obj = f
        for part in parts:
            obj = obj[part]

        if idx is not None:
            audio = obj[idx]
        else:
            audio = obj[:]

        # Get sampling rate
        if 'sr' in f:
            sr = int(np.array(f['sr']).flat[0])
        elif 'config_tone' in f and 'fs' in f['config_tone']:
            sr = int(np.array(f['config_tone']['fs']).flat[0])
        else:
            sr = 32000  # default

    return audio, sr


def get_hdf5_num_stimuli(hdf5_path, audio_key):
    """Get the number of stimuli in an HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        parts = audio_key.split('/')
        obj = f
        for part in parts:
            obj = obj[part]
        return obj.shape[0]


def ensure_mono(audio):
    """
    Convert audio to mono.

    Args:
        audio: numpy array of shape (samples,) or (channels, samples)

    Returns:
        1D numpy array
    """
    if audio.ndim > 1:
        # Take mean across channels
        audio = np.mean(audio, axis=0)
    return audio.astype(np.float32)


def resample_audio(audio, orig_sr, target_sr=8000):
    """
    Resample audio to target sampling rate.

    Args:
        audio: 1D numpy array
        orig_sr: Original sampling rate
        target_sr: Target sampling rate (default 8000 for JDC)

    Returns:
        1D numpy array at target_sr
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    return librosa.resample(
        audio.astype(np.float64),
        orig_sr=orig_sr,
        target_sr=target_sr,
    ).astype(np.float32)


def extract_jdc_feature(
    audio,
    sr=8000,
    n_frames=31,
    hop_length=80,
    n_fft=1024,
    mean_path=None,
    std_path=None,
):
    """
    Extract dB-scale log-magnitude spectrogram features for JDC, following the
    official preprocessing pipeline (featureExtraction.py).

    Steps:
      1. STFT with 1024-pt Hann window, 80-sample hop, win_length=1024
      2. Magnitude spectrum → dB scale via librosa.power_to_db(ref=np.max)
      3. Take first 513 frequency bins (0–4000 Hz)
      4. Zero-pad END to make total frames divisible by n_frames
      5. Split into non-overlapping n_frames chunks (step=n_frames)
      6. Z-score normalize using pre-computed training statistics
      7. Add channel dimension: (N, n_frames, 513, 1)

    Args:
        audio: 1D numpy array, already resampled to 8 kHz
        sr: Sampling rate (must be 8000)
        n_frames: Frames per chunk (31)
        hop_length: STFT hop in samples (80 = 10 ms at 8 kHz)
        n_fft: FFT window size (1024)
        mean_path: Path to x_data_mean_total_31.npy
        std_path: Path to x_data_std_total_31.npy

    Returns:
        x_test: numpy array of shape (N, 31, 513, 1)
    """
    assert sr == 8000, f"JDC requires 8 kHz audio, got {sr} Hz"

    # 1. STFT: 1024-pt Hann window, 80-sample hop, win_length=1024
    D = librosa.stft(
        audio.astype(np.float64),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hann',
    )
    magnitude = np.abs(D)  # shape: (513, T)

    # 2. Convert to dB scale: 10 * log10(|S| / max(|S|))
    #    This is NOT np.log() — it's power_to_db (dB scale)
    log_spec = librosa.power_to_db(magnitude, ref=np.max)  # shape: (513, T)
    log_spec = log_spec.astype(np.float32)

    num_frames_total = log_spec.shape[1]

    # 3. Pad END with zeros to make total frames divisible by n_frames
    pad_num = num_frames_total % n_frames
    if pad_num != 0:
        len_pad = n_frames - pad_num
        padding = np.zeros(shape=(513, len_pad), dtype=np.float32)
        log_spec = np.concatenate((log_spec, padding), axis=1)
        num_frames_total = num_frames_total + len_pad

    # 4. Split into NON-OVERLAPPING chunks (step = n_frames)
    x_test = []
    for j in range(0, num_frames_total, n_frames):
        chunk = log_spec[:, j:j + n_frames].T  # shape: (n_frames, 513)
        x_test.append(chunk)
    x_test = np.array(x_test, dtype=np.float32)  # shape: (N, 31, 513)

    # 5. Z-score normalize using pre-computed training statistics
    if mean_path is not None and std_path is not None:
        x_train_mean = np.load(mean_path)
        x_train_std = np.load(std_path)
        x_test = (x_test - x_train_mean) / (x_train_std + 1e-4)
    else:
        warnings.warn(
            "No mean/std paths provided; skipping Z-score normalization. "
            "Predictions may be unreliable."
        )

    # 6. Add channel dimension
    x_test = x_test[:, :, :, np.newaxis]  # shape: (N, 31, 513, 1)

    return x_test


def get_num_original_frames(audio_samples, sr=8000, hop_length=80, n_fft=1024):
    """
    Calculate the number of STFT frames for given audio length without padding.

    Args:
        audio_samples: Number of audio samples
        sr: Sampling rate
        hop_length: STFT hop length
        n_fft: FFT size

    Returns:
        Number of STFT frames
    """
    # Following librosa's frame calculation
    n_frames = 1 + (audio_samples - n_fft) // hop_length
    return max(0, n_frames)
