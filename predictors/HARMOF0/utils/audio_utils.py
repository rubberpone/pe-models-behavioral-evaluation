"""
Audio utility functions for HarmoF0 psychophysics pipeline.

Provides functions for loading HDF5 audio, resampling, frame extraction,
and zero-padding for HarmoF0 inference.
"""

import numpy as np
import h5py
from scipy.signal import resample_poly


def load_audio_from_hdf5(hdf5_path, idx):
    """
    Load a single audio waveform from an HDF5 dataset.

    Args:
        hdf5_path: path to the HDF5 file
        idx: row index to load

    Returns:
        audio: numpy array of shape (samples,) at original sampling rate
        sr: original sampling rate (Hz)
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Determine the audio key by inspecting the file structure
        audio_key = _find_audio_key(f)
        audio_data = f[audio_key]

        # Extract the indexed row
        audio = audio_data[idx].astype(np.float32)

        # Try to extract sampling rate from metadata
        sr = _get_sampling_rate(f)

    audio = ensure_mono(audio)
    return audio, sr


def load_all_audio_from_hdf5(hdf5_path, audio_key):
    """
    Load all audio waveforms at once from an HDF5 dataset.

    Args:
        hdf5_path: path to the HDF5 file
        audio_key: key path like 'tone_in_noise' or 'stimuli/signal'

    Returns:
        audio: numpy array of shape (N, samples) at original sampling rate
    """
    with h5py.File(hdf5_path, 'r') as f:
        parts = audio_key.split('/')
        obj = f
        for part in parts:
            obj = obj[part]
        audio = obj[:].astype(np.float32)

    # Ensure mono (if stereo, take first channel or average)
    if audio.ndim == 3:
        audio = audio.squeeze(1)  # (N, channels, samples) -> handle
    return audio


def load_all_metadata_from_hdf5(hdf5_path):
    """
    Load all 1D metadata arrays from an HDF5 file.

    Args:
        hdf5_path: path to the HDF5 file

    Returns:
        metadata: dict of {key: numpy_array} for all 1D datasets
    """
    metadata = {}
    with h5py.File(hdf5_path, 'r') as f:
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset) and item.ndim <= 1:
                data = item[:]
                # Convert to list for JSON serialization
                if data.dtype == np.int64 or data.dtype == np.int32:
                    metadata[key] = data.tolist()
                elif data.dtype == np.float64 or data.dtype == np.float32:
                    metadata[key] = data.tolist()
                else:
                    metadata[key] = data.tolist()
    return metadata


def resample_audio(audio, orig_sr, target_sr=16000):
    """
    Resample audio to target sampling rate.

    Args:
        audio: 1D numpy array
        orig_sr: original sampling rate (Hz)
        target_sr: target sampling rate (Hz), default 16000

    Returns:
        1D numpy array resampled to target_sr
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    # Calculate resampling ratio
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    # Use scipy's resample_poly for efficient resampling
    audio_64 = audio.astype(np.float64)
    resampled = resample_poly(audio_64, up, down)
    return resampled.astype(np.float32)


def extract_overlapping_frames(audio, sr=16000, window_ms=64, hop_ms=10, num_frames=5):
    """
    Extract overlapping frames from a short audio clip (multi-frame mode).

    Generates num_frames overlapping windows centred around the audio,
    each of length window_ms. Zero-pads boundaries as needed.

    Args:
        audio: 1D numpy array (e.g., 800 samples for 50ms at 16kHz)
        sr: sampling rate (Hz), default 16000
        window_ms: window length in ms, default 64
        hop_ms: hop length in ms, default 10
        num_frames: number of frames to extract, default 5

    Returns:
        List of numpy arrays, each of length window_len (1024 samples for 64ms)
    """
    window_len = int(sr * window_ms / 1000)   # 1024 samples at 16kHz
    hop_len = int(sr * hop_ms / 1000)         # 160 samples at 16kHz
    audio_len = len(audio)

    # Center the audio within the window
    center_shift = (window_len - audio_len) // 2  # 112 samples

    frames = []
    for i in range(-(num_frames // 2), num_frames // 2 + 1):
        offset = i * hop_len  # -320, -160, 0, 160, 320
        frame = np.zeros(window_len, dtype=np.float32)
        src_start = 0
        src_end = audio_len
        dst_start = center_shift + offset
        dst_end = dst_start + audio_len

        # Handle boundary cases (frame partially outside window)
        if dst_start < 0:
            src_start = -dst_start
            dst_start = 0
        if dst_end > window_len:
            src_end = window_len - dst_start
            dst_end = window_len

        # Copy the valid portion
        valid_len = min(src_end - src_start, dst_end - dst_start)
        if valid_len > 0:
            frame[dst_start:dst_start + valid_len] = audio[src_start:src_start + valid_len]

        frames.append(frame)

    return frames


def pad_to_1024(audio, sr=16000):
    """
    Symmetrically zero-pad 50ms audio (800 samples at 16kHz) to 1024 samples (64ms).

    For single-frame mode.

    Args:
        audio: 1D numpy array
        sr: sampling rate (Hz), default 16000

    Returns:
        1D numpy array of exactly 1024 samples
    """
    target_len = 1024
    if len(audio) < target_len:
        pad_len = target_len - len(audio)
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    elif len(audio) > target_len:
        # Center-crop to 1024
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    return audio.astype(np.float32)


def ensure_mono(audio):
    """
    Ensure audio is single-channel (mono).

    Args:
        audio: numpy array, possibly multi-channel

    Returns:
        1D numpy array of mono audio
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # If shape is (channels, samples), average channels
        if audio.shape[0] <= 2:
            return np.mean(audio, axis=0).astype(audio.dtype)
        # If shape is (samples, channels), average channels
        else:
            return np.mean(audio, axis=1).astype(audio.dtype)
    elif audio.ndim == 3:
        # (N, 1, samples) or similar
        return audio.squeeze()
    return audio


def _find_audio_key(hdf5_file):
    """
    Find the audio dataset key in an HDF5 file.
    Returns the path to the largest 2D dataset.
    """
    best_key = None
    best_size = 0

    def _search(name, obj):
        nonlocal best_key, best_size
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            size = obj.shape[0] * obj.shape[1] if obj.ndim == 2 else obj.shape[0]
            if size > best_size:
                best_size = size
                best_key = name

    hdf5_file.visititems(_search)
    return best_key


def _get_sampling_rate(hdf5_file):
    """
    Extract sampling rate from HDF5 metadata.
    Falls back to 32000 Hz (default for psychophysics datasets).
    """
    # Try to find sr in metadata
    if 'sr' in hdf5_file:
        return int(hdf5_file['sr'][0])
    # Try config_tone/fs
    if 'config_tone' in hdf5_file:
        config = hdf5_file['config_tone']
        if 'fs' in config:
            return int(config['fs'][0])
    # Default for psychophysics datasets
    return 32000
