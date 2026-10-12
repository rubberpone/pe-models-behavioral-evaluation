"""Audio utility functions for the PENN psychophysics pipeline.

Handles:
  - Loading audio from HDF5 stimulus files
  - Mono conversion
  - Multi-window extraction (for original 16 kHz CREPE/DeepF0 configs only;
    not needed for ++ models)
"""

import numpy as np
import h5py


def load_audio_from_hdf5(hdf5_path: str, audio_key: str):
    """Load audio waveform from an HDF5 stimulus file.

    Args:
        hdf5_path: Path to the HDF5 file.
        audio_key: Key path inside the HDF5 file, e.g. 'tone_in_noise'
                   or 'stimuli/signal'.

    Returns:
        numpy.ndarray of shape (N, samples) and float32 dtype, or None on failure.
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            parts = audio_key.split('/')
            obj = f
            for part in parts:
                obj = obj[part]
            audio = obj[:]
        return np.asarray(audio, dtype=np.float32)
    except Exception as e:
        print(f"  [ERROR] Failed to read '{audio_key}' from {hdf5_path}: {e}")
        return None


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels.

    Args:
        audio: numpy array of shape (..., channels) or (samples,).

    Returns:
        1D numpy array of shape (samples,).
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        # Shape could be (samples, channels) or (channels, samples).
        # For our HDF5 data, shape is (N_stimuli, samples) when loading
        # a single stimulus; ensure_mono is called per-stimulus.
        if audio.shape[0] < audio.shape[1]:
            # (channels, samples) — average across channels
            return audio.mean(axis=0)
        else:
            # (samples, channels) — average across channels
            return audio.mean(axis=1)
    # Higher-dimensional: flatten trailing dims by taking first channel
    return audio[..., 0]


def extract_overlapping_frames(audio, sr=16000, window_ms=64, hop_ms=10, num_frames=5):
    """Extract overlapping frames from a short audio clip.

    This is used ONLY for the original CREPE/DeepF0 configs (16 kHz, 64 ms window).
    All ++ models (FCNF0++, CREPE++, DeepF0++) do NOT need this function —
    they use Strategy 1 (direct inference) with 128 ms window at 8 kHz.

    Args:
        audio: 1D numpy array.
        sr: Sampling rate in Hz (default 16000 for original configs).
        window_ms: Window length in ms (default 64).
        hop_ms: Hop length in ms (default 10).
        num_frames: Number of frames to extract (default 5).

    Returns:
        List of 1D numpy arrays, each of length `int(sr * window_ms / 1000)`.
    """
    window_len = int(sr * window_ms / 1000)   # 1024 at 16 kHz
    hop_len = int(sr * hop_ms / 1000)         # 160 at 16 kHz
    audio_len = len(audio)
    center_shift = (window_len - audio_len) // 2

    frames = []
    for i in range(-(num_frames // 2), num_frames // 2 + 1):
        offset = i * hop_len
        frame = np.zeros(window_len, dtype=np.float32)
        src_start = 0
        src_end = audio_len
        dst_start = center_shift + offset
        dst_end = dst_start + audio_len

        if dst_start < 0:
            src_start = -dst_start
            dst_start = 0
        if dst_end > window_len:
            src_end = window_len - dst_start
            dst_end = window_len

        frame[dst_start:dst_end] = audio[src_start:src_end]
        frames.append(frame)

    return frames
