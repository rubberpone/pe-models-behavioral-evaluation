#!/usr/bin/env python3
"""
utils/audio_utils.py — Audio helper functions (standby).

The psychophysical stimuli are pre-generated and stored in HDF5 files,
so these synthesis functions are not needed for the main inference pipeline.
They are provided for reference and potential future use.
"""

import numpy as np


def generate_harmonic(
    f0: float,
    duration: float = 0.05,
    sr: int = 32000,
    n_harmonics: int = 12,
    phase_mode: str = 'sine',
    roll_off: float = 0.0,
) -> np.ndarray:
    """
    Generate a harmonic complex tone.

    Args:
        f0: Fundamental frequency in Hz.
        duration: Duration in seconds (default 50 ms).
        sr: Sampling rate in Hz.
        n_harmonics: Number of harmonics to include.
        phase_mode: 'sine' (all cosine-phase), 'random', or 'alternating'.
        roll_off: Spectral roll-off in dB per octave (0 = flat).

    Returns:
        Audio waveform as float32 numpy array.
    """
    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr
    waveform = np.zeros(n_samples, dtype=np.float64)

    for h in range(1, n_harmonics + 1):
        freq = f0 * h
        if freq >= sr / 2:
            break

        # Amplitude with optional roll-off
        amp = 10.0 ** (-roll_off * np.log2(max(h, 1)) / 20.0)

        # Phase
        if phase_mode == 'sine':
            phase = 0.0
        elif phase_mode == 'alternating':
            phase = 0.0 if h % 2 == 0 else np.pi / 2
        elif phase_mode == 'random':
            phase = np.random.uniform(0, 2 * np.pi)
        else:
            phase = 0.0

        waveform += amp * np.cos(2 * np.pi * freq * t + phase)

    # Normalize
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform /= peak

    return waveform.astype(np.float32)


def generate_transposed(
    f_carrier: float,
    f_envelope: float,
    duration: float = 0.05,
    sr: int = 32000,
) -> np.ndarray:
    """
    Generate a transposed tone (Oxenham et al. 2004 style).

    A high-frequency carrier is half-wave rectified and low-pass filtered
    to create a temporal envelope, which then modulates a tone at f_carrier.

    Args:
        f_carrier: Carrier frequency in Hz.
        f_envelope: Envelope repetition rate in Hz (perceived pitch).
        duration: Duration in seconds.
        sr: Sampling rate in Hz.

    Returns:
        Audio waveform as float32 numpy array.
    """
    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr

    # Half-wave rectified sinusoid at f_envelope
    envelope = np.maximum(0.0, np.sin(2 * np.pi * f_envelope * t))
    # Simple low-pass filter the envelope
    envelope = np.convolve(envelope, np.ones(50) / 50, mode='same')

    # Apply envelope to carrier
    carrier = np.sin(2 * np.pi * f_carrier * t)
    waveform = envelope * carrier

    # Normalize
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform /= peak

    return waveform.astype(np.float32)


def ensure_mono(waveform: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel audio to mono by averaging channels.

    Args:
        waveform: Array of shape (samples,), (channels, samples), or (batch, channels, samples).

    Returns:
        Mono waveform.
    """
    waveform = np.asarray(waveform)

    if waveform.ndim == 1:
        return waveform
    elif waveform.ndim == 2:
        # (channels, samples) or (batch, samples)
        # If first dim is small, treat as channels
        if waveform.shape[0] <= 16:
            return waveform.mean(axis=0)
        else:
            return waveform  # assume (batch, samples)
    elif waveform.ndim == 3:
        # (batch, channels, samples)
        return waveform.mean(axis=1)
    else:
        raise ValueError(f'Unexpected waveform shape: {waveform.shape}')
