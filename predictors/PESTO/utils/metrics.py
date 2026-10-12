#!/usr/bin/env python3
"""
utils/metrics.py — F0 conversion and metric computation helpers.

Provides conversion functions between:
  - F0 (Hz) and MIDI note numbers
  - Hz and cents (relative to a reference frequency)
Useful for psychophysical pitch analysis and threshold calculations.
"""

import numpy as np

# ---------------------------------------------------------------------------
# F0 ↔ MIDI
# ---------------------------------------------------------------------------

def hz_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    """
    Convert fundamental frequency in Hz to MIDI note number.

    MIDI note 69 = A4 = 440 Hz.

    Args:
        f0_hz: F0 values in Hz (scalar or array).

    Returns:
        MIDI note numbers (float).
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    # MIDI formula: m = 69 + 12 * log2(f / 440)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 69.0 + 12.0 * np.log2(np.maximum(f0_hz, 1e-12) / 440.0)
    return result


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """
    Convert MIDI note number to fundamental frequency in Hz.

    Args:
        midi: MIDI note numbers (scalar or array).

    Returns:
        F0 values in Hz.
    """
    midi = np.asarray(midi, dtype=np.float64)
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# ---------------------------------------------------------------------------
# Hz ↔ Cents
# ---------------------------------------------------------------------------

def hz_to_cents(f0_hz: np.ndarray, f_ref_hz: float = 440.0) -> np.ndarray:
    """
    Convert frequency in Hz to cents relative to a reference frequency.

    One cent is 1/100 of a semitone.
    cents = 1200 * log2(f0 / f_ref)

    Args:
        f0_hz: F0 values in Hz.
        f_ref_hz: Reference frequency in Hz (default: 440 Hz).

    Returns:
        Cents relative to f_ref_hz.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 1200.0 * np.log2(np.maximum(f0_hz, 1e-12) / f_ref_hz)
    return result


def cents_to_hz(cents: np.ndarray, f_ref_hz: float = 440.0) -> np.ndarray:
    """
    Convert cents (relative to f_ref_hz) to frequency in Hz.

    Args:
        cents: Cents relative to f_ref_hz.
        f_ref_hz: Reference frequency in Hz (default: 440 Hz).

    Returns:
        Frequency in Hz.
    """
    cents = np.asarray(cents, dtype=np.float64)
    return f_ref_hz * (2.0 ** (cents / 1200.0))


# ---------------------------------------------------------------------------
# Cent difference between two frequencies
# ---------------------------------------------------------------------------

def cent_difference(f1_hz: np.ndarray, f2_hz: np.ndarray) -> np.ndarray:
    """
    Compute the absolute cent difference between two frequencies.

    Args:
        f1_hz, f2_hz: Frequencies in Hz.

    Returns:
        Absolute difference in cents.
    """
    f1_hz = np.asarray(f1_hz, dtype=np.float64)
    f2_hz = np.asarray(f2_hz, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        diff = np.abs(1200.0 * np.log2(
            np.maximum(f1_hz, 1e-12) / np.maximum(f2_hz, 1e-12)
        ))
    return diff


# ---------------------------------------------------------------------------
# Convenience: aggregate frame-level predictions
# ---------------------------------------------------------------------------

def aggregate_pitch(pitch_hz: np.ndarray, method: str = 'median') -> float:
    """
    Aggregate frame-level pitch predictions into a single F0 value.

    For steady-state 50-ms stimuli, PESTO outputs ~5 frames of predictions.
    The median is the recommended aggregation to mitigate VQT edge effects.

    Args:
        pitch_hz: Per-frame pitch predictions in Hz.
        method: Aggregation method — 'median' (default) or 'mean'.

    Returns:
        Aggregated F0 value in Hz.
    """
    pitch_hz = np.asarray(pitch_hz, dtype=np.float64)
    valid = pitch_hz[pitch_hz > 0]  # exclude zero/unvoiced frames

    if len(valid) == 0:
        return 0.0

    if method == 'median':
        return float(np.median(valid))
    elif method == 'mean':
        return float(np.mean(valid))
    else:
        raise ValueError(f"Unknown aggregation method: '{method}'")
