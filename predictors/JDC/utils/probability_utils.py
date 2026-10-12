"""
Probability utility functions for mapping JDC's 722-bin output to the
700-bin framework used by the Josh2021 psychophysical experiments.

Core function: jdc_to_700 — Gaussian interpolation from 721 pitch bins
(excluding non-voice class at index 0) to 700 equally-spaced cents bins.
"""

import numpy as np
import warnings


def get_cents_from_hz(freq_hz, ref_hz=27.5):
    """
    Convert frequency in Hz to cents relative to a reference frequency.

    Args:
        freq_hz: Frequency in Hz (scalar or array)
        ref_hz: Reference frequency in Hz (default: 27.5 Hz = A0)

    Returns:
        Cents value(s)
    """
    return 1200.0 * np.log2(np.maximum(freq_hz, 1e-10) / ref_hz)


def midi_to_hz(midi_note):
    """
    Convert MIDI note number to frequency in Hz.

    Args:
        midi_note: MIDI note number (69 = A4 = 440 Hz)

    Returns:
        Frequency in Hz
    """
    return 440.0 * (2 ** ((midi_note - 69.0) / 12.0))


def hz_to_midi(freq_hz):
    """
    Convert frequency in Hz to MIDI note number.

    Args:
        freq_hz: Frequency in Hz

    Returns:
        MIDI note number
    """
    return 69.0 + 12.0 * np.log2(np.maximum(freq_hz, 1e-10) / 440.0)


def jdc_to_700(activation_722, sigma=1.5):
    """
    Convert JDC's 722-bin activation (index 0 = non-voice, indices 1–721 = pitch)
    to a 700-bin probability distribution over the Josh2021 grid.

    This performs distribution→distribution Gaussian interpolation, NOT
    argmax-then-Gaussian (which is what PESTO does).

    Args:
        activation_722: numpy array of shape (722,) — softmax probabilities
                         from the JDC model. DO NOT re-apply softmax.
        sigma: Smoothing parameter in target bins (default: 1.5 bins ≈ 9.3 cents)

    Returns:
        labels_pred: int, index (0–699) of bin with maximum probability
        probs_700: numpy array of shape (700,) — normalized probability distribution
    """
    # 1. Extract pitch probabilities (EXCLUDE non-voice class at index 0)
    #    CRITICAL: Use activation_722[1:] (indices 1..721), NOT activation_722[:721]
    pitch_probs = activation_722[1:].copy()  # shape: (721,)

    # 2. JDC source grid: 721 pitch bins starting from D2 (MIDI 38)
    #    pitch_probs[0] → MIDI 38 (D2, 73.416 Hz)
    #    pitch_probs[720] → MIDI 83 (B5, 987.77 Hz)
    #    Resolution: 1/16 semitone = 6.25 cents
    midi_notes = 38.0 + np.arange(721) / 16.0
    freqs_jdc = midi_to_hz(midi_notes)

    # 3. Josh target grid: 700 bins from 80 Hz, 6.25 cents spacing
    cents_per_bin_josh = 100.0 / 16.0  # 6.25 cents
    n_bins_josh = 700
    freqs_josh = 80.0 * (2 ** (np.arange(n_bins_josh) * cents_per_bin_josh / 1200.0))

    # 4. Unified cent reference at 27.5 Hz (consistent with Josh2021 framework)
    ref_freq = 27.5
    cents_jdc = get_cents_from_hz(freqs_jdc, ref_freq)
    cents_josh = get_cents_from_hz(freqs_josh, ref_freq)

    # 5. Gaussian interpolation: distribution → distribution
    sigma_cents = sigma * cents_per_bin_josh  # ≈ 9.375 cents for sigma=1.5
    probs_700 = np.zeros(n_bins_josh, dtype=np.float64)

    for j, cj in enumerate(cents_josh):
        distances = cents_jdc - cj
        mask = np.abs(distances) < 3.0 * sigma_cents
        if not np.any(mask):
            continue
        weights = np.exp(-(distances[mask] ** 2) / (2.0 * sigma_cents ** 2))
        probs_700[j] = np.sum(pitch_probs[mask] * weights)

    # 6. Normalize
    if pitch_probs.sum() == 0:
        warnings.warn("pitch_probs sum is zero; returning uniform distribution")
        return 0, np.ones(n_bins_josh, dtype=np.float64) / n_bins_josh

    if probs_700.sum() > 0:
        probs_700 /= probs_700.sum()
    else:
        # Fallback: uniform distribution
        warnings.warn("probs_700 sum is zero after interpolation; returning uniform distribution")
        probs_700 = np.ones(n_bins_josh, dtype=np.float64) / n_bins_josh

    labels_pred = int(np.argmax(probs_700))
    return labels_pred, probs_700.astype(np.float64)
