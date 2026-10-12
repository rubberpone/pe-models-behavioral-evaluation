"""Probability mapping utilities for PENN psychophysics pipeline.

Core functionality:
  - Gaussian interpolation from PENN posteriorgram to 700-class distribution
  - Frequency ↔ cents conversion
  - Convenience wrappers for ++ models (1440 bins) and original configs (360 bins)

Key parameters:
  - All ++ models: 1440 bins, 5-cent spacing, fmin=31.0 Hz
  - Original CREPE/DeepF0: 360 bins, 20-cent spacing, fmin=32.70 Hz
  - Target (Josh2021): 700 bins, 6.25-cent (100/16) spacing, fmin=80.0 Hz
  - Reference frequency for cents: 10.0 Hz (unified)
  - Gaussian sigma: 1.5 target bins (~9.3 cents)
"""

import numpy as np
import warnings


def get_cents_from_hz(freq_hz, ref_hz=10.0):
    """Convert frequency in Hz to cents relative to a reference.

    Args:
        freq_hz: Frequency in Hz (scalar or array).
        ref_hz: Reference frequency in Hz (default 10.0).

    Returns:
        Cents value(s): 1200 * log2(freq_hz / ref_hz).
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    return 1200.0 * np.log2(freq_hz / ref_hz)


def penn_posteriorgram_to_700(posteriorgram, fmin_source=31.0, cents_per_bin=5.0,
                                n_bins_target=700, fmin_target=80.0,
                                cents_per_bin_target=100 / 16, sigma=1.5):
    """Convert a PENN posteriorgram to a 700-bin probability distribution.

    Uses Gaussian interpolation (distribution → distribution) to map from
    PENN's native bin grid to the Josh2021 700-class framework.

    Args:
        posteriorgram: numpy array of shape (N_bins,) — already normalized
                       (Softmax for ++ models, Sigmoid for original configs).
        fmin_source: Minimum frequency of source (PENN) bins in Hz.
                     Default 31.0 Hz for ++ models.
        cents_per_bin: Bin spacing in cents for source.
                       Default 5.0 for ++ models.
        n_bins_target: Number of target bins (default 700).
        fmin_target: Minimum frequency of target bins in Hz (default 80.0).
        cents_per_bin_target: Target bin spacing in cents
                              (default 100/16 ≈ 6.25).
        sigma: Gaussian smoothing width in target bins (default 1.5).

    Returns:
        tuple (labels_pred: int, probs_700: np.ndarray of shape (700,))
        - labels_pred: index (0–699) of the bin with maximum probability
        - probs_700: 700-element probability vector summing to 1
    """
    n_bins_source = len(posteriorgram)
    ref_freq = 10.0  # Unified cent reference

    # --- Source (PENN) grid ---
    freqs_source = fmin_source * (2 ** (np.arange(n_bins_source) * cents_per_bin / 1200))
    cents_source = 1200.0 * np.log2(freqs_source / ref_freq)

    # --- Target (Josh2021) grid ---
    freqs_target = fmin_target * (2 ** (np.arange(n_bins_target) * cents_per_bin_target / 1200))
    cents_target = 1200.0 * np.log2(freqs_target / ref_freq)

    # --- Gaussian interpolation ---
    sigma_cents = sigma * cents_per_bin_target  # ≈ 9.3 cents for default
    probs_700 = np.zeros(n_bins_target, dtype=np.float64)

    for j, cj in enumerate(cents_target):
        distances = cents_source - cj
        mask = np.abs(distances) < 3 * sigma_cents
        if not np.any(mask):
            continue
        weights = np.exp(-(distances[mask] ** 2) / (2 * sigma_cents ** 2))
        probs_700[j] = np.sum(posteriorgram[mask] * weights)

    # --- Edge case: zero-sum posteriorgram ---
    if posteriorgram.sum() == 0:
        warnings.warn("posteriorgram sum is zero; returning uniform distribution")
        return 0, np.ones(n_bins_target) / n_bins_target

    # --- Normalize ---
    if probs_700.sum() > 0:
        probs_700 /= probs_700.sum()
    else:
        warnings.warn("probs_700 sum is zero; returning uniform distribution")
        return 0, np.ones(n_bins_target) / n_bins_target

    return int(np.argmax(probs_700)), probs_700


def fcnf0pp_to_700(posteriorgram_1440):
    """Convenience wrapper for all ++ models (FCNF0++ / CREPE++ / DeepF0++).

    All ++ models share the same output grid: 1440 bins, 5-cent spacing,
    starting at 31.0 Hz.

    Args:
        posteriorgram_1440: numpy array of shape (1440,) — Softmax-normalized
                            posteriorgram from a ++ model.

    Returns:
        tuple (labels_pred: int, probs_700: np.ndarray of shape (700,))
    """
    return penn_posteriorgram_to_700(
        posteriorgram_1440,
        fmin_source=31.0,
        cents_per_bin=5.0,
    )


def deepf0_to_700(activation_360):
    """Convenience wrapper for original CREPE/DeepF0 configs.

    Original configs: 360 bins, 20-cent spacing, starting at 32.70 Hz.
    The caller should apply Sigmoid to raw logits before calling this function
    (original configs use BCE training).

    Args:
        activation_360: numpy array of shape (360,) — Sigmoid-normalized
                        activations from original CREPE/DeepF0.

    Returns:
        tuple (labels_pred: int, probs_700: np.ndarray of shape (700,))
    """
    # Optionally renormalize (Sigmoid outputs are independent per bin)
    activation_360 = np.asarray(activation_360, dtype=np.float64)
    if activation_360.sum() > 0:
        activation_360 = activation_360 / activation_360.sum()
    return penn_posteriorgram_to_700(
        activation_360,
        fmin_source=32.70,
        cents_per_bin=20.0,
    )
