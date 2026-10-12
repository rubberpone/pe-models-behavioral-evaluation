"""
Probability utility functions for HarmoF0 psychophysics pipeline.

Core function: harmof0_to_700 — converts HarmoF0's 352-bin activation
to a 700-bin probability distribution via Gaussian interpolation.
"""

import numpy as np
import warnings

# -----------------------------------------------------------------------
# Pre-compute Gaussian interpolation kernel (352 → 700)
# This is constant regardless of the activation values.
# -----------------------------------------------------------------------

# HarmoF0 grid: 352 bins from 27.5 Hz, 25-cent spacing (Q=48)
_FREQS_HARMOF0 = 27.5 * (2.0 ** (np.arange(352) / 48.0))
_CENTS_HARMOF0 = 1200.0 * np.log2(_FREQS_HARMOF0 / 27.5)

# Josh2021 grid: 700 bins from 80 Hz, 1/16 semitone spacing (6.25 cents)
_CENTS_PER_BIN_JOSH = 100.0 / 16.0  # 6.25 cents
_FREQS_JOSH = 80.0 * (2.0 ** (np.arange(700) * _CENTS_PER_BIN_JOSH / 1200.0))
_CENTS_JOSH = 1200.0 * np.log2(_FREQS_JOSH / 27.5)

_SIGMA_DEFAULT = 1.5
_SIGMA_CENTS = _SIGMA_DEFAULT * _CENTS_PER_BIN_JOSH  # sigma in cent units

# Pre-compute the Gaussian weight matrix: shape (700, 352)
# Each row j contains weights from each of the 352 HarmoF0 bins to target bin j.
# Weights outside ±3*sigma are set to 0.
_KERNEL = np.zeros((700, 352), dtype=np.float64)
for j in range(700):
    cj = _CENTS_JOSH[j]
    distances = _CENTS_HARMOF0 - cj
    mask = np.abs(distances) < (3.0 * _SIGMA_CENTS)
    if np.any(mask):
        _KERNEL[j, mask] = np.exp(-(distances[mask] ** 2) / (2.0 * _SIGMA_CENTS ** 2))


def get_cents_from_hz(freq_hz, ref_hz=27.5):
    """
    Convert frequency in Hz to cents relative to a reference frequency.

    Args:
        freq_hz: frequency in Hz (scalar or array)
        ref_hz: reference frequency in Hz, default 27.5

    Returns:
        cents value(s)
    """
    return 1200.0 * np.log2(freq_hz / ref_hz)


def get_hz_from_cents(cents, ref_hz=27.5):
    """
    Convert cents relative to reference back to Hz.

    Args:
        cents: cents value(s)
        ref_hz: reference frequency in Hz, default 27.5

    Returns:
        frequency in Hz
    """
    return ref_hz * (2.0 ** (cents / 1200.0))


def harmof0_to_700(activation_352, sigma=1.5):
    """
    Convert HarmoF0's 352-bin activation to a 700-bin probability distribution
    using Gaussian interpolation.

    This performs a distribution→distribution mapping, NOT a scalar→distribution
    mapping. The 352-bin activation vector (softmax over HarmoF0 frequency bins)
    is interpolated onto the 700-bin Josh2021 grid.

    Args:
        activation_352: numpy array of shape (352,), probability distribution
                         (should sum to ~1)
        sigma: Gaussian smoothing parameter in target bin units (default: 1.5)

    Returns:
        labels_pred: int, index (0–699) of bin with maximum probability
        probs_700: numpy array of shape (700,), sum to ~1
    """
    # Recompute kernel if sigma differs from default
    if abs(sigma - _SIGMA_DEFAULT) > 1e-6:
        sigma_cents = sigma * _CENTS_PER_BIN_JOSH
        kernel = np.zeros((700, 352), dtype=np.float64)
        for j in range(700):
            cj = _CENTS_JOSH[j]
            distances = _CENTS_HARMOF0 - cj
            mask = np.abs(distances) < (3.0 * sigma_cents)
            if np.any(mask):
                kernel[j, mask] = np.exp(-(distances[mask] ** 2) / (2.0 * sigma_cents ** 2))
    else:
        kernel = _KERNEL

    # Matrix multiplication: (700, 352) @ (352,) → (700,)
    if np.sum(activation_352) == 0:
        warnings.warn("activation_352 sum is zero; returning uniform distribution")
        return 0, np.ones(700, dtype=np.float64) / 700

    probs_700 = kernel.dot(activation_352.astype(np.float64))

    # Normalize
    if probs_700.sum() > 0:
        probs_700 /= probs_700.sum()

    labels_pred = int(np.argmax(probs_700))

    return labels_pred, probs_700


def harmof0_to_700_batch(activations_352, sigma=1.5):
    """
    Batched version: convert multiple 352-bin activations to 700-bin distributions.

    Args:
        activations_352: numpy array of shape (N, 352), probability distributions
        sigma: Gaussian smoothing parameter (default: 1.5)

    Returns:
        labels_pred: numpy array of shape (N,), predicted bin indices
        probs_700: numpy array of shape (N, 700), probability distributions
    """
    # Recompute kernel if sigma differs from default
    if abs(sigma - _SIGMA_DEFAULT) > 1e-6:
        sigma_cents = sigma * _CENTS_PER_BIN_JOSH
        kernel = np.zeros((700, 352), dtype=np.float64)
        for j in range(700):
            cj = _CENTS_JOSH[j]
            distances = _CENTS_HARMOF0 - cj
            mask = np.abs(distances) < (3.0 * sigma_cents)
            if np.any(mask):
                kernel[j, mask] = np.exp(-(distances[mask] ** 2) / (2.0 * sigma_cents ** 2))
    else:
        kernel = _KERNEL

    # Matrix multiplication: (N, 352) @ (352, 700) → (N, 700)
    probs_700 = activations_352.astype(np.float64) @ kernel.T  # (N, 700)

    # Normalize each row
    row_sums = probs_700.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    probs_700 /= row_sums

    # Argmax per row
    labels_pred = np.argmax(probs_700, axis=1).astype(int)

    return labels_pred, probs_700
