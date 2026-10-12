#!/usr/bin/env python3
"""
Probability utility functions for RMVPE psychophysics pipeline.

Core: 360 → 700 dimensional Gaussian interpolation.
Maps RMVPE's 360-bin sigmoid activation onto the Josh2021 700-bin grid
(80–1000 Hz, 1/16 semitone spacing).

Key constants from official RMVPE implementation (src/constants.py):
  - CONST = 1997.3794084376191  (cent offset of bin 0 relative to 10 Hz)
  - cents_per_bin = 20
  - reference frequency = 10 Hz
"""

import numpy as np
import warnings

# ============================================================================
# RMVPE Official Constants (from src/constants.py)
# ============================================================================
RMVPE_CONST = 1997.3794084376191
RMVPE_CENTS_PER_BIN = 20         # 20 cents between adjacent bins
RMVPE_N_BINS = 360
RMVPE_REF_FREQ = 10.0             # Reference frequency in Hz

# ============================================================================
# Josh2021 700-class grid parameters
# ============================================================================
JOSH_CENTS_PER_BIN = 100.0 / 16.0  # = 6.25 cents
JOSH_N_BINS = 700
JOSH_FMIN = 80.0                   # Starting frequency in Hz


def get_cents_from_hz(freq_hz: float, ref_hz: float = RMVPE_REF_FREQ) -> float:
    """
    Convert frequency in Hz to cents relative to a reference frequency.

    Args:
        freq_hz: Frequency in Hz.
        ref_hz: Reference frequency in Hz (default: 10.0 Hz).

    Returns:
        Cents value = 1200 * log2(freq_hz / ref_hz).
    """
    if freq_hz <= 0:
        return 0.0
    return 1200.0 * np.log2(freq_hz / ref_hz)


def get_hz_from_cents(cents: float, ref_hz: float = RMVPE_REF_FREQ) -> float:
    """
    Convert cents to frequency in Hz.

    Args:
        cents: Cents value relative to ref_hz.
        ref_hz: Reference frequency in Hz (default: 10.0 Hz).

    Returns:
        Frequency in Hz = ref_hz * 2^(cents / 1200).
    """
    return ref_hz * (2.0 ** (cents / 1200.0))


def build_rmvpe_cents_grid() -> np.ndarray:
    """
    Build the RMVPE cent grid (360 bins, 20-cent spacing, CONST offset).

    Official mapping: cents[i] = i * 20 + CONST, where CONST = 1997.3794084376191.
    Reference frequency: 10 Hz.
    Bin 0 frequency ≈ 31.7 Hz.

    Returns:
        numpy array of shape (360,) with cent values.
    """
    return np.arange(RMVPE_N_BINS) * RMVPE_CENTS_PER_BIN + RMVPE_CONST


def build_josh_cents_grid(ref_hz: float = RMVPE_REF_FREQ) -> np.ndarray:
    """
    Build the Josh2021 700-bin cent grid.

    The Josh grid starts at 80 Hz and uses 6.25-cent spacing (1/16 semitone).
    All cent values are computed relative to the same reference frequency
    used by RMVPE (10 Hz) for consistency.

    Args:
        ref_hz: Reference frequency for cent calculation (default: 10.0 Hz).

    Returns:
        numpy array of shape (700,) with cent values relative to ref_hz.
    """
    # Josh bin j frequency: f_j = 80 * 2^(j * 6.25 / 1200)
    freqs_josh = JOSH_FMIN * (2.0 ** (np.arange(JOSH_N_BINS) * JOSH_CENTS_PER_BIN / 1200.0))
    # Convert to cents relative to the same reference as RMVPE
    cents_josh = 1200.0 * np.log2(freqs_josh / ref_hz)
    return cents_josh


def rmvpe_to_700(
    activation_360: np.ndarray,
    sigma: float = 1.5,
) -> tuple:
    """
    Convert RMVPE's 360-bin sigmoid activation to a 700-bin distribution.

    This implements a distribution-to-distribution Gaussian interpolation,
    NOT a scalar-to-distribution fit. Each 360-bin activation value is
    spread to nearby 700-bin positions via a Gaussian kernel.

    Method:
      For each target bin j (cent position c_j):
        prob_700[j] = sum_i( activation_360[i] * exp(-(c_j - c_i)^2 / (2*sigma_cents^2)) )
      where c_i are RMVPE cent positions, c_j are Josh cent positions,
      and sigma_cents = sigma * 6.25 ≈ 9.4 cents.

      Only source bins within ±3*sigma_cents contribute (range clipping).

    Important constraints (from README §7.3):
      - Reference frequency: 10 Hz (same as RMVPE grid)
      - CONST = 1997.3794084376191 (official value, NOT 2065.5 derived from 32.7 Hz)
      - sigma = 1.5 target bins ≈ 9.4 cents
      - Result is normalized to sum to 1

    Args:
        activation_360: numpy array of shape (360,) — RMVPE sigmoid activation
                        (independent per-bin confidences, NOT a softmax).
        sigma: Smoothing parameter in units of target bins (default: 1.5).

    Returns:
        labels_pred: int (0–699), index of the bin with maximum probability.
        probs_700: numpy array of shape (700,), normalized probability distribution.
    """
    if activation_360.ndim != 1 or len(activation_360) != RMVPE_N_BINS:
        raise ValueError(
            f"activation_360 must be a 1-D array of length {RMVPE_N_BINS}, "
            f"got shape {activation_360.shape}"
        )

    # 1. Build cent grids (both relative to 10 Hz reference)
    cents_rmvpe = build_rmvpe_cents_grid()   # (360,)
    cents_josh = build_josh_cents_grid()      # (700,)

    # 2. Gaussian interpolation parameters
    sigma_cents = sigma * JOSH_CENTS_PER_BIN  # ≈ 1.5 * 6.25 = 9.375 cents
    clip_range = 3.0 * sigma_cents            # ≈ 28.1 cents

    # 3. Interpolate: for each Josh bin, sum weighted contributions
    probs_700 = np.zeros(JOSH_N_BINS, dtype=np.float64)

    for j in range(JOSH_N_BINS):
        cj = cents_josh[j]
        distances = cents_rmvpe - cj
        mask = np.abs(distances) < clip_range

        if not np.any(mask):
            continue

        weights = np.exp(-(distances[mask] ** 2) / (2.0 * sigma_cents ** 2))
        probs_700[j] = np.sum(activation_360[mask] * weights)

    # 4. Normalize
    total_activation = activation_360.sum()
    if total_activation <= 0:
        warnings.warn(
            "activation_360 sum is zero; returning uniform distribution. "
            "This may indicate a silent/unvoiced frame."
        )
        return 0, np.ones(JOSH_N_BINS) / JOSH_N_BINS

    if probs_700.sum() > 0:
        probs_700 /= probs_700.sum()
    else:
        # All target bins got zero weight (shouldn't happen with normal input)
        warnings.warn(
            "probs_700 sum is zero after interpolation; returning uniform distribution."
        )
        probs_700 = np.ones(JOSH_N_BINS) / JOSH_N_BINS

    labels_pred = int(np.argmax(probs_700))
    return labels_pred, probs_700.astype(np.float64)


# ============================================================================
# Cached grids (computed once, reused across calls)
# ============================================================================
class _CachedGrids:
    """Singleton for pre-computed cent grids to avoid repeated computation."""
    _rmvpe_cents = None
    _josh_cents = None

    @classmethod
    def get_rmvpe_cents(cls) -> np.ndarray:
        if cls._rmvpe_cents is None:
            cls._rmvpe_cents = build_rmvpe_cents_grid()
        return cls._rmvpe_cents

    @classmethod
    def get_josh_cents(cls) -> np.ndarray:
        if cls._josh_cents is None:
            cls._josh_cents = build_josh_cents_grid()
        return cls._josh_cents


def rmvpe_to_700_fast(
    activation_360: np.ndarray,
    sigma: float = 1.5,
) -> tuple:
    """
    Optimized version of rmvpe_to_700 using pre-computed grids and vectorization.

    Same semantics as rmvpe_to_700() but faster for repeated calls.

    Args:
        activation_360: numpy array of shape (360,).
        sigma: Smoothing parameter in target bins (default: 1.5).

    Returns:
        labels_pred: int (0–699).
        probs_700: numpy array of shape (700,), normalized.
    """
    cents_rmvpe = _CachedGrids.get_rmvpe_cents()   # (360,)
    cents_josh = _CachedGrids.get_josh_cents()      # (700,)

    sigma_cents = sigma * JOSH_CENTS_PER_BIN
    clip_range = 3.0 * sigma_cents

    probs_700 = np.zeros(JOSH_N_BINS, dtype=np.float64)

    for j in range(JOSH_N_BINS):
        cj = cents_josh[j]
        distances = cents_rmvpe - cj
        mask = np.abs(distances) < clip_range
        if not np.any(mask):
            continue
        weights = np.exp(-(distances[mask] ** 2) / (2.0 * sigma_cents ** 2))
        probs_700[j] = np.sum(activation_360[mask] * weights)

    if activation_360.sum() <= 0:
        return 0, np.ones(JOSH_N_BINS) / JOSH_N_BINS

    if probs_700.sum() > 0:
        probs_700 /= probs_700.sum()

    return int(np.argmax(probs_700)), probs_700
