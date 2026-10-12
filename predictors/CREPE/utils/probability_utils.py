"""
Probability utility functions for CREPE psychophysics pipeline.

Core functionality: 360→700 dimensional Gaussian interpolation
mapping CREPE's activation vector to the 700-bin F0 classification
framework used in Josh2021 experiments.
"""

import numpy as np


# Constants for the 360-bin CREPE output
CREPE_N_BINS = 360
CREPE_CENTS_PER_BIN = 20.0  # 20 cents per bin
CREPE_F_MIN = 32.70  # C1 in Hz

# Constants for the 700-bin target
TARGET_N_BINS = 700
TARGET_CENTS_PER_BIN = 100.0 / 16.0  # 1/16 semitone ≈ 6.25 cents
TARGET_F_MIN = 80.0  # Hz
TARGET_F_MAX = 1000.0  # Hz

# Reference frequency for cent calculation (10 Hz)
REF_HZ = 10.0

# Default sigma for Gaussian interpolation (in target bins)
DEFAULT_SIGMA = 1.5


def get_cent_from_hz(freq_hz, ref_hz=REF_HZ):
    """
    Convert frequency in Hz to cents relative to a reference frequency.

    Args:
        freq_hz: Frequency in Hz (float or array-like).
        ref_hz: Reference frequency in Hz (default: 10 Hz).

    Returns:
        Cents value = 1200 * log2(freq_hz / ref_hz).
    """
    return 1200.0 * np.log2(np.asarray(freq_hz) / ref_hz)


def get_hz_from_cent(cents, ref_hz=REF_HZ):
    """
    Convert cents (relative to ref_hz) back to frequency in Hz.

    Args:
        cents: Cents value (float or array-like).
        ref_hz: Reference frequency in Hz (default: 10 Hz).

    Returns:
        Frequency in Hz = ref_hz * 2^(cents / 1200).
    """
    return ref_hz * (2.0 ** (np.asarray(cents) / 1200.0))


def _compute_bin_centers(n_bins, f_min, cents_per_bin):
    """
    Compute the cent positions of bin centers for a logarithmic frequency grid.

    Args:
        n_bins: Number of bins.
        f_min: Minimum frequency in Hz.
        cents_per_bin: Cents per bin.

    Returns:
        numpy.ndarray of shape (n_bins,) with cent positions relative to REF_HZ.
    """
    # Bin centers in Hz
    bin_centers_hz = f_min * (2.0 ** (np.arange(n_bins) * cents_per_bin / 1200.0))
    # Convert to cents relative to REF_HZ
    return get_cent_from_hz(bin_centers_hz)


def crepe_to_700(activation_360, sigma=DEFAULT_SIGMA):
    """
    Map CREPE's 360-dimensional activation vector to a 700-dimensional
    probability distribution via Gaussian interpolation.

    This is a distribution-to-distribution interpolation (NOT a scalar-to-
    distribution fit like PESTO). For each target bin j:

        prob_700[j] = Σ_i act_360[i] · exp(-(c_j - c_i)² / (2σ²))

    where c_j and c_i are cent positions of target and source bins.

    Args:
        activation_360: numpy.ndarray of shape (360,) — aggregated CREPE
                        activation (softmax probabilities).
        sigma: Gaussian smoothing width in target bins (default: 1.5,
               ≈ 9.3 cents). Controls the interpolation smoothness.

    Returns:
        tuple (label_pred, probs_700):
          - label_pred: int in [0, 699], predicted class index.
          - probs_700: numpy.ndarray of shape (700,), normalized probability
                       distribution summing to 1.
    """
    activation_360 = np.asarray(activation_360, dtype=np.float64)

    # Compute cent positions for 360 CREPE bins
    cents_360 = _compute_bin_centers(
        CREPE_N_BINS, CREPE_F_MIN, CREPE_CENTS_PER_BIN
    )

    # Compute cent positions for 700 target bins
    cents_700 = _compute_bin_centers(
        TARGET_N_BINS, TARGET_F_MIN, TARGET_CENTS_PER_BIN
    )

    # Pre-compute sigma in cents for the Gaussian
    # 1 target bin = TARGET_CENTS_PER_BIN cents
    sigma_cents = sigma * TARGET_CENTS_PER_BIN

    # Compute pairwise squared differences in cents
    # Shape: (700, 360)
    diff_cents = cents_700[:, np.newaxis] - cents_360[np.newaxis, :]

    # Gaussian weights: exp(-d² / (2σ²))
    weights = np.exp(-0.5 * (diff_cents / sigma_cents) ** 2)

    # Weighted sum: prob_700[j] = Σ_i act_360[i] * weight[j, i]
    probs_700 = weights @ activation_360

    # Normalize to ensure sum-to-1
    prob_sum = probs_700.sum()
    if prob_sum > 0:
        probs_700 /= prob_sum
    else:
        # Fallback: uniform distribution
        probs_700 = np.ones(TARGET_N_BINS) / TARGET_N_BINS

    # Predicted label: argmax
    label_pred = int(np.argmax(probs_700))

    return label_pred, probs_700.astype(np.float64)


def label_to_frequency(label, n_bins=TARGET_N_BINS,
                       f_min=TARGET_F_MIN, cents_per_bin=TARGET_CENTS_PER_BIN):
    """
    Convert a 700-bin class label (0-699) back to frequency in Hz.

    Args:
        label: Class index (0 to n_bins-1).
        n_bins: Number of bins (default: 700).
        f_min: Minimum frequency in Hz (default: 80).
        cents_per_bin: Cents per bin (default: 100/16).

    Returns:
        Frequency in Hz.
    """
    return f_min * (2.0 ** (label * cents_per_bin / 1200.0))
