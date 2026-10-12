"""
Probability utility functions for FCPE psychophysics pipeline.

Core functions:
- local_argmax: Local weighted average pitch decoding (in cent space).
- fcpe_to_700: Gaussian interpolation from 360-bin sigmoid activation to 700-bin vector.
- get_cents_from_hz: Frequency-to-cents conversion (relative to reference).

IMPORTANT: FCPE uses sigmoid activation (per-bin binary classification),
NOT softmax. The 360-dim output does NOT sum to 1, and the mapped 700-dim
vector also does NOT sum to 1 by default.

IMPORTANT (cent_table): FCPE's cent_table is NOT simply `np.arange(360) * 20`.
It is computed as:
    linspace(1200*log2(f0_min/10), 1200*log2(f0_max/10), 360)
where f0_min = 32.70 Hz, f0_max = 1975.5 Hz.
This gives ~19.78 cent spacing starting at ~2051 cents (32.70 Hz),
NOT 20 cents starting at 0 cents (10 Hz).
"""

import numpy as np
import warnings


# Default FCPE model parameters
FCPE_F0_MIN = 32.70   # Hz
FCPE_F0_MAX = 1975.5  # Hz
FCPE_REF_FREQ = 10.0  # Hz
FCPE_N_BINS = 360


def _compute_default_cent_table():
    """
    Compute the default FCPE cent table matching the model's internal
    cent_table computation: linspace(f0_to_cent(f0_min), f0_to_cent(f0_max), 360).

    Returns:
        numpy array of shape (360,), cents relative to 10 Hz.
    """
    cents_start = 1200.0 * np.log2(FCPE_F0_MIN / FCPE_REF_FREQ)
    cents_end = 1200.0 * np.log2(FCPE_F0_MAX / FCPE_REF_FREQ)
    return np.linspace(cents_start, cents_end, FCPE_N_BINS)


def local_argmax(activation: np.ndarray,
                 cent_table: np.ndarray = None) -> tuple:
    """
    Decode pitch from a 360-dim sigmoid activation vector using local
    weighted average (local argmax).

    This mirrors FCPE's latent2cents_local_decoder: ±4 bins around the peak,
    weighted average in cent space, then convert to Hz.

    Decoding formula:
        c_hat = sum(y_i * c_i) / sum(y_i)  for i in [m-4, m+4]
        m = argmax_i(y_i)
        f_hat = 10 * 2^(c_hat / 1200)

    where y_i is the sigmoid activation of bin i, and c_i is the cent value
    from the FCPE cent_table (NOT simply i * 20).

    Args:
        activation: numpy array of shape (360,) — sigmoid activations
                    (NOT softmax probabilities). Each element ∈ [0, 1],
                    and the sum is NOT 1.
        cent_table: numpy array of shape (360,) — cent values for each bin
                    relative to 10 Hz. If None, uses default FCPE cent_table.

    Returns:
        cents: Pitch in cents relative to 10 Hz (float).
        pitch_hz: Pitch in Hz (float).
    """
    if cent_table is None:
        cent_table = _compute_default_cent_table()

    n_bins = len(activation)

    # Find peak activation bin
    m = np.argmax(activation)

    # Local weighted average around peak (±4 bins = 9-bin window)
    start = max(0, m - 4)
    end = min(n_bins, m + 5)
    weights = activation[start:end]

    if weights.sum() > 0:
        weighted_cents = np.sum(weights * cent_table[start:end]) / np.sum(weights)
    else:
        weighted_cents = cent_table[m]

    # Convert cents to Hz: f = 10 * 2^(cents / 1200)
    pitch_hz = FCPE_REF_FREQ * (2 ** (weighted_cents / 1200.0))

    return weighted_cents, pitch_hz


def get_cents_from_hz(freq_hz: float, ref_hz: float = 10.0) -> float:
    """
    Convert frequency in Hz to cents relative to a reference frequency.

    Args:
        freq_hz: Frequency in Hz.
        ref_hz: Reference frequency in Hz (default: 10 Hz).

    Returns:
        Cents relative to ref_hz.
    """
    if freq_hz <= 0:
        return 0.0
    return 1200.0 * np.log2(freq_hz / ref_hz)


def fcpe_to_700(activation_360: np.ndarray, sigma: float = 1.5,
                cent_table: np.ndarray = None) -> tuple:
    """
    Convert FCPE's 360-bin sigmoid activation to a 700-bin vector
    via Gaussian interpolation.

    IMPORTANT: FCPE uses sigmoid (per-bin binary classification), NOT softmax.
    The input activation_360 does NOT sum to 1, and the output mapped_700
    also does NOT sum to 1 by default. Each element is a Gaussian-weighted
    sum of neighboring sigmoid activations.

    Grid definitions:
    - FCPE: 360 bins from 32.70 Hz to 1975.5 Hz
            cent_table = linspace(1200*log2(32.70/10), 1200*log2(1975.5/10), 360)
            ~19.78 cent spacing, reference = 10 Hz
    - Josh: 700 bins from 80 Hz
            6.25-cent spacing (1/16 semitone = 100/16 cents)
            f_j = 80 * 2^(j * 6.25 / 1200)

    Interpolation formula:
        mapped_700[j] = sum_i(activation_360[i] * exp(-(c_j - c_i)^2 / (2*sigma^2)))

    where c_j and c_i are cent positions relative to 10 Hz reference.

    Args:
        activation_360: numpy array of shape (360,), sigmoid activations ∈ [0, 1].
                        NOT a probability distribution — does NOT sum to 1.
        sigma: Smoothing parameter in target bins (default: 1.5).
               sigma_cents = sigma * 6.25 ≈ 9.375 cents.
        cent_table: numpy array of shape (360,) — cent values for each FCPE bin
                    relative to 10 Hz. If None, uses default FCPE cent_table.

    Returns:
        labels_pred: int, predicted class index (0–699), argmax of mapped_700.
        mapped_700: numpy array of shape (700,), Gaussian-interpolated values.
                    NOT a probability distribution — does NOT sum to 1 by default.
    """
    if cent_table is None:
        cent_table = _compute_default_cent_table()

    ref_freq = FCPE_REF_FREQ  # 10 Hz
    n_bins_fcpe = len(cent_table)
    n_bins_josh = 700
    cents_per_bin_josh = 100.0 / 16.0  # 6.25 cents

    # --- 1. FCPE grid: bin cent values from the cent_table (relative to 10 Hz) ---
    cents_fcpe_ref10 = cent_table  # shape (360,), cents relative to 10 Hz
    freqs_fcpe = ref_freq * (2 ** (cents_fcpe_ref10 / 1200.0))

    # --- 2. Josh grid: 700 bins, 6.25-cent spacing, starting at 80 Hz ---
    cents_josh_from_80 = np.arange(n_bins_josh) * cents_per_bin_josh
    freqs_josh = 80.0 * (2 ** (cents_josh_from_80 / 1200.0))

    # --- 3. Unified cent reference at 10 Hz ---
    cents_josh_ref10 = 1200.0 * np.log2(freqs_josh / ref_freq)

    # --- 4. Gaussian interpolation ---
    # sigma in cents = sigma_target_bins * cents_per_target_bin
    sigma_cents = sigma * cents_per_bin_josh  # ≈ 9.375 cents
    mapped_700 = np.zeros(n_bins_josh, dtype=np.float64)

    for j, cj in enumerate(cents_josh_ref10):
        # Distance from each FCPE bin to this target bin (in cents)
        distances = cents_fcpe_ref10 - cj
        # Only consider source bins within ±3*sigma for efficiency
        mask = np.abs(distances) < 3.0 * sigma_cents
        if not np.any(mask):
            continue
        weights = np.exp(-(distances[mask] ** 2) / (2.0 * sigma_cents ** 2))
        mapped_700[j] = np.sum(activation_360[mask] * weights)

    # --- 5. Handle edge cases ---
    if np.max(activation_360) == 0:
        warnings.warn("activation_360 is all zeros; returning zero vector")
        return 0, mapped_700

    # --- 6. Argmax for label prediction ---
    labels_pred = int(np.argmax(mapped_700))

    return labels_pred, mapped_700
