"""
Probability utility functions for SwiftF0 psychophysics pipeline.

Implements 200→700 dimensional Gaussian interpolation from SwiftF0's
output space to the standard 700-bin psychophysics evaluation grid.
"""

import numpy as np
import warnings


def swiftf0_to_700(activation_200: np.ndarray, sigma: float = 2.0):
    """
    Map SwiftF0's 200-bin softmax distribution to the 700-bin psychophysics grid
    using Gaussian (distribution→distribution) interpolation.

    Args:
        activation_200: numpy array of shape (200,) — SwiftF0 softmax probabilities.
        sigma: Gaussian kernel width in target bin units. Default 2.0 (~12.5 cents).

    Returns:
        (labels_pred, probs_out) tuple:
            - labels_pred: int, predicted class index (0–699)
            - probs_out: numpy array of shape (700,), normalized probability distribution
    """
    # 1. SwiftF0 grid: 200 bins, starting at 46.875 Hz, 33.05 cents spacing
    f_min = 46.875
    f_max = 2093.75
    n_bins_swift = 200
    delta = np.log2(f_max / f_min) / (n_bins_swift - 1)
    freqs_swift = f_min * (2 ** (np.arange(n_bins_swift) * delta))

    # 2. Josh / psychophysics grid: 700 bins, starting at 80 Hz, 6.25 cents spacing
    cents_per_bin_josh = 100.0 / 16.0  # 6.25 cents
    n_bins_josh = 700
    freqs_josh = 80.0 * (2 ** (np.arange(n_bins_josh) * cents_per_bin_josh / 1200.0))

    # 3. Unified cents reference: 46.875 Hz (consistent with SwiftF0 grid origin)
    ref_freq = 46.875
    cents_swift = 1200.0 * np.log2(freqs_swift / ref_freq)
    cents_josh = 1200.0 * np.log2(freqs_josh / ref_freq)

    # 4. Gaussian interpolation (fully vectorized)
    sigma_cents = sigma * cents_per_bin_josh  # 2.0 * 6.25 = 12.5 cents

    # Compute pairwise distance matrix: (700, 200)
    # distances[j, i] = cents_josh[j] - cents_swift[i]
    distances = cents_josh[:, np.newaxis] - cents_swift[np.newaxis, :]  # (700, 200)

    # Mask: only keep bins within ±3σ
    mask = np.abs(distances) < 3.0 * sigma_cents

    # Compute Gaussian weights where mask is True
    weights = np.exp(-(distances ** 2) / (2.0 * sigma_cents ** 2))  # (700, 200)

    # Apply mask and compute weighted sum
    # probs_700[j] = sum_i activation_200[i] * weights[j, i] * mask[j, i]
    probs_700 = np.sum(activation_200[np.newaxis, :] * weights * mask, axis=1)  # (700,)

    # 5. Edge case: zero activation sum
    if activation_200.sum() == 0:
        warnings.warn("activation_200 sum is zero; returning uniform distribution")
        return 0, np.ones(700, dtype=np.float64) / 700.0

    # 6. Normalize
    if probs_700.sum() > 0:
        probs_700 /= probs_700.sum()

    labels_pred = int(np.argmax(probs_700))

    return labels_pred, probs_700


def get_cents_from_hz(freq_hz: float, ref_hz: float = 46.875) -> float:
    """
    Convert a frequency in Hz to cents relative to a reference frequency.

    Args:
        freq_hz: Frequency in Hz.
        ref_hz: Reference frequency in Hz. Default 46.875 (SwiftF0 bin 0).

    Returns:
        Cents value.
    """
    if freq_hz <= 0:
        return 0.0
    return 1200.0 * np.log2(freq_hz / ref_hz)
