"""
Probability mapping utilities: 486→700 dimensional Gaussian interpolation.

Converts FCN's 486-bin per-bin sigmoid activation vector to the 700-bin
distribution used by the Josh2021 psychophysical analysis framework.

Key specifications:
  - FCN grid:  486 bins, 30–1000 Hz, 12.5-cent spacing (96 bins/octave)
  - Josh grid: 700 bins, 80–1000 Hz, 6.25-cent spacing (192 bins/octave)
  - Method:    Gaussian interpolation (distribution → distribution)
  - Sigma:     1.5 target bins (~9.3 cents)
"""

import numpy as np
import warnings


def get_fcn_frequencies():
    """
    Compute FCN frequency grid: 486 bins from 30 Hz, 12.5-cent spacing.

    f(i) = 30 * 2^(i/96)  where i ∈ [0, 485]

    Returns:
        numpy array of shape (486,), frequencies in Hz
    """
    return 30.0 * (2 ** (np.arange(486) / 96))


def get_josh_frequencies():
    """
    Compute Josh2021 frequency grid: 700 bins from 80 Hz, 6.25-cent spacing.

    Returns:
        numpy array of shape (700,), frequencies in Hz
    """
    cents_per_bin = 100.0 / 16.0  # 6.25 cents
    return 80.0 * (2 ** (np.arange(700) * cents_per_bin / 1200.0))


def get_cents_from_hz(freq_hz, ref_hz=30.0):
    """
    Convert frequency in Hz to cents relative to a reference frequency.

    Args:
        freq_hz: frequency in Hz (scalar or array)
        ref_hz: reference frequency in Hz (default: 30 Hz, matching FCN grid start)

    Returns:
        cents value(s)
    """
    return 1200.0 * np.log2(np.asarray(freq_hz) / ref_hz)


def fcn_to_700(activation_486, sigma=1.5):
    """
    Convert FCN's 486-bin activation to a 700-bin probability distribution.

    Uses Gaussian interpolation in the cents domain. Each target bin j
    receives a weighted sum of source bin activations, where weights
    are Gaussian functions of the cent distance between bins.

    prob_700[j] = Σ_i act_486[i] * exp(-(c_j - c_i)² / (2σ²))

    where:
      - c_j, c_i are cent positions (reference: 30 Hz)
      - σ = 1.5 target bins (~9.3 cents)
      - Only source bins within ±3σ contribute (range clipping)

    Args:
        activation_486: numpy array of shape (486,), per-bin sigmoid activations
                        (values in [0,1]; does NOT need to sum to 1)
        sigma: smoothing parameter in target bins (default: 1.5)

    Returns:
        tuple of (labels_pred, probs_700)
          labels_pred: int index (0–699) of the bin with maximum probability
          probs_700:   numpy array of shape (700,), normalized to sum to 1
    """
    # Input validation
    if activation_486.ndim != 1 or len(activation_486) != 486:
        raise ValueError(
            f"activation_486 must be 1D array of length 486, "
            f"got shape {activation_486.shape}"
        )

    # Handle all-zero input
    if activation_486.sum() == 0:
        warnings.warn("activation_486 sum is zero; returning uniform distribution")
        return 0, np.ones(700) / 700

    # 1. FCN grid: 486 bins, 30 Hz start, 12.5-cent spacing
    freqs_fcn = get_fcn_frequencies()

    # 2. Josh grid: 700 bins, 80 Hz start, 6.25-cent spacing
    n_bins_josh = 700
    cents_per_bin_josh = 100.0 / 16.0  # 6.25 cents
    freqs_josh = get_josh_frequencies()

    # 3. Unified cent reference at 30 Hz
    #    (Official freq2cents uses f_ref=10 Hz, but the reference only adds
    #     a constant offset — relative distances between bins are unaffected)
    ref_freq = 30.0
    cents_fcn = 1200.0 * np.log2(freqs_fcn / ref_freq)
    cents_josh = 1200.0 * np.log2(freqs_josh / ref_freq)

    # 4. Gaussian interpolation with range clipping
    sigma_cents = sigma * cents_per_bin_josh  # ~9.375 cents
    clip_range = 3 * sigma_cents               # ±3σ

    probs_700 = np.zeros(n_bins_josh, dtype=np.float64)

    for j, cj in enumerate(cents_josh):
        distances = cents_fcn - cj
        mask = np.abs(distances) < clip_range
        if not np.any(mask):
            continue
        weights = np.exp(-(distances[mask] ** 2) / (2 * sigma_cents ** 2))
        probs_700[j] = np.sum(activation_486[mask] * weights)

    # 5. Normalize
    if probs_700.sum() > 0:
        probs_700 /= probs_700.sum()
    else:
        # This should not happen if activation_486.sum() > 0 and sigma > 0,
        # but handle it gracefully
        warnings.warn("probs_700 sum is zero after interpolation; returning uniform")
        return 0, np.ones(700) / 700

    return int(np.argmax(probs_700)), probs_700.astype(np.float64)


def continuous_f0_from_activation(activation_486, return_hz=True):
    """
    Decode a continuous F0 estimate from the 486-bin activation vector.

    Uses the official local weighted average method (to_local_average_cents):
    weighted average of cents values within ±4 bins of the argmax,
    converted back to Hz.

    f0_hz = 10.0 * 2^(local_cents / 1200)

    Args:
        activation_486: numpy array of shape (486,)
        return_hz: if True, return Hz; if False, return cents

    Returns:
        float: F0 estimate in Hz (or cents if return_hz=False)
    """
    probs = np.asarray(activation_486, dtype=np.float64)

    center = int(np.argmax(probs))
    start = max(0, center - 4)
    end = min(486, center + 5)  # +5 because slice is exclusive

    # Cents mapping relative to 10 Hz (official freq2cents)
    freqs = 30.0 * (2 ** (np.arange(486) / 96))
    cents_mapping = 1200.0 * np.log2(freqs / 10.0)

    local_probs = probs[start:end]
    local_cents = cents_mapping[start:end]

    total = np.sum(local_probs)
    if total == 0:
        # Fallback: just use argmax
        return 10.0 * (2 ** (cents_mapping[center] / 1200))

    weighted_cents = np.sum(local_probs * local_cents) / total

    if not return_hz:
        return weighted_cents

    f0_hz = 10.0 * (2 ** (weighted_cents / 1200))
    return f0_hz


if __name__ == '__main__':
    # Quick test
    print("FCN grid (first 5):", get_fcn_frequencies()[:5])
    print("FCN grid (last 5):", get_fcn_frequencies()[-5:])
    print("Josh grid (first 5):", get_josh_frequencies()[:5])
    print("Josh grid (last 5):", get_josh_frequencies()[-5:])

    # Test with a synthetic activation (Gaussian peak at bin 200)
    synth = np.zeros(486)
    synth[195:206] = np.exp(-((np.arange(195, 206) - 200) ** 2) / (2 * 2 ** 2))
    # Add some noise
    synth += np.abs(np.random.randn(486) * 0.01)

    label, probs = fcn_to_700(synth)
    print(f"\nSynthetic test:")
    print(f"  Peak FCN bin: {np.argmax(synth)}")
    print(f"  Predicted label: {label}")
    print(f"  probs_700 sum: {probs.sum():.6f}")
    print(f"  probs_700 shape: {probs.shape}")
    print(f"  Top 5 probs bins: {np.argsort(probs)[-5:][::-1]}")

    f0_hz = continuous_f0_from_activation(synth)
    print(f"  Continuous F0: {f0_hz:.2f} Hz")
