#!/usr/bin/env python3
"""
Probability mapping utilities for FT Attention pipeline.

Maps 321-dim FT Attention softmax output (1 unvoiced + 320 voiced bins)
to the 700-dim psychophysics grid via Gaussian interpolation.
"""

import numpy as np


# ============================================================================
# Constants
# ============================================================================

# FT Attention voiced grid: 320 bins starting at 31 Hz, 60 bins/octave
FTATT_N_VOICED = 320
FTATT_START_FREQ = 31.0
FTATT_BINS_PER_OCTAVE = 60
FTATT_CENTS_PER_BIN = 100 * 12 / FTATT_BINS_PER_OCTAVE  # 20 cents per bin

# Josh psychophysics grid: 700 bins starting at 80 Hz, 6.25 cents per bin
JOSH_N_BINS = 700
JOSH_START_FREQ = 80.0
JOSH_CENTS_PER_BIN = 100 / 16  # 6.25 cents per bin


def get_cents_from_hz(freq_hz, ref_hz=31.0):
    """
    Convert frequency in Hz to cents relative to a reference frequency.

    Args:
        freq_hz: frequency (or array of frequencies) in Hz
        ref_hz: reference frequency in Hz (default 31.0)

    Returns:
        cents value(s)
    """
    return 1200.0 * np.log2(np.asarray(freq_hz) / ref_hz)


def ft_attention_to_700(activation_321, sigma=2.0):
    """
    Map FT Attention's 321-dim softmax output to the 700-dim psychophysics grid
    using distribution-to-distribution Gaussian interpolation.

    The FT Attention output has 321 bins:
      - bin 0: unvoiced (CenFreq[0] = 0 Hz)
      - bins 1-320: voiced (31 Hz to 1250 Hz, 60 bins/octave)

    The target grid has 700 bins from 80 Hz to ~1000 Hz, 6.25 cent spacing.

    Method:
      1. Discard unvoiced bin (index 0), keep 320 voiced bins
      2. Gaussian interpolation: prob_700[j] = sum_i act_320[i] * exp(-(cj-ci)^2/(2*sigma^2))
      3. sigma in target-bin units (2.0 target bins = ~12.5 cents)
      4. Range clipping: only source bins within ±3*sigma contribute

    Args:
        activation_321: ndarray of shape (321,) — FT Attention softmax probabilities
                         (index 0 = unvoiced, indices 1-320 = voiced)
        sigma: Gaussian sigma in target bin units (default 2.0 ≈ 12.5 cents)

    Returns:
        (label_pred, probs_700) tuple:
          - label_pred: int, argmax index into the 700-dim grid (0-699)
          - probs_700: ndarray of shape (700,), normalized probability distribution
    """
    # 1. Discard unvoiced bin (index 0), keep only voiced bins
    activation_320 = activation_321[1:].copy()  # (320,)

    # Handle edge case: all-zero activation
    if activation_320.sum() == 0:
        return 0, np.ones(JOSH_N_BINS) / JOSH_N_BINS

    # 2. Build source frequency grid (FT Attention voiced bins)
    #    freq_src[i] = 31.0 * 2^(i / 60)  for i in [0, 319]
    freqs_src = FTATT_START_FREQ * (2 ** (np.arange(FTATT_N_VOICED) / FTATT_BINS_PER_OCTAVE))

    # 3. Build target frequency grid (Josh grid: 700 bins, 80 Hz start, 6.25 cent step)
    freqs_tgt = JOSH_START_FREQ * (2 ** (np.arange(JOSH_N_BINS) * JOSH_CENTS_PER_BIN / 1200))

    # 4. Convert both grids to cents relative to 31 Hz (unified reference)
    cents_src = get_cents_from_hz(freqs_src, ref_hz=FTATT_START_FREQ)
    cents_tgt = get_cents_from_hz(freqs_tgt, ref_hz=FTATT_START_FREQ)

    # 5. Gaussian interpolation
    sigma_cents = sigma * JOSH_CENTS_PER_BIN  # sigma in cents (~12.5 cents)
    cutoff = 3 * sigma_cents

    probs_700 = np.zeros(JOSH_N_BINS)

    for j, cj in enumerate(cents_tgt):
        dist = cents_src - cj  # (320,)
        mask = np.abs(dist) < cutoff
        if not np.any(mask):
            continue
        weights = np.exp(-(dist[mask] ** 2) / (2 * sigma_cents ** 2))
        probs_700[j] = np.sum(activation_320[mask] * weights)

    # 6. Normalize
    if probs_700.sum() > 0:
        probs_700 /= probs_700.sum()
    else:
        probs_700 = np.ones(JOSH_N_BINS) / JOSH_N_BINS

    label_pred = int(np.argmax(probs_700))

    return label_pred, probs_700
