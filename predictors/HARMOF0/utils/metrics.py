"""
Metrics utility functions for HarmoF0 psychophysics pipeline.

Provides helper functions for F0 conversion and metric calculation.
These are optional but useful for evaluation.
"""

import numpy as np


def bin_to_freq(bin_idx, n_bins=700, f_min=80.0, cents_per_bin=100.0 / 16.0):
    """
    Convert a 700-bin index to frequency in Hz.

    Args:
        bin_idx: index (0 to n_bins-1)
        n_bins: number of bins, default 700
        f_min: frequency of bin 0, default 80 Hz
        cents_per_bin: cents per bin, default 6.25 (1/16 semitone)

    Returns:
        frequency in Hz
    """
    return f_min * (2.0 ** (bin_idx * cents_per_bin / 1200.0))


def freq_to_bin(freq_hz, n_bins=700, f_min=80.0, cents_per_bin=100.0 / 16.0):
    """
    Convert a frequency in Hz to the nearest 700-bin index.

    Args:
        freq_hz: frequency in Hz
        n_bins: number of bins, default 700
        f_min: frequency of bin 0, default 80 Hz
        cents_per_bin: cents per bin, default 6.25

    Returns:
        bin index (clamped to [0, n_bins-1])
    """
    if freq_hz <= 0:
        return 0
    cents = 1200.0 * np.log2(freq_hz / f_min)
    bin_idx = int(round(cents / cents_per_bin))
    return max(0, min(n_bins - 1, bin_idx))


def compute_error_cents(freq_pred_hz, freq_label_hz):
    """
    Compute prediction error in cents.

    Args:
        freq_pred_hz: predicted frequency in Hz
        freq_label_hz: ground-truth frequency in Hz

    Returns:
        error in cents (positive = predicted higher than label)
    """
    if freq_label_hz <= 0 or freq_pred_hz <= 0:
        return 0.0
    return 1200.0 * np.log2(freq_pred_hz / freq_label_hz)


def compute_bin_error(labels_pred, labels_gt):
    """
    Compute bin-level prediction error.

    Args:
        labels_pred: predicted bin index
        labels_gt: ground-truth bin index

    Returns:
        absolute error in bins
    """
    return abs(labels_pred - labels_gt)
