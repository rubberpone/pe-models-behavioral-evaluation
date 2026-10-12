"""
Metrics and helper functions for F0 conversion and evaluation.

Optional utility module for converting between F0 representations.
"""

import numpy as np


# 700-bin grid constants
N_BINS = 700
F_MIN = 80.0   # Hz
F_MAX = 1000.0  # Hz
CENTS_PER_BIN = 100.0 / 16.0  # 1/16 semitone
REF_HZ = 10.0


def frequency_to_label(freq_hz):
    """
    Convert a continuous frequency in Hz to the nearest 700-bin class index.

    Args:
        freq_hz: Frequency in Hz (float or array-like).

    Returns:
        Integer class label (0-699) or array of labels.
    """
    freq_hz = np.asarray(freq_hz)
    # label = round(1200 / cents_per_bin * log2(f / f_min))
    label = np.round(
        1200.0 / CENTS_PER_BIN * np.log2(freq_hz / F_MIN)
    ).astype(int)
    return np.clip(label, 0, N_BINS - 1)


def label_to_frequency(label):
    """
    Convert a 700-bin class label back to frequency in Hz.

    Args:
        label: Class index (0-699).

    Returns:
        Frequency in Hz.
    """
    return F_MIN * (2.0 ** (np.asarray(label) * CENTS_PER_BIN / 1200.0))


def cent_error(freq_pred_hz, freq_true_hz, ref_hz=REF_HZ):
    """
    Compute pitch error in cents between predicted and true frequencies.

    Args:
        freq_pred_hz: Predicted frequency in Hz.
        freq_true_hz: True frequency in Hz.
        ref_hz: Reference frequency for cent conversion.

    Returns:
        Error in cents = 1200 * |log2(f_pred / f_true)|.
    """
    return 1200.0 * np.abs(np.log2(
        np.asarray(freq_pred_hz) / np.asarray(freq_true_hz)
    ))
