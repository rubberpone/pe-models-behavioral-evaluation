"""
Optional metrics and conversion helpers for F0 evaluation.

Utilities for converting between F0 representations and computing
evaluation metrics.
"""

import numpy as np


def hz_to_label(f0_hz):
    """
    Convert a continuous F0 in Hz to the corresponding 700-class label index.

    The Josh2021 grid: 700 bins from 80 Hz, 6.25-cent spacing.
    bin_freq = 80 * 2^(label * 6.25 / 1200)

    Args:
        f0_hz: frequency in Hz (scalar or array)

    Returns:
        int or array: label index (0–699)
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    cents_per_bin = 100.0 / 16.0  # 6.25 cents
    # Invert: label = (1200 / 6.25) * log2(f / 80)
    label = np.round((1200.0 / cents_per_bin) * np.log2(f0_hz / 80.0))
    label = np.clip(label, 0, 699)
    return label.astype(int)


def label_to_hz(label):
    """
    Convert a 700-class label index to frequency in Hz.

    Args:
        label: int or array, index (0–699)

    Returns:
        float or array: frequency in Hz
    """
    label = np.asarray(label, dtype=np.float64)
    cents_per_bin = 100.0 / 16.0
    return 80.0 * (2 ** (label * cents_per_bin / 1200.0))


def compute_accuracy(labels_true, labels_pred):
    """
    Compute classification accuracy.

    Args:
        labels_true: ground-truth label indices
        labels_pred: predicted label indices

    Returns:
        float: accuracy (0.0 to 1.0)
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    return np.mean(labels_true == labels_pred)


def compute_mae_hz(f0_true_hz, f0_pred_hz):
    """
    Compute Mean Absolute Error in Hz.

    Args:
        f0_true_hz: ground-truth F0 in Hz
        f0_pred_hz: predicted F0 in Hz

    Returns:
        float: MAE in Hz
    """
    return np.mean(np.abs(np.asarray(f0_true_hz) - np.asarray(f0_pred_hz)))


def compute_rmse_hz(f0_true_hz, f0_pred_hz):
    """
    Compute Root Mean Square Error in Hz.

    Args:
        f0_true_hz: ground-truth F0 in Hz
        f0_pred_hz: predicted F0 in Hz

    Returns:
        float: RMSE in Hz
    """
    diff = np.asarray(f0_true_hz) - np.asarray(f0_pred_hz)
    return np.sqrt(np.mean(diff ** 2))


if __name__ == '__main__':
    # Quick tests
    print("80 Hz → label:", hz_to_label(80.0))
    print("1000 Hz → label:", hz_to_label(1000.0))
    print("Label 0 → Hz:", label_to_hz(0))
    print("Label 699 → Hz:", label_to_hz(699))
    print("Label 350 → Hz:", label_to_hz(350))
