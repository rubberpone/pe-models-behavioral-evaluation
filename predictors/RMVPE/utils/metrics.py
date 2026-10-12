#!/usr/bin/env python3
"""
Optional metrics helper functions for RMVPE psychophysics evaluation.

Provides:
  - F0 conversion utilities (Hz ↔ cents)
  - Label ↔ Hz mapping from reference templates
  - Accuracy computation helpers
"""

import numpy as np
import json
import os
from typing import Tuple, Dict


def hz_to_cents(freq_hz: np.ndarray, ref_hz: float = 10.0) -> np.ndarray:
    """Convert Hz to cents relative to reference frequency."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    result = np.zeros_like(freq_hz)
    mask = freq_hz > 0
    result[mask] = 1200.0 * np.log2(freq_hz[mask] / ref_hz)
    return result


def cents_to_hz(cents: np.ndarray, ref_hz: float = 10.0) -> np.ndarray:
    """Convert cents to Hz."""
    return ref_hz * (2.0 ** (np.asarray(cents) / 1200.0))


def build_label_to_hz_mapping(reference_dir: str, experiment_name: str) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Build per-experiment label → Hz mapping from reference template.

    Uses the reference template's `f0_label` and `f0` arrays to determine
    which Hz value corresponds to each class label (0–699).
    Missing labels are filled by log-linear interpolation.

    Args:
        reference_dir: Path to directory with reference JSON files.
        experiment_name: Name of experiment (e.g. 'BernsteinOxenham2005').

    Returns:
        label_to_hz: np.ndarray of shape (700,) — label → Hz.
        label_pairs: Dict[int, float] — known label → Hz mappings.
    """
    ref_path = os.path.join(reference_dir, f'predictions_mert_{experiment_name}.json')
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f'Reference template not found: {ref_path}')

    with open(ref_path, 'r') as f:
        ref = json.load(f)

    f0_labels = np.array(ref['f0_label'], dtype=np.int64)
    f0_hz = np.array(ref['f0'], dtype=np.float64)

    # Build unique label → Hz mapping
    pairs: Dict[int, float] = {}
    for lab, hz in zip(f0_labels, f0_hz):
        if lab not in pairs:
            pairs[lab] = float(hz)

    sorted_labels = sorted(pairs.keys())
    n_classes = 700

    # Build full 700-class mapping via log-linear interpolation
    label_to_hz = np.zeros(n_classes, dtype=np.float64)
    for i in range(n_classes):
        if i in pairs:
            label_to_hz[i] = pairs[i]
        elif i < sorted_labels[0]:
            label_to_hz[i] = pairs[sorted_labels[0]]
        elif i > sorted_labels[-1]:
            label_to_hz[i] = pairs[sorted_labels[-1]]
        else:
            lo = max(l for l in sorted_labels if l < i)
            hi = min(l for l in sorted_labels if l > i)
            frac = (i - lo) / (hi - lo)
            label_to_hz[i] = pairs[lo] * (pairs[hi] / pairs[lo]) ** frac

    return label_to_hz, pairs


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance_cents: float = 50.0,
    label_to_hz: np.ndarray = None,
) -> float:
    """
    Compute pitch estimation accuracy within a cent tolerance.

    Args:
        y_true: Ground truth class labels (0–699).
        y_pred: Predicted class labels (0–699).
        tolerance_cents: Tolerance in cents (default: 50 = quarter-tone).
        label_to_hz: Optional (700,) array mapping labels to Hz.
                     If None, uses equal-tempered mapping from 80 Hz.

    Returns:
        Accuracy as a fraction in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if label_to_hz is None:
        # Default: 80 Hz * 2^(label * 6.25/1200)
        label_to_hz = 80.0 * (2.0 ** (np.arange(700) * 6.25 / 1200.0))

    hz_true = label_to_hz[y_true]
    hz_pred = label_to_hz[y_pred]

    cents_true = hz_to_cents(hz_true)
    cents_pred = hz_to_cents(hz_pred)

    error = np.abs(cents_true - cents_pred)
    accuracy = np.mean(error <= tolerance_cents)

    return float(accuracy)
