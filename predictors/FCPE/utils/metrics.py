"""
Optional metrics and helper functions for F0 evaluation.

Provides utilities for F0 conversion and pitch-related calculations.
"""

import numpy as np
from typing import Tuple


def hz_to_cents(freq_hz: float, ref_hz: float = 10.0) -> float:
    """
    Convert frequency in Hz to cents relative to a reference.

    Args:
        freq_hz: Frequency in Hz.
        ref_hz: Reference frequency in Hz (default: 10 Hz).

    Returns:
        Cents value.
    """
    if freq_hz <= 0:
        return 0.0
    return 1200.0 * np.log2(freq_hz / ref_hz)


def cents_to_hz(cents: float, ref_hz: float = 10.0) -> float:
    """
    Convert cents (relative to ref_hz) to frequency in Hz.

    Args:
        cents: Cents value relative to ref_hz.
        ref_hz: Reference frequency in Hz (default: 10 Hz).

    Returns:
        Frequency in Hz.
    """
    return ref_hz * (2.0 ** (cents / 1200.0))


def josh_bin_to_hz(bin_idx: int) -> float:
    """
    Convert a Josh 700-grid bin index to frequency in Hz.

    Josh grid: 700 bins, 6.25-cent spacing, starting at 80 Hz.
    f_j = 80 * 2^(j * 6.25 / 1200)

    Args:
        bin_idx: Bin index (0–699).

    Returns:
        Frequency in Hz.
    """
    cents_per_bin = 100.0 / 16.0  # 6.25 cents
    return 80.0 * (2.0 ** (bin_idx * cents_per_bin / 1200.0))


def fcpe_bin_to_hz(bin_idx: int) -> float:
    """
    Convert an FCPE 360-grid bin index to frequency in Hz.

    FCPE grid: 360 bins, 20-cent spacing, reference = 10 Hz.
    f_i = 10 * 2^(i * 20 / 1200) = 10 * 2^(i / 60)

    Args:
        bin_idx: Bin index (0–359).

    Returns:
        Frequency in Hz.
    """
    ref_freq = 10.0
    return ref_freq * (2.0 ** (bin_idx * 20.0 / 1200.0))
