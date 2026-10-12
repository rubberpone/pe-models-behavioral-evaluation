"""
Optional metrics helpers for F0 conversion and evaluation.
"""

import numpy as np


def hz_to_cents(freq_hz: float, ref_hz: float = 10.0) -> float:
    """Convert frequency in Hz to cents relative to reference."""
    if freq_hz <= 0:
        return 0.0
    return 1200.0 * np.log2(freq_hz / ref_hz)


def cents_to_hz(cents: float, ref_hz: float = 10.0) -> float:
    """Convert cents value to frequency in Hz."""
    return ref_hz * (2.0 ** (cents / 1200.0))


def bin_to_freq(bin_idx: int, f_min: float = 80.0, cents_per_bin: float = 6.25) -> float:
    """Convert a 700-grid bin index to frequency in Hz."""
    return f_min * (2.0 ** (bin_idx * cents_per_bin / 1200.0))


def freq_to_bin(freq_hz: float, f_min: float = 80.0, cents_per_bin: float = 6.25) -> int:
    """Convert a frequency in Hz to the nearest 700-grid bin index."""
    if freq_hz <= 0:
        return 0
    cents = 1200.0 * np.log2(freq_hz / f_min)
    return int(round(cents / cents_per_bin))
