"""
(Optional) Metrics and helper functions for F0 conversion and evaluation.
"""

import numpy as np


def hz_to_cents(freq_hz, ref_hz=27.5):
    """Convert Hz to cents relative to a reference."""
    return 1200.0 * np.log2(np.maximum(freq_hz, 1e-10) / ref_hz)


def cents_to_hz(cents, ref_hz=27.5):
    """Convert cents to Hz."""
    return ref_hz * (2 ** (cents / 1200.0))


def midi_to_hz(midi_note):
    """Convert MIDI note to Hz."""
    return 440.0 * (2 ** ((midi_note - 69.0) / 12.0))


def hz_to_midi(freq_hz):
    """Convert Hz to MIDI note."""
    return 69.0 + 12.0 * np.log2(np.maximum(freq_hz, 1e-10) / 440.0)
