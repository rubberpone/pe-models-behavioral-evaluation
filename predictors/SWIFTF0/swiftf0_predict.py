"""
SwiftF0Predictor — Convenience wrapper for SwiftF0 pitch prediction.

Encapsulates model loading and provides a simple predict() interface
that returns both the scalar F0 estimate (Hz) and the 200-dim activation.
"""

import numpy as np
from typing import Tuple, Optional
import os

from swift_f0_activation import SwiftF0Activation, PitchResultWithActivation


class SwiftF0Predictor:
    """High-level predictor wrapping SwiftF0Activation for psychophysics.

    Handles model loading, audio preprocessing (mono + resample),
    and returns both F0 estimates and raw activations.

    Note: SwiftF0 runs on CPU via ONNX Runtime by default.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.9,
        fmin: float = 46.875,
        fmax: float = 2093.75,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to modified ONNX model. Uses default if None.
            confidence_threshold: Confidence threshold for voicing (0-1).
            fmin: Minimum voiced frequency in Hz.
            fmax: Maximum voiced frequency in Hz.
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'model_with_activation.onnx'
            )

        self.detector = SwiftF0Activation(
            confidence_threshold=confidence_threshold,
            fmin=fmin,
            fmax=fmax,
            model_path=model_path,
        )
        self.model_path = model_path

    def predict(
        self, audio: np.ndarray, sr: int = 32000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict F0 and return activations for a single audio trial.

        Args:
            audio: 1D numpy array of audio samples.
            sr: Sample rate in Hz. Default 32000 (original psychophysics data rate).

        Returns:
            Tuple of (f0_hz, activation_200):
                - f0_hz: numpy array of F0 estimates per frame (Hz), shape (T,)
                - activation_200: 200-dim softmax probability distribution per frame,
                  shape (T, 200)
        """
        result = self.detector.detect_from_array(audio, sample_rate=sr)
        return result.pitch_hz, result.activation

    def predict_aggregated(
        self, audio: np.ndarray, sr: int = 32000
    ) -> Tuple[float, np.ndarray]:
        """
        Predict F0 and return time-aggregated activation for a single audio trial.

        For short audio (e.g., 50-150ms), typically produces 1-9 frames.
        Aggregates by averaging activations across all time frames.

        Args:
            audio: 1D numpy array of audio samples.
            sr: Sample rate in Hz.

        Returns:
            Tuple of (f0_hz_median, activation_200_mean):
                - f0_hz_median: Median F0 across frames (Hz), scalar float
                - activation_200_mean: Mean 200-dim softmax across frames, shape (200,)
        """
        f0_hz, activation = self.predict(audio, sr)

        # Aggregate across time frames
        f0_median = float(np.median(f0_hz))
        activation_mean = activation.mean(axis=0)  # (200,)

        # Re-normalize after averaging
        if activation_mean.sum() > 0:
            activation_mean = activation_mean / activation_mean.sum()

        return f0_median, activation_mean
