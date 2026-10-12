#!/usr/bin/env python3
"""
FT Attention Predictor — encapsulates the FT Attention model for F0 estimation.

Handles: model loading, CFP computation, inference, and frame aggregation.
Uses TensorFlow/Keras to load the official .h5 pre-trained weights.

Input:  32 kHz mono audio numpy array
Output: (f0_hz, activation_321) — F0 estimate in Hz and 321-dim softmax probabilities
"""

import os
import sys
import numpy as np

# Ensure FTANet-melodic-main is on the path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_FTANET_PATH = os.path.join(_PROJECT_ROOT, "FTANet-melodic-main")
if _FTANET_PATH not in sys.path:
    sys.path.insert(0, _FTANET_PATH)

from utils.audio_utils import (
    resample_audio,
    ensure_mono,
    compute_cfp,
    pad_audio_centered,
)


class FTAttentionPredictor:
    """
    FT Attention F0 predictor wrapping the official TF/Keras model.

    Usage:
        predictor = FTAttentionPredictor(model_path='FTANet-melodic-main/model/ftanet.h5')
        f0_hz, activation_321 = predictor.predict(audio_32k, sr=32000)
    """

    def __init__(self, model_path=None):
        """
        Initialize and load the FT Attention model.

        Args:
            model_path: path to .h5 weights file.
                        Defaults to FTANet-melodic-main/model/ftanet.h5
        """
        # Suppress TF logging
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

        import tensorflow as tf
        # Import model definition from official code
        from network.ftanet import create_model
        from constant import IN_SHAPE  # (320, 128, 3)

        if model_path is None:
            model_path = os.path.join(
                _PROJECT_ROOT, "FTANet-melodic-main", "model", "ftanet.h5"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        print(f"[FTAttentionPredictor] Loading model from: {model_path}")
        print(f"[FTAttentionPredictor] Input shape (training): {IN_SHAPE}")

        # Create model with fixed time dimension (the architecture uses //
        # operations on shape dims and cannot handle None)
        self.model = create_model(input_shape=IN_SHAPE)  # (320, 128, 3)
        self.model.load_weights(model_path)
        print("[FTAttentionPredictor] Model loaded successfully.")

        # Store config
        self.sr_cfp = 8000
        self.hop = 80
        self.window_size = 768
        self.fmin = 31.0
        self.fmax = 1250.0
        self.n_bins_per_octave = 60
        self.n_voiced = 320
        self.n_total = 321

        # Target padding: 10320 samples at 8kHz → exactly 128 CFP frames at hop=80
        # (ceil(10320/80)=129, arange gives 128 frames; must match model IN_SHAPE)
        self.target_samples = 10320  # gives exactly 128 CFP frames
        self.time_segment = 128      # fixed T for model input

    def predict(self, audio_32k, sr=32000):
        """
        Run FT Attention inference on a single audio segment.

        Full pipeline:
          1. Ensure mono
          2. Resample 32kHz → 8kHz
          3. Zero-pad to ~1.28s (centered)
          4. Compute CFP 3-channel representation
          5. Model forward pass → (321, T) softmax probabilities
          6. Aggregate frames: median Hz for voiced, mean activation

        Args:
            audio_32k: 1-D ndarray, audio waveform at original sr
            sr: original sampling rate in Hz (default 32000)

        Returns:
            (f0_hz, activation_321) tuple:
              - f0_hz: float, estimated F0 in Hz (median over voiced frames)
              - activation_321: ndarray of shape (321,), frame-mean softmax probabilities
        """
        # Step 1: Ensure mono
        audio = ensure_mono(audio_32k)

        # Step 2: Resample 32kHz → 8kHz
        audio_8k = resample_audio(audio, orig_sr=sr, target_sr=self.sr_cfp)

        # Step 3: Zero-pad to ~1.28s
        audio_padded = pad_audio_centered(audio_8k, self.target_samples)

        # Step 4: Compute CFP representation
        cfp, cen_freq, time = compute_cfp(
            audio_padded,
            sr=self.sr_cfp,
            hop=self.hop,
            window_size=self.window_size,
            fmin=self.fmin,
            fmax=self.fmax,
            n_bins_per_octave=self.n_bins_per_octave,
        )
        # cfp shape: (3, 320, T) — need to transpose for Keras input (batch, F, T, C)
        cfp_input = np.transpose(cfp, (1, 2, 0))  # (320, T, 3)

        # Pad/trim time dimension to exactly 128 frames for fixed model input
        T_actual = cfp_input.shape[1]
        if T_actual < self.time_segment:
            # Zero-pad along time axis
            pad_width = self.time_segment - T_actual
            cfp_input = np.pad(cfp_input, ((0, 0), (0, pad_width), (0, 0)),
                               mode='constant', constant_values=0)
        elif T_actual > self.time_segment:
            # Trim time axis
            cfp_input = cfp_input[:, :self.time_segment, :]

        cfp_input = cfp_input[np.newaxis, ...]     # (1, 320, 128, 3)

        # Step 5: Model forward pass
        # Output shape: (1, 321, 128) — softmax already applied
        probs = self.model.predict(cfp_input, verbose=0)  # (1, 321, 128)

        # Step 6: Aggregate frames
        # Use only the valid (non-padded) frames for aggregation
        probs = np.squeeze(probs, axis=0)  # (321, 128)
        probs_valid = probs[:, :T_actual]   # (321, T_actual) — discard padded frames
        T = probs_valid.shape[1]

        if T == 1:
            # Single frame: use directly
            activation_321 = probs_valid[:, 0]  # (321,)
            # Decode F0 from voiced bins (indices 1-320)
            voiced_probs = activation_321[1:]  # (320,)
            if voiced_probs.max() > 0:
                bin_idx = np.argmax(voiced_probs)  # 0-based index in [0, 319]
                f0_hz = self.fmin * (2 ** (bin_idx / self.n_bins_per_octave))
            else:
                f0_hz = 0.0
        else:
            # Multi-frame: mean activation, median voiced frequency
            activation_321 = probs_valid.mean(axis=1)  # (321,)

            # Per-frame F0 estimates (voiced bins only, ignoring unvoiced)
            per_frame_hz = []
            for t in range(T):
                frame_probs = probs_valid[:, t]  # (321,)
                voiced_probs = frame_probs[1:]  # (320,)
                if voiced_probs.max() > 0:
                    bin_idx = np.argmax(voiced_probs)
                    hz = self.fmin * (2 ** (bin_idx / self.n_bins_per_octave))
                    per_frame_hz.append(hz)

            if per_frame_hz:
                f0_hz = float(np.median(per_frame_hz))
            else:
                f0_hz = 0.0

        return f0_hz, activation_321
