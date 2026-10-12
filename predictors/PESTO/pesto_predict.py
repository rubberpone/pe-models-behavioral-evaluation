#!/usr/bin/env python3
"""
pesto_predict.py — Wrapper for PESTO pitch prediction.

Provides a unified interface for single-file and batch processing,
handling mono conversion, audio loading, and prediction aggregation.

Usage:
    from pesto_predict import PestoPredictor

    predictor = PestoPredictor(step_size=10)
    f0_hz, confidence = predictor.predict(waveform, sr)
    f0_median = predictor.predict_aggregated(waveform, sr, method='median')
"""

import numpy as np
import torch
import pesto


class PestoPredictor:
    """
    Thin wrapper around pesto.predict() for psychophysical experiments.

    PESTO (Pitch Estimation with Self-supervised Transposition-equivariant
    Objective) is a lightweight (130k param) end-to-end pitch estimator that
    processes VQT frames and outputs pitch probability distributions.

    Parameters
    ----------
    step_size : float
        Hop size in milliseconds between consecutive frames.
        Default 10 ms (as used in the PESTO paper).
    reduction : str
        Pitch decoding method. Default 'alwa' (Argmax-Local Weighted
        Averaging, same as CREPE).
    model_name : str
        Name of the PESTO model checkpoint. Default 'mir-1k_g7'.
    gpu : int
        GPU device ID. -1 for CPU (default).
    """

    def __init__(
        self,
        step_size: float = 10.0,
        reduction: str = 'alwa',
        model_name: str = 'mir-1k_g7',
        gpu: int = -1,
    ):
        self.step_size = step_size
        self.reduction = reduction
        self.model_name = model_name
        self.gpu = gpu
        self._device = torch.device(
            f'cuda:{gpu}' if (gpu >= 0 and torch.cuda.is_available()) else 'cpu'
        )

    def predict(
        self,
        waveform: np.ndarray,
        sr: int = 32000,
    ):
        """
        Run PESTO pitch prediction on a waveform.

        PESTO accepts shapes: (samples,) or (batch_size, samples).
        All inputs are treated as mono.

        Args:
            waveform: numpy array of shape (samples,) or (batch, samples).
            sr: Sampling rate in Hz (default 32000).

        Returns:
            tuple: (timesteps, pitch_hz, confidence, activations)
                - timesteps: time stamps in seconds (1D array)
                - pitch_hz: pitch predictions in Hz (1D for single, 2D for batch)
                - confidence: voicing confidence (1D for single, 2D for batch)
                - activations: raw pitch activations (2D or 3D array)
        """
        # Convert to torch tensor — PESTO handles (samples,) and (batch, samples)
        x = torch.as_tensor(np.asarray(waveform, dtype=np.float32),
                            dtype=torch.float32).to(self._device)

        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, samples)

        timesteps, pitch, confidence, activations = pesto.predict(
            x,
            sr,
            step_size=self.step_size,
            reduction=self.reduction,
            model_name=self.model_name,
        )

        # Move to CPU and convert to numpy
        timesteps = timesteps.cpu().numpy()
        pitch = pitch.cpu().numpy()
        confidence = confidence.cpu().numpy()
        activations = activations.cpu().numpy()

        # Squeeze batch dim for single-stimulus inputs
        if waveform.ndim == 1:
            pitch = pitch[0]
            confidence = confidence[0] if confidence.ndim >= 1 else confidence

        return timesteps, pitch, confidence, activations

    def predict_aggregated(
        self,
        waveform: np.ndarray,
        sr: int = 32000,
        method: str = 'median',
    ) -> float:
        """
        Run PESTO prediction on a single waveform and aggregate frame outputs.

        For steady-state stimuli, the median over all time frames provides
        a robust estimate of the (constant) F0.  Zero-valued frames are
        excluded before aggregation.

        Args:
            waveform: Single waveform as numpy array of shape (samples,).
            sr: Sampling rate in Hz.
            method: Aggregation method — 'median' (default) or 'mean'.

        Returns:
            Aggregated F0 value in Hz (float).
        """
        _, pitch_hz, confidence, _ = self.predict(waveform, sr)

        # pitch_hz is now 1D (n_frames,) for single input
        # Filter by confidence — keep only frames model considers voiced
        if confidence is not None and len(confidence) > 0:
            voiced_mask = confidence > 0.1
            if voiced_mask.any():
                pitch_hz = pitch_hz[voiced_mask]

        # Exclude zero / silent frames
        valid = pitch_hz[pitch_hz > 0]
        if len(valid) == 0:
            return 0.0

        if method == 'median':
            return float(np.median(valid))
        elif method == 'mean':
            return float(np.mean(valid))
        else:
            raise ValueError(f"Unknown aggregation method: '{method}'")

    @property
    def device(self) -> torch.device:
        return self._device
