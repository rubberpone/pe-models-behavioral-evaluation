"""
FCPE Predictor — Wrapper for FCPE (Fast Context-based Pitch Estimation) inference.

Encapsulates model loading via torchfcpe's spawn_bundled_infer_model() and
provides a high-level predict() method that:
  1. Runs the raw waveform through the internal Wav2Mel + CFNaiveMelPE pipeline.
  2. Decodes per-frame pitch using the model's built-in latent2cents_local_decoder
     (local weighted average in cent space, ±4 bins = 9-bin window).
  3. Aggregates multi-frame outputs: median of Hz, mean of sigmoid activations.
     (Custom aggregation — official FCPE outputs per-frame f0 without aggregation.)

Key insight: FCPE's cent_table is NOT `np.arange(360) * 20`. It is:
    linspace(1200*log2(f0_min/10), 1200*log2(f0_max/10), 360)
    where f0_min = 32.70 Hz, f0_max = 1975.5 Hz.
This gives ~19.78 cent spacing, NOT exactly 20 cents.

Reference: Luo et al. (2025). FCPE: A Fast Context-based Pitch Estimation Model.
           arXiv:2509.15140. Code: https://github.com/CNChTu/FCPE
"""

import numpy as np
import torch
from typing import Tuple, Optional


class FCPEPredictor:
    """
    Wrapper for FCPE pitch estimation model using the official torchfcpe package.

    Usage:
        predictor = FCPEPredictor(device='cuda')
        f0_hz, activation_360 = predictor.predict(audio_1d_numpy)
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize the FCPE predictor.

        Args:
            device: Device string ('cuda' or 'cpu').
        """
        from torchfcpe import spawn_bundled_infer_model

        self.device = device
        print(f"[FCPEPredictor] Loading FCPE model on {device}...")

        # spawn_bundled_infer_model returns an InferCFNaiveMelPE instance
        # that wraps both Wav2Mel preprocessor and CFNaiveMelPE model.
        # Bundled pre-trained weights are loaded automatically.
        self.model = spawn_bundled_infer_model(device=device)
        self.model.eval()

        # Extract the cent_table from the internal CFNaiveMelPE model.
        # This is the canonical mapping from bin index → cents.
        # cent_table shape: (360,), values in cents relative to 10 Hz.
        self.cent_table = self.model.model.cent_table.cpu().numpy()

        print(f"[FCPEPredictor] Model loaded: {type(self.model).__name__}")
        print(f"[FCPEPredictor] cent_table: [{self.cent_table[0]:.1f}, ..., "
              f"{self.cent_table[-1]:.1f}] cents "
              f"(step ≈ {(self.cent_table[-1] - self.cent_table[0]) / 359:.2f} cents)")

    def predict(self, audio: np.ndarray, sr: int = 32000,
                threshold: float = 0.006) -> Tuple[float, np.ndarray]:
        """
        Run FCPE inference on a single audio segment.

        Pipeline:
          1. Convert numpy audio to torch tensor (1, n_samples, 1).
          2. Pass through model.wav2mel() for internal feature extraction.
          3. Pass features through model.model() to get 360-dim sigmoid activations.
          4. Use model.model.latent2cents_local_decoder() for per-frame Hz
             (official local weighted average in cent space).
          5. Aggregate: median of per-frame Hz, element-wise mean of activations.

        Args:
            audio: 1-D numpy array of audio samples. Can be at any sample rate —
                   the model's internal Wav2Mel resamples to 16 kHz automatically.
            sr: Sample rate of the input audio (default: 32000 for our stimuli).
            threshold: V/UV confidence threshold (default: 0.006, matching FCPE).
                       Frames with max activation ≤ threshold are treated as unvoiced
                       (their f0 is set to 0 Hz in the decoder output).

        Returns:
            f0_hz: Continuous pitch estimate in Hz (median of per-frame estimates,
                   excluding unvoiced frames).
            activation_360: numpy array of shape (360,) — element-wise mean of
                           per-frame sigmoid activations. Values ∈ [0, 1];
                           does NOT sum to 1 (sigmoid, not softmax).
        """
        # Ensure clean 1-D float32 array
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D audio, got shape {audio.shape}")

        # Normalize peak amplitude to avoid clipping
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = audio / peak * 0.95

        # Convert to tensor: shape (1, n_samples, 1)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1)
        audio_tensor = audio_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            # Step 1: Internal feature extraction (resample + STFT + mel/STFT filterbank + log)
            mels = self.model.wav2mel(audio_tensor, sr)  # shape: (1, T, C)

            # Step 2: CFNaiveMelPE forward pass → sigmoid activations
            latent = self.model.model(mels)  # shape: (1, T, 360)

            # Step 3: Official local_argmax decoder → cents → Hz per frame
            cents = self.model.model.latent2cents_local_decoder(
                latent, threshold=threshold, mask=True
            )  # shape: (1, T, 1), unvoiced frames have -inf cents
            f0_per_frame = self.model.model.cent_to_f0(cents)  # shape: (1, T, 1)

        # Move to CPU for numpy processing
        latent_np = latent.squeeze(0).cpu().numpy()  # shape: (T, 360)
        f0_np = f0_per_frame.squeeze(0).squeeze(-1).cpu().numpy()  # shape: (T,)

        n_frames = latent_np.shape[0]

        if n_frames == 0:
            return 0.0, np.zeros(360, dtype=np.float64)

        # --- Per-frame aggregation ---
        # Custom aggregation (not part of official FCPE):
        # - Median of per-frame Hz estimates (excluding unvoiced: f0 == 0)
        # - Element-wise mean of per-frame sigmoid activations

        # Filter out unvoiced frames for Hz aggregation
        valid_mask = f0_np > 0
        if np.any(valid_mask):
            f0_hz = float(np.median(f0_np[valid_mask]))
        else:
            # All frames unvoiced — use best-effort with threshold=0 (no masking)
            cents_all = self.model.model.latent2cents_local_decoder(
                latent, threshold=0.0, mask=False
            )
            f0_all = self.model.model.cent_to_f0(cents_all)
            f0_all_np = f0_all.squeeze(0).squeeze(-1).cpu().numpy()
            f0_hz = float(np.median(f0_all_np))

        # Element-wise mean of sigmoid activations
        activation_360 = np.mean(latent_np, axis=0).astype(np.float64)
        # activation_360 shape: (360,), values ∈ [0, 1], does NOT sum to 1 (sigmoid)

        return f0_hz, activation_360

    def get_cent_table(self) -> np.ndarray:
        """
        Return the model's cent_table for use in 360→700 mapping.

        The cent_table maps each of the 360 FCPE bins to its cent value
        relative to 10 Hz. This is needed for accurate Gaussian interpolation
        to the 700-bin Josh grid.

        Returns:
            numpy array of shape (360,), cents relative to 10 Hz.
        """
        return self.cent_table.copy()
