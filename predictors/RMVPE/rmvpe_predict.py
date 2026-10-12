#!/usr/bin/env python3
"""
RMVPEPredictor — wrapper for RMVPE E2E model inference.

Encapsulates:
  - Model definition (matching publicly available pretrained checkpoint)
  - Model loading (handles DataParallel prefix)
  - Audio preprocessing (resample, mono, zero-pad)
  - Model inference (single-frame mode for short stimuli)
  - Pitch decoding (local weighted average, peak ±4 bins)
  - Voicing detection (confidence threshold)

Architecture notes:
  The publicly shared pretrained checkpoint (lj1995/VoiceConversionWebUI)
  was trained with N_MELS=128, WINDOW_LENGTH=1024, hop_length=160.
  This differs from the official RMVPE repo (N_MELS=256, WINDOW=2048, hop=320)
  but the checkpoint determines the actual architecture.

Key design decisions:
  - Input: raw 16 kHz mono audio (model computes mel internally)
  - Single-frame mode: zero-pad then feed entire waveform
  - Output: Sigmoid activation (independent per-bin, NOT softmax)
  - Pitch decode: local weighted average using CONST=1997.3794084376191
  - Voicing threshold: 0.5 (project default; official default is 0.0)
  - Mel parameters from official code: HTK formula, NO center padding, clamp(1e-5)
    (using the mel formula the paper describes, applied to checkpoint's N_MELS=128)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from scipy.signal import get_window
import warnings

# ============================================================================
# Model architecture constants (matching the publicly available checkpoint)
# ============================================================================
SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 128           # Checkpoint expects 128 mel bins (GRU input = 3*128 = 384)
MEL_FMIN = 30
MEL_FMAX = 8000        # SAMPLE_RATE // 2
WINDOW_LENGTH = 1024   # Checkpoint trained with 1024-point STFT window
HOP_LENGTH = 160       # 10 ms at 16 kHz (checkpoint was trained with this)

# Pitch decoding constants (from official RMVPE src/constants.py)
RMVPE_CONST = 1997.3794084376191
RMVPE_REF_FREQ = 10.0
# Pre-compute cent mapping for local weighted average
CENTS_MAPPING = np.linspace(0, 7180, N_CLASS) + RMVPE_CONST  # (360,)


# ============================================================================
# RMVPE Model Definition (self-contained, matches checkpoint architecture)
# ============================================================================

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.is_shortcut = (in_channels != out_channels)
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        # Determine output_padding based on stride
        if stride == (1, 2):
            out_padding = (0, 1)
        elif stride == (2, 2):
            out_padding = (1, 1)
        elif stride == (2, 1):
            out_padding = (1, 0)
        else:
            out_padding = (1, 1)  # default fallback
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=stride, padding=(1, 1),
                               output_padding=out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks,
                 out_channels=16, momentum=0.01):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(
                in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            skip, x = self.layers[i](x)
            concat_tensors.append(skip)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(
            in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(
                out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(
                in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    """Deep U-Net WITHOUT TimbreFilter (matches publicly available checkpoint)."""
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4,
                 in_channels=1, en_out_channels=16):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size,
                               n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2,
                                         self.encoder.out_channel, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features,
                          num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class E2E(nn.Module):
    """
    End-to-end RMVPE model matching the publicly available checkpoint architecture.

    Key differences from official RMVPE repo:
      - N_MELS=128 (GRU input = 3*128 = 384 matches checkpoint)
      - WINDOW_LENGTH=1024
      - hop_length=160 (10ms at 16kHz)
      - DeepUnet WITHOUT TimbreFilter
      - MelSpectrogram with corrected librosa API calls for v0.11
    """
    def __init__(self, hop_length=160, n_blocks=4, n_gru=1, kernel_size=(2, 2),
                 en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E, self).__init__()
        self.mel = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length,
            mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX,
        )
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers,
                             in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))

        n_mel_input = 3 * N_MELS  # = 384
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(n_mel_input, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(n_mel_input, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, x):
        mel_out = self.mel(x.reshape(-1, x.shape[-1]))
        # Pad time frames to multiple of 32 for U-Net
        n_frames = mel_out.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel_out = F.pad(mel_out, (0, n_pad), mode='constant')
        mel_out = mel_out.transpose(-1, -2).unsqueeze(1)  # (B, 1, T, n_mels)
        x_unet = self.unet(mel_out)
        if n_pad > 0:
            x_unet = x_unet[:, :, :-n_pad, :]
        x_out = self.cnn(x_unet).transpose(1, 2).flatten(-2)  # (B, T, 3*n_mels)

        hidden_vec = 0
        if len(self.fc) == 4:  # has BiGRU
            for i in range(len(self.fc)):
                x_out = self.fc[i](x_out)
                if i == 0:
                    hidden_vec = x_out
        else:
            x_out = self.fc(x_out)

        return hidden_vec, x_out


# ============================================================================
# MelSpectrogram (adapted for librosa 0.11+ compatibility)
# ============================================================================

class MelSpectrogram(torch.nn.Module):
    """
    Mel-spectrogram computation matching RMVPE training setup.

    Uses:
      - HTK mel formula (htk=True, NOT default Slaney)
      - NO STFT center padding (matches commented-out padding in official code)
      - log clamp at 1e-5
      - librosa mel filterbank

    Adapted for librosa >= 0.10 API changes.
    """
    def __init__(self, n_mels, sample_rate, filter_length, hop_length,
                 win_length=None, mel_fmin=0.0, mel_fmax=None):
        super(MelSpectrogram, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else filter_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # STFT basis
        self.register_buffer('forward_basis', self._build_stft_basis())

        # Mel filterbank (HTK formula)
        mel_basis = mel(
            sr=sample_rate, n_fft=filter_length, n_mels=n_mels,
            fmin=mel_fmin, fmax=mel_fmax, htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def _build_stft_basis(self):
        """Build Fourier basis for STFT (same logic as official spec.py)."""
        filter_length = self.filter_length
        win_length = self.win_length

        fourier_basis = np.fft.fft(np.eye(filter_length))
        cutoff = int((filter_length / 2 + 1))
        fourier_basis = np.vstack([
            np.real(fourier_basis[:cutoff, :]),
            np.imag(fourier_basis[:cutoff, :]),
        ])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        # Hann window
        fft_window = get_window('hann', win_length, fftbins=True)
        # librosa's pad_center changed API in 0.10+; handle both cases
        try:
            from librosa.util import pad_center
            fft_window = pad_center(fft_window, size=filter_length)
        except TypeError:
            fft_window = pad_center(fft_window, filter_length)
        fft_window = torch.from_numpy(fft_window).float()
        forward_basis *= fft_window

        return forward_basis.float()

    def forward(self, y):
        """
        Compute mel-spectrogram from raw audio.

        Args:
            y: (B, T) float tensor in range [-1, 1]

        Returns:
            mel_output: (B, n_mels, T_frames) log-mel spectrogram
        """
        assert torch.min(y) >= -1, f"Input min {torch.min(y)} < -1"
        assert torch.max(y) <= 1, f"Input max {torch.max(y)} > 1"

        num_batches = y.size(0)
        num_samples = y.size(1)

        # STFT — NO center padding (as in official code, explicitly commented out)
        y_reshaped = y.view(num_batches, 1, num_samples)

        forward_transform = F.conv1d(
            y_reshaped,
            self.forward_basis,
            stride=self.hop_length,
            padding=0,
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

        # Mel filterbank + log
        mel_output = torch.matmul(self.mel_basis, magnitude)
        mel_output = torch.log(torch.clamp(mel_output, min=1e-5))

        return mel_output


# ============================================================================
# RMVPEPredictor
# ============================================================================

class RMVPEPredictor:
    """
    Predictor wrapper for the RMVPE E2E model.

    Usage:
        predictor = RMVPEPredictor(
            model_path='pretrained_models/rmvpe.pth',
            device='cuda',
            voicing_threshold=0.5,
        )
        f0_hz, confidence, activation_360 = predictor.predict(audio, sr=32000)
    """

    def __init__(
        self,
        model_path: str = 'pretrained_models/rmvpe.pth',
        device: str = 'cuda',
        voicing_threshold: float = 0.5,
        hop_length: int = HOP_LENGTH,
        n_blocks: int = 4,
        n_gru: int = 1,
        kernel_size: tuple = (2, 2),
    ):
        """
        Initialize RMVPE predictor.

        Args:
            model_path: Path to pretrained weights (.pth file).
            device: 'cuda' or 'cpu'.
            voicing_threshold: Confidence threshold for voicing (default: 0.5).
            hop_length: Hop length in samples (default: 160 = 10ms @ 16kHz).
            n_blocks: Number of residual blocks in U-Net.
            n_gru: Number of BiGRU layers (1 = with BiGRU).
            kernel_size: Pooling kernel size in U-Net.
        """
        self.device = torch.device(
            device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        )
        self.voicing_threshold = voicing_threshold
        self.hop_length = hop_length

        # Build model
        self.model = E2E(
            hop_length=hop_length,
            n_blocks=n_blocks,
            n_gru=n_gru,
            kernel_size=kernel_size,
        )

        # Load weights
        self._load_weights(model_path)

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, model_path: str):
        """
        Load pretrained weights.

        The checkpoint (lj1995/VoiceConversionWebUI) does NOT include
        mel.stft.forward_basis or mel.mel_basis (these are computed at init).
        It also omits TimbreFilter layers (DeepUnet without tf).

        We load with strict=False and filter by matching shapes.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                f"Please download from: "
                f"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
            )

        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

        # Handle 'module.' prefix from DataParallel wrapping
        if isinstance(state_dict, dict):
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Filter to matching keys
        model_dict = self.model.state_dict()
        filtered_dict = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                skipped.append(k)

        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict, strict=False)

        n_model = len(model_dict)
        n_loaded = len(filtered_dict)
        if n_loaded < n_model - 5:  # Allow ~5 missing buffers (mel basis, forward_basis)
            warnings.warn(
                f"Loaded only {n_loaded}/{n_model} parameter tensors. "
                f"The checkpoint may be incompatible. "
                f"Missing keys include: {skipped[:5]}..."
            )

    def predict(
        self,
        audio: np.ndarray,
        sr: int = 32000,
        min_len: int = 2048,
    ):
        """
        Run inference on a single audio stimulus.

        Pipeline:
          1. Convert to mono (average channels if stereo)
          2. Resample to 16 kHz (if not already)
          3. Zero-pad to ≥ min_len samples (if needed)
          4. Feed as raw audio to E2E.forward() (model computes mel internally)
          5. Extract first frame's activation (single-frame mode)
          6. Decode pitch via local weighted average
          7. Apply voicing threshold

        Args:
            audio: Raw audio as numpy array (any shape, any sample rate).
            sr: Sample rate of the input audio (default: 32000).
            min_len: Minimum length in samples after resampling (default: 2048).

        Returns:
            f0_hz: float — estimated F0 in Hz (0.0 if unvoiced/error).
            confidence: float — max sigmoid activation value.
            activation_360: numpy array of shape (360,) — raw sigmoid activation.
        """
        try:
            # --- 1. Convert to mono numpy ---
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                if audio.shape[0] > 1:
                    audio = audio.mean(axis=0)
                else:
                    audio = audio.squeeze(0)

            # --- 2. Resample to 16 kHz ---
            if sr != SAMPLE_RATE:
                import torchaudio
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                audio = resampler(audio_tensor).squeeze(0).numpy()

            # --- 3. Normalize to [-1, 1] ---
            audio_max = np.max(np.abs(audio))
            if audio_max > 1.0:
                audio = audio / audio_max

            # --- 4. Zero-pad to minimum length ---
            if len(audio) < min_len:
                pad_total = min_len - len(audio)
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                audio = np.pad(audio, (pad_left, pad_right),
                              mode='constant', constant_values=0)

            # --- 5. Prepare tensor ---
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
            audio_tensor = audio_tensor.to(self.device)

            # --- 6. Forward pass ---
            with torch.inference_mode():
                hidden_vec, output = self.model(audio_tensor)
                # output: (1, T_frames, 360) — sigmoid activation

            # --- 7. Single-frame extraction ---
            # Take first frame (appropriate for short stimuli)
            activation = output.squeeze(0)[0, :].cpu().numpy()  # (360,)

            # --- 8. Pitch decoding ---
            f0_hz, confidence = self.decode_pitch(
                activation, threshold=self.voicing_threshold)

            return float(f0_hz), float(confidence), activation.astype(np.float64)

        except Exception as e:
            warnings.warn(f"RMVPE prediction failed: {e}")
            return 0.0, 0.0, np.zeros(N_CLASS, dtype=np.float64)

    def decode_pitch(
        self,
        activation: np.ndarray,
        threshold: float = 0.0,
    ):
        """
        Decode pitch from RMVPE activation using local weighted average.

        Implements the official RMVPE pitch decoding from src/utils.py
        (to_local_average_cents).

        Algorithm:
          1. Find argmax bin m
          2. Compute weighted average of cents over bins [m-4, m+5)
          3. Convert cents to Hz using ref_freq=10 Hz
          4. If max activation ≤ threshold, output 0 Hz (unvoiced)

        Cent mapping: cents[i] = i * 20 + CONST, CONST = 1997.3794084376191
        Reference frequency: 10 Hz

        Args:
            activation: numpy array of shape (360,) or (T, 360).
            threshold: Voicing threshold (project default: 0.5, official: 0.0).

        Returns:
            f0_hz: float or numpy array of shape (T,) — F0 in Hz.
            confidence: float or numpy array — max activation value.
        """
        # Handle multi-frame input
        if activation.ndim == 2:
            T = activation.shape[0]
            f0s = np.zeros(T, dtype=np.float64)
            confs = np.zeros(T, dtype=np.float64)
            for t in range(T):
                f0s[t], confs[t] = self.decode_pitch(
                    activation[t, :], threshold)
            return f0s, confs

        # --- Single frame ---
        m = int(np.argmax(activation))
        confidence = float(activation[m])

        # Voicing check
        if confidence <= threshold:
            return 0.0, confidence

        # Local weighted average over [m-4, m+5)
        start = max(0, m - 4)
        end = min(N_CLASS, m + 5)
        weights = activation[start:end]
        cent_values = CENTS_MAPPING[start:end]

        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return 0.0, confidence

        c_hat = np.sum(weights * cent_values) / weight_sum
        f0_hz = RMVPE_REF_FREQ * (2.0 ** (c_hat / 1200.0))

        return float(f0_hz), confidence

    def predict_batch(
        self,
        audios: np.ndarray,
        sr: int = 32000,
        min_len: int = 2048,
    ):
        """
        Run inference on a batch of audio stimuli.

        Args:
            audios: numpy array of shape (B, T_samples) at original sample rate.
            sr: Sample rate of the input audio.
            min_len: Minimum length after resampling.

        Returns:
            List of (f0_hz, confidence, activation_360) tuples.
        """
        results = []
        for i in range(len(audios)):
            result = self.predict(audios[i], sr=sr, min_len=min_len)
            results.append(result)
        return results
