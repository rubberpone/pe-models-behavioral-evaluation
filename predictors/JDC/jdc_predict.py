#!/usr/bin/env python3
"""
JDC Predictor — Wrapper for JDC model loading and inference.

Provides a clean prediction interface that handles the full pipeline:
  audio → STFT → power_to_db → Z-score norm → chunking → JDC → pitch decoding → medfilt
"""

import os
import sys
import numpy as np
import torch
from scipy.signal import medfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.jdc import JDC
from utils.audio_utils import extract_jdc_feature, get_num_original_frames


class JDCPredictor:
    """
    Wrapper for JDC (Joint Detection and Classification) pitch estimation.

    Handles:
      - Model loading (from PyTorch checkpoint)
      - Spectrogram feature extraction (STFT → power_to_db → Z-score → chunks)
      - Dual-output inference (pitch 722 + voice detection 2)
      - Pitch decoding (argmax → MIDI → Hz → medfilt)
    """

    def __init__(
        self,
        checkpoint_path=None,
        device=None,
        mean_path=None,
        std_path=None,
    ):
        """
        Args:
            checkpoint_path: Path to .pt weights file
            device: torch device (auto-detected if None)
            mean_path: Path to x_data_mean_total_31.npy
            std_path: Path to x_data_std_total_31.npy
        """
        self.project_root = os.path.dirname(os.path.abspath(__file__))

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.project_root, 'pretrained_models', 'jdc_weights.pt'
            )
        if mean_path is None:
            mean_path = os.path.join(
                self.project_root, 'melodyExtraction_JDC-master',
                'x_data_mean_total_31.npy'
            )
        if std_path is None:
            std_path = os.path.join(
                self.project_root, 'melodyExtraction_JDC-master',
                'x_data_std_total_31.npy'
            )

        self.checkpoint_path = checkpoint_path
        self.mean_path = mean_path
        self.std_path = std_path

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load model
        self.model = JDC.load_from_checkpoint(checkpoint_path, device=self.device)

        # Build pitch range (matching official implementation)
        # Index 0 = non-voice (pitch=0)
        # Indices 1..721 = MIDI 38..83 with 1/16 semitone resolution
        self.pitch_range = np.arange(38.0, 83.0 + 0.5 / 16.0, 1.0 / 16.0)
        self.pitch_range = np.concatenate([np.zeros(1), self.pitch_range])  # 722 total

    def predict(self, audio, sr=8000, batch_size=64, verbose=False):
        """
        Predict F0 for an audio signal using JDC.

        Args:
            audio: 1D numpy array (mono, 8 kHz)
            sr: Sampling rate (must be 8000)
            batch_size: Batch size for processing multiple chunks
            verbose: Print progress information

        Returns:
            dict with keys:
              - 'f0_hz': 1D numpy array of pitch estimates per frame (Hz)
              - 'activation_722': (N, 722) array of per-chunk activations
              - 'activation_V': (N, 2) array of per-chunk voice detection probs
              - 'n_original_frames': Number of original (non-padded) STFT frames
        """
        assert sr == 8000, f"JDC requires 8 kHz audio, got {sr} Hz"

        # 1. Extract features: (N, 31, 513, 1) dB-scale, Z-score normalized
        x_test = extract_jdc_feature(
            audio,
            sr=sr,
            n_frames=31,
            hop_length=80,
            n_fft=1024,
            mean_path=self.mean_path,
            std_path=self.std_path,
        )

        n_original_frames = get_num_original_frames(
            len(audio), sr=8000, hop_length=80, n_fft=1024
        )

        if x_test.shape[0] == 0:
            return {
                'f0_hz': np.array([]),
                'activation_722': np.array([]),
                'activation_V': np.array([]),
                'n_original_frames': n_original_frames,
            }

        # 2. Transpose to PyTorch channel-first: (N, 1, 31, 513)
        input_tensor = torch.FloatTensor(x_test).permute(0, 3, 1, 2).to(self.device)

        # 3. Run inference with batching
        all_outputs = []
        all_outputs_V = []

        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i:i + batch_size]
            with torch.no_grad():
                output, output_V = self.model(batch)
                # output:    (B, 31, 722)
                # output_V:  (B, 31, 2)
            all_outputs.append(output.cpu().numpy())
            all_outputs_V.append(output_V.cpu().numpy())

        outputs = np.concatenate(all_outputs, axis=0)    # (N, 31, 722)
        outputs_V = np.concatenate(all_outputs_V, axis=0)  # (N, 31, 2)

        # 4. Decode pitch following official implementation
        # Flatten all frames across all chunks
        num_total = outputs.shape[0] * outputs.shape[1]  # N * 31
        output_flat = outputs.reshape(num_total, 722)

        est_pitch = np.zeros(num_total)
        for i in range(num_total):
            idx = np.argmax(output_flat[i, :])
            pitch_midi = self.pitch_range[int(idx)]
            if pitch_midi >= 38.0 and pitch_midi <= 83.0:
                est_pitch[i] = 2 ** ((pitch_midi - 69.0) / 12.0) * 440.0

        # 5. Apply 5-sample median filter (as in official implementation)
        est_pitch = medfilt(est_pitch, 5)

        return {
            'f0_hz': est_pitch,
            'activation_722': output_flat,        # (N*31, 722)
            'activation_V': outputs_V.reshape(-1, 2),  # (N*31, 2)
            'n_original_frames': n_original_frames,
            'n_chunks': x_test.shape[0],
        }

    def predict_batch_from_features(self, features, n_original_frames_list=None):
        """
        Run inference on a batch of precomputed spectrogram features.

        Each stimulus produces 1 chunk of (1, 31, 513), so we stack them
        into a batch of (B, 1, 31, 513) for efficient GPU inference.

        Args:
            features: numpy array of shape (B, 31, 513, 1) —
                       precomputed features for B stimuli
            n_original_frames_list: list of ints with original frame counts per stimulus

        Returns:
            list of dicts with keys: f0_hz, activation_722, non_voice_prob
        """
        if features.shape[0] == 0:
            return []

        # Transpose to channel-first: (B, 1, 31, 513)
        input_tensor = torch.FloatTensor(features).permute(0, 3, 1, 2).to(self.device)

        # Model inference
        with torch.no_grad():
            output, _ = self.model(input_tensor)
            # output: (B, 31, 722)

        activations = output.cpu().numpy()  # (B, 31, 722)
        B = activations.shape[0]

        # Pitch range for argmax
        pitch_range = self.pitch_range  # (722,)

        results = []
        for i in range(B):
            act = activations[i]  # (31, 722)

            # Number of valid frames for this stimulus
            n_orig = n_original_frames_list[i] if n_original_frames_list else 31
            n_valid = min(n_orig, 31)

            valid_act = act[:n_valid]  # (n_valid, 722)

            # Decode pitch per frame
            est_pitch = np.zeros(n_valid)
            for t in range(n_valid):
                idx = np.argmax(valid_act[t])
                pitch_midi = pitch_range[int(idx)]
                if pitch_midi >= 38.0 and pitch_midi <= 83.0:
                    est_pitch[t] = 2 ** ((pitch_midi - 69.0) / 12.0) * 440.0

            # Median filter
            est_pitch = medfilt(est_pitch, 5) if len(est_pitch) >= 5 else est_pitch

            # Aggregate
            voiced_pitch = est_pitch[est_pitch > 0]
            f0_hz = float(np.median(voiced_pitch)) if len(voiced_pitch) > 0 else 0.0

            activation_722 = valid_act.mean(axis=0)
            non_voice_prob = float(activation_722[0])

            results.append({
                'f0_hz': f0_hz,
                'activation_722': activation_722,
                'non_voice_prob': non_voice_prob,
            })

        return results

    def predict_stimulus(self, audio, sr=8000):
        """
        Predict F0 for a single 50-ms stimulus.

        Args:
            audio: 1D numpy array (mono, 8 kHz)

        Returns:
            dict with f0_hz, activation_722, non_voice_prob
        """
        result = self.predict(audio, sr=sr)
        n_orig = result['n_original_frames']

        if n_orig == 0 or result['activation_722'].shape[0] == 0:
            return {
                'f0_hz': 0.0,
                'activation_722': np.ones(722, dtype=np.float64) / 722,
                'non_voice_prob': 0.0,
            }

        n_valid = min(n_orig, result['activation_722'].shape[0])

        valid_activations = result['activation_722'][:n_valid]
        valid_pitch = result['f0_hz'][:n_valid]

        voiced_pitch = valid_pitch[valid_pitch > 0]
        f0_hz = float(np.median(voiced_pitch)) if len(voiced_pitch) > 0 else 0.0

        activation_722 = valid_activations.mean(axis=0)
        non_voice_prob = float(activation_722[0])

        return {
            'f0_hz': f0_hz,
            'activation_722': activation_722,
            'non_voice_prob': non_voice_prob,
        }
