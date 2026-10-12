"""
FCN predictor wrapper: model loading, inference, and pitch decoding.

Encapsulates FCNPitchEstimator with a convenient predict() interface
that takes preprocessed numpy audio and returns both a continuous
F0 estimate (via local weighted average decoding) and the raw 486-bin
activation vector.
"""

import os
import numpy as np
import torch

from fcn_model import FCNPitchEstimator
from utils.probability_utils import continuous_f0_from_activation


class FCNPredictor:
    """
    Wrapper for FCN model loading and inference.

    Handles:
      - Model loading from .pt checkpoint
      - Device management (CPU/CUDA)
      - Single-segment and batch inference
      - Pitch decoding via local weighted average
    """

    def __init__(self, model_type='fcn_993', checkpoint_dir=None, device=None):
        """
        Args:
            model_type: 'fcn_1953', 'fcn_993', or 'fcn_929'
            checkpoint_dir: directory containing {model_type}.pt files
            device: torch device ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_type = model_type

        # Device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        # Build model
        self.model = FCNPitchEstimator(model_type=model_type)

        # Load weights
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'pretrained_models'
            )

        checkpoint_path = os.path.join(checkpoint_dir, f'{model_type}.pt')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Run 'python utils/convert_weights.py' first to convert "
                f"Keras weights to PyTorch format."
            )

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.min_input_length = self.model.min_input_length

    def predict(self, audio_norm):
        """
        Run inference on a single preprocessed audio segment.

        Args:
            audio_norm: 1D numpy array of length min_input_length,
                        already resampled to 8 kHz, padded, and normalized

        Returns:
            tuple of (f0_hz, activation_486)
              f0_hz:         continuous F0 estimate in Hz
              activation_486: numpy array of shape (486,), raw sigmoid activations
        """
        # Convert to tensor
        if not isinstance(audio_norm, np.ndarray):
            audio_norm = np.asarray(audio_norm, dtype=np.float32)
        else:
            audio_norm = audio_norm.astype(np.float32)

        input_tensor = torch.from_numpy(audio_norm).unsqueeze(0).unsqueeze(0)
        # shape: (1, 1, samples)
        input_tensor = input_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)  # shape: (1, 486)

        activation = output.squeeze().cpu().numpy()  # (486,)

        # Decode continuous F0
        f0_hz = continuous_f0_from_activation(activation, return_hz=True)

        return f0_hz, activation

    def predict_batch(self, audio_batch, verbose=True):
        """
        Run inference on a batch of preprocessed audio segments.

        Args:
            audio_batch: numpy array of shape (N, min_input_length)
            verbose: show progress bar

        Returns:
            tuple of (f0_hz_list, activations_array)
              f0_hz_list:        list of N float F0 estimates
              activations_array: numpy array of shape (N, 486)
        """
        from tqdm import tqdm

        N = audio_batch.shape[0]
        all_f0 = []
        all_activations = np.zeros((N, 486), dtype=np.float32)
        errors = []

        iterator = range(N)
        if verbose:
            iterator = tqdm(iterator, desc="  Predicting")

        for i in iterator:
            try:
                f0_hz, act = self.predict(audio_batch[i])
                all_f0.append(f0_hz)
                all_activations[i] = act
            except Exception as e:
                all_f0.append(None)
                all_activations[i] = np.zeros(486, dtype=np.float32)
                errors.append(i)
                if verbose:
                    print(f"  [ERROR] Segment {i}: {e}")

        if errors and verbose:
            print(f"  ⚠️  {len(errors)}/{N} segments had errors")

        return all_f0, all_activations

    def get_model_info(self):
        """Return model metadata."""
        info = self.model.get_model_info()
        info['device'] = str(self.device)
        return info


if __name__ == '__main__':
    # Quick test
    print("Testing FCNPredictor...")
    import sys

    for mt in ['fcn_1953', 'fcn_993', 'fcn_929']:
        try:
            predictor = FCNPredictor(model_type=mt, device='cpu')
            info = predictor.get_model_info()
            min_len = info['min_input_length']

            # Test with random normalized audio
            test_audio = np.random.randn(min_len).astype(np.float32)
            f0, act = predictor.predict(test_audio)

            print(f"  {mt}: min_len={min_len}, f0={f0:.2f} Hz, "
                  f"act_shape={act.shape}, act_range=[{act.min():.4f}, {act.max():.4f}]")
        except FileNotFoundError as e:
            print(f"  {mt}: [SKIP] {e}")
