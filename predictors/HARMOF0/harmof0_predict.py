"""
harmof0_predict.py — Batched HarmoF0 inference wrapper.

Provides a batched interface for running HarmoF0 inference on many
audio frames simultaneously, avoiding per-frame overhead.
"""

import numpy as np
import torch
import torch.nn.functional as F
import harmof0


class HarmoF0BatchPredictor:
    """
    Batched HarmoF0 predictor that processes multiple audio frames
    in a single forward pass for maximum throughput.

    The HarmoF0 model accepts [1, total_frames, 1024] input and produces
    [1, total_frames, 352] output. We batch all frames together and
    split the results afterward.
    """

    def __init__(self, device='cuda'):
        """
        Initialize the batched HarmoF0 predictor.

        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device

        # Load HarmoF0 model directly (avoid PitchTracker overhead)
        self.net = harmof0.HarmoF0()

        # Load checkpoint
        import os
        package_dir = os.path.dirname(os.path.realpath(harmof0.__file__))
        checkpoint_path = os.path.join(package_dir, 'checkpoints', 'mdb-stem-synth.pth')
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.net = self.net.to(device)
        self.net.eval()

    def predict_batch(self, frames, batch_size=5000):
        """
        Run HarmoF0 inference on a batch of audio frames.

        Args:
            frames: numpy array of shape (total_frames, 1024)
            batch_size: max frames per GPU batch

        Returns:
            activation_map: numpy array of shape (total_frames, 352), sigmoid output
        """
        total_frames = frames.shape[0]
        all_outputs = []

        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            batch_frames = frames[start:end]

            # Convert to tensor: [1, batch_frames, 1024]
            batch_tensor = torch.from_numpy(batch_frames).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                # HarmoF0.forward returns (activation, specgram)
                # activation shape: [1, batch_frames, 352]
                activation, _ = self.net(batch_tensor)

            # Move to CPU
            all_outputs.append(activation.squeeze(0).cpu().numpy())

        # Concatenate all batches
        activation_map = np.concatenate(all_outputs, axis=0)
        return activation_map  # shape: (total_frames, 352)

    def predict_multi_frame_batch(self, all_frames, n_frames_per_sample=5, batch_size=5000):
        """
        Run batched multi-frame inference and aggregate.

        Processes all frames from all samples in a single batched call,
        then aggregates per sample.

        Args:
            all_frames: numpy array of shape (N * n_frames_per_sample, 1024)
            n_frames_per_sample: number of frames per audio sample (default 5)
            batch_size: max frames per GPU batch

        Returns:
            aggregated_freqs: numpy array of shape (N,) — median Hz per sample
            aggregated_activations: numpy array of shape (N, 352) — mean softmax activation
        """
        # Step 1: Run inference on all frames
        activation_map = self.predict_batch(all_frames, batch_size=batch_size)
        # shape: (N * 5, 352)

        # Step 2: Convert sigmoid -> softmax (per-frame probability distribution)
        activation_tensor = torch.from_numpy(activation_map)
        activation_softmax = F.softmax(activation_tensor, dim=-1).numpy()
        # shape: (N * 5, 352)

        # Step 3: Reshape to (N, 5, 352)
        n_samples = activation_softmax.shape[0] // n_frames_per_sample
        activation_reshaped = activation_softmax.reshape(n_samples, n_frames_per_sample, 352)

        # Step 4: Aggregate across frames
        # Mean activation
        aggregated_activations = activation_reshaped.mean(axis=1)  # (N, 352)
        # Normalize
        sums = aggregated_activations.sum(axis=1, keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        aggregated_activations /= sums

        # Median frequency (argmax per frame, decode, then median)
        # Frequencies from activation_map (sigmoid) via argmax for each frame
        activation_reshaped_sigmoid = activation_map.reshape(n_samples, n_frames_per_sample, 352)
        best_bins = np.argmax(activation_reshaped_sigmoid, axis=2)  # (N, 5)
        freqs_per_frame = 27.5 * (2.0 ** (best_bins.astype(np.float64) / 48.0))
        aggregated_freqs = np.median(freqs_per_frame, axis=1)  # (N,)

        return aggregated_freqs, aggregated_activations
