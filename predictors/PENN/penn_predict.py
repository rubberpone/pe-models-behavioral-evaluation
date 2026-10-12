"""PENN model prediction wrapper.

Provides a unified interface for F0 estimation using the PENN framework.
Supports FCNF0++ (default, pre-trained), CREPE++, and DeepF0++.

Key design:
  - Uses penn.core.infer() to get raw posteriorgram logits
  - Computes pitch (Hz) from argmax of frame-averaged posteriorgram
  - Computes periodicity from entropy of the posteriorgram (built-in method)
  - All ++ models use Strategy 1 (direct inference, 128 ms window)
  - Aggregation: median for scalar estimates, mean for posteriorgram

This eliminates the redundant penn.from_audio() call (which would run
inference a second time) by computing pitch and periodicity directly
from the posteriorgram.
"""

import numpy as np
import torch
import penn


# ---------------------------------------------------------------------------
# Model configuration lookup
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    'fcn': {
        'name': 'FCNF0++',
        'checkpoint': None,   # Override with local path when HuggingFace unavailable
        'config': None,
        'decoder': 'viterbi',
        'sample_rate': 8000,
        'n_bins': 1440,
        'cents_per_bin': 5.0,
        'fmin': 31.0,
        'output_dir': 'PENN_FCN_psy_predictions',
        'json_prefix': 'predictions_penn_fcn',
    },
    'crepe++': {
        'name': 'CREPE++',
        'checkpoint': 'runs/crepe++/checkpoints/best.pt',
        'config': 'config/crepe++.py',
        'decoder': 'argmax',
        'sample_rate': 8000,
        'n_bins': 1440,
        'cents_per_bin': 5.0,
        'fmin': 31.0,
        'output_dir': 'PENN_CREPE_PP_psy_predictions',
        'json_prefix': 'predictions_penn_crepe++',
    },
    'deepf0++': {
        'name': 'DeepF0++',
        'checkpoint': 'runs/deepf0++/checkpoints/best.pt',
        'config': 'config/deepf0++.py',
        'decoder': 'argmax',
        'sample_rate': 8000,
        'n_bins': 1440,
        'cents_per_bin': 5.0,
        'fmin': 31.0,
        'output_dir': 'PENN_DEEPF0_PP_psy_predictions',
        'json_prefix': 'predictions_penn_deepf0++',
    },
}


def get_model_config(model_name: str) -> dict:
    """Look up configuration for a PENN model variant.

    Args:
        model_name: One of 'fcn', 'crepe++', 'deepf0++'.

    Returns:
        dict with model configuration.

    Raises:
        ValueError: if model_name is unknown.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )
    return dict(MODEL_CONFIGS[model_name])


def _entropy_based_periodicity(probs, dim=-1):
    """Compute periodicity from posteriorgram entropy.

    Low entropy → high periodicity (clear pitch).
    High entropy → low periodicity (noisy/unvoiced).

    Args:
        probs: Probability tensor of shape (..., n_bins) summing to 1 along dim.
        dim: The bin dimension.

    Returns:
        Periodicity tensor in [0, 1].
    """
    n_bins = probs.shape[dim]
    # Clip to avoid log(0)
    eps = 1e-8
    probs = probs.clamp(min=eps)
    entropy = -(probs * probs.log()).sum(dim=dim)  # shape: (...)
    max_entropy = np.log(n_bins)
    periodicity = 1.0 - entropy / max_entropy
    return periodicity


class PENNPredictor:
    """Wrapper around PENN for single-stimulus F0 prediction.

    Runs a single inference pass (penn.core.infer) and derives pitch,
    periodicity, and posteriorgram from the output logits.
    """

    def __init__(self, model_name='fcn', gpu=0, checkpoint_override=None):
        """Initialise the predictor.

        Args:
            model_name: Model variant ('fcn', 'crepe++', 'deepf0++').
            gpu: GPU index (None for CPU, 0 for first GPU).
            checkpoint_override: Path to a local checkpoint file.
        """
        self.model_name = model_name
        self.gpu = gpu
        self.device = 'cpu' if gpu is None else f'cuda:{gpu}'

        cfg = get_model_config(model_name)
        self.model_config = cfg

        # Resolve checkpoint
        self.checkpoint = checkpoint_override or cfg['checkpoint']
        self.fmin = cfg['fmin']
        self.cents_per_bin = cfg['cents_per_bin']
        self.n_bins = cfg['n_bins']

    def predict(self, audio, sample_rate=32000):
        """Run PENN on a single audio stimulus.

        Args:
            audio: 1D numpy array of float32 samples.
            sample_rate: Original sample rate in Hz (default 32000).

        Returns:
            tuple (f0_hz, periodicity, posteriorgram):
                f0_hz: float — pitch estimate (Hz), from argmax of
                       frame-averaged posteriorgram.
                periodicity: float — median entropy-based periodicity (0–1).
                posteriorgram: np.ndarray of shape (1440,), Softmax-normalised
                               mean posteriorgram across frames.
        """
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio.copy()).float().unsqueeze(0)
        else:
            audio_tensor = audio.float().unsqueeze(0)

        # ---- Single inference pass via low-level API ----
        frames_list = list(penn.core.preprocess(
            audio_tensor, sample_rate, center='half-window'
        ))

        all_logits = []
        for frames in frames_list:
            frames = frames.to(self.device)
            logits = penn.core.infer(frames, checkpoint=self.checkpoint)
            # logits shape: (N_preprocess_frames, 1440, 1)
            logits = logits.squeeze(-1)  # → (N_frames, 1440)
            all_logits.append(logits.cpu())

        # Concatenate across preprocess batches → (total_frames, 1440)
        combined_logits = torch.cat(all_logits, dim=0)  # (total_frames, 1440)

        # Softmax across bins (CCE-trained ++ models)
        probs = torch.nn.functional.softmax(combined_logits.float(), dim=1)

        # ---- Frame-averaged posteriorgram ----
        posteriorgram = probs.mean(dim=0).numpy()  # shape: (1440,)

        # ---- Pitch: argmax of frame-averaged posteriorgram → Hz ----
        best_bin = int(posteriorgram.argmax())
        # Convert bin index to frequency: fmin * 2^(bin * cents_per_bin / 1200)
        f0_hz = float(self.fmin * (2 ** (best_bin * self.cents_per_bin / 1200)))

        # ---- Periodicity: entropy-based (built-in method) ----
        per_frame_periodicity = _entropy_based_periodicity(probs, dim=1)
        periodicity_val = float(per_frame_periodicity.float().median().item())

        return f0_hz, periodicity_val, posteriorgram

    def predict_batch(self, audio_batch, sample_rate=32000):
        """Run PENN on a batch of audio stimuli.

        Args:
            audio_batch: numpy array of shape (B, samples).
            sample_rate: Original sample rate in Hz.

        Returns:
            list of (f0_hz, periodicity, posteriorgram) tuples.
        """
        results = []
        for i in range(audio_batch.shape[0]):
            results.append(self.predict(audio_batch[i], sample_rate))
        return results
