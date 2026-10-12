"""
CREPE prediction wrapper for psychophysical experiments.

Encapsulates torchcrepe inference with proper frame aggregation
and model configuration matching official CREPE behavior.

Key design: torchcrepe.predict() does NOT return the 360-dim activation
vector. We access the model's internal logits via torchcrepe.infer(),
apply sigmoid to get per-bin probabilities (the "activation"), and
separately decode pitch and periodicity.

The sigmoid outputs are per-bin independent probabilities. We normalize
them to sum to 1 to form a proper probability distribution for downstream
Gaussian interpolation to 700 bins.
"""

import numpy as np
import torch
import torchcrepe


# CREPE model capacity: 'full' is the default and most accurate
MODEL_CAPACITY = 'full'

# Default step size in milliseconds
DEFAULT_STEP_SIZE = 10

# CREPE constants
CREPE_SR = 16000
WINDOW_SIZE = 1024
PITCH_BINS = 360


def predict_from_audio(audio, sr, step_size=DEFAULT_STEP_SIZE, device='cuda'):
    """
    Run CREPE inference on single audio trial and aggregate across frames.

    Uses torchcrepe's internal preprocessing and model, but captures
    the 360-dim activation (sigmoid probabilities) before decoding.

    Args:
        audio: numpy.ndarray of shape (n_samples,) — mono audio at `sr` Hz.
        sr: Sampling rate of the input audio in Hz.
        step_size: Frame shift in milliseconds (default: 10 ms).
        device: Device for inference ('cuda' or 'cpu').

    Returns:
        tuple (pitch_median, activation_mean, periodicity_mean):
          - pitch_median: float, median pitch across frames in Hz.
          - activation_mean: numpy.ndarray of shape (360,), element-wise mean
                             of the 360-dim normalized activation vectors.
          - periodicity_mean: float, mean periodicity (confidence) across frames.

        Returns (None, None, None) on failure.
    """
    try:
        # Convert hop length from milliseconds to samples
        hop_length = int(sr * step_size / 1000)

        # Convert audio to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio.copy()).float()
        else:
            audio_tensor = audio.float()

        # Add batch dimension: (1, n_samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        with torch.no_grad():

            # --- Step 1: Preprocess (replicate torchcrepe.preprocess) ---
            # Resample if needed
            if sr != CREPE_SR:
                audio_tensor = torchcrepe.resample(audio_tensor, sr)
                hop_length = int(hop_length * CREPE_SR / sr)
                sr = CREPE_SR

            # Zero-pad symmetrically (center the signal)
            total_frames = 1 + int(audio_tensor.size(1) // hop_length)
            audio_tensor = torch.nn.functional.pad(
                audio_tensor,
                (WINDOW_SIZE // 2, WINDOW_SIZE // 2)
            )

            # Extract frames via unfold
            frames = torch.nn.functional.unfold(
                audio_tensor[:, None, None, :],
                kernel_size=(1, WINDOW_SIZE),
                stride=(1, hop_length)
            )
            # frames shape: (1, 1024, n_frames)
            frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE)
            # frames shape: (n_frames, 1024)

            # Mean-center and scale (as done in torchcrepe.preprocess)
            frames = frames.to(device)
            frames -= frames.mean(dim=1, keepdim=True)
            frames /= torch.max(
                torch.tensor(1e-10, device=frames.device),
                frames.std(dim=1, keepdim=True)
            )

            # --- Step 2: Inference ---
            # Get logits from the model
            logits = torchcrepe.infer(frames, model=MODEL_CAPACITY,
                                      device=device, embed=False)
            # logits shape: (n_frames, 360)

            # --- Step 3: Activation (sigmoid) ---
            # Apply sigmoid to get per-bin probabilities
            activation = torch.sigmoid(logits)
            # activation shape: (n_frames, 360)

            # --- Step 4: Decode pitch and periodicity ---
            # Reshape logits for postprocessing: (1, 360, n_frames)
            logits_3d = logits.T.unsqueeze(0)  # (1, 360, n_frames)

            # Decode pitch using weighted_argmax
            pitch = torchcrepe.decode.weighted_argmax(logits_3d)[1]
            # pitch shape: (1, n_frames)

            # Compute periodicity from decoded bins and logits
            # (inline torchcrepe.core.periodicity, not publicly exported)
            bins = logits_3d.argmax(dim=1)  # (1, n_frames)
            probs_stacked = logits_3d.transpose(1, 2).reshape(-1, PITCH_BINS)
            bins_stacked = bins.reshape(-1, 1).to(torch.int64)
            periodicity = probs_stacked.gather(1, bins_stacked)
            periodicity = periodicity.reshape(bins.size(0), bins.size(1))
            # periodicity shape: (1, n_frames)

            # Remove batch dimension
            pitch = pitch.squeeze(0)          # (n_frames,)
            periodicity = periodicity.squeeze(0)
            activation = activation            # (n_frames, 360)

            # Move to CPU
            pitch_np = pitch.cpu().numpy()
            periodicity_np = periodicity.cpu().numpy()
            activation_np = activation.cpu().numpy()

            # --- Step 5: Aggregate across frames ---
            # Pitch (Hz): median
            pitch_median = float(np.median(pitch_np))

            # Activation (360-dim): element-wise mean
            activation_mean = activation_np.mean(axis=0)  # (360,)

            # Normalize to sum to 1 (treat as probability distribution)
            act_sum = activation_mean.sum()
            if act_sum > 0:
                activation_mean = activation_mean / act_sum

            # Periodicity: mean
            periodicity_mean = float(periodicity_np.mean())

            return pitch_median, activation_mean, periodicity_mean

    except Exception as e:
        print(f"  [ERROR] CREPE prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def predict_batch(audio_batch, sr, step_size=DEFAULT_STEP_SIZE, device='cuda'):
    """
    Run CREPE inference on a batch of audio trials and aggregate per trial.

    Each trial in the batch is processed independently (since torchcrepe's
    pipeline works best with single-trial audio).

    Args:
        audio_batch: numpy.ndarray of shape (B, n_samples) — B trials.
        sr: Sampling rate of the input audio in Hz.
        step_size: Frame shift in milliseconds (default: 10 ms).
        device: Device for inference ('cuda' or 'cpu').

    Returns:
        tuple (pitches, activations, periodicities):
          - pitches: list of float (median pitch per trial in Hz).
          - activations: list of (360,) ndarray (normalized activation per trial).
          - periodicities: list of float (mean periodicity per trial).
    """
    B = audio_batch.shape[0]
    hop_length = int(sr * step_size / 1000)

    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio_batch.copy()).float()

    # Resample if needed
    if sr != CREPE_SR:
        audio_tensor = torchcrepe.resample(audio_tensor, sr)
        hop_length = int(hop_length * CREPE_SR / sr)

    with torch.no_grad():
        # Pad symmetrically
        audio_tensor = torch.nn.functional.pad(
            audio_tensor,
            (WINDOW_SIZE // 2, WINDOW_SIZE // 2)
        )

        # Extract frames for entire batch
        frames = torch.nn.functional.unfold(
            audio_tensor[:, None, None, :],
            kernel_size=(1, WINDOW_SIZE),
            stride=(1, hop_length)
        )
        # frames shape: (B, 1024, n_frames)
        frames = frames.transpose(1, 2)  # (B, n_frames, 1024)

        # Reshape to (B * n_frames, 1024) for inference
        B_size, n_frames_total, _ = frames.shape
        frames = frames.reshape(-1, WINDOW_SIZE)

        # Mean-center and scale
        frames = frames.to(device)
        frames -= frames.mean(dim=1, keepdim=True)
        frames /= torch.max(
            torch.tensor(1e-10, device=frames.device),
            frames.std(dim=1, keepdim=True)
        )

        # Inference on all frames
        logits = torchcrepe.infer(frames, model=MODEL_CAPACITY,
                                  device=device, embed=False)
        # logits shape: (B * n_frames, 360)

        # Activation (sigmoid)
        activation = torch.sigmoid(logits)
        # activation shape: (B * n_frames, 360)

        # Reshape back to (B, n_frames, 360) for per-trial aggregation
        activation_3d = activation.reshape(B_size, n_frames_total, PITCH_BINS)

        # For pitch: reshape logits to (B, 360, n_frames) for decoder
        logits_3d = logits.reshape(B_size, n_frames_total, PITCH_BINS)
        logits_3d = logits_3d.transpose(1, 2)  # (B, 360, n_frames)

        # Decode pitch and periodicity per trial
        pitches = []
        activations = []
        periodicities = []

        for i in range(B_size):
            # Pitch decoding
            trial_logits = logits_3d[i:i+1]  # (1, 360, n_frames)
            pitch_i = torchcrepe.decode.weighted_argmax(trial_logits)[1]
            pitch_i = pitch_i.squeeze(0).cpu().numpy()

            # Periodicity: value at argmax bin (inline implementation
            # of torchcrepe.core.periodicity, which is not publicly exported)
            bins_i = trial_logits.argmax(dim=1)  # (1, n_frames)
            probs_stacked = trial_logits.transpose(1, 2).reshape(-1, PITCH_BINS)
            bins_stacked = bins_i.reshape(-1, 1).to(torch.int64)
            periodicity_i = probs_stacked.gather(1, bins_stacked)
            periodicity_i = periodicity_i.reshape(bins_i.size(0), bins_i.size(1))
            periodicity_i = periodicity_i.squeeze(0).cpu().numpy()

            # Activation aggregation: mean across frames, then normalize
            act_i = activation_3d[i].cpu().numpy()  # (n_frames, 360)
            act_mean = act_i.mean(axis=0)
            act_sum = act_mean.sum()
            if act_sum > 0:
                act_mean = act_mean / act_sum

            pitches.append(float(np.median(pitch_i)))
            activations.append(act_mean)
            periodicities.append(float(periodicity_i.mean()))

        return pitches, activations, periodicities
