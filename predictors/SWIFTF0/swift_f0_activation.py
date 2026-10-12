"""
SwiftF0Activation — Extended SwiftF0 wrapper that exposes 200-dim activations.

Uses a modified ONNX model (model_with_activation.onnx) that exposes the
internal /Softmax_output_0 tensor as an additional output 'activation_200'.

This class provides the same API as the official SwiftF0 class, but the
returned result object includes an additional `.activation` field containing
the raw 200-dimensional softmax probability distribution for each time frame.
"""

import onnxruntime
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class PitchResultWithActivation:
    """Extended pitch detection result container.

    Includes all standard SwiftF0 fields plus raw 200-dim activations.

    Attributes:
        pitch_hz: Estimated F0 in Hz for each frame, shape (T,)
        confidence: Model confidence score [0-1] for each frame, shape (T,)
        timestamps: Time positions (seconds) for each frame, shape (T,)
        voicing: Boolean voicing decisions for each frame, shape (T,)
        activation: Raw 200-dim softmax probabilities, shape (T, 200)
    """

    pitch_hz: np.ndarray
    confidence: np.ndarray
    timestamps: np.ndarray
    voicing: np.ndarray
    activation: np.ndarray


class SwiftF0Activation:
    """Extended SwiftF0 detector that returns 200-dim softmax activations.

    Uses a modified ONNX model with an additional 'activation_200' output
    that exposes the internal Softmax layer output.

    The model uses STFT-based processing internally:
    - Sample rate: 16kHz
    - STFT frame length: 1024 samples
    - Hop size: 256 samples
    - Padding: 384 samples (symmetric)
    - Center offset: 127.5 samples
    """

    # Audio processing constants (matching official SwiftF0)
    TARGET_SAMPLE_RATE = 16000
    HOP_LENGTH = 256
    FRAME_LENGTH = 1024
    STFT_PADDING = (FRAME_LENGTH - HOP_LENGTH) // 2  # 384
    MIN_AUDIO_LENGTH = 256
    CENTER_OFFSET = (FRAME_LENGTH - 1) / 2 - STFT_PADDING  # 127.5

    # Model frequency limits
    MODEL_FMIN = 46.875
    MODEL_FMAX = 2093.75

    # Defaults
    DEFAULT_CONFIDENCE_THRESHOLD = 0.9
    DEFAULT_FMIN = MODEL_FMIN
    DEFAULT_FMAX = MODEL_FMAX

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize SwiftF0Activation with a modified ONNX model.

        Args:
            confidence_threshold: Confidence threshold (0-1) for voicing.
            fmin: Minimum frequency (Hz) to consider voiced.
            fmax: Maximum frequency (Hz) to consider voiced.
            model_path: Path to modified ONNX model. If None, looks for
                       'model_with_activation.onnx' in the current directory.
        """
        # Validate confidence threshold
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError(
                    f"confidence_threshold ({confidence_threshold}) must be between 0.0 and 1.0"
                )
            self.confidence_threshold = confidence_threshold
        else:
            self.confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD

        # Set frequency limits
        self.fmin = fmin or self.DEFAULT_FMIN
        self.fmax = fmax or self.DEFAULT_FMAX

        # Validate frequency ranges
        if self.fmin < self.MODEL_FMIN:
            raise ValueError(
                f"fmin ({self.fmin} Hz) is below model minimum ({self.MODEL_FMIN} Hz)."
            )
        if self.fmax > self.MODEL_FMAX:
            raise ValueError(
                f"fmax ({self.fmax} Hz) is above model maximum ({self.MODEL_FMAX} Hz)."
            )
        if self.fmin > self.fmax:
            raise ValueError(
                f"fmin ({self.fmin} Hz) cannot be greater than fmax ({self.fmax} Hz)"
            )

        # Locate the modified ONNX model
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'model_with_activation.onnx'
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modified model not found at: {model_path}. "
                "Run modify_onnx.py first to generate it."
            )

        # Initialize ONNX Runtime session (suppress shape mismatch warnings)
        session_options = onnxruntime.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1
        session_options.log_severity_level = 3  # ERROR only (suppress warnings)
        self.pitch_session = onnxruntime.InferenceSession(
            model_path, session_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.pitch_input_name = self.pitch_session.get_inputs()[0].name

        # Verify the model has the activation_200 output
        output_names = [out.name for out in self.pitch_session.get_outputs()]
        if 'activation_200' not in output_names:
            raise ValueError(
                f"Modified model does not have 'activation_200' output. "
                f"Available outputs: {output_names}. "
                "Ensure you used the correct modified model."
            )

    def _run_model(self, audio_16k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the ONNX model, requesting all three outputs.

        Args:
            audio_16k: Mono audio at 16kHz, shape (n_samples,)

        Returns:
            Tuple of (pitch_hz, confidence, activation_200)
        """
        if audio_16k.ndim != 1:
            raise ValueError("Input audio must be 1D (mono)")
        if len(audio_16k) == 0:
            raise ValueError("Input audio cannot be empty")

        # Pad audio to minimum required length if needed
        if len(audio_16k) < self.MIN_AUDIO_LENGTH:
            audio_16k = np.pad(
                audio_16k,
                (0, max(0, self.MIN_AUDIO_LENGTH - len(audio_16k))),
                mode="constant",
            )

        # Prepare input and run model — request all three outputs
        ort_inputs = {self.pitch_input_name: audio_16k[None, :].astype(np.float32)}
        outputs = self.pitch_session.run(
            ['pitch_hz', 'confidence', 'activation_200'],
            ort_inputs
        )

        # outputs are: pitch_hz, confidence, activation_200
        # Each has shape (1, T) or (1, T, 200) — strip batch dim
        pitch_hz = outputs[0][0]       # (T,)
        confidence = outputs[1][0]     # (T,)
        activation = outputs[2][0]     # (T, 200)

        return pitch_hz, confidence, activation

    def _compute_voicing(
        self, pitch_hz: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
        """Compute voicing mask."""
        return (
            (confidence > self.confidence_threshold)
            & (pitch_hz >= self.fmin)
            & (pitch_hz <= self.fmax)
        )

    def _calculate_timestamps(self, n_frames: int) -> np.ndarray:
        """Calculate frame timestamps accounting for STFT center positions."""
        frame_centers = np.arange(n_frames) * self.HOP_LENGTH + self.CENTER_OFFSET
        return frame_centers / self.TARGET_SAMPLE_RATE

    def detect_from_array(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> PitchResultWithActivation:
        """
        Detect pitch and return 200-dim activations from a numpy array.

        Args:
            audio_array: Input audio as 1D or 2D numpy array.
            sample_rate: Sample rate of the input audio.

        Returns:
            PitchResultWithActivation with pitch_hz, confidence, timestamps,
            voicing, and activation (200-dim softmax per frame).
        """
        if audio_array.size == 0:
            raise ValueError("Input audio cannot be empty")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")

        # Convert to mono if needed
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=-1)

        # Resample to 16kHz if needed
        if sample_rate != self.TARGET_SAMPLE_RATE:
            try:
                import librosa
                audio_16k = librosa.resample(
                    audio_array.astype(np.float64),
                    orig_sr=sample_rate,
                    target_sr=self.TARGET_SAMPLE_RATE,
                ).astype(np.float32)
            except ImportError:
                raise ImportError(
                    "librosa required for resampling. Install with: pip install librosa"
                )
        else:
            audio_16k = audio_array.astype(np.float32)

        # Run model to get pitch, confidence, and activations
        pitch_hz, confidence, activation = self._run_model(audio_16k)

        # Compute voicing and timestamps
        voicing = self._compute_voicing(pitch_hz, confidence)
        timestamps = self._calculate_timestamps(len(pitch_hz))

        return PitchResultWithActivation(
            pitch_hz=pitch_hz,
            confidence=confidence,
            timestamps=timestamps,
            voicing=voicing,
            activation=activation,
        )

    def detect_from_file(self, audio_path: str) -> PitchResultWithActivation:
        """
        Detect pitch and return 200-dim activations from an audio file.

        Args:
            audio_path: Path to audio file (any format supported by librosa).

        Returns:
            PitchResultWithActivation with pitch_hz, confidence, timestamps,
            voicing, and activation (200-dim softmax per frame).
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa required for file loading. Install with: pip install librosa"
            )
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return self.detect_from_array(audio, sr)
