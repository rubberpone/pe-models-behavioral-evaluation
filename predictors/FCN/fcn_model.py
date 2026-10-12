"""
FCN (Fully-Convolutional Network) model definition for F0 estimation.

PyTorch Conv1d implementation equivalent to the official Keras Conv2D
architecture (kernel shape [width, 1] → Conv1d kernel width).

Reference: Ardaillon & Roebel (2019), Interspeech 2019.
Official code: https://github.com/ardaillon/FCN-f0
"""

import torch
import torch.nn as nn


class FCNPitchEstimator(nn.Module):
    """
    Fully-Convolutional Network for pitch estimation.

    Three variants are supported:
      - fcn_1953: Full architecture (6 conv + 1 classifier)
      - fcn_993:  Same filters as 1953, all kernel widths = 32
      - fcn_929:  5 conv + 1 classifier, MaxPool only on first 2 layers

    Input:  raw waveform at 8 kHz
    Output: per-bin sigmoid activations (486 bins, 30–1000 Hz, 12.5-cent spacing)
    """

    MODEL_CONFIGS = {
        'fcn_1953': {
            'filters': [256, 32, 32, 128, 256, 512],
            'widths': [32, 64, 64, 64, 64, 64],
            'maxpool_layers': 3,  # MaxPool after first 3 conv layers
            'min_input_length': 1953,
        },
        'fcn_993': {
            'filters': [256, 32, 32, 128, 256, 512],
            'widths': [32, 32, 32, 32, 32, 32],
            'maxpool_layers': 3,
            'min_input_length': 993,
        },
        'fcn_929': {
            'filters': [256, 32, 128, 256, 512],
            'widths': [32, 64, 64, 64, 64],
            'maxpool_layers': 2,  # MaxPool only after first 2 conv layers
            'min_input_length': 929,
        },
    }

    def __init__(self, model_type='fcn_993'):
        """
        Args:
            model_type: one of 'fcn_1953', 'fcn_993', 'fcn_929'
        """
        super(FCNPitchEstimator, self).__init__()

        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Choose from {list(self.MODEL_CONFIGS.keys())}"
            )

        cfg = self.MODEL_CONFIGS[model_type]
        self.model_type = model_type
        self.min_input_length = cfg['min_input_length']
        self.num_layers = len(cfg['filters'])

        # Build convolutional layers
        layers = []
        in_channels = 1  # mono audio

        for l_idx, (num_filters, kernel_width) in enumerate(
            zip(cfg['filters'], cfg['widths'])
        ):
            layer_num = l_idx + 1

            # Conv1d with valid padding (no zero-padding), stride=1
            # Equivalent to Keras Conv2D(filters, (width, 1), padding='valid')
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_width,
                stride=1,
                padding=0,  # valid convolution
            )
            layers.append((f'conv{layer_num}', conv))

            # ReLU activation (applied in Conv2D params in Keras, explicit in PyTorch)
            layers.append((f'conv{layer_num}_relu', nn.ReLU()))

            # MaxPool (2x, stride 2) after first N conv layers
            if l_idx < cfg['maxpool_layers']:
                pool = nn.MaxPool1d(kernel_size=2, stride=2)
                layers.append((f'conv{layer_num}_maxpool', pool))

            # Batch Normalization
            bn = nn.BatchNorm1d(num_features=num_filters)
            layers.append((f'conv{layer_num}_bn', bn))

            in_channels = num_filters

        # Classifier: Conv1d(512 → 486, kernel_size=4, padding=0) + Sigmoid
        self.classifier = nn.Conv1d(
            in_channels=in_channels,
            out_channels=486,
            kernel_size=4,
            stride=1,
            padding=0,  # valid convolution
        )
        self.sigmoid = nn.Sigmoid()

        # Register all convolutional + BN layers as a ModuleDict
        self.conv_stack = nn.ModuleDict(layers)

        # Store config for reference
        self._layer_keys = [name for name, _ in layers]

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input tensor of shape (batch, 1, samples) at 8 kHz

        Returns:
            tensor of shape (batch, 486) — per-bin sigmoid activations
        """
        for name in self._layer_keys:
            layer = self.conv_stack[name]
            x = layer(x)

        # Classifier + Sigmoid
        x = self.classifier(x)  # (batch, 486, time)
        x = self.sigmoid(x)     # (batch, 486, time)

        # Squeeze temporal dimension (should be 1 for min-length inputs)
        x = x.squeeze(-1)  # (batch, 486)

        # If batch_size=1 and 486-dim, ensure it's (batch, 486)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return x

    def get_model_info(self):
        """Return model configuration details."""
        cfg = self.MODEL_CONFIGS[self.model_type]
        return {
            'model_type': self.model_type,
            'filters': cfg['filters'],
            'widths': cfg['widths'],
            'maxpool_layers': cfg['maxpool_layers'],
            'min_input_length': cfg['min_input_length'],
            'num_conv_layers': len(cfg['filters']),
            'output_bins': 486,
        }


def _compute_output_length(input_length, kernels, maxpool_layers):
    """Helper to compute output length given valid convolutions and maxpool."""
    length = input_length
    for i, k in enumerate(kernels):
        length = length - k + 1  # valid conv
        if i < maxpool_layers:
            length = length // 2  # MaxPool(2x, stride 2)
    length = length - 4 + 1  # classifier kernel_size=4
    return length


if __name__ == '__main__':
    # Quick test
    for mt in ['fcn_1953', 'fcn_993', 'fcn_929']:
        model = FCNPitchEstimator(model_type=mt)
        min_len = model.min_input_length
        x = torch.randn(1, 1, min_len)
        with torch.no_grad():
            out = model(x)
        out_len = _compute_output_length(
            min_len,
            model.MODEL_CONFIGS[mt]['widths'],
            model.MODEL_CONFIGS[mt]['maxpool_layers'],
        )
        print(f"{mt}: input={min_len}, output={out.shape}, "
              f"expected_time_dim={out_len}")
        info = model.get_model_info()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params={n_params:,}, filters={info['filters']}, "
              f"widths={info['widths']}")
