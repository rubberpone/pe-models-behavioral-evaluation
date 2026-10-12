"""
JDC (Joint Detection and Classification) PyTorch Reimplementation.

Architecture based on:
  Kum, S., & Nam, J. (2019). Joint Detection and Classification of Singing Voice
  Melody Using Convolutional Recurrent Neural Networks. Applied Sciences, 9(7), 1324.

This is a faithful PyTorch port of the official Keras implementation from:
  https://github.com/keums/melodyExtraction_JDC

Key architecture features:
  - Dual-output: main pitch (722 classes) + auxiliary voice detection (2 classes)
  - ConvBlock → 3× ResBlocks with skip connections → Bi-LSTM → FC
  - Joint branch fuses multi-scale features from all blocks for voice detection
  - Softmax is built into the model (do NOT re-apply)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Initial convolution block: 2 × Conv3×3 → BN → LReLU → Conv3×3
    (No BN/activation after second conv, matching Keras implementation)
    """

    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.conv2(x)  # No BN/LReLU after second conv
        return x


class ResBlock(nn.Module):
    """
    Residual block:
      BN → LReLU → MaxPool(1,4) → 1×1 Conv skip → 3×3 Conv → BN → LReLU → 3×3 Conv → add(skip, output)
    """

    def __init__(self, in_channels, out_channels, block_id):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pool = nn.MaxPool2d(kernel_size=(1, 4))
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.block_id = block_id

    def forward(self, x):
        # BN → LReLU → MaxPool
        out = self.bn1(x)
        out = F.leaky_relu(out, negative_slope=0.01)
        out = self.pool(out)

        # Skip connection (1×1 Conv)
        skip = self.conv_skip(out)

        # Main path: Conv3×3 → BN → LReLU → Conv3×3
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.leaky_relu(out, negative_slope=0.01)
        out = self.conv2(out)

        # Residual addition
        out = out + skip
        return out


class JDC(nn.Module):
    """
    JDC (Joint Detection and Classification) network.

    Input:  (B, 1, 31, 513) — dB-scale, Z-score normalized spectrogram chunks
    Output: (output, output_V)
      - output:   (B, 31, 722) — pitch class probabilities (softmax built-in)
      - output_V: (B, 31, 2)   — voice/non-voice detection probabilities (softmax built-in)
    """

    def __init__(self, num_pitch_classes=722):
        super().__init__()

        # === ConvBlock ===
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False)

        # === ResBlock 2: 64→128 ===
        self.bn_r2_1 = nn.BatchNorm2d(64)
        self.pool_r2 = nn.MaxPool2d(kernel_size=(1, 4))
        self.conv2_1x1 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False)
        self.bn_r2_2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)

        # === ResBlock 3: 128→192 ===
        self.bn_r3_1 = nn.BatchNorm2d(128)
        self.pool_r3 = nn.MaxPool2d(kernel_size=(1, 4))
        self.conv3_1x1 = nn.Conv2d(128, 192, kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(128, 192, kernel_size=3, padding='same', bias=False)
        self.bn_r3_2 = nn.BatchNorm2d(192)
        self.conv3_2 = nn.Conv2d(192, 192, kernel_size=3, padding='same', bias=False)

        # === ResBlock 4: 192→256 ===
        self.bn_r4_1 = nn.BatchNorm2d(192)
        self.pool_r4 = nn.MaxPool2d(kernel_size=(1, 4))
        self.conv4_1x1 = nn.Conv2d(192, 256, kernel_size=1, bias=False)
        self.conv4_1 = nn.Conv2d(192, 256, kernel_size=3, padding='same', bias=False)
        self.bn_r4_2 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False)

        # === Final Pool ===
        self.bn_final = nn.BatchNorm2d(256)
        self.pool_final = nn.MaxPool2d(kernel_size=(1, 4))
        self.dropout_final = nn.Dropout(0.5)

        # === Main Branch Bi-LSTM ===
        # Input: (B, 31, 512) — 31 time steps × (2 * 256 channels after final pool)
        self.lstm1 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # LSTM → FC 722 + softmax
        self.fc_main = nn.Linear(512, num_pitch_classes)  # 512 = 2*256 (Bi-LSTM output)

        # === Joint Branch ===
        # Multi-scale pooling: downsample block_1..block_4 to uniform 2 freq bins
        self.joint_pool_1 = nn.MaxPool2d(kernel_size=(1, 256))   # 513 → 2
        self.joint_pool_2 = nn.MaxPool2d(kernel_size=(1, 64))    # 128 → 2
        self.joint_pool_3 = nn.MaxPool2d(kernel_size=(1, 16))    # 32  → 2
        # block_4 is already pool'd to ~2 freq bins (no extra pool)

        # 1×1 Conv: 640 → 256
        self.conv_joint = nn.Conv2d(640, 256, kernel_size=1, bias=False)
        self.bn_joint = nn.BatchNorm2d(256)
        self.dropout_joint = nn.Dropout(0.5)

        # Joint Bi-LSTM: (B, 31, 512) → (B, 31, 64)
        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # Joint FC: 2 classes
        self.fc_joint = nn.Linear(64, 2)  # 64 = 2*32 (Bi-LSTM output)

    def forward(self, x):
        """
        Args:
            x: (B, 1, 31, 513) — dB-scale, Z-score normalized spectrograms

        Returns:
            output:   (B, 31, 722) — pitch probabilities (softmax built-in)
            output_V: (B, 31, 2)   — voice detection probabilities
        """
        B = x.size(0)

        # ================================================================
        # ConvBlock
        # ================================================================
        block_1 = self.conv1_1(x)                         # (B, 64, 31, 513)
        block_1 = self.bn1(block_1)
        block_1 = F.leaky_relu(block_1, negative_slope=0.01)
        block_1 = self.conv1_2(block_1)                    # (B, 64, 31, 513)

        # ================================================================
        # ResBlock 2: 64 → 128
        # ================================================================
        r2 = self.bn_r2_1(block_1)
        r2 = F.leaky_relu(r2, negative_slope=0.01)
        r2 = self.pool_r2(r2)                              # (B, 64, 31, 128)
        skip2 = self.conv2_1x1(r2)                         # (B, 128, 31, 128)
        r2 = self.conv2_1(r2)                              # (B, 128, 31, 128)
        r2 = self.bn_r2_2(r2)
        r2 = F.leaky_relu(r2, negative_slope=0.01)
        r2 = self.conv2_2(r2)                              # (B, 128, 31, 128)
        block_2 = r2 + skip2                               # (B, 128, 31, 128)

        # ================================================================
        # ResBlock 3: 128 → 192
        # ================================================================
        r3 = self.bn_r3_1(block_2)
        r3 = F.leaky_relu(r3, negative_slope=0.01)
        r3 = self.pool_r3(r3)                              # (B, 128, 31, 32)
        skip3 = self.conv3_1x1(r3)                         # (B, 192, 31, 32)
        r3 = self.conv3_1(r3)                              # (B, 192, 31, 32)
        r3 = self.bn_r3_2(r3)
        r3 = F.leaky_relu(r3, negative_slope=0.01)
        r3 = self.conv3_2(r3)                              # (B, 192, 31, 32)
        block_3 = r3 + skip3                               # (B, 192, 31, 32)

        # ================================================================
        # ResBlock 4: 192 → 256
        # ================================================================
        r4 = self.bn_r4_1(block_3)
        r4 = F.leaky_relu(r4, negative_slope=0.01)
        r4 = self.pool_r4(r4)                              # (B, 192, 31, 8)
        skip4 = self.conv4_1x1(r4)                         # (B, 256, 31, 8)
        r4 = self.conv4_1(r4)                              # (B, 256, 31, 8)
        r4 = self.bn_r4_2(r4)
        r4 = F.leaky_relu(r4, negative_slope=0.01)
        r4 = self.conv4_2(r4)                              # (B, 256, 31, 8)
        block_4 = r4 + skip4                               # (B, 256, 31, 8)

        # ================================================================
        # Final Pool (after ResBlock 4)
        # ================================================================
        out_pooled = self.bn_final(block_4)
        out_pooled = F.leaky_relu(out_pooled, negative_slope=0.01)
        out_pooled = self.pool_final(out_pooled)            # (B, 256, 31, 2)
        out_pooled = self.dropout_final(out_pooled)

        # ================================================================
        # Main Branch: Reshape → Bi-LSTM → FC → Softmax
        # ================================================================
        # (B, 256, 31, 2) → (B, 31, 512)
        main_in = out_pooled.permute(0, 2, 1, 3).contiguous()  # (B, 31, 256, 2)
        main_in = main_in.view(B, 31, -1)                       # (B, 31, 512)

        main_out, _ = self.lstm1(main_in)                       # (B, 31, 512)
        main_out = self.fc_main(main_out)                       # (B, 31, 722)
        output = F.softmax(main_out, dim=-1)                    # (B, 31, 722)

        # ================================================================
        # Joint Branch: Multi-scale features → Voice Detection
        # ================================================================

        # Downsample each block to uniform 2 freq bins
        # block_1: (B, 64, 31, 513) → MaxPool(1, 256) → (B, 64, 31, 2)
        j1 = self.joint_pool_1(block_1)
        # block_2: (B, 128, 31, 128) → MaxPool(1, 64) → (B, 128, 31, 2)
        j2 = self.joint_pool_2(block_2)
        # block_3: (B, 192, 31, 32) → MaxPool(1, 16) → (B, 192, 31, 2)
        j3 = self.joint_pool_3(block_3)
        # block_4 (out_pooled): (B, 256, 31, 2) — already has 2 freq bins
        j4 = out_pooled

        # Concatenate along channel dim: 64 + 128 + 192 + 256 = 640
        joint = torch.cat([j1, j2, j3, j4], dim=1)          # (B, 640, 31, 2)

        # 1×1 Conv: 640 → 256
        joint = self.conv_joint(joint)                       # (B, 256, 31, 2)
        joint = self.bn_joint(joint)
        joint = F.leaky_relu(joint, negative_slope=0.01)
        joint = self.dropout_joint(joint)

        # Reshape for Bi-LSTM: (B, 256, 31, 2) → (B, 31, 512)
        joint_in = joint.permute(0, 2, 1, 3).contiguous()    # (B, 31, 256, 2)
        joint_in = joint_in.view(B, 31, -1)                   # (B, 31, 512)

        joint_out, _ = self.lstm2(joint_in)                   # (B, 31, 64)
        joint_out = self.fc_joint(joint_out)                  # (B, 31, 2)
        output_V_raw = F.softmax(joint_out, dim=-1)           # (B, 31, 2)

        # ================================================================
        # Voice Fusion: Main non-voice prob → Joint voice detection
        # ================================================================
        # output_NS: non-voice probability from main branch (index 0)
        output_NS = output[:, :, 0:1]                         # (B, 31, 1)
        output_S = 1.0 - output_NS                            # (B, 31, 1)
        output_VV = torch.cat([output_NS, output_S], dim=-1)  # (B, 31, 2)

        # Add and re-softmax
        output_V = output_V_raw + output_VV
        output_V = F.softmax(output_V, dim=-1)                # (B, 31, 2)

        return output, output_V

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device=None):
        """
        Load JDC model from a PyTorch checkpoint (state_dict).

        Args:
            checkpoint_path: Path to .pt weights file
            device: torch device

        Returns:
            JDC model instance with loaded weights
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cls()
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
