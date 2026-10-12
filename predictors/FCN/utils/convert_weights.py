"""
Weight conversion script: Keras .h5 (Conv2D) → PyTorch .pt (Conv1d).

Converts the official Keras/TensorFlow FCN weights to PyTorch format.
Handles the necessary reshaping from Conv2D (kernel shape [width, 1])
to Conv1d.

Usage:
    python utils/convert_weights.py

Output:
    pretrained_models/fcn_1953.pt
    pretrained_models/fcn_993.pt
    pretrained_models/fcn_929.pt
"""

import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcn_model import FCNPitchEstimator


def load_keras_weights(h5_path):
    """
    Load Keras weights from .h5 file.

    Returns a dict mapping layer_name → list of weight arrays.

    Keras weight format for Conv2D: [kernel, bias]
      - kernel shape: (width, 1, in_channels, out_channels)
    Keras weight format for BatchNormalization: [gamma, beta, moving_mean, moving_var]
    """
    import h5py

    weights = {}
    with h5py.File(h5_path, 'r') as f:
        # Navigate to the weights group
        def collect_weights(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Get layer name from path
                # Paths look like: /layer_name/layer_name/weight_name:0
                parts = name.split('/')
                if len(parts) >= 3:
                    layer_name = parts[1]  # e.g., 'conv1', 'conv1-BN'
                    weight_name = parts[-1]  # e.g., 'kernel:0', 'bias:0'

                    # Clean up weight_name
                    weight_name = weight_name.replace(':0', '')

                    full_key = f"{layer_name}/{weight_name}"
                    weights[full_key] = obj[:]
                elif len(parts) == 2:
                    # Top level or differently structured
                    pass

        f.visititems(collect_weights)

    return weights


def convert_conv2d_to_conv1d(keras_kernel):
    """
    Convert Keras Conv2D kernel to PyTorch Conv1d kernel.

    Keras Conv2D kernel shape: (width, 1, in_channels, out_channels)
    PyTorch Conv1d kernel shape: (out_channels, in_channels, width)

    Since the Keras kernel has shape (width, 1, in_c, out_c), we:
      1. Squeeze the singleton dimension (the '1')
      2. Transpose to (out_c, in_c, width)
    """
    # keras_kernel shape: (width, 1, in_channels, out_channels)
    # Remove the singleton dimension (height=1)
    if keras_kernel.ndim == 4:
        kernel_3d = np.squeeze(keras_kernel, axis=1)  # (width, in_c, out_c)
        # Transpose to (out_c, in_c, width)
        kernel_1d = np.transpose(kernel_3d, (2, 1, 0))
    elif keras_kernel.ndim == 3:
        # Already squeezed or different format
        kernel_1d = np.transpose(keras_kernel, (2, 1, 0))
    else:
        raise ValueError(f"Unexpected kernel shape: {keras_kernel.shape}")

    return kernel_1d


def convert_model(model_type, keras_dir, output_dir):
    """
    Convert a single FCN model variant from Keras to PyTorch.

    Args:
        model_type: 'fcn_1953', 'fcn_993', or 'fcn_929'
        keras_dir: path to the official_keras directory
        output_dir: path to save the .pt file
    """
    model_name_map = {
        'fcn_1953': 'FCN_1953',
        'fcn_993': 'FCN_993',
        'fcn_929': 'FCN_929',
    }

    keras_model_dir = os.path.join(keras_dir, model_name_map[model_type])
    weights_path = os.path.join(keras_model_dir, 'weights.h5')

    if not os.path.exists(weights_path):
        print(f"  [ERROR] Keras weights not found: {weights_path}")
        return False

    print(f"\nConverting {model_type}...")
    print(f"  Keras weights: {weights_path}")

    # Load Keras weights
    keras_weights = load_keras_weights(weights_path)
    print(f"  Loaded {len(keras_weights)} weight tensors")

    # Create PyTorch model
    model = FCNPitchEstimator(model_type=model_type)
    cfg = model.MODEL_CONFIGS[model_type]

    # Build state dict
    state_dict = {}
    missing = []
    extra = []

    num_layers = len(cfg['filters'])

    for l_idx in range(num_layers):
        layer_num = l_idx + 1

        # Conv layer
        # Keras key format: "conv1/kernel", "conv1/bias"
        conv_name = f"conv{layer_num}"
        conv_keras_key = f"{conv_name}/kernel"
        bias_keras_key = f"{conv_name}/bias"

        pytorch_conv_key = f"conv_stack.conv{layer_num}.weight"
        pytorch_bias_key = f"conv_stack.conv{layer_num}.bias"

        if conv_keras_key in keras_weights:
            keras_kernel = keras_weights[conv_keras_key]
            pt_kernel = convert_conv2d_to_conv1d(keras_kernel)
            state_dict[pytorch_conv_key] = torch.from_numpy(pt_kernel.copy())

            if bias_keras_key in keras_weights:
                keras_bias = keras_weights[bias_keras_key]
                state_dict[pytorch_bias_key] = torch.from_numpy(keras_bias.copy())
            else:
                missing.append(pytorch_bias_key)
        else:
            missing.append(pytorch_conv_key)

        # BatchNorm layer
        # Keras key format: "conv1-BN/gamma", "conv1-BN/beta", ...
        bn_name = f"conv{layer_num}-BN"
        bn_keras_keys = {
            f"{bn_name}/gamma": f"conv_stack.conv{layer_num}_bn.weight",
            f"{bn_name}/beta": f"conv_stack.conv{layer_num}_bn.bias",
            f"{bn_name}/moving_mean": f"conv_stack.conv{layer_num}_bn.running_mean",
            f"{bn_name}/moving_variance": f"conv_stack.conv{layer_num}_bn.running_var",
        }

        for keras_key, pt_key in bn_keras_keys.items():
            if keras_key in keras_weights:
                state_dict[pt_key] = torch.from_numpy(
                    keras_weights[keras_key].copy()
                )
            else:
                missing.append(pt_key)

    # Classifier layer
    # Keras key format: "classifier/kernel", "classifier/bias"
    cls_keras_keys = {
        "classifier/kernel": "classifier.weight",
        "classifier/bias": "classifier.bias",
    }

    for keras_key, pt_key in cls_keras_keys.items():
        if keras_key in keras_weights:
            if 'kernel' in keras_key:
                keras_kernel = keras_weights[keras_key]
                pt_kernel = convert_conv2d_to_conv1d(keras_kernel)
                state_dict[pt_key] = torch.from_numpy(pt_kernel.copy())
            else:
                state_dict[pt_key] = torch.from_numpy(
                    keras_weights[keras_key].copy()
                )
        else:
            missing.append(pt_key)

    # Check for extra keys in Keras weights not used
    used_keras_keys = set()
    for l_idx in range(num_layers):
        ln = l_idx + 1
        cn = f"conv{ln}"
        bn = f"conv{ln}-BN"
        used_keras_keys.add(f"{cn}/kernel")
        used_keras_keys.add(f"{cn}/bias")
        used_keras_keys.add(f"{bn}/gamma")
        used_keras_keys.add(f"{bn}/beta")
        used_keras_keys.add(f"{bn}/moving_mean")
        used_keras_keys.add(f"{bn}/moving_variance")
    used_keras_keys.add("classifier/kernel")
    used_keras_keys.add("classifier/bias")

    for k in keras_weights:
        if k not in used_keras_keys:
            extra.append(k)

    # Load state dict into model
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  [ERROR] Failed to load state dict: {e}")

    # Verify by forward pass
    min_len = model.min_input_length
    x = torch.randn(1, 1, min_len)
    with torch.no_grad():
        out = model(x)
    print(f"  Forward pass: input (1,1,{min_len}) → output {list(out.shape)}")

    # Report
    print(f"  State dict keys: {len(state_dict)}")
    if missing:
        print(f"  [WARNING] Missing keys ({len(missing)}): {missing}")
    if extra:
        print(f"  [INFO] Extra Keras keys (not used): {extra}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_type}.pt")
    torch.save(model.state_dict(), output_path)
    print(f"  ✅ Saved: {output_path}")

    return True


def main():
    """Convert all three FCN variants from Keras to PyTorch."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keras_dir = os.path.join(project_root, 'pretrained_models', 'official_keras')
    output_dir = os.path.join(project_root, 'pretrained_models')

    print("=" * 60)
    print("FCN Weight Conversion: Keras → PyTorch")
    print(f"Keras dir: {keras_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    success = True
    for model_type in ['fcn_1953', 'fcn_993', 'fcn_929']:
        try:
            ok = convert_model(model_type, keras_dir, output_dir)
            if not ok:
                success = False
        except Exception as e:
            print(f"  [ERROR] Failed to convert {model_type}: {e}")
            import traceback
            traceback.print_exc()
            success = False

    if success:
        print(f"\n{'='*60}")
        print("✅ All models converted successfully!")
        print(f"   Outputs: fcn_1953.pt, fcn_993.pt, fcn_929.pt")
    else:
        print(f"\n{'='*60}")
        print("⚠️  Some conversions failed. Check errors above.")


if __name__ == '__main__':
    main()
