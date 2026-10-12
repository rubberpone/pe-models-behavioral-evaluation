import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import json
import glob
from pathlib import Path
import math

class FlattenTF(nn.Module):
    """修复的TensorFlow风格展平层"""
    def __init__(self, name='flatten'):
        super(FlattenTF, self).__init__()
        self.name = name
    
    def forward(self, x):
        # TensorFlow风格: (B, C, H, W) -> (B, H*W*C)
        batch_size = x.size(0)
        return x.view(batch_size, -1)

class HWAPool2d(nn.Module):
    """修复的自定义池化层 - 模拟TensorFlow的hpool"""
    def __init__(self, name='hpool', pool_size=(8, 8), strides=(2, 2), padding='SAME'):
        super(HWAPool2d, self).__init__()
        self.name = name
        self.pool_size = pool_size if isinstance(pool_size, (list, tuple)) else (pool_size, pool_size)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.padding = padding.upper()  # 确保大写
        
        # 创建Hanning窗口核
        kh, kw = self.pool_size
        if kh > 1:
            window_h = 0.5 * (1 - torch.cos(2.0 * math.pi * torch.arange(kh) / (kh - 1)))
        else:
            window_h = torch.ones(1)
        if kw > 1:
            window_w = 0.5 * (1 - torch.cos(2.0 * math.pi * torch.arange(kw) / (kw - 1)))
        else:
            window_w = torch.ones(1)
            
        window = torch.outer(window_h, window_w)
        
        # 归一化
        if window.sum() > 0:
            window = window / window.sum()
            
        # 扩展为深度卷积核形状: (out_channels, 1, kh, kw)
        self.register_buffer('kernel', window.unsqueeze(0).unsqueeze(0))  # (1, 1, kh, kw)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        kh, kw = self.pool_size
        sh, sw = self.strides
        
        # 扩展核到正确的通道数
        kernel = self.kernel.repeat(channels, 1, 1, 1)  # (channels, 1, kh, kw)
        
        # 计算padding
        if self.padding == 'SAME':
            # TensorFlow的SAME padding
            out_h = math.ceil(height / sh)
            out_w = math.ceil(width / sw)
            
            pad_h = max((out_h - 1) * sh + kh - height, 0)
            pad_w = max((out_w - 1) * sw + kw - width, 0)
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            x_padded = nnf.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        else:  # VALID
            x_padded = x
        
        # 使用深度卷积进行池化
        return nnf.conv2d(
            x_padded, 
            kernel, 
            stride=self.strides, 
            groups=channels,  # 深度卷积
            padding=0
        )

# 在 Conv2DWrapper 类中，修改 __init__ 方法：
class Conv2DWrapper(nn.Module):
    """包装卷积层以匹配TensorFlow的行为"""
    def __init__(self, name, filters, kernel_size, strides=(1, 1), padding='SAME', 
                 use_bias=True, dilation_rate=(1, 1)):
        super(Conv2DWrapper, self).__init__()
        self.name = name
        self.padding = padding.upper()  # 确保大写
        self.dilation_rate = dilation_rate
        
        # 确保kernel_size是tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        # 确保strides是tuple
        if isinstance(strides, int):
            strides = (strides, strides)
        
        # 确保dilation_rate是tuple
        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate, dilation_rate)
            
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        
        # 重要修改：不要在这里创建卷积层，延迟到forward中创建
        self.filters = filters
        self.use_bias = use_bias
        
        # 这些将在build时设置
        self.conv = None
        self.in_channels = None
    
    def build(self, in_channels):
        """构建卷积层，使用正确的输入通道数"""
        if self.conv is None or self.in_channels != in_channels:
            self.in_channels = in_channels
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                padding=0,  # 手动处理padding
                dilation=self.dilation_rate,
                bias=self.use_bias
            )
            # 初始化权重
            nn.init.xavier_uniform_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        # 如果卷积层还没有构建，先构建
        if self.conv is None:
            self.build(x.size(1))
        
        kh, kw = self.kernel_size
        sh, sw = self.strides
        dh, dw = self.dilation_rate
        
        # 调整kernel大小考虑dilation
        effective_kh = kh + (kh - 1) * (dh - 1)
        effective_kw = kw + (kw - 1) * (dw - 1)
        
        if self.padding == 'SAME':
            # TensorFlow的SAME padding
            out_h = math.ceil(x.size(2) / sh)
            out_w = math.ceil(x.size(3) / sw)
            
            pad_h = max((out_h - 1) * sh + effective_kh - x.size(2), 0)
            pad_w = max((out_w - 1) * sw + effective_kw - x.size(3), 0)
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            x_padded = nnf.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        else:  # VALID
            x_padded = x
        
        return self.conv(x_padded)

class PitchNet(nn.Module):
    """修复后的PitchNet模型类 - 更接近TensorFlow实现"""
    
    # 添加类常量
    EPSILON = 1e-3
    MOMENTUM = 1e-2
    F0_LOWER_LIMIT = 80
    F0_UPPER_LIMIT = 1000
    BIN_IN_SEMITONE = 16
    
    def __init__(self, name='', n_classes=700, input_shape=(1, 100, 1000)):  # 改为700！
        super(PitchNet, self).__init__()
        self.name = name
        self.n_classes = n_classes  # 现在应该是700
        self.input_shape = input_shape
        self.layers = nn.ModuleDict()
        self.layer_order = []
        self._hooks = []
        
        self._f0classes = None
        
    @property
    def f0classes(self):
        """延迟初始化f0classes"""
        if self._f0classes is None:
            self._f0classes = torch.tensor(self.get_labels(), dtype=torch.float32)
        return self._f0classes
    
    @classmethod
    def get_labels(cls):
        """计算频率标签 - 需要根据700分类调整"""
        bin_in_octave = 12 * cls.BIN_IN_SEMITONE
        max_octave = np.log2(cls.F0_UPPER_LIMIT / cls.F0_LOWER_LIMIT)
        bins = 2 ** np.arange(0, max_octave, 1 / bin_in_octave)
        
        labels = cls.F0_LOWER_LIMIT * bins
        
        # 问题：这里用 cls.n_classes，但这是实例属性！
        # 改为使用固定值700
        if len(labels) != 700:  # 直接使用700
            # 从F0_LOWER_LIMIT到F0_UPPER_LIMIT在对数空间均匀分布生成700个值
            log_min = np.log(cls.F0_LOWER_LIMIT)
            log_max = np.log(cls.F0_UPPER_LIMIT)
            log_labels = np.linspace(log_min, log_max, 700)
            labels = np.exp(log_labels)
        
        return labels
    
    def add_layer(self, name, layer):
        """添加层并保持顺序"""
        self.layers[name] = layer
        self.layer_order.append(name)
    
# 在 _get_layer_by_type 方法中，完整添加以下层类型处理：

    def _get_layer_by_type(self, layer_type, layer_args):
        """根据TensorFlow层类型创建PyTorch层"""
        name = layer_args['name']
        
        # 1. 卷积层
        if layer_type == 'tf.layers.conv2d':
            filters = layer_args['filters']
            kernel_size = layer_args['kernel_size']
            strides = layer_args.get('strides', (1, 1))
            padding = layer_args.get('padding', 'SAME')
            use_bias = layer_args.get('use_bias', True)
            dilation_rate = layer_args.get('dilation_rate', (1, 1))
            
            print(f"[DEBUG] 创建Conv2D: filters={filters}, kernel={kernel_size}, strides={strides}")
            
            return Conv2DWrapper(
                name=name,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                dilation_rate=dilation_rate
            )
        
        # 2. ReLU激活函数
        elif layer_type == 'tf.nn.relu':
            return nn.ReLU(inplace=True)
        
        # 3. BatchNorm层
        elif layer_type == 'tf.layers.batch_normalization':
            momentum = layer_args.get('momentum', 0.99)
            epsilon = layer_args.get('epsilon', 1e-3)
            
            class SimpleBatchNorm(nn.Module):
                """简化的BatchNorm实现，在构建时跳过"""
                def __init__(self, momentum=0.99, epsilon=1e-3):
                    super(SimpleBatchNorm, self).__init__()
                    self.momentum = momentum
                    self.epsilon = epsilon
                    self.bn_layer = None
                    self.num_features = None
                    self.input_dim = None
                
                def forward(self, x):
                    # 延迟创建BatchNorm层
                    if self.bn_layer is None:
                        if x.dim() == 2:  # [B, features]
                            self.bn_layer = nn.BatchNorm1d(
                                x.size(1),
                                eps=self.epsilon,
                                momentum=1-self.momentum
                            )
                        elif x.dim() == 4:  # [B, C, H, W]
                            self.bn_layer = nn.BatchNorm2d(
                                x.size(1),
                                eps=self.epsilon,
                                momentum=1-self.momentum
                            )
                        
                        # 初始化参数
                        self.bn_layer.weight.data.fill_(1.0)
                        self.bn_layer.bias.data.zero_()
                        
                        # 移动bn_layer到与x相同的设备
                        if x.is_cuda:
                            self.bn_layer.cuda()
                    
                    # 在构建时（batch_size=1）或eval模式下，直接返回x
                    if x.size(0) == 1 and self.training:
                        return x
                    else:
                        return self.bn_layer(x)
            
            return SimpleBatchNorm(momentum=momentum, epsilon=epsilon)
        
        # 4. 自定义池化层
        elif layer_type == 'hpool':
            pool_size = layer_args['pool_size']
            strides = layer_args['strides']
            padding = layer_args.get('padding', 'SAME')
            
            return HWAPool2d(
                name=name,
                pool_size=pool_size,
                strides=strides,
                padding=padding
            )
        
        # 5. 展平层
        elif layer_type == 'tf.layers.flatten':
            return FlattenTF(name=name)
        
        # 6. 全连接层
        elif layer_type == 'tf.layers.dense':
            units = layer_args['units']
            use_bias = layer_args.get('use_bias', True)
            return nn.Linear(0, units, bias=use_bias)
        
        # 7. Dropout层
        elif layer_type == 'tf.layers.dropout':
            rate = layer_args['rate']
            return nn.Dropout(rate)
        
        # 8. 输出层
        elif layer_type == 'fc_top_classification':
            return nn.Linear(0, self.n_classes, bias=True)  # n_classes现在应该是700
        
        else:
            raise ValueError(f"未支持的层类型: {layer_type}")
    
    def _initialize_conv_weights(self, conv_layer):
        """初始化卷积层权重以匹配TensorFlow"""
        # TensorFlow默认使用glorot_uniform初始化
        nn.init.xavier_uniform_(conv_layer.weight)
        if conv_layer.bias is not None:
            nn.init.zeros_(conv_layer.bias)
    
    def _initialize_dense_weights(self, linear_layer):
        """初始化全连接层权重以匹配TensorFlow"""
        nn.init.xavier_uniform_(linear_layer.weight)
        if linear_layer.bias is not None:
            nn.init.zeros_(linear_layer.bias)
    
    def _calculate_output_shape(self, input_shape, layer_type, layer_args):
        """计算层的输出形状（模拟TensorFlow行为）"""
        h, w = input_shape[-2], input_shape[-1]
        
        if layer_type == 'tf.layers.conv2d':
            kernel_size = layer_args['kernel_size']
            strides = layer_args.get('strides', (1, 1))
            padding = layer_args.get('padding', 'SAME').upper()
            dilation_rate = layer_args.get('dilation_rate', (1, 1))
            
            kh, kw = kernel_size
            sh, sw = strides
            dh, dw = dilation_rate
            
            # 考虑dilation的有效核大小
            effective_kh = kh + (kh - 1) * (dh - 1)
            effective_kw = kw + (kw - 1) * (dw - 1)
            
            if padding == 'VALID':
                out_h = (h - effective_kh) // sh + 1
                out_w = (w - effective_kw) // sw + 1
            else:  # SAME
                out_h = math.ceil(h / sh)
                out_w = math.ceil(w / sw)
            
            return (layer_args['filters'], out_h, out_w)
            
        elif layer_type == 'hpool':
            pool_size = layer_args['pool_size']
            strides = layer_args['strides']
            padding = layer_args.get('padding', 'SAME').upper()
            
            ph, pw = pool_size
            sh, sw = strides
            
            if padding == 'VALID':
                out_h = (h - ph) // sh + 1
                out_w = (w - pw) // sw + 1
            else:  # SAME
                out_h = math.ceil(h / sh)
                out_w = math.ceil(w / sw)
            
            return (input_shape[0], out_h, out_w)  # 保持通道数不变
            
        elif layer_type == 'tf.layers.flatten':
            # 展平层
            return (np.prod(input_shape),)
            
        elif layer_type == 'tf.layers.dense':
            # 全连接层
            return (layer_args['units'],)
            
        elif layer_type == 'fc_top_classification':
            # 输出层
            return (self.n_classes,)
            
        else:
            # 对于其他层（ReLU, BatchNorm, Dropout），形状不变
            return input_shape
    

    def build_from_architecture(self, architecture, input_shape=None, batch_size=None):
        """从架构配置构建模型，修复批次维度问题"""
        print(f"构建模型架构: {self.name}")
        
        # 清空现有层
        self.layers = nn.ModuleDict()
        self.layer_order = []
        
        # 设置输入形状
        if input_shape is not None:
            self.input_shape = input_shape
        
        # 构建时使用batch_size=2，避免BatchNorm问题
        build_batch_size = 2 if batch_size is None else max(2, batch_size)
        
        # 创建虚拟输入来计算形状
        x = torch.randn(build_batch_size, *self.input_shape)
        print(f"构建输入形状: {x.shape}")
        
        # 在构建时强制设置为eval模式
        original_training = self.training
        self.eval()  # 关键：构建时使用eval模式
        
        current_channels = self.input_shape[0]
        
        # 构建每一层
        for i, layer_config in enumerate(architecture):
            layer_type = layer_config['layer_type']
            layer_args = layer_config['args']
            layer_name = layer_args['name']
            
            print(f"添加层 {i+1}: {layer_type} - {layer_name}")
            
            # 创建层
            layer = self._get_layer_by_type(layer_type, layer_args)
            
            # 特殊处理：为Linear层设置正确的输入特征数
            if isinstance(layer, nn.Linear):
                # 对于Linear层，需要知道输入特征数
                if x.dim() > 1:
                    # 展平最后一个维度之前的所有维度
                    x_flat = x.view(build_batch_size, -1)
                    in_features = x_flat.shape[1]
                else:
                    in_features = x.numel()
                
                # 创建新的Linear层
                units = layer_args.get('units', self.n_classes)
                use_bias = layer_args.get('use_bias', True)
                new_layer = nn.Linear(in_features, units, bias=use_bias)
                
                # 初始化权重
                if in_features > 0 and units > 0:
                    nn.init.xavier_uniform_(new_layer.weight)
                    if new_layer.bias is not None:
                        nn.init.zeros_(new_layer.bias)
                
                layer = new_layer
            
            # 添加到模型
            self.add_layer(layer_name, layer)
            
            # 前向传播以获取下一层的输入形状
            try:
                with torch.no_grad():
                    # 对于某些层需要特殊处理
                    if isinstance(layer, Conv2DWrapper):
                        # Conv2DWrapper会在第一次forward时自动构建
                        if layer.conv is None:
                            layer.build(x.size(1))  # 输入通道数
                    
                    # 关键修改：在构建时，BatchNorm层直接通过，不做计算
                    # 因为在eval模式下，BatchNorm不会计算统计量
                    # 而且batch_size=2可以避免"Expected more than 1 value"错误
                    
                    # 执行前向传播
                    x = layer(x)
                    print(f"  {layer_type}: 输出形状: {x.shape}")
            except Exception as e:
                print(f"  ❌ 前向传播失败: {e}")
                print(f"    当前形状: {x.shape}")
                print(f"    层参数: {layer_args}")
                # 对于某些错误，尝试跳过继续构建
                if "Expected more than 1 value" in str(e):
                    print(f"  ⚠️  BatchNorm警告，继续构建...")
                    # 保持x形状不变
                else:
                    raise
            
            # 更新当前通道数（对于卷积层）
            if layer_type == 'tf.layers.conv2d':
                current_channels = layer_args['filters']
        
        # 恢复原始训练状态
        if original_training:
            self.train()
        
        print(f"✅ 模型构建完成! 最终输出形状: {x.shape}")
        
        # 检查输出形状是否正确
        if x.dim() >= 2 and x.shape[1] != self.n_classes:
            print(f"⚠️  警告: 输出特征数 ({x.shape[1] if x.dim() >= 2 else x.shape[0]}) 不等于类别数 ({self.n_classes})")
        
        return self
    

    def load_architecture(self, arch_path, input_shape=None, batch_size=1):
        """从JSON文件加载架构"""
        try:
            with open(arch_path, 'r', encoding='utf-8') as f:
                architecture = json.load(f)
            
            return self.build_from_architecture(architecture, input_shape, batch_size)
            
        except Exception as e:
            print(f"❌ 加载架构失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def forward(self, x):
        """前向传播 - 按顺序通过所有层"""
        for layer_name in self.layer_order:
            x = self.layers[layer_name](x)
        return x
    
    def save_weights_npz(self, filepath):
        """保存权重为npz格式（兼容TensorFlow格式）"""
        weights_dict = {}
        
        for name, param in self.named_parameters():
            param_data = param.data.cpu().numpy()
            
            # 转换为TensorFlow格式的名称
            tf_name = name.replace('.', '/')
            
            if 'weight' in name:
                if 'conv' in name:
                    # 卷积层权重: PyTorch (out_c, in_c, h, w) -> TF (h, w, in_c, out_c)
                    weights_dict[f'{tf_name}/kernel'] = param_data.transpose(2, 3, 1, 0)
                elif 'Linear' in name:
                    # 全连接层权重: PyTorch (out_features, in_features) -> TF (in_features, out_features)
                    weights_dict[f'{tf_name}/kernel'] = param_data.T
            elif 'bias' in name:
                weights_dict[f'{tf_name}/bias'] = param_data
        
        # 保存BatchNorm参数
        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                base_name = name.replace('.', '/')
                weights_dict[f'{base_name}/gamma'] = module.weight.data.cpu().numpy()
                weights_dict[f'{base_name}/beta'] = module.bias.data.cpu().numpy()
                weights_dict[f'{base_name}/moving_mean'] = module.running_mean.cpu().numpy()
                weights_dict[f'{base_name}/moving_variance'] = module.running_var.cpu().numpy()
        
        np.savez(filepath, **weights_dict)
        print(f"✅ 权重已保存: {filepath}")
    
    def predict_f0(self, x):
        """预测基频"""
        with torch.no_grad():
            output = self.forward(x)
            predictions = torch.argmax(output, dim=1)
            return self.f0classes[predictions]