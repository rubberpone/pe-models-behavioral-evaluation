"""
PyTorch模型心理物理学实验评估器 

检查实验文件 → 查找模型 → 加载模型(从架构路径/data/zyf/pitchnet_torch/models加载) → 预处理数据 → 预测 → 保存结果 → 生成报告

# 运行评估
cd /data/zyf/pitchnet_torch
python psychophysics_evaluator.py

# 或者指定自定义路径
python psychophysics_evaluator.py \
    --models_dir /data/zyf/pitchnet_torch/trained_models \
    --nervegrams_dir /data/zyf/pitchnet/data/psychophysics_nervegrams \
    --output_dir /data/zyf/pitchnet/data/model_predictions \
    --batch_size 100

输入
selected_checkpoints/
├── arch_0190_step040000_e03_b19_best.pth
└── ...

输出
model_psy_predictions_global/
├── predictions_arch_0190_BernsteinOxenham2005.json
├── predictions_arch_0190_OxenhamEtAl2004.json
├── predictions_arch_0346_BernsteinOxenham2005.json
└── predictions_arch_0346_OxenhamEtAl2004.json
└── ...

单个预测文件结构（JSON格式）
{
  "f0_label:labels_pred": [123, 234, 345, ...],
  "f0_label:probs_out": [[0.01, 0.02, ...], [0.03, 0.04, ...], ...],
  "experiment_name": "BernsteinOxenham2005",
  "model_name": "arch_0190",
  "timestamp": "2024-01-15T10:30:00",
  "f0": [100.0, 200.0, 300.0, ...],
  "level": [50, 55, 60, ...],
  "f0_ref": [100.0, 100.0, 100.0, ...]
}

评估报告文件
{
  "evaluation_date": "2024-01-15T10:35:00",
  "models_directory": "/data/zyf/pitchnet_torch/trained_models",
  "nervegrams_directory": "/data/zyf/pitchnet/data/psychophysics_nervegrams",
  "output_directory": "/data/zyf/pitchnet_torch/model_psy_predictions_global",
  "total_predictions_attempted": 25,
  "successful_predictions": 20,
  "success_rate": 80.0,
  "experiments": ["BernsteinOxenham2005", "OxenhamEtAl2004", ...],
  "device_used": "cuda",
  "generated_files": ["predictions_arch_0190_BernsteinOxenham2005.json", ...],
  "total_generated_files": 20
}

"""

import os
import sys
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
import argparse
from tqdm import tqdm
import copy

# 添加您的模型路径
# 策略 B 副本：DNN_MODEL.py 与本脚本同目录，脚本目录已在 sys.path 中
from DNN_MODEL import PitchNet

# 策略 B 副本（2026-08-14）：本机驱动 550.163 与 pitchnet_pytorch 的
# cuDNN 9.1.0.70 不兼容（CUDNN_STATUS_NOT_INITIALIZED；9.0.1 兼容、
# 9.1 需更新驱动）——禁用 cuDNN 走原生 CUDA 核，已实测 GPU conv 正常。
torch.backends.cudnn.enabled = False

class PyTorchPsychophysicsEvaluator:
    """PyTorch心理物理学实验评估器"""
    
    def __init__(self, models_dir, nervegrams_dir, output_dir):
        """
        初始化评估器
        
        Args:
            models_dir: 训练好的模型目录
            nervegrams_dir: 神经图数据目录  
            output_dir: 预测结果输出目录
        """
        self.models_dir = Path(models_dir)
        self.nervegrams_dir = Path(nervegrams_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.experiments = {
            'BernsteinOxenham2005': 'nervegram_stim_BernsteinOxenham2005.hdf5',
            'OxenhamEtAl2004': 'nervegram_stim_OxenhamEtAl2004.hdf5',
            'ShackletonCarlyon1994': 'nervegram_stim_ShackletonCarlyon1994.hdf5',
            'MooreEtAl1985': 'nervegram_stim_MooreEtAl1985.hdf5',
            'MooreMoore2003': 'nervegram_stim_MooreMoore2003.hdf5',
            # 策略 B 副本新增：B6 校准集神经图
            'ERB_B6': 'nervegram_stim_ERB_B6.hdf5'
        }
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 使用设备: {self.device}")
        
    def find_trained_models(self):
        """查找所有训练好的模型"""
        print("🔍 查找训练好的模型...")
        
        model_dirs = []
        
        # # 使用正确的模型权重目录
        # models_base_dir = Path('/data/zyf/pitchnet_torch/trained_models_0109_global/selected_checkpoints')
        
        # if not models_base_dir.exists():
        #     print(f"❌ 模型目录不存在: {models_base_dir}")
        #     return model_dirs
        
        # print(f"📁 扫描目录: {models_base_dir}")

        models_base_dir = Path(self.models_dir)
    
        if not models_base_dir.exists():
            print(f"❌ 模型目录不存在: {models_base_dir}")
            return model_dirs
        
        print(f"📁 扫描目录: {models_base_dir}")
        
        # 定义要查找的模型架构列表（与架构文件目录匹配）
        target_archs = ['arch_0190', 'arch_0083', 'arch_0338', 'arch_0288', 
                    'arch_0286', 'arch_0191', 'arch_0302', 'arch_0154',
                    'arch_0335', 'arch_0346']
        # 策略 B 副本新增: arch 过滤
        if getattr(self, 'arch_filter', None) is not None:
            target_archs = [a for a in target_archs if a in self.arch_filter]
        
        # 查找所有.pth文件
        pth_files = list(models_base_dir.glob('*.pth'))
        print(f"  找到 {len(pth_files)} 个.pth文件")
        
        # 按架构组织文件
        arch_files = {}
        for pth_file in pth_files:
            filename = pth_file.name
            # 从文件名提取架构名称
            for arch in target_archs:
                if filename.startswith(arch + '_'):
                    if arch not in arch_files:
                        arch_files[arch] = []
                    arch_files[arch].append(pth_file)
                    break
        
        # 为每个架构选择最佳文件
        for arch_name, files in arch_files.items():
            if files:
                # 选择"best"文件或最新的
                best_file = None
                for file in files:
                    if 'best' in file.name.lower():
                        best_file = file
                        break
                
                # 如果没有找到best文件，选择第一个
                if best_file is None:
                    best_file = files[0]
                
                # 创建一个虚拟的模型目录（用于兼容现有代码）
                virtual_model_dir = models_base_dir / arch_name
                model_dirs.append((arch_name, virtual_model_dir, best_file))
                print(f"  ✅ 找到模型: {arch_name} -> {best_file.name}")
            else:
                print(f"  ❌ 未找到架构 {arch_name} 的权重文件")
        
        print(f"📊 总共找到 {len(model_dirs)} 个模型")
        return model_dirs
    
    def load_pytorch_model(self, model_dir, checkpoint_path):
        """加载PyTorch模型和权重"""
        try:
            # 从目录名获取架构名称
            arch_name = model_dir.name
            
            print(f"  正在加载模型 {arch_name}...")
            print(f"  权重文件: {checkpoint_path}")
            
            # 确定输入形状
            input_shape = (1, 100, 1000)  # 通道数, 高度, 宽度
            
            # 创建PitchNet模型实例
            model = PitchNet(
                name=arch_name,
                n_classes=700,  # 700个类别
                input_shape=input_shape
            )
            
            # 修改这里：使用新的架构文件路径 /data/zyf/pitchnet_torch/models
            base_models_path = Path('/data/zyf/pitchnet_torch/models')
            arch_file = base_models_path / arch_name / "brain_arch.json"
            
            if not arch_file.exists():
                print(f"  ⚠️ 架构文件不存在: {arch_file}")
                # 尝试其他可能的架构文件名
                alt_arch_file = base_models_path / arch_name / f"arch_{arch_name}.json"
                if alt_arch_file.exists():
                    arch_file = alt_arch_file
                else:
                    print(f"  ❌ 无法找到架构文件，尝试使用默认架构...")
                    # 可以在这里添加默认架构创建逻辑
                    return None
            
            # 如果存在架构文件，加载架构
            if arch_file and arch_file.exists():
                print(f"  加载架构文件: {arch_file}")
                try:
                    model.load_architecture(arch_file, input_shape, batch_size=2)  # 添加batch_size参数
                except Exception as e:
                    print(f"  ❌ 加载架构失败: {e}")
                    # 尝试使用其他参数
                    try:
                        model.load_architecture(arch_file, input_shape)
                    except Exception as e2:
                        print(f"  ❌ 再次尝试加载架构失败: {e2}")
                        return None
            else:
                print(f"  ❌ 架构文件不存在，无法加载模型")
                return None
            
            # 加载.pth权重文件
            print(f"  加载权重文件: {checkpoint_path}")
            
            # 检查文件大小
            file_size = checkpoint_path.stat().st_size / (1024**2)  # MB
            print(f"  文件大小: {file_size:.2f} MB")
            
            # 加载checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 打印checkpoint的键，帮助我们了解结构
            print(f"  📝 checkpoint中的键: {list(checkpoint.keys())}")
            
            # 从checkpoint中提取状态字典
            # 关键修改：根据您的输出，权重在'model_state_dict'中
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"  📝 使用'model_state_dict'中的权重")
            elif 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
                print(f"  📝 使用'model_state'中的权重")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"  📝 使用'state_dict'中的权重")
            else:
                # 尝试直接作为状态字典使用
                state_dict = checkpoint
                print(f"  📝 直接使用checkpoint作为状态字典")
            
            # 检查状态字典的键
            state_dict_keys = list(state_dict.keys())
            print(f"  📝 状态字典中的键 (前10个): {state_dict_keys[:10]}")
            
            # 获取模型的状态字典
            model_state_dict = model.state_dict()
            model_keys = list(model_state_dict.keys())
            print(f"  📝 模型期望的键 (前10个): {model_keys[:10] if model_keys else '[]'}")
            
            # 检查模型是否有参数
            if len(model_keys) == 0:
                print(f"  ❌ 模型没有参数，架构可能未正确加载")
                return None
            
            # 键名映射 - 根据需要调整
            # 你可能需要将checkpoint中的键名映射到模型中的键名
            key_mapping = {}
            
            # 常见的键名映射
            for key in state_dict_keys:
                # 如果checkpoint中的键包含layers，但格式不同
                if 'layers' not in key and any(layer_name in key for layer_name in ['conv', 'bn', 'fc']):
                    # 尝试推断映射
                    new_key = key
                    if key.startswith('conv') or key.startswith('bn') or key.startswith('fc'):
                        new_key = f'layers.{key}'
                    key_mapping[key] = new_key
            
            # 应用键名映射
            new_state_dict = {}
            for key, value in state_dict.items():
                if key in key_mapping:
                    new_key = key_mapping[key]
                    new_state_dict[new_key] = value
                else:
                    # 尝试其他可能的映射
                    # 移除可能的'module.'前缀（如果是用DataParallel训练的）
                    if key.startswith('module.'):
                        new_key = key[7:]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
            
            # 尝试加载状态字典，设置strict=False以允许部分加载
            try:
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                
                if missing_keys:
                    print(f"  ⚠️ 缺少的键 ({len(missing_keys)}个):")
                    for i, key in enumerate(missing_keys[:5]):  # 只显示前5个
                        print(f"    - {key}")
                    if len(missing_keys) > 5:
                        print(f"    ... 还有 {len(missing_keys)-5} 个")
                
                if unexpected_keys:
                    print(f"  ⚠️ 意外的键 ({len(unexpected_keys)}个):")
                    for i, key in enumerate(unexpected_keys[:5]):  # 只显示前5个
                        print(f"    - {key}")
                    if len(unexpected_keys) > 5:
                        print(f"    ... 还有 {len(unexpected_keys)-5} 个")
                
                print(f"  ✅ 权重加载成功")
                
            except Exception as e:
                print(f"  ❌ 加载状态字典失败: {e}")
                # 尝试更宽松的加载方式
                print(f"  🔄 尝试更宽松的加载方式...")
                
                # 手动将权重分配给模型
                loaded_count = 0
                for key, value in state_dict.items():
                    # 尝试找到对应的模型参数
                    found = False
                    for model_key, model_param in model.named_parameters():
                        # 简单匹配逻辑
                        if key.endswith(model_key.split('.')[-1]) or model_key.split('.')[-1] in key:
                            # 检查形状是否匹配
                            if value.shape == model_param.shape:
                                model_param.data.copy_(value)
                                found = True
                                loaded_count += 1
                                if loaded_count <= 5:  # 只显示前几个
                                    print(f"    ✅ 手动加载: {key} -> {model_key}")
                                break
                    
                    if not found and 'running_mean' not in key and 'running_var' not in key:
                        # 只显示非BatchNorm统计量的未匹配键
                        if loaded_count < 10:  # 限制显示数量
                            print(f"    ❌ 无法找到匹配的参数: {key} (形状: {value.shape})")
                
                print(f"  📊 手动加载了 {loaded_count} 个参数")
            
            # 打印模型信息
            print(f"  ✅ 模型加载完成: {arch_name}")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    总参数数量: {total_params:,}")
            print(f"    可训练参数: {trainable_params:,}")
            
            # 检查参数数量是否合理
            if total_params == 0:
                print(f"  ❌ 错误: 模型参数数量为0，加载失败!")
                return None
            
            # 将模型移到设备并设为评估模式
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"  ❌ 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def load_nervegram_data(self, experiment_name):
        """加载神经图数据"""
        nervegram_file = self.nervegrams_dir / self.experiments[experiment_name]
        
        if not nervegram_file.exists():
            print(f"  ❌ 神经图文件不存在: {nervegram_file}")
            return None
        
        try:
            # 首先打印文件信息
            print(f"  🔍 加载实验数据: {experiment_name}")
            print(f"    文件路径: {nervegram_file}")
            
            # 尝试打开文件并列出所有key
            with h5py.File(nervegram_file, 'r') as f:
                print(f"    文件keys: {list(f.keys())}")
                
                # 检查数据组是否存在
                if 'nervegram_meanrates' not in f:
                    print(f"    ❌ 'nervegram_meanrates' 不存在")
                    # 查看实际的数据组名称
                    for key in f.keys():
                        if 'nervegram' in key.lower():
                            print(f"    找到可能的神经图数据组: {key}")
                    return None
                
                # 加载神经图数据
                nervegrams = f['nervegram_meanrates'][:]
                print(f"    神经图形状: {nervegrams.shape}")
                
                # 加载所有特征数据
                features = {}
                for key in f.keys():
                    if key not in ['nervegram_meanrates', 'signal_fs', 'nervegram_fs', 'cf_list']:
                        try:
                            # 尝试以不同方式访问数据
                            if isinstance(f[key], h5py.Dataset):
                                features[key] = f[key][:]
                            elif isinstance(f[key], h5py.Group):
                                # 如果是组，可能需要特殊处理
                                print(f"    警告: {key} 是Group而不是Dataset")
                                # 尝试获取组内的数据
                                for subkey in f[key].keys():
                                    if isinstance(f[key][subkey], h5py.Dataset):
                                        features[f"{key}/{subkey}"] = f[key][subkey][:]
                            else:
                                # 对于标量数据
                                features[key] = f[key][()]
                        except Exception as e:
                            print(f"    警告: 无法加载 {key}: {e}")
                
                print(f"    加载的特征数量: {len(features)}")
                
                return {
                    'nervegrams': nervegrams,
                    'features': features,
                    'experiment_name': experiment_name
                }
                    
        except Exception as e:
            print(f"  ❌ 加载神经图数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_nervegrams(self, nervegrams):
        """预处理神经图数据以适应模型输入"""
        # 转换为PyTorch张量
        nervegrams_tensor = torch.from_numpy(nervegrams).float()
        
        # 根据您的模型输入形状 (1, 100, 1000) 进行调整
        # 假设nervegrams形状为 [batch, 100, 1000]
        
        if nervegrams_tensor.dim() == 3:
            # 如果形状是 [batch, 100, 1000]，添加通道维度 -> [batch, 1, 100, 1000]
            nervegrams_tensor = nervegrams_tensor.unsqueeze(1)
        elif nervegrams_tensor.dim() == 2:
            # 如果是单个样本 [100, 1000]，添加batch和通道维度 -> [1, 1, 100, 1000]
            nervegrams_tensor = nervegrams_tensor.unsqueeze(0).unsqueeze(0)
        
        return nervegrams_tensor.to(self.device)
    
    def predict_f0(self, model, nervegrams_batch):
        """使用模型预测F0"""
        with torch.no_grad():
            # 前向传播
            outputs = model(nervegrams_batch)
            
            # 获取预测结果
            if isinstance(outputs, tuple):
                logits = outputs[0]  # 如果是多个输出，取第一个
            else:
                logits = outputs
                
            # 应用softmax获取概率
            probs = torch.softmax(logits, dim=-1)
            
            # 获取预测标签
            pred_labels = torch.argmax(probs, dim=-1)
            
            return pred_labels.cpu().numpy(), probs.cpu().numpy()
    
    def process_single_experiment(self, model, model_name, experiment_data, batch_size=100):
        """处理单个实验"""
        experiment_name = experiment_data['experiment_name']
        nervegrams = experiment_data['nervegrams']
        features = experiment_data['features']
        
        print(f"  🎯 处理实验: {experiment_name}")
        print(f"    总样本数: {len(nervegrams)}")
        print(f"    批次大小: {batch_size}")
        
        n_samples = len(nervegrams)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # 初始化结果存储
        all_pred_labels = []
        all_pred_probs = []
        
        # 分批处理
        for batch_idx in tqdm(range(n_batches), desc=f"处理批次"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            # 获取当前批次
            batch_nervegrams = nervegrams[start_idx:end_idx]
            
            # 预处理
            batch_tensor = self.preprocess_nervegrams(batch_nervegrams)
            
            # 预测
            pred_labels, pred_probs = self.predict_f0(model, batch_tensor)
            
            # 收集结果
            all_pred_labels.extend(pred_labels)
            all_pred_probs.extend(pred_probs)
        
        # 转换为numpy数组
        all_pred_labels = np.array(all_pred_labels)
        all_pred_probs = np.array(all_pred_probs)
        
        print(f"    ✅ 预测完成: {all_pred_labels.shape} 个预测")
        
        # 准备结果字典
        results = {
            'f0_label:labels_pred': all_pred_labels.tolist(),
            'f0_label:probs_out': all_pred_probs.tolist(),
            'experiment_name': experiment_name,
            'model_name': model_name,
            'timestamp': str(np.datetime64('now'))
        }
        
        # 添加原始特征数据
        for key, value in features.items():
            if isinstance(value, np.ndarray) and len(value) == n_samples:
                results[key] = value.tolist()
            else:
                results[key] = value  # 标量或配置参数
        
        return results
    
    def save_prediction_results(self, results, model_name, experiment_name):
        """保存预测结果到JSON文件"""
        # 生成输出文件名
        output_filename = f"predictions_{model_name}_{experiment_name}.json"
        output_path = self.output_dir / output_filename
        
        # 保存结果
        with open(output_path, 'w') as f:
            # 使用自定义的JSON编码器处理numpy数据类型
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.bool_)):
                        return bool(obj)
                    elif isinstance(obj, (bytes, np.bytes_)):
                        return obj.decode('utf-8')
                        return bool(obj)
                    return super().default(obj)
            
            json.dump(results, f, cls=NumpyEncoder, indent=2)
        
        print(f"    💾 预测结果保存到: {output_path}")
        return output_path
    
    def validate_prediction_file(self, file_path, expected_samples):
        """验证预测文件的有效性"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 检查必需字段
            required_fields = ['f0_label:labels_pred', 'f0_label:probs_out', 'f0']
            for field in required_fields:
                if field not in data:
                    print(f"    ❌ 缺少必需字段: {field}")
                    return False
            
            # 检查数据长度
            pred_labels = data['f0_label:labels_pred']
            if len(pred_labels) != expected_samples:
                print(f"    ❌ 预测数量不匹配: {len(pred_labels)} != {expected_samples}")
                return False
            
            # 检查概率形状
            pred_probs = data['f0_label:probs_out']
            if len(pred_probs) != expected_samples:
                print(f"    ❌ 概率数量不匹配: {len(pred_probs)} != {expected_samples}")
                return False
            
            print(f"    ✅ 文件验证通过: {len(pred_labels)} 个预测")
            return True
            
        except Exception as e:
            print(f"    ❌ 文件验证失败: {e}")
            return False
    

    def check_experiment_files(self):
        """检查实验文件是否存在"""
        print("📁 检查实验文件...")
        for exp_name, exp_file in self.experiments.items():
            if getattr(self, 'exp_filter', None) is not None and \
                    exp_name not in self.exp_filter:
                continue
            file_path = self.nervegrams_dir / exp_file
            if file_path.exists():
                print(f"  ✅ {exp_name}: {exp_file} (存在)")
                # 检查文件大小
                file_size = file_path.stat().st_size / (1024**2)  # MB
                print(f"      文件大小: {file_size:.2f} MB")
            else:
                print(f"  ❌ {exp_name}: {exp_file} (不存在!)")


    def run_evaluation(self):
        """运行完整的评估流程"""
        print("🚀 开始心理物理学实验评估")
        print("=" * 60)
        
        # 0. 首先检查实验文件
        self.check_experiment_files()

        # 1. 查找模型
        model_dirs = self.find_trained_models()
        if not model_dirs:
            print("❌ 没有找到可用的模型")
            return
        
        # 2. 加载实验数据
        experiment_data = {}
        for exp_name, exp_file in self.experiments.items():
            if getattr(self, 'exp_filter', None) is not None and \
                    exp_name not in self.exp_filter:
                continue
            data = self.load_nervegram_data(exp_name)
            if data is not None:
                experiment_data[exp_name] = data
            else:
                print(f"❌ 无法加载实验数据: {exp_name}")
        
        if not experiment_data:
            print("❌ 没有可用的实验数据")
            return
        
        print(f"📊 将评估 {len(model_dirs)} 个模型在 {len(experiment_data)} 个实验上")
        print(f"🎯 总共将生成 {len(model_dirs) * len(experiment_data)} 个预测文件")
        
        # 3. 逐个模型和实验进行评估
        successful_predictions = 0
        total_predictions = len(model_dirs) * len(experiment_data)
        
        for model_idx, (arch_name, model_dir, checkpoint_path) in enumerate(model_dirs, 1):
            print(f"\n{'='*50}")
            print(f"🧠 处理模型 {model_idx}/{len(model_dirs)}: {arch_name}")
            print(f"检查点: {checkpoint_path.name}")
            print('='*50)
            
            # 加载模型
            model = self.load_pytorch_model(model_dir, checkpoint_path)
            if model is None:
                continue
            
            for exp_idx, (exp_name, exp_data) in enumerate(experiment_data.items(), 1):
                print(f"\n  🔬 实验 {exp_idx}/{len(experiment_data)}: {exp_name}")
                
                try:
                    # 处理实验
                    results = self.process_single_experiment(model, arch_name, exp_data)
                    
                    # 保存结果
                    output_file = self.save_prediction_results(results, arch_name, exp_name)
                    
                    # 验证文件
                    expected_samples = len(exp_data['nervegrams'])
                    if self.validate_prediction_file(output_file, expected_samples):
                        successful_predictions += 1
                        print(f"    ✅ 成功生成预测文件")
                    else:
                        print(f"    ⚠️  文件生成但验证有警告")
                    
                except Exception as e:
                    print(f"    ❌ 处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 4. 生成评估报告
        self.generate_evaluation_report(successful_predictions, total_predictions)
        
        print(f"\n🎉 评估完成!")
        print(f"📊 成功率: {successful_predictions}/{total_predictions} "
              f"({successful_predictions/total_predictions*100:.1f}%)")
    
    def generate_evaluation_report(self, successful, total):
        """生成评估报告"""
        report = {
            'evaluation_date': str(np.datetime64('now')),
            'models_directory': str(self.models_dir),
            'nervegrams_directory': str(self.nervegrams_dir),
            'output_directory': str(self.output_dir),
            'total_predictions_attempted': total,
            'successful_predictions': successful,
            'success_rate': successful / total * 100 if total > 0 else 0,
            'experiments': list(self.experiments.keys()),
            'device_used': str(self.device)
        }
        
        # 列出生成的文件
        prediction_files = list(self.output_dir.glob('predictions_*.json'))
        report['generated_files'] = [f.name for f in prediction_files]
        report['total_generated_files'] = len(prediction_files)
        
        # 保存报告
        report_file = self.output_dir / 'evaluation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 评估报告已保存: {report_file}")
        
        # 打印摘要
        print(f"\n📊 评估摘要:")
        print(f"  输出目录: {self.output_dir}")
        print(f"  生成文件: {len(prediction_files)} 个JSON文件")
        print(f"  成功率: {report['success_rate']:.1f}%")
        
        return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyTorch心理物理学实验评估')
    parser.add_argument('--models_dir', type=str, 
                       default='/data/zyf/pitchnet_torch/trained_models_0109_global_log/selected_checkpoints',
                       help='训练好的模型目录')
    parser.add_argument('--nervegrams_dir', type=str,
                       default='/data/zyf/pitchnet/data/psychophysics_nervegrams_log', 
                       help='神经图数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='/data/zyf/pitchnet_torch/model_psy_predictions_0109_global_log',
                       help='预测结果输出目录')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='推理批次大小')
    parser.add_argument('--archs', type=str, default=None,
                       help='逗号分隔的架构过滤（如 arch_0191）；默认 None = 全部')
    parser.add_argument('--experiments', type=str, default=None,
                       help='逗号分隔的实验名过滤（如 ERB_B6）；默认 None = 全部')
    
    args = parser.parse_args()
    
    print("🔧 PyTorch心理物理学实验评估器")
    print("=" * 60)
    print(f"模型目录: {args.models_dir}")
    print(f"神经图目录: {args.nervegrams_dir}") 
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print("=" * 60)
    
    # 创建评估器
    evaluator = PyTorchPsychophysicsEvaluator(
        models_dir=args.models_dir,
        nervegrams_dir=args.nervegrams_dir,
        output_dir=args.output_dir
    )

    # 策略 B 副本新增: arch/实验过滤
    evaluator.arch_filter = (set(s.strip() for s in args.archs.split(','))
                             if args.archs else None)
    evaluator.exp_filter = (set(s.strip() for s in args.experiments.split(','))
                            if args.experiments else None)
    
    # 运行评估
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()