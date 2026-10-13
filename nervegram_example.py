"""
行为实验神经图生成器 - 适配五个行为实验文件
Windows移植版本
基于原始代码风格，适配不同文件结构
"""

import os
import sys
import glob
import multiprocessing as mp
import time
import h5py
import numpy as np
import signal
import traceback
from datetime import datetime
import shutil
import tempfile

# 添加必要的路径 - Windows路径格式
sys.path.append(r'D:\\Events\\pitchnet\\packages\\bez2018model')
# 或者使用相对路径
# sys.path.append(os.path.join('C:', 'data', 'zyf', 'pitchnet', 'packages', 'bez2018model'))

try:
    import bez2018model_run_hdf5_dataset as nervegen
except ImportError as e:
    print(f"❌ 无法导入nervegen模块: {e}")
    print("请确保bez2018model模块已正确安装并配置")
    sys.exit(1)

# 添加辅助函数生成对数尺度CF
# def get_log_cf_list(num_cf=100, min_cf=125.0, max_cf=14000.0):
#     return np.logspace(np.log10(min_cf), np.log10(max_cf), num=num_cf).tolist()

# # 配置参数 - 与您提供的代码保持一致
# BEZ2018_CONFIG = {
#     'nervegram_dur': 0.050,
#     'nervegram_fs': 20e3,
#     'buffer_start_dur': 0.070,
#     'buffer_end_dur': 0.010,
#     'pin_fs': 100e3,
#     'pin_dBSPL_flag': 0,
#     'pin_dBSPL': None,
#     'species': 2,
#     'bandwidth_scale_factor': 1.0,
#     'cf_list': get_log_cf_list(num_cf=100, min_cf=125.0, max_cf=14000.0),# 对数尺度模式下传入CF列表,ERB模式下设置None
#     'num_cf': None,# ERB模式下设置100, 对数尺度模式下设置None
#     'min_cf': 125.0,
#     'max_cf': 14e3,
#     'synapseMode': 0,
#     'max_spikes_per_train': 1000,
#     'num_spike_trains': 40,
#     'cohc': 1.0,
#     'cihc': 1.0,
#     'IhcLowPass_cutoff': 3e3,
#     'IhcLowPass_order': 7,
#     'spont': 70.0,
#     'noiseType': 1,
#     'implnt': 0,
#     'tabs': 6e-4,
#     'trel': 6e-4,
#     'random_seed': None,
#     'return_vihcs': False,
#     'return_meanrates': True,
#     'return_spike_times': False,
#     'return_spike_tensor_sparse': False,
#     'return_spike_tensor_dense': False,
#     'nervegram_spike_tensor_fs': 100e3,
#     'squeeze_channel_dim': True
# }

BEZ2018_CONFIG = {
    'nervegram_dur': 0.050,
    'nervegram_fs': 20e3, # 时间采样率20khz
    'buffer_start_dur': 0.070,
    'buffer_end_dur': 0.010,
    'pin_fs': 100e3,
    'pin_dBSPL_flag': 0,
    'pin_dBSPL': None,
    'species': 2,
    'bandwidth_scale_factor': 1.0,
    'cf_list': None,
    'num_cf': 100, # fiber数量，默认100（如果cf_list为None）
    'min_cf': 125.0,
    'max_cf': 14e3,
    'synapseMode': 0,
    'max_spikes_per_train': 1000,
    'num_spike_trains': 40,
    'cohc': 1.0,
    'cihc': 1.0,
    'IhcLowPass_cutoff': 3e3,# IHC截止频率（默认3000 Hz）
    'IhcLowPass_order': 7,
    'spont': 70.0,
    'noiseType': 1,
    'implnt': 0,
    'tabs': 6e-4,
    'trel': 6e-4,
    'random_seed': None,
    'return_vihcs': False,
    'return_meanrates': True,
    'return_spike_times': False,
    'return_spike_tensor_sparse': False,
    'return_spike_tensor_dense': False,
    'nervegram_spike_tensor_fs': 100e3,
    'squeeze_channel_dim': True
}

class PsychophysicsNervegramGenerator:
    """行为实验神经图生成器 - 适配五个特定实验文件"""

    def __init__(self, input_dir, output_dir, max_workers=4, start_idx=None, end_idx=None, part_suffix=None):  # 【修改1】添加分批处理参数
        """
        初始化处理器

        Args:
            input_dir: 输入数据目录
            output_dir: 输出数据目录
            max_workers: 最大并行进程数（Windows建议设为4）
            start_idx: 起始索引（None表示从0开始）
            end_idx: 结束索引（None表示处理到最后）
            part_suffix: 输出文件后缀（如'_part2'，避免覆盖前半段结果）
        """
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.max_workers = max_workers
        self.start_idx = start_idx  # 【修改1】记录起始索引
        self.end_idx = end_idx        # 【修改1】记录结束索引
        self.part_suffix = part_suffix  # 【修改1】记录文件后缀
        self.start_time = time.time()

        # 【修改1】打印分批处理信息
        if start_idx is not None or end_idx is not None:
            print(f"📦 分批处理模式:")
            print(f"   起始索引: {start_idx if start_idx is not None else '0 (开头)'}")
            print(f"   结束索引: {end_idx if end_idx is not None else '末尾'}")
            print(f"   输出后缀: {part_suffix if part_suffix else '无'}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义五个行为实验文件的信号键映射
        # 基于分析结果，每个文件的信号位置不同
        self.experiment_configs = {
            'stim_BernsteinOxenham2005.hdf5': {
                'signal_key': 'tone_in_noise',
                'sr_key': 'config_tone/fs',  # 注意：这个文件采样率在config_tone/fs
                'sr_default': 32000,
                'batch_size': 100,  # 批次大小
                'description': 'Bernstein & Oxenham (2005) 实验'
            }
            # 'stim_OxenhamEtAl2004.hdf5': {
            #     'signal_key': 'stimuli/signal_in_noise',  # 优先使用带噪声的信号
            #     'alternative_signal_key': 'stimuli/signal',
            #     'sr_key': 'sr',
            #     'sr_default': 32000,
            #     'batch_size': 100,
            #     'description': 'Oxenham et al. (2004) 实验'
            # },
            # 'stim_ShackletonCarlyon1994.hdf5': {
            #     'signal_key': 'tone_in_noise',
            #     'sr_key': 'sr',
            #     'sr_default': 32000,
            #     'batch_size': 100,
            #     'description': 'Shackleton & Carlyon (1994) 实验'
            # },
            # 'stim_MooreEtAl1985.hdf5': {
            #     'signal_key': 'stimuli/signal',
            #     'sr_key': 'sr',
            #     'sr_default': 32000,
            #     'batch_size': 100,
            #     'description': 'Moore et al. (1985) 实验'
            # },
            # 'stim_MooreMoore2003.hdf5': {
            #     'signal_key': 'stimuli/signal',
            #     'sr_key': 'sr',
            #     'sr_default': 32000,
            #     'batch_size': 100,
            #     'description': 'Moore & Moore (2003) 实验'
            # }
        }
        
        # 需要复制的特征键列表
        self.feature_keys_to_copy = [
            'f0', 'f0_label', 'f0_log2', 'f0_lognormal',
            'f0_label_024', 'f0_label_048', 'f0_label_096',
            'f0_label_192', 'f0_label_384'
        ]
        
        # 实验特定特征（每个实验可能有不同）
        self.experiment_specific_features = {
            'stim_BernsteinOxenham2005.hdf5': [
                'base_f0', 'delta_f0', 'low_harm', 
                'max_audible_harm', 'min_audible_harm', 'phase_mode'
            ],
            'stim_OxenhamEtAl2004.hdf5': [
                'f0_label_coarse', 'f_carrier', 'f_envelope'
            ],
            'stim_ShackletonCarlyon1994.hdf5': [
                'filter_fh', 'filter_fl', 'phase_mode'
            ],
            'stim_MooreEtAl1985.hdf5': [
                'mistuned_harm', 'mistuned_pct'
            ],
            'stim_MooreMoore2003.hdf5': [
                'f0_shift', 'spectral_envelope_bandwidth_in_harmonics',
                'spectral_envelope_centered_harmonic',
                'spectral_envelope_f_bandwidth', 'spectral_envelope_f_center'
            ]
        }
        
        # 进度跟踪
        self.total_experiments = len(self.experiment_configs)
        self.completed_experiments = 0
        self.successful_experiments = 0
        self.failed_experiments = 0
    
    def signal_handler(self, signum, frame):
        """Windows信号处理（简化版）"""
        print(f"\n⚠️ 收到退出信号，正在保存进度...")
        self.save_progress()
        print("程序退出")
        sys.exit(1)
    
    def save_progress(self):
        """保存处理进度"""
        progress_file = os.path.join(self.output_dir, 'psychophysics_progress.txt')
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                f.write(f"总实验数: {self.total_experiments}\n")
                f.write(f"已完成实验: {self.completed_experiments}\n")
                f.write(f"成功实验: {self.successful_experiments}\n")
                f.write(f"失败实验: {self.failed_experiments}\n")
                f.write(f"开始时间: {datetime.fromtimestamp(self.start_time)}\n")
                f.write(f"已运行时间: {time.time() - self.start_time:.1f} 秒\n")
                # 【修改6】添加分批处理信息
                if self.start_idx is not None or self.end_idx is not None:
                    f.write(f"分批处理: start_idx={self.start_idx}, end_idx={self.end_idx}, suffix={self.part_suffix}\n")
                f.write(f"\n处理详情:\n")

                for exp_file, config in self.experiment_configs.items():
                    exp_name = os.path.splitext(exp_file)[0]
                    # 【修改6】使用part_suffix构造输出文件名
                    suffix = self.part_suffix if self.part_suffix else ''
                    output_file = os.path.join(self.output_dir, f'nervegram_{exp_name}{suffix}.hdf5')
                    if os.path.exists(output_file):
                        status = "✅ 已完成"
                    else:
                        status = "⏳ 未开始"
                    f.write(f"  {exp_file}: {status}\n")
            print(f"📁 进度已保存到: {progress_file}")
        except Exception as e:
            print(f"❌ 保存进度失败: {e}")
    
    def load_progress(self):
        """加载处理进度"""
        progress_file = os.path.join(self.output_dir, 'psychophysics_progress.txt')
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    print("📋 加载之前的进度...")
                    for line in f:
                        print(f"  {line.strip()}")
                return True
            except Exception as e:
                print(f"⚠️ 加载进度文件失败: {e}")
                return False
        return False
    
    def get_experiment_info(self, filename):
        """获取实验文件信息"""
        filepath = os.path.join(self.input_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  ❌ 文件不存在: {filepath}")
            return None
        
        try:
            with h5py.File(filepath, 'r') as f:
                config = self.experiment_configs[filename]
                
                # 获取信号数据信息
                signal_key = config['signal_key']
                signal_data = None
                
                # 检查主要信号键
                if signal_key in f:
                    signal_data = f[signal_key]
                elif 'alternative_signal_key' in config and config['alternative_signal_key'] in f:
                    signal_key = config['alternative_signal_key']
                    signal_data = f[signal_key]
                    print(f"  ℹ️  使用备用信号键: {signal_key}")
                else:
                    # 尝试自动寻找信号
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset) and f[key].dtype in [np.float32, np.float64]:
                            if len(f[key].shape) == 2 and f[key].shape[1] == 4800:  # 典型信号形状
                                signal_key = key
                                signal_data = f[key]
                                print(f"  🔍 自动找到信号键: {signal_key}")
                                break
                
                if signal_data is None:
                    print(f"  ❌ 未找到信号数据")
                    return None
                
                # 获取采样率
                sr_value = config['sr_default']
                if config['sr_key'] in f:
                    sr_data = f[config['sr_key']]
                    if isinstance(sr_data, h5py.Dataset):
                        sr_value = sr_data[()] if sr_data.shape else sr_data
                else:
                    # 尝试从config_tone获取
                    if 'config_tone/fs' in f:
                        sr_data = f['config_tone/fs']
                        sr_value = sr_data[()] if sr_data.shape else sr_data
                
                # 获取特征
                available_features = {}
                all_features = self.feature_keys_to_copy + self.experiment_specific_features.get(filename, [])
                
                for feat_key in all_features:
                    if feat_key in f:
                        feat_data = f[feat_key]
                        if isinstance(feat_data, h5py.Dataset):
                            available_features[feat_key] = feat_data.shape
                
                return {
                    'filename': filename,
                    'filepath': filepath,
                    'signal_key': signal_key,
                    'signal_shape': signal_data.shape,
                    'num_signals': signal_data.shape[0],
                    'signal_length': signal_data.shape[1],
                    'sr_value': sr_value,
                    'available_features': available_features,
                    'config': config
                }
                
        except Exception as e:
            print(f"  ❌ 读取文件信息失败: {e}")
            traceback.print_exc()
            return None
    
    def prepare_experiment_data(self, exp_info):
        """准备实验数据，创建临时文件适配nervegen输入格式"""
        print(f"\n🔄 准备实验数据: {exp_info['filename']}")

        # Windows兼容的临时文件名
        pid = os.getpid()
        timestamp = int(time.time())
        base_name = exp_info['filename'].replace('.hdf5', '')

        # 使用tempfile获取临时目录
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f'psychophysics_{base_name}_{pid}_{timestamp}.hdf5')

        # 【修改2】计算实际要处理的数据范围
        start_idx = self.start_idx if self.start_idx is not None else 0
        end_idx = self.end_idx if self.end_idx is not None else exp_info['num_signals']

        print(f"  📊 数据范围: 索引 {start_idx} 到 {end_idx-1} (共 {end_idx-start_idx} 条)")

        try:
            with h5py.File(exp_info['filepath'], 'r') as src_f:
                with h5py.File(temp_file, 'w') as dst_f:
                    # 1. 复制信号数据（只复制指定范围）【修改2】
                    signal_data = src_f[exp_info['signal_key']][start_idx:end_idx]
                    dst_f.create_dataset('/stimuli/signal_in_noise', data=signal_data)
                    print(f"  ✅ 复制信号: {signal_data.shape} (原始范围: 0-{exp_info['num_signals']-1})")

                    # 2. 写入采样率
                    dst_f.create_dataset('/sr', data=[exp_info['sr_value']])
                    print(f"  ✅ 采样率: {exp_info['sr_value']} Hz")

                    # 3. 复制特征数据（只复制指定范围）【修改2】
                    features_copied = 0
                    all_features = self.feature_keys_to_copy + self.experiment_specific_features.get(exp_info['filename'], [])

                    for feat_key in all_features:
                        if feat_key in src_f:
                            feat_data = src_f[feat_key]
                            if isinstance(feat_data, h5py.Dataset):
                                # 只复制指定范围的特征数据
                                dst_f.create_dataset(f'/{feat_key}', data=feat_data[start_idx:end_idx])
                                features_copied += 1

                    print(f"  ✅ 复制 {features_copied} 个特征")

                    # 4. 复制配置参数（如有）
                    if 'config_tone' in src_f:
                        config_group = src_f['config_tone']
                        dst_config = dst_f.create_group('/config_tone')

                        for key in config_group.keys():
                            if isinstance(config_group[key], h5py.Dataset):
                                dst_config.create_dataset(key, data=config_group[key][:])

                        print(f"  ✅ 复制配置参数")

                    # 【修改2】在临时文件中记录原始索引范围（用于拼接时参考）
                    if start_idx > 0 or end_idx < exp_info['num_signals']:
                        grp = dst_f.create_group('_batch_info')
                        grp.create_dataset('start_idx', data=start_idx)
                        grp.create_dataset('end_idx', data=end_idx)
                        grp.create_dataset('original_num_signals', data=exp_info['num_signals'])
                        print(f"  📝 记录批处理信息: start_idx={start_idx}, end_idx={end_idx}")

                    return temp_file

        except Exception as e:
            print(f"  ❌ 准备数据失败: {e}")
            traceback.print_exc()

            # 清理临时文件
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e2:
                    print(f"  ⚠️  清理临时文件失败: {e2}")

            return None
    
    def process_single_experiment(self, exp_info, exp_idx):
        """处理单个行为实验"""
        print(f"\n{'='*70}")
        print(f"🧪 处理行为实验 {exp_idx+1}/{self.total_experiments}")
        print(f"文件: {exp_info['filename']}")
        print(f"描述: {exp_info['config']['description']}")
        print(f"{'='*70}")

        exp_start_time = time.time()

        # 【修改3】检查输出文件是否已存在（使用part_suffix避免覆盖）
        exp_name = os.path.splitext(exp_info['filename'])[0]
        suffix = self.part_suffix if self.part_suffix else ''
        output_file = os.path.join(self.output_dir, f'nervegram_{exp_name}{suffix}.hdf5')

        if os.path.exists(output_file):
            # 验证现有文件
            try:
                with h5py.File(output_file, 'r') as f:
                    # 【修改3】计算实际处理的信号数量
                    actual_num_signals = exp_info['num_signals']
                    if self.start_idx is not None or self.end_idx is not None:
                        start_idx = self.start_idx if self.start_idx is not None else 0
                        end_idx = self.end_idx if self.end_idx is not None else exp_info['num_signals']
                        actual_num_signals = end_idx - start_idx

                    if '/nervegram_meanrates' in f and f['/nervegram_meanrates'].shape[0] == actual_num_signals:
                        print(f"  ⏭️  实验已处理且数据完整，跳过")
                        self.completed_experiments += 1
                        self.successful_experiments += 1
                        return True
            except Exception as e:
                print(f"  ⚠️  验证现有文件失败: {e}")
                pass  # 文件可能损坏，重新处理

        # 准备数据
        temp_input_file = self.prepare_experiment_data(exp_info)
        if not temp_input_file:
            print(f"  ❌ 数据准备失败")
            self.completed_experiments += 1
            self.failed_experiments += 1
            return False

        try:
            print(f"  🚀 开始神经图生成...")
            # 【修改3】显示实际处理的信号数量
            actual_num_signals = exp_info['num_signals']
            if self.start_idx is not None or self.end_idx is not None:
                start_idx = self.start_idx if self.start_idx is not None else 0
                end_idx = self.end_idx if self.end_idx is not None else exp_info['num_signals']
                actual_num_signals = end_idx - start_idx

            print(f"    信号数: {actual_num_signals} (原始: {exp_info['num_signals']})")
            print(f"    批次大小: {exp_info['config']['batch_size']}")
            print(f"    输出文件: {output_file}")

            # 准备要复制的特征键列表
            source_keys = self.feature_keys_to_copy + self.experiment_specific_features.get(exp_info['filename'], [])

            # 调用nervegen处理（这是核心处理逻辑）
            print(f"    调用nervegen.run_dataset_generation...")
            nervegen.run_dataset_generation(
                source_hdf5_filename=temp_input_file,
                dest_hdf5_filename=output_file,
                source_key_signal='/stimuli/signal_in_noise',
                source_key_signal_fs='/sr',
                kwargs_nervegram=BEZ2018_CONFIG,
                data_key_pair_list=[
                    ('/nervegram_meanrates', 'nervegram_meanrates'),
                    ('/pin_dBSPL', 'pin_dBSPL'),
                ],
                source_keys_to_copy=source_keys  # 复制所有特征
            )

            # 验证结果
            if os.path.exists(output_file):
                verification = self.verify_output_file(output_file, exp_info)

                if verification['success']:
                    exp_time = time.time() - exp_start_time
                    print(f"  ✅ 神经图生成成功!")
                    print(f"     耗时: {exp_time:.1f} 秒")
                    print(f"     神经图形状: {verification['nervegram_shape']}")
                    print(f"     特征数量: {verification['features_count']}")
                    print(f"     输出文件: nervegram_{exp_name}{suffix}.hdf5")

                    self.successful_experiments += 1
                else:
                    print(f"  ❌ 神经图验证失败: {verification.get('message', '未知错误')}")
                    self.failed_experiments += 1

            else:
                print(f"  ❌ 神经图生成失败，输出文件不存在")
                self.failed_experiments += 1

        except Exception as e:
            print(f"  ❌ 处理过程中出错: {e}")
            traceback.print_exc()
            self.failed_experiments += 1

        finally:
            # 清理临时文件
            if os.path.exists(temp_input_file):
                try:
                    os.remove(temp_input_file)
                    print(f"  🗑️  清理临时文件")
                except Exception as e:
                    print(f"  ⚠️  清理临时文件失败: {e}")

            self.completed_experiments += 1
            exp_time = time.time() - exp_start_time
            print(f"  ⏱️  本实验总耗时: {exp_time:.1f} 秒")

        return True
    
    def verify_output_file(self, output_file, exp_info):
        """验证输出文件的质量和完整性"""
        print(f"  🔍 验证输出文件...")

        try:
            with h5py.File(output_file, 'r') as f:
                verification = {
                    'success': False,
                    'nervegram_shape': None,
                    'features_count': 0,
                    'message': ''
                }

                # 1. 检查nervegram数据
                if '/nervegram_meanrates' not in f:
                    verification['message'] = '缺少nervegram_meanrates数据'
                    return verification

                nervegram_data = f['/nervegram_meanrates'][:]
                verification['nervegram_shape'] = nervegram_data.shape

                # 【修改5】计算期望的信号数量（考虑分批处理）
                expected_num_signals = exp_info['num_signals']
                if self.start_idx is not None or self.end_idx is not None:
                    start_idx = self.start_idx if self.start_idx is not None else 0
                    end_idx = self.end_idx if self.end_idx is not None else exp_info['num_signals']
                    expected_num_signals = end_idx - start_idx

                # 检查信号数量是否匹配
                if nervegram_data.shape[0] != expected_num_signals:
                    verification['message'] = f'信号数量不匹配: 期望{expected_num_signals}, 实际{nervegram_data.shape[0]}'
                    return verification

                # 2. 检查基本特征
                required_features = ['f0', 'f0_label', 'f0_log2']
                features_found = []

                for feat in required_features:
                    if f'/{feat}' in f:
                        feat_data = f[f'/{feat}'][:]
                        if feat_data.shape[0] == expected_num_signals:
                            features_found.append(feat)

                verification['features_count'] = len(features_found)

                if len(features_found) < 2:
                    verification['message'] = f'基本特征不足: 找到{len(features_found)}/{len(required_features)}个'
                    return verification

                # 3. 数据质量检查
                # 计算nervegram数据的统计信息
                mean_val = np.mean(nervegram_data)
                std_val = np.std(nervegram_data)

                # 检查是否有异常值
                if mean_val < 0 or mean_val > 1000:  # 合理的范围检查
                    verification['message'] = f'均值异常: {mean_val:.2f}'
                    return verification

                verification['success'] = True
                verification['message'] = f'验证通过: 形状{nervegram_data.shape}, 均值{mean_val:.2f}, 标准差{std_val:.2f}'
                return verification

        except Exception as e:
            verification['message'] = f'验证过程出错: {str(e)}'
            return verification
    
    def run_batch_processing(self, exp_info):
        """批量处理实验数据（如果数据量太大）"""
        # 如果信号数量超过5000，使用批量处理
        if exp_info['num_signals'] > 5000:
            print(f"  📦 数据量较大 ({exp_info['num_signals']}个信号)，启用批量处理...")
            
            # 这里可以添加批量处理逻辑
            # 但基于您的原始代码，nervegen.run_dataset_generation应该能处理大数据
            return False  # 暂时不使用批量处理
        
        return False  # 直接处理
    
    def process_all_experiments(self):
        """处理所有行为实验"""
        print(f"\n{'='*70}")
        print(f"🚀 开始处理 {self.total_experiments} 个行为实验")
        print(f"{'='*70}")
        
        # 按文件大小排序，先处理小的
        experiments_to_process = []
        
        for filename in self.experiment_configs.keys():
            exp_info = self.get_experiment_info(filename)
            if exp_info:
                # 添加文件大小信息用于排序
                file_size = os.path.getsize(exp_info['filepath'])
                exp_info['file_size'] = file_size
                experiments_to_process.append(exp_info)
        
        # 按文件大小排序（从小到大）
        experiments_to_process.sort(key=lambda x: x['file_size'])
        
        print(f"\n📊 实验处理顺序（按文件大小排序）:")
        for i, exp_info in enumerate(experiments_to_process):
            size_mb = exp_info['file_size'] / (1024**2)
            print(f"  {i+1}. {exp_info['filename']} ({size_mb:.1f} MB)")
        
        # 处理每个实验
        for exp_idx, exp_info in enumerate(experiments_to_process):
            print(f"\n{'='*70}")
            print(f"🔬 实验详情:")
            print(f"  文件名: {exp_info['filename']}")
            print(f"  信号键: {exp_info['signal_key']}")
            print(f"  信号数: {exp_info['num_signals']}")
            print(f"  采样率: {exp_info['sr_value']} Hz")
            print(f"  可用特征: {len(exp_info['available_features'])} 个")
            print(f"{'='*70}")
            
            self.process_single_experiment(exp_info, exp_idx)
            
            # 保存进度
            self.save_progress()
            
            # 显示总体进度
            total_time = time.time() - self.start_time
            progress = self.completed_experiments / self.total_experiments * 100
            
            if self.completed_experiments > 0:
                estimated_total = total_time / self.completed_experiments * self.total_experiments
                remaining = estimated_total - total_time
            else:
                remaining = total_time * (self.total_experiments - 1)
            
            print(f"\n📊 总体进度: {progress:.1f}% ({self.completed_experiments}/{self.total_experiments})")
            print(f"  成功: {self.successful_experiments} | 失败: {self.failed_experiments}")
            print(f"  耗时: {total_time/60:.1f} 分钟 | 剩余: {remaining/60:.1f} 分钟")
            
            # 实验间短暂暂停
            if exp_idx < len(experiments_to_process) - 1:
                print(f"\n⏸️  暂停3秒，准备下一个实验...")
                time.sleep(3)
    
    def validate_all_outputs(self):
        """验证所有输出文件"""
        print(f"\n{'='*70}")
        print(f"🔍 验证所有输出文件")
        print(f"{'='*70}")

        validation_results = {}

        for filename in self.experiment_configs.keys():
            exp_name = os.path.splitext(filename)[0]
            # 【修改7】使用part_suffix构造输出文件名
            suffix = self.part_suffix if self.part_suffix else ''
            output_file = os.path.join(self.output_dir, f'nervegram_{exp_name}{suffix}.hdf5')

            print(f"\n📁 {filename}:")

            if os.path.exists(output_file):
                try:
                    with h5py.File(output_file, 'r') as f:
                        # 获取实验信息用于验证
                        exp_info = self.get_experiment_info(filename)

                        if exp_info:
                            verification = self.verify_output_file(output_file, exp_info)

                            if verification['success']:
                                print(f"  ✅ {verification['message']}")
                                validation_results[filename] = {
                                    'status': '✅ 有效',
                                    'shape': verification['nervegram_shape'],
                                    'features': verification['features_count']
                                }
                            else:
                                print(f"  ❌ {verification['message']}")
                                validation_results[filename] = {
                                    'status': f'❌ 无效 - {verification["message"]}'
                                }
                        else:
                            print(f"  ⚠️  无法获取实验信息")
                            validation_results[filename] = {
                                'status': '⚠️ 无法验证'
                            }
                except Exception as e:
                    print(f"  ❌ 读取失败: {str(e)}")
                    validation_results[filename] = {
                        'status': f'❌ 读取失败 - {str(e)}'
                    }
            else:
                print(f"  ❌ 文件不存在")
                validation_results[filename] = {
                    'status': '❌ 文件不存在'
                }

        return validation_results
    
    def generate_final_report(self, validation_results):
        """生成最终报告"""
        report_file = os.path.join(self.output_dir, 'psychophysics_final_report.txt')
        total_time = time.time() - self.start_time

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("行为实验神经图生成最终报告\n")
                f.write("="*60 + "\n\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总运行时间: {total_time/60:.2f} 分钟\n\n")
                f.write(f"输入目录: {self.input_dir}\n")
                f.write(f"输出目录: {self.output_dir}\n\n")

                # 【修改8】添加分批处理信息
                if self.start_idx is not None or self.end_idx is not None:
                    f.write(f"分批处理: start_idx={self.start_idx}, end_idx={self.end_idx}, suffix={self.part_suffix}\n\n")

                f.write(f"实验总数: {self.total_experiments}\n")
                f.write(f"成功实验: {self.successful_experiments}\n")
                f.write(f"失败实验: {self.failed_experiments}\n")
                if self.total_experiments > 0:
                    f.write(f"成功率: {self.successful_experiments/self.total_experiments*100:.1f}%\n\n")
                else:
                    f.write(f"成功率: 0.0%\n\n")

                f.write("各实验处理状态:\n")
                f.write("-"*60 + "\n")

                total_output_size = 0
                suffix = self.part_suffix if self.part_suffix else ''  # 【修改8】获取后缀
                for filename in self.experiment_configs.keys():
                    exp_name = os.path.splitext(filename)[0]
                    output_file = os.path.join(self.output_dir, f'nervegram_{exp_name}{suffix}.hdf5')

                    status = validation_results.get(filename, {}).get('status', '❓ 未知')

                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file) / (1024**2)
                        total_output_size += file_size

                        f.write(f"✓ {filename}\n")
                        f.write(f"  输出文件: nervegram_{exp_name}{suffix}.hdf5 ({file_size:.2f} MB)\n")
                        f.write(f"  状态: {status}\n")

                        if 'shape' in validation_results.get(filename, {}):
                            f.write(f"  神经图形状: {validation_results[filename]['shape']}\n")

                        if 'features' in validation_results.get(filename, {}):
                            f.write(f"  特征数量: {validation_results[filename]['features']}\n")

                    else:
                        f.write(f"✗ {filename}\n")
                        f.write(f"  状态: {status}\n")

                    f.write("\n")

                f.write("-"*60 + "\n")
                f.write(f"输出文件总大小: {total_output_size:.2f} MB\n\n")

                # 配置信息
                f.write("处理配置:\n")
                f.write(f"  听觉外周模型: bez2018model\n")
                f.write(f"  通道数 (num_cf): {BEZ2018_CONFIG['num_cf']}\n")
                f.write(f"  频率范围: {BEZ2018_CONFIG['min_cf']:.0f}-{BEZ2018_CONFIG['max_cf']:.0f} Hz\n")
                f.write(f"  神经图时长: {BEZ2018_CONFIG['nervegram_dur']*1000:.0f} ms\n")
                f.write(f"  神经图采样率: {BEZ2018_CONFIG['nervegram_fs']/1000:.0f} kHz\n")
                f.write(f"  最大并行进程: {self.max_workers}\n")

            print(f"\n📄 详细报告已保存到: {report_file}")
            return report_file

        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
            return None
    
    def run(self):
        """运行处理流程"""
        print("🚀 行为实验神经图生成器 (Windows版本)")
        print("=" * 70)
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"最大并行进程: {self.max_workers}")
        print(f"系统CPU核心数: {mp.cpu_count()}")
        print(f"Python版本: {sys.version}")
        print(f"操作系统: {os.name}")
        print("=" * 70)
        
        # Windows信号处理（简化）
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
        except AttributeError:
            print("⚠️ Windows不支持所有POSIX信号，使用简化信号处理")
        
        # 检查输入文件
        missing_files = []
        for filename in self.experiment_configs.keys():
            filepath = os.path.join(self.input_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        if missing_files:
            print(f"\n⚠️  以下文件缺失:")
            for f in missing_files:
                print(f"  - {f}")
            
            if len(missing_files) == len(self.experiment_configs):
                print("❌ 没有找到任何实验文件")
                return
        
        # 加载进度
        self.load_progress()
        
        # 处理所有实验
        self.process_all_experiments()
        
        # 验证输出
        print(f"\n{'='*70}")
        print(f"🎯 处理完成，开始验证输出文件")
        print(f"{'='*70}")
        
        validation_results = self.validate_all_outputs()
        
        # 生成最终报告
        report_file = self.generate_final_report(validation_results)
        
        # 总结
        print(f"\n{'='*70}")
        print(f"🎉 行为实验神经图生成完成!")
        print(f"{'='*70}")
        print(f"总实验数: {self.total_experiments}")
        print(f"成功: {self.successful_experiments}")
        print(f"失败: {self.failed_experiments}")
        print(f"输出目录: {self.output_dir}")
        if report_file:
            print(f"查看报告: {report_file}")
        else:
            print("⚠️ 报告生成失败")

def main():
    # Windows配置参数 - 请根据实际情况修改
    # 建议使用绝对路径
    input_dir = r'D:\Events\pitchnet\data\dataset_psychophysics'
    output_dir = r'D:\Events\pitchnet\psychophysics_nervegrams_1000fiber_1000hz'

    # 或者使用相对路径
    # input_dir = os.path.join('data', 'dataset_psychophysics')
    # output_dir = os.path.join('data', 'psychophysics_nervegrams')

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 【修改4】分批处理参数配置（另一台电脑上运行）
    # 说明：原始数据共72600条，这台电脑从第50000条开始处理
    # 第一台电脑：0-4999（已完成或正在进行）
    # 第二台电脑（本机）：5000-72599（从第5000条开始到结束）
    START_IDX = 36507      # 起始索引（None表示从0开始）
    END_IDX = 50000        # 结束索引（None表示处理到最后）；= 50000，到49999结束（不包含50000）
    PART_SUFFIX = '_part3'  # 输出文件后缀，避免覆盖（如：nervegram_stim_BernsteinOxenham2005_part2.hdf5）

    # 如果要在本机从头开始运行，设置：
    # START_IDX = None
    # END_IDX = None
    # PART_SUFFIX = None

    print(f"\n{'='*60}")
    print(f"📦 分批处理配置:")
    print(f"   起始索引: {START_IDX if START_IDX is not None else '0 (开头)'}")
    print(f"   结束索引: {END_IDX if END_IDX is not None else '末尾'}")
    print(f"   文件后缀: {PART_SUFFIX if PART_SUFFIX else '无（会覆盖原文件）'}")
    print(f"{'='*60}\n")

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        print("请确保输入目录存在且包含以下文件:")
        for filename in ['stim_BernsteinOxenham2005.hdf5',
                        'stim_OxenhamEtAl2004.hdf5',
                        'stim_ShackletonCarlyon1994.hdf5',
                        'stim_MooreEtAl1985.hdf5',
                        'stim_MooreMoore2003.hdf5']:
            print(f"  - {filename}")
        return

    # 【修改4】创建处理器 - 传入分批处理参数
    processor = PsychophysicsNervegramGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=16,  # Windows建议减少并行进程数
        start_idx=START_IDX,      # 【修改4】起始索引
        end_idx=END_IDX,          # 【修改4】结束索引
        part_suffix=PART_SUFFIX   # 【修改4】输出文件后缀
    )

    # 运行处理
    processor.run()

if __name__ == "__main__":
    # Windows多进程设置
    try:
        # Windows默认使用'spawn'，无需强制设置
        mp.freeze_support()  # 打包exe时需要
    except:
        pass
    
    main()