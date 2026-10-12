"""
B6 神经图生成脚本（交付 PC 端运行，2026-08-14）。

基于 nervegram_example.py 精简改编：只处理 stim_ERB_B6.hdf5，
BEZ2018_CONFIG 用与 Josh 既有心理物理神经图完全一致的规范配置
（100 fibers / 20 kHz / 50 ms / IHC 3 kHz / cf 125–14000），
该配置已与 /data/zyf/pitchnet/data/psychophysics_nervegrams/
nervegram_stim_BernsteinOxenham2005.hdf5 内置的 config_bez2018model
逐项核对一致。

用法（PC 端）:
  1. 把 stim_ERB_B6.hdf5 从服务器拷到 PC（如 D:/ERB/stim_ERB_B6.hdf5）
  2. 修改下方 INPUT_HDF5 / OUTPUT_HDF5 / BEZ2018_PATH 三个路径
  3. python nervegram_generate_B6.py

输出规格（Josh 评估器 PSYCHO_EVALUATOR_global.py 的读取要求）:
  nervegram_stim_ERB_B6.hdf5
    /nervegram_meanrates  (8470, 100, 1000) float —— 100 fibers × 1000 时间点
    /cf_list              (1, 100) 125–14000 Hz
    /nervegram_fs         [20000]  /signal_fs [32000]
    + 特征列（base_f0/delta_f0/low_harm/min_audible_harm/max_audible_harm/
      phase_mode/f0/f0_label*/f0_log2/f0_lognormal）
    + config_bez2018model 组
完成后拷回服务器: /data/zyf/PE_model_ERB_estimation/data/nervegrams/
"""

import os
import sys
import time
import h5py
import numpy as np
import traceback
import tempfile

# ============================================================
# 【PC 端修改这三个路径】
# ============================================================
INPUT_HDF5 = r'D:\ERB\stim_ERB_B6.hdf5'        # B6 刺激文件（从服务器拷来）
OUTPUT_HDF5 = r'D:\ERB\nervegram_stim_ERB_B6.hdf5'  # 神经图输出
BEZ2018_PATH = r'D:\Events\pitchnet\packages\bez2018model'  # bez2018 包路径
# ============================================================

sys.path.append(BEZ2018_PATH)
import bez2018model_run_hdf5_dataset as nervegen

# 与 Josh 既有神经图逐项核对一致的规范配置（2026-08-14 实测 config_bez2018model 组）
BEZ2018_CONFIG = {
    'nervegram_dur': 0.050,
    'nervegram_fs': 20e3,          # 神经图时间采样率 20 kHz
    'buffer_start_dur': 0.070,
    'buffer_end_dur': 0.010,
    'pin_fs': 100e3,
    'pin_dBSPL_flag': 0,
    'pin_dBSPL': None,
    'species': 2,
    'bandwidth_scale_factor': 1.0,
    'cf_list': None,
    'num_cf': 100,                 # 100 fibers（ERB 模式，125–14000 Hz）
    'min_cf': 125.0,
    'max_cf': 14e3,
    'synapseMode': 0,
    'max_spikes_per_train': 1000,
    'num_spike_trains': 40,
    'cohc': 1.0,
    'cihc': 1.0,
    'IhcLowPass_cutoff': 3e3,      # IHC 低通截止 3 kHz（默认值）
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
    'squeeze_channel_dim': True,
}

# B6 特征列（与 BernsteinOxenham2005 同构）
FEATURE_KEYS = [
    'f0', 'f0_label', 'f0_log2', 'f0_lognormal',
    'f0_label_024', 'f0_label_048', 'f0_label_096',
    'f0_label_192', 'f0_label_384',
    'base_f0', 'delta_f0', 'low_harm',
    'max_audible_harm', 'min_audible_harm', 'phase_mode',
]

EXPECTED_N = 8470  # B6 刺激条数


def main():
    t0 = time.time()
    print("=" * 70)
    print("B6 神经图生成（规范配置：100 fibers / 20 kHz / 50 ms）")
    print("=" * 70)

    if not os.path.exists(INPUT_HDF5):
        print(f"❌ 输入文件不存在: {INPUT_HDF5}")
        sys.exit(1)

    # 1. 准备 bez2018 输入格式的临时文件（照 nervegram_example.py 的 prepare 步骤）
    pid = os.getpid()
    temp_file = os.path.join(tempfile.gettempdir(),
                             f'erb_B6_{pid}_{int(time.time())}.hdf5')
    try:
        with h5py.File(INPUT_HDF5, 'r') as src_f, h5py.File(temp_file, 'w') as dst_f:
            signal = src_f['tone_in_noise'][:]
            assert signal.shape[0] == EXPECTED_N, \
                f"条数不符: {signal.shape[0]} != {EXPECTED_N}"
            dst_f.create_dataset('/stimuli/signal_in_noise', data=signal)
            dst_f.create_dataset('/sr', data=[32000.0])
            n_feat = 0
            for k in FEATURE_KEYS:
                if k in src_f:
                    dst_f.create_dataset(f'/{k}', data=src_f[k][:])
                    n_feat += 1
            if 'config_tone' in src_f:
                grp = dst_f.create_group('/config_tone')
                for k in src_f['config_tone'].keys():
                    grp.create_dataset(k, data=src_f['config_tone'][k][:])
            print(f"✅ 临时输入就绪: {signal.shape}, {n_feat} 个特征")
    except Exception as e:
        print(f"❌ 准备数据失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. 调 bez2018 生成神经图
    try:
        print("🚀 开始神经图生成（8,470 条 × 100 fibers，耗时视机器而定）...")
        nervegen.run_dataset_generation(
            source_hdf5_filename=temp_file,
            dest_hdf5_filename=OUTPUT_HDF5,
            source_key_signal='/stimuli/signal_in_noise',
            source_key_signal_fs='/sr',
            kwargs_nervegram=BEZ2018_CONFIG,
            data_key_pair_list=[
                ('/nervegram_meanrates', 'nervegram_meanrates'),
                ('/pin_dBSPL', 'pin_dBSPL'),
            ],
            source_keys_to_copy=FEATURE_KEYS,
        )
    except Exception as e:
        print(f"❌ 神经图生成失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # 3. 验证输出规格
    with h5py.File(OUTPUT_HDF5, 'r') as f:
        ng = f['/nervegram_meanrates']
        ok_shape = ng.shape == (EXPECTED_N, 100, 1000)
        ok_fs = float(f['/nervegram_fs'][()]) == 20000.0
        cf = f['/cf_list'][:]
        ok_cf = cf.size == 100 and cf.min() >= 125.0 and cf.max() <= 14001.0
        print(f"  神经图形状: {ng.shape} {'✓' if ok_shape else '✗ 期望 (8470, 100, 1000)'}")
        print(f"  nervegram_fs: {f['/nervegram_fs'][()]} {'✓' if ok_fs else '✗'}")
        print(f"  cf_list: n={cf.size}, [{cf.min():.0f}, {cf.max():.0f}] "
              f"{'✓' if ok_cf else '✗'}")
        if not (ok_shape and ok_fs and ok_cf):
            print("❌ 验证未通过")
            sys.exit(1)

    print(f"✅ 完成: {OUTPUT_HDF5}（{time.time()-t0:.0f} 秒）")
    print("拷回服务器: /data/zyf/PE_model_ERB_estimation/data/nervegrams/")


if __name__ == "__main__":
    main()
