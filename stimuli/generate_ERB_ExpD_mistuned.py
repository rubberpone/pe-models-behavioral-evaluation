#!/usr/bin/env python3
"""
策略 B4: Exp D 失谐扩展 — 单谐波失谐扫参刺激生成。

参数空间（README_strategy_B.md §5.3 B4）:
  45 f0（共享网格，80–1000 Hz）
  × mistuned_harm H = 1..12 连续整数（HDF5 超集口径，无缺口；
    12×1000 = 12 kHz < 16 kHz，全 F0 无约束）
  × mistuned_pct = linspace(-8, 8, 13)（对称，镜像 HDF5 超集设计）
  × 5 个 ±4% log-uniform 浮动变体（含中心值，锚点聚合）
  = 35,100 条

关键口径:
  - 12 个等幅谐波（60 dB SPL/谐波）sine 相位，单谐波失谐 (1+pct/100)×
    ——每刺激恒有 12 个谐波（≥3 规则不适用：Exp D 是 N 维实验非频段实验）
  - 无噪声（HDF5 实测 stimuli 组只有 signal；评估器用键 stimuli/signal）
  - 梯形补全已否决（12 谐波是范式定义，保 A4 基线可比性；
    窄 ERB 衰减点由 B1 补全跨实验承担）

f0_label_* 列由 stimuli_f0_labels.py 生成后统一后处理。

用法:
    python generate_ERB_ExpD_mistuned.py data/stim_ERB_B4.hdf5
"""

import sys
import os
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_stimulus_utils import (
    initialize_hdf5_file, write_example_to_hdf5,
    complex_tone,
)


def generate_ExpD_ERB_mistuned_dataset(
    hdf5_filename,
    fs=32000, dur=0.150,
    f0_min=80.0, f0_max=1000.0, f0_n=45,
    low_harm=1, upp_harm=12,
    mistuned_pct_min=-8.0, mistuned_pct_max=8.0, mistuned_pct_n=13,
    n_jitter=5, jitter_pct=4.0,
    harmonic_dBSPL=60.0, phase_mode='sine',
    disp_step=500
):
    """生成单谐波失谐扫参刺激集（策略 B4，镜像 Josh HDF5 超集口径）."""
    f0_nominal_list = np.geomspace(f0_min, f0_max, f0_n)
    list_mistuned_harm = np.arange(low_harm, upp_harm + 1, dtype=int)
    list_mistuned_pct = np.linspace(mistuned_pct_min, mistuned_pct_max,
                                    mistuned_pct_n)

    # ±4% log-uniform 浮动变体（含中心值；恢复 Josh 原设计 178 变体 → 5 个）
    jitter_ratio = 1.0 + jitter_pct / 100.0
    f0_variants = np.geomspace(f0_nominal_list / jitter_ratio,
                               f0_nominal_list * jitter_ratio, n_jitter).T

    N = f0_n * len(list_mistuned_harm) * len(list_mistuned_pct) * n_jitter
    print(f"[B4] Total stimuli: {N:,}")

    # 12 个等幅谐波（60 dB SPL/谐波），sine 相位
    harmonic_numbers = np.arange(low_harm, upp_harm + 1, dtype=int)
    amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL / 20)) * np.ones_like(
        harmonic_numbers, dtype=np.float64)

    config_dict = {
        'config_tone/fs': fs, 'config_tone/dur': dur,
        'config_tone/harmonic_dBSPL': harmonic_dBSPL,
        'config_tone/phase_mode': 0,  # sine
        'config_tone/low_harm': low_harm,
        'config_tone/upp_harm': upp_harm,
        'config_tone/f0_min': f0_min, 'config_tone/f0_max': f0_max,
        'config_tone/f0_n': f0_n,
        'config_tone/n_jitter': n_jitter,
        'config_tone/jitter_pct': jitter_pct,
    }
    config_key_pair_list = [(k, k) for k in config_dict.keys()]

    itrN = 0
    for i_f0, f0_nominal in enumerate(f0_nominal_list):
        for f0 in f0_variants[i_f0]:
            for mistuned_harm in list_mistuned_harm:
                for mistuned_pct in list_mistuned_pct:
                    harmonic_freqs = f0 * harmonic_numbers
                    mistuned_index = harmonic_numbers == mistuned_harm
                    harmonic_freqs[mistuned_index] = (
                        (1.0 + mistuned_pct / 100.0) * harmonic_freqs[mistuned_index])
                    signal = complex_tone(f0, fs, dur, harmonic_numbers=None,
                                         frequencies=harmonic_freqs,
                                         amplitudes=amplitudes,
                                         phase_mode=phase_mode)

                    data_dict = {
                        'stimuli/signal': signal.astype(np.float32),
                        'f0': float(f0),
                        'f0_nominal': float(f0_nominal),
                        'mistuned_harm': int(mistuned_harm),
                        'mistuned_pct': float(mistuned_pct),
                    }
                    data_dict.update(config_dict)

                    if itrN == 0:
                        data_key_pair_list = [
                            (k, k) for k in data_dict
                            if not k.startswith('config_tone/')
                        ]
                        initialize_hdf5_file(hdf5_filename, N, data_dict,
                                            data_key_pair_list=data_key_pair_list,
                                            config_key_pair_list=config_key_pair_list)
                        hdf5_f = h5py.File(hdf5_filename, 'r+')

                    write_example_to_hdf5(hdf5_f, data_dict, itrN,
                                         data_key_pair_list=data_key_pair_list)
                    if itrN % disp_step == 0:
                        print(f'  [{itrN}/{N}] f0_nominal={f0_nominal:.1f}, '
                              f'H={mistuned_harm}, pct={mistuned_pct}')
                    itrN += 1

    hdf5_f.close()
    print(f"[B4] Done: {hdf5_filename}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_ERB_ExpD_mistuned.py <hdf5_filename>")
        sys.exit(1)
    generate_ExpD_ERB_mistuned_dataset(sys.argv[1])
