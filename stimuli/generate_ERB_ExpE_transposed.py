#!/usr/bin/env python3
"""
策略 B5: Exp E 转置音扩展 — 移调音扫参刺激生成。

参数空间（README_strategy_B.md §5.3 B5）:
  11 类（纯音 fc=0 + 10 转置载波 fc∈{1000,1500,2000,3000,4000,5000,
    6350,8000,10080,12000}——原 3 个 {4000,6350,10080} 保 A5/Josh 可比，
    新增 7 个按需填充，整体近似对数均匀）
  × 包络 F0 密集格 1/3072 oct、80–min(1000, 0.2×fc)（纯音 = 80–1000 全格；
    0.2fc 是包络低通截止约束，扩展至 1000 是科学必需——原 80–320
    对任何载波都到不了膝点）
  = ~103,352 条

关键口径:
  - 无 δ——F0DL 靠密集格 pairwise（±2.7 semitones 窗口，参照点 = 共享
    45 网格），复用 Exp A 的 2AFC 机器（Josh 代码实测 f0dl_transposed_tones
    直接复用 f0dl_bernox 全套函数），B5 不另设计
  - 70 dB SPL、UMNm 噪声（纯音也加——Josh 特意说明）、offset_start
    随机起始相位、包络经 4 阶 Butterworth filtfilt 低通（0.2fc）——原样
  - 正确答案 = 包络频率

f0_label_* 列由 stimuli_f0_labels.py 生成后统一后处理。

用法:
    python generate_ERB_ExpE_transposed.py data/stim_ERB_B5.hdf5
"""

import sys
import os
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_stimulus_utils import (
    initialize_hdf5_file, write_example_to_hdf5,
    get_Oxenham2004_transposed_tone,
    modified_uniform_masking_noise,
)

# 纯音 + 10 转置载波（原 3 个保可比性，新增 7 个按需填充）
CARRIER_LIST = [0.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0,
                5000.0, 6350.0, 8000.0, 10080.0, 12000.0]


def envelope_f0_range(f_carrier, f0_min=80.0, f0_max=1000.0,
                      step_size_in_octaves=1/3072):
    """包络 F0 密集格：80–min(1000, 0.2×fc)；纯音 (fc=0) = 80–1000 全格."""
    if f_carrier > 0:
        f0_max = min(f0_max, 0.2 * f_carrier)
    f0s = np.arange(0, np.log2(f0_max / f0_min), step_size_in_octaves)
    return f0_min * np.power(2, f0s)


def generate_ExpE_ERB_transposed_dataset(
    hdf5_filename,
    fs=32000, dur=0.150, buffer_dur=1.0,
    carrier_list=None,
    f0_min=80.0, f0_max=1000.0, step_size_in_octaves=1/3072,
    dBSPL=70.0,
    noise_dBHzSPL=15.0, noise_attenuation_start=600.0,
    noise_attenuation_slope=2.0,
    disp_step=1000
):
    """生成移调音扫参刺激集（策略 B5，Exp E 扩展版）."""
    if carrier_list is None:
        carrier_list = CARRIER_LIST

    list_f_envelope_per_carrier = {
        fc: envelope_f0_range(fc, f0_min=f0_min, f0_max=f0_max,
                              step_size_in_octaves=step_size_in_octaves)
        for fc in carrier_list
    }
    N = sum(len(v) for v in list_f_envelope_per_carrier.values())
    print(f"[B5] Total stimuli: {N:,}")
    for fc in carrier_list:
        print(f'      fc={fc:>7.0f}: {len(list_f_envelope_per_carrier[fc]):,} '
              f'envelopes ('
              f'{list_f_envelope_per_carrier[fc][0]:.2f}–'
              f'{list_f_envelope_per_carrier[fc][-1]:.2f} Hz)')

    config_dict = {
        'sr': fs,
        'config_tone/fs': fs,
        'config_tone/dur': dur,
        'config_tone/dBSPL': dBSPL,
        'config_tone/noise_dBHzSPL': noise_dBHzSPL,
        'config_tone/noise_attenuation_start': noise_attenuation_start,
        'config_tone/noise_attenuation_slope': noise_attenuation_slope,
    }
    config_key_pair_list = [(k, k) for k in config_dict.keys()]

    itrN = 0
    for f_carrier in carrier_list:
        for f_envelope in list_f_envelope_per_carrier[f_carrier]:
            y = get_Oxenham2004_transposed_tone(
                f_carrier, f_envelope, fs=fs, dur=dur, buffer_dur=buffer_dur,
                dBSPL=dBSPL, offset_start=True,
                lowpass_filter_envelope=True)
            # UMNm 噪声（纯音也加——Josh 特意说明）
            noise = modified_uniform_masking_noise(
                fs, dur, dBHzSPL=noise_dBHzSPL,
                attenuation_start=noise_attenuation_start,
                attenuation_slope=noise_attenuation_slope)
            signal_in_noise = (y + noise).astype(np.float32)

            data_dict = {
                'stimuli/signal': y.astype(np.float32),
                'stimuli/signal_in_noise': signal_in_noise,
                'f0': float(f_envelope),
                'f_envelope': float(f_envelope),
                'f_carrier': float(f_carrier),
            }
            data_dict.update(config_dict)

            if itrN == 0:
                data_key_pair_list = [
                    (k, k) for k in data_dict
                    if not k.startswith('config_tone/') and k != 'sr'
                ]
                initialize_hdf5_file(hdf5_filename, N, data_dict,
                                    data_key_pair_list=data_key_pair_list,
                                    config_key_pair_list=config_key_pair_list)
                hdf5_f = h5py.File(hdf5_filename, 'r+')

            write_example_to_hdf5(hdf5_f, data_dict, itrN,
                                 data_key_pair_list=data_key_pair_list)
            if itrN % disp_step == 0:
                print(f'  [{itrN}/{N}] fc={f_carrier:.0f}, '
                      f'f_envelope={f_envelope:.2f}')
            itrN += 1

    hdf5_f.close()
    print(f"[B5] Done: {hdf5_filename}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_ERB_ExpE_transposed.py <hdf5_filename>")
        sys.exit(1)
    generate_ExpE_ERB_transposed_dataset(sys.argv[1])
