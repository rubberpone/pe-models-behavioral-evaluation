#!/usr/bin/env python3
"""
策略 B6: Josh CNN 校准集 — 验证 B 管道的可行性（原 B4 重命名为 B6）。

参数（README_strategy_B.md §5.3 B6）:
  base_f0 = {100, 150, 200, 250, 300}（精确取自 Josh Exp A）
  × low_harm = {1, 5, 10, 15, 20, 25, 30}
  × 2 phases (sine, rand) × 121 delta_f0 (linspace 0.94–1.06)
  = 8,470 条

用途与使用顺序（三时机）:
  ① 最先跑：生成后、全量推理前，用 B6 把新链（生成→推理→npz→JSON→
    2AFC→ERB）端到端跑通（Josh CNN 测出 ERB ≈ 人类 ERB, r > 0.8 才放行）
  ② A-B 等价性检验：10 模型全跑 B6，同条件 F0DL 与 Exp A 旧预测对比
  ③ 双带宽对照（待定项，§13）：B6 同时生成原版 + 2000 Hz 宽版，
    数据驱动 B1 的滤波器带宽决策

循环顺序与 Josh Exp A 生成器一致（lh → base_f0 → δ → ph），
便于与 Josh HDF5 逐行对照。

f0_label_* 列由 stimuli_f0_labels.py 生成后统一后处理。

用法:
    python generate_ERB_calibration.py data/stim_ERB_B6.hdf5
"""

import sys
import os
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_stimulus_utils import (
    initialize_hdf5_file, write_example_to_hdf5,
    get_bandpass_filter_frequency_response,
    shift_bandpass_filter_frequency_response,
    bernox2005_bandpass_complex_tone,
    modified_uniform_masking_noise,
)


def generate_calibration_dataset(
    hdf5_filename,
    fs=32000, dur=0.150,
    base_f0_list=None,
    low_harm_list=None,
    phase_modes=None,
    delta_f0_min=0.94, delta_f0_max=1.06, delta_f0_n=121,
    highpass_filter_cutoff=2500.0, lowpass_filter_cutoff=3500.0,
    filter_order=4,
    threshold_dBSPL=33.3, component_dBSL=15.0,
    noise_dBHzSPL=15.0, noise_attenuation_start=600.0,
    noise_attenuation_slope=2.0,
    disp_step=500
):
    """生成 Josh CNN 校准刺激集（策略 B6，Josh Exp A 精确子集）."""
    if base_f0_list is None:
        base_f0_list = [100.0, 150.0, 200.0, 250.0, 300.0]
    if low_harm_list is None:
        low_harm_list = [1, 5, 10, 15, 20, 25, 30]
    if phase_modes is None:
        phase_modes = ['sine', 'rand']

    phase_mode_encoding = {'sine': 0, 'rand': 1, 'sch': 2, 'cos': 3, 'alt': 4}
    phase_mode_decoding = {v: k for k, v in phase_mode_encoding.items()}
    unique_ph_list = np.array([phase_mode_encoding[p] for p in phase_modes])
    delta_f0_list = np.linspace(delta_f0_min, delta_f0_max, delta_f0_n)

    N = len(base_f0_list) * len(low_harm_list) * len(unique_ph_list) * len(delta_f0_list)
    print(f"[B6 Calibration] Total stimuli: {N:,}")

    baseline_freq_response = get_bandpass_filter_frequency_response(
        highpass_filter_cutoff, lowpass_filter_cutoff,
        fs=fs, order=filter_order)

    config_dict = {
        'config_tone/fs': fs, 'config_tone/dur': dur,
        'config_tone/base_f0_list': str(list(base_f0_list)),
        'config_tone/low_harm_list': str(list(low_harm_list)),
        'config_tone/delta_f0_min': delta_f0_min,
        'config_tone/delta_f0_max': delta_f0_max,
        'config_tone/delta_f0_n': delta_f0_n,
        'config_tone/unshifted_highpass_filter_cutoff': highpass_filter_cutoff,
        'config_tone/unshifted_lowpass_filter_cutoff': lowpass_filter_cutoff,
        'config_tone/unshifted_filter_order': filter_order,
        'config_tone/threshold_dBSPL': threshold_dBSPL,
        'config_tone/component_dBSL': component_dBSL,
        'config_tone/noise_dBHzSPL': noise_dBHzSPL,
        'config_tone/noise_attenuation_start': noise_attenuation_start,
        'config_tone/noise_attenuation_slope': noise_attenuation_slope,
    }
    config_kps = [(k, k) for k in config_dict.keys()]

    itrN = 0
    for lh in low_harm_list:
        for base_f0 in base_f0_list:
            # 与 Josh Exp A 生成器一致的平移
            desired_fl = base_f0 * lh
            fixed_freq_response = shift_bandpass_filter_frequency_response(
                desired_fl, -1 * component_dBSL, fs=fs,
                unshifted_passband=None,
                frequency_response_in_dB=baseline_freq_response)

            for delta_f0 in delta_f0_list:
                for ph in unique_ph_list:
                    f0 = base_f0 * delta_f0
                    signal, audible_harmonic_numbers = bernox2005_bandpass_complex_tone(
                        f0, fs, dur,
                        frequency_response_in_dB=fixed_freq_response,
                        threshold_dBSPL=threshold_dBSPL,
                        component_dBSL=component_dBSL,
                        strict_low_harm=None,
                        strict_audible_harm=False,
                        phase_mode=phase_mode_decoding[int(ph)])
                    noise = modified_uniform_masking_noise(
                        fs, dur, dBHzSPL=noise_dBHzSPL,
                        attenuation_start=noise_attenuation_start,
                        attenuation_slope=noise_attenuation_slope)
                    tone_in_noise = (signal + noise).astype(np.float32)

                    data_dict = {
                        'tone_in_noise': tone_in_noise,
                        'f0': float(f0), 'base_f0': float(base_f0),
                        'delta_f0': float(delta_f0),
                        'phase_mode': int(ph), 'low_harm': int(lh),
                        'min_audible_harm': int(np.min(audible_harmonic_numbers)),
                        'max_audible_harm': int(np.max(audible_harmonic_numbers)),
                    }
                    data_dict.update(config_dict)

                    if itrN == 0:
                        data_kps = [(k, k) for k in data_dict
                                   if not k.startswith('config_tone/')]
                        initialize_hdf5_file(hdf5_filename, N, data_dict,
                                            data_key_pair_list=data_kps,
                                            config_key_pair_list=config_kps)
                        hdf5_f = h5py.File(hdf5_filename, 'r+')

                    write_example_to_hdf5(hdf5_f, data_dict, itrN,
                                         data_key_pair_list=data_kps)
                    if itrN % disp_step == 0:
                        print(f'  [{itrN}/{N}] lh={lh}, base_f0={base_f0}, '
                              f'audible=[{audible_harmonic_numbers[0]},'
                              f'{audible_harmonic_numbers[-1]}]')
                    itrN += 1

    hdf5_f.close()
    print(f"[B6] Done: {hdf5_filename}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_ERB_calibration.py <hdf5_filename>")
        sys.exit(1)
    generate_calibration_dataset(sys.argv[1])
