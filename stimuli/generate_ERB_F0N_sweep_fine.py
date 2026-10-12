#!/usr/bin/env python3
"""
策略 B1_fine: F0-N 二维扫参刺激生成——categorical perception 细网格版
（反馈⑥，revision_plan_advisor_2026-08-19.md §7，2026-08-19 用户启动令）。

本文件是 generate_ERB_F0N_sweep.py（B1 生成器）的**副本**，唯一差异 =
base_f0 网格：45 点 geomspace(80,1000) → **177 点 MIDI A4=440 锚
0.25 半音网格**（440×2^(k/48)，k=−119..57，78.88–1002.2 Hz；
k≡0 (mod 4) 即 MIDI 乐音）。其余全部同口径：

  × low_harm N = 1..min(40, ⌊16000/F0⌋−1) + 低 F0 行稀疏补全
    {45,55,70,90,115,145,185}（各受 ⌊16000/F0⌋−1 截断；顶角退化条件
    与 B1 同规则——空 audible 写 −1 哨兵，分析层跳过）
  × 2 phases (sine, rand) × 121 delta_f0 (linspace 0.94–1.06)
  滤波器 4 阶高通(2500)×4 阶低通(3500) 级联平移（−15 dB 截止 =
  N×base_f0）、δ 集、相位、掩蔽噪声、谐波电平（滤前 48.3 dB SPL）
  全部原样继承（B1 版 docstring 的顶角退化分析同样适用于本网格）。

设计定稿（2026-08-19，用户）：
  - 锚点 MIDI A4=440：整数半音 = 真实乐音，与训练 GT/MIDI 标注对齐，
    锯齿与网格线对齐（80 Hz 锚的 0.51 半音相位偏移问题不复存在）
  - N 全保留（不牺牲 N 细粒度）；单模型串行执行（每模型完成后停下
    汇报，JSON+npz 由用户自行备份/删除）
  - 磁盘账：~1.57M 条、HDF5 ~28 GB（一次性、全模型共享）

f0_label_* 列由 stimuli_f0_labels.py 生成后统一后处理（与 Josh 工作流一致）。

用法:
    python generate_ERB_F0N_sweep_fine.py data/stim_ERB_B1_fine.hdf5
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

SPARSE_COMPLETION = (45, 55, 70, 90, 115, 145, 185)  # 低 F0 行稀疏高 N 补全

# MIDI A4=440 锚、0.25 半音步长：k 范围覆盖 80–1000 Hz
FINE_K_START = -119
FINE_K_END = 57  # inclusive


def generate_ERB_F0N_sweep_dataset(
    hdf5_filename,
    fs=32000, dur=0.150,
    phase_modes=None,
    low_harm_max=40,
    sparse_completion=SPARSE_COMPLETION,
    base_f0_min=80.0, base_f0_max=1000.0, base_f0_n=45,
    delta_f0_min=0.94, delta_f0_max=1.06, delta_f0_n=121,
    highpass_filter_cutoff=2500.0, lowpass_filter_cutoff=3500.0,
    filter_order=4,
    threshold_dBSPL=33.3, component_dBSL=15.0,
    noise_dBHzSPL=15.0, noise_attenuation_start=600.0,
    noise_attenuation_slope=2.0,
    disp_step=2000
):
    """生成 F0-N 二维扫参刺激集（B1_fine：MIDI 锚 0.25 半音细网格）。"""
    if phase_modes is None:
        phase_modes = ['sine', 'rand']

    phase_mode_encoding = {'sine': 0, 'rand': 1, 'sch': 2, 'cos': 3, 'alt': 4}
    phase_mode_decoding = {v: k for k, v in phase_mode_encoding.items()}

    unique_ph_list = np.array([phase_mode_encoding[p] for p in phase_modes])
    # 与 B1 的唯一差异：base_f0 网格（MIDI 440 锚 0.25 半音，177 点）
    base_f0_list = 440.0 * np.power(2.0,
                                    np.arange(FINE_K_START,
                                              FINE_K_END + 1) / 48.0)
    base_f0_min = float(base_f0_list[0])
    base_f0_max = float(base_f0_list[-1])
    base_f0_n = len(base_f0_list)
    delta_f0_list = np.linspace(delta_f0_min, delta_f0_max, delta_f0_n)

    # 每行的 low_harm 条件表：1..min(40, ⌊16000/F0⌋−1) + 稀疏补全
    # （各受 Nyquist 帽截断；帽取 ⌊16000/F0⌋−1 的原因见 B1 版 docstring
    # —— 对任何 δ∈[0.94,1.06] 都保证 ≥1 个可听谐波）
    low_harm_lists = {}
    for base_f0 in base_f0_list:
        n_max_nyq = int(np.floor((fs / 2) / base_f0))
        base_lh = np.arange(1, min(low_harm_max, n_max_nyq - 1) + 1)
        sparse_lh = np.array([n for n in sparse_completion if n <= n_max_nyq - 1])
        low_harm_lists[base_f0] = np.array(
            sorted(set(list(base_lh) + list(sparse_lh))))

    N = sum(len(lh) for lh in low_harm_lists.values())
    N = N * len(unique_ph_list) * len(delta_f0_list)
    print(f"[B1_fine] base_f0 网格: {base_f0_n} 点, "
          f"{base_f0_min:.2f}–{base_f0_max:.2f} Hz, "
          f"步长 0.25 半音 (MIDI A4=440 锚)")
    print(f"[B1_fine] Total stimuli: {N:,}")

    # 基线滤波器只计算一次；每个条件整体平移，宽度不随 F0 调整
    baseline_freq_response = get_bandpass_filter_frequency_response(
        highpass_filter_cutoff, lowpass_filter_cutoff,
        fs=fs, order=filter_order)

    data_dict = {
        'config_tone/fs': fs, 'config_tone/dur': dur,
        'config_tone/low_harm_min': 1,
        'config_tone/low_harm_max': low_harm_max,
        'config_tone/sparse_completion': str(list(sparse_completion)),
        'config_tone/base_f0_min': base_f0_min,
        'config_tone/base_f0_max': base_f0_max,
        'config_tone/base_f0_n': base_f0_n,
        'config_tone/base_f0_grid': 'midi_a440_qsemitone',  # B1_fine 新键
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
    config_key_pair_list = [(k, k) for k in data_dict.keys()]

    itrN = 0
    for base_f0 in base_f0_list:
        for lh in low_harm_lists[base_f0]:
            # 滤波器平移：−15 dB 低频截止点 = low_harm × base_f0
            desired_fl = base_f0 * lh
            desired_fl_gain_in_dB = -1 * component_dBSL
            fixed_freq_response = shift_bandpass_filter_frequency_response(
                desired_fl, desired_fl_gain_in_dB, fs=fs,
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

                    # 空 audible 哨兵（同 B1 规则）：个别 (F0, lh, δ) 组合的
                    # 谐波网格整体滑出滤波器 −15dB 轮廓——刺激按 Josh 原样
                    # 生成（纯噪声），元数据写 −1 哨兵，分析层跳过该行。
                    if len(audible_harmonic_numbers) == 0:
                        min_audible_harm = -1
                        max_audible_harm = -1
                    else:
                        min_audible_harm = int(np.min(audible_harmonic_numbers))
                        max_audible_harm = int(np.max(audible_harmonic_numbers))

                    noise = modified_uniform_masking_noise(
                        fs, dur, dBHzSPL=noise_dBHzSPL,
                        attenuation_start=noise_attenuation_start,
                        attenuation_slope=noise_attenuation_slope)

                    tone_in_noise = (signal + noise).astype(np.float32)

                    data_dict.update({
                        'tone_in_noise': tone_in_noise,
                        'f0': float(f0), 'base_f0': float(base_f0),
                        'delta_f0': float(delta_f0),
                        'phase_mode': int(ph), 'low_harm': int(lh),
                        'min_audible_harm': min_audible_harm,
                        'max_audible_harm': max_audible_harm,
                    })

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
                        print(f'  [{itrN}/{N}] base_f0={base_f0:.1f}, '
                              f'lh={lh}, f0={f0:.1f}, audible=['
                              f'{audible_harmonic_numbers[0]},'
                              f'{audible_harmonic_numbers[-1]}]')
                    itrN += 1

    hdf5_f.close()
    print(f"[B1_fine] Done: {hdf5_filename}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_ERB_F0N_sweep_fine.py <hdf5_filename>")
        sys.exit(1)
    generate_ERB_F0N_sweep_dataset(sys.argv[1])
