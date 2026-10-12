#!/usr/bin/env python3
"""
策略 B1: F0-N 二维扫参刺激生成（Exp A 扩展版）。

参数空间（README_strategy_B.md §5.3 B1）:
  45 base_f0 (80–1000 Hz, np.geomspace, ~1 半音步长)
  × low_harm N = 1..min(40, ⌊16000/F0⌋−1) + 低 F0 行稀疏补全 {45,55,70,90,115,145,185}
    （各受 ⌊16000/F0⌋−1 截断；见下方"顶角退化条件"注）
  × 2 phases (sine, rand) × 121 delta_f0 (linspace 0.94–1.06)
  = 397,364 条

顶角退化条件（环节 0/1 发现的两层边缘，2026-08-13/14）:
  ① lh×F0 = 16000 恰顶存储 Nyquist 时，滤波器 −15dB 低截止在 16 kHz，
    可听谐波区间 [16000, ~19800] 全部落在合成范围之外 → audible 恒为空。
    Josh Exp A 不会遇到（max 30×300 = 9000 Hz）。修复：帽取 ⌊16000/F0⌋−1。
    17 行 F0≥399 Hz 各失去顶角条件（F0=1000 行 lh 1..15），二维图梯形
    顶点由 (1000,16) 内移到 (1000,15)——边界线本身是数学线，不是数据点。
  ② ⌊−1⌋ 帽仍留 1 个残角：frac(16000/F0)<0.06 的行（F0=841.8,
    frac=0.0056）在 δ∈(q/⌊q⌋−1… 窗口内 h_max 掉落两级 → 3 个 δ
    {1.056,1.057,1.058} × 2 相位 = 6 条刺激 audible 为空（全 121 δ ×
    1,642 条件实证扫描确认，仅此 3 个组合）。处置：哨兵方案——生成器
    照常生成（纯噪声刺激，Josh 原样），min/max_audible_harm 写 −1 哨兵，
    分析层跳过这 6 行（不进 2AFC 配对），二维图点 (841.8, 18) 保留
    （该条件 236/242 刺激正常）。

与 Josh Exp A 的差异（仅三处，其余照猫画虎）:
  ① base_f0 10 点 (100–300) → 45 点 (80–1000)
  ② low_harm 上限 30 → min(40, ⌊16000/F0⌋−1)
  ③ 低 F0 行稀疏高 N 补全 {45..185}（窄 ERB 膝点覆盖，如 PESTO N_knee(80)≈62）
  滤波器 4 阶高通(2500)×4 阶低通(3500) 级联平移（−15 dB 截止 = N×base_f0）、
  δ 集、相位、掩蔽噪声、谐波电平（滤前 48.3 dB SPL）全部原样继承。

f0_label_* 列由 stimuli_f0_labels.py 生成后统一后处理（与 Josh 工作流一致）。

用法:
    python generate_ERB_F0N_sweep.py data/stim_ERB_B1.hdf5
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
    """生成 F0-N 二维扫参刺激集（策略 B1，Josh Exp A 生成方式原样继承）."""
    if phase_modes is None:
        phase_modes = ['sine', 'rand']

    phase_mode_encoding = {'sine': 0, 'rand': 1, 'sch': 2, 'cos': 3, 'alt': 4}
    phase_mode_decoding = {v: k for k, v in phase_mode_encoding.items()}

    unique_ph_list = np.array([phase_mode_encoding[p] for p in phase_modes])
    base_f0_list = np.geomspace(base_f0_min, base_f0_max, base_f0_n)
    delta_f0_list = np.linspace(delta_f0_min, delta_f0_max, delta_f0_n)

    # 每行的 low_harm 条件表：1..min(40, ⌊16000/F0⌋−1) + 稀疏补全
    # （各受 Nyquist 帽截断）。帽取 ⌊16000/F0⌋−1 而非 ⌊16000/F0⌋ 的原因：
    # lh×F0 = 16000 恰顶 Nyquist 时，滤波器 −15dB 低截止在 16 kHz，
    # 可听谐波区间 [16000, ~19800] 全部落在合成范围之外 → audible 恒为空
    # （纯噪声刺激，F0DL 无意义）。Josh Exp A 不会遇到（max 30×300=9000 Hz），
    # 是本扩展区的新边缘：数学上可证 ⌊16000/F0⌋−1 对任何 δ∈[0.94,1.06]
    # 都保证 ≥1 个可听谐波（顶部谐波间隙 ≤1.06F0 < 滤波边缘距 Nyquist ≥F0）。
    low_harm_lists = {}
    for base_f0 in base_f0_list:
        n_max_nyq = int(np.floor((fs / 2) / base_f0))
        base_lh = np.arange(1, min(low_harm_max, n_max_nyq - 1) + 1)
        sparse_lh = np.array([n for n in sparse_completion if n <= n_max_nyq - 1])
        low_harm_lists[base_f0] = np.array(
            sorted(set(list(base_lh) + list(sparse_lh))))

    N = sum(len(lh) for lh in low_harm_lists.values())
    N = N * len(unique_ph_list) * len(delta_f0_list)
    print(f"[B1] Total stimuli: {N:,}")

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

                    # 空 audible 哨兵：个别 (F0, lh, δ) 组合的谐波网格整体滑出
                    # 滤波器 −15dB 轮廓（全 121 δ 实证扫描：仅 (841.8, 18,
                    # δ∈{1.056..1.058}) 3 个 δ × 2 相位 = 6 条，占 0.0015%）。
                    # 刺激本身按 Josh 原样生成（纯噪声），元数据写 −1 哨兵，
                    # 分析层跳过这 6 行（不进 2AFC 配对），二维图点保留。
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
    print(f"[B1] Done: {hdf5_filename}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_ERB_F0N_sweep.py <hdf5_filename>")
        sys.exit(1)
    generate_ERB_F0N_sweep_dataset(sys.argv[1])
