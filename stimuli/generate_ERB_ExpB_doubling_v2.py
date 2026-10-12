#!/usr/bin/env python3
"""
策略 B2-v2: Exp B 修缮版 — 频带集 = 12 v1 基础带 + 68 谐波锚定带
（2026-08-19，revision_plan_advisor_2026-08-19.md §2 定稿，log_revision §1）。

参数空间（v2 扩展）:
  45 f0（共享网格，80–1000 Hz）
  × 80 band（12 v1 基础带（保 A1/Josh 可比）
    + 68 谐波锚定带（band = [h1·F0, (h1+k−1)·F0]，h1∈{1..12} 步 0.25、
      k∈{3,4,5,6}——带内谐波窗 3–6 个，谐波帽 ≤6；读
      results/analysis/strategy_B/b2_v2_design.json，由
      analysis_strategy_B/design_B2_v2_bands.py 确定性贪心生成））
  × 2 phases（alt(4) + sine(0)）
  × 7 个 ±4% log-uniform 浮动变体（含中心值，锚点聚合）
  = 80 × 45 × 2 × 7 = 50,400 条（v1 为 7,560）

关键口径（与 v1 完全一致，仅 BAND_LIST 不同）:
  - 滤波器 = 高通×低通级联、order=8、无 0dB 重缩放（AltPhaseHarmonics 版，
    rescale_max=False——与 Bernstein/Exp A 版不同）
  - 谐波电平 = 通带内每谐波平坦 50 dB SPL + 滤波器响应（无阈值结构）
  - 分析时每个 (band, 标称 F0) 的 ratio 按 7 个浮动变体取 median；
    二维图点仅在 floor(fh/F0)−ceil(fl/F0)+1 ≥ 3 时有效；
    v2 分析转折点提取另加谐波帽 n_harm ≤ 6（锚定带口径）

f0_label_* 列由 stimuli_f0_labels.py 生成后统一后处理。

用法:
    python generate_ERB_ExpB_doubling_v2.py data/stim_ERB_B2_v2.hdf5
"""

import json
import sys
import os
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_stimulus_utils import (
    initialize_hdf5_file, write_example_to_hdf5,
    get_bandpass_filter_frequency_response,
    complex_tone, modified_uniform_masking_noise,
)

# v1 的 12 基础带（锚定带 + 等比系列 + 高 F0 补充带，与 v1 完全一致）
BAND_LIST_V1 = [
    (125.0, 625.0), (1375.0, 1875.0), (3900.0, 5400.0),  # Josh 原 band
    (800.0, 1200.0), (1200.0, 1800.0), (1800.0, 2700.0),
    (2700.0, 4050.0), (4050.0, 6075.0), (6075.0, 9112.5),
    (9112.5, 13668.75), (13668.75, 16000.0),               # i=6,7（上缘 16 kHz 截断）
    (5000.0, 7500.0),                                     # 高 F0 行低 N 侧补充
]

# 68 谐波锚定带：读定稿清单（design_B2_v2_bands.py 输出）
_DESIGN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "results", "analysis", "strategy_B",
                       "b2_v2_design.json")
with open(_DESIGN) as _fh:
    _anchored = json.load(_fh)["bands"]
assert len(_anchored) == 68, f"锚定带数异常: {len(_anchored)}"
BAND_LIST = [(float(b[0]), float(b[1])) for b in BAND_LIST_V1 + _anchored]


def generate_ExpB_ERB_doubling_dataset(
    hdf5_filename,
    fs=32000, dur=0.150,
    f0_min=80.0, f0_max=1000.0, f0_n=45,
    band_list=None,
    n_jitter=7, jitter_pct=4.0,
    filter_order=8, passband_component_dBSPL=50.0,
    noise_dBHzSPL=15.0, noise_attenuation_start=600.0,
    noise_attenuation_slope=2.0,
    disp_step=500
):
    """生成连续 filter band 扫参的 ALT 刺激集（策略 B2）."""
    if band_list is None:
        band_list = BAND_LIST

    phase_mode_encoding = {'sine': 0, 'rand': 1, 'sch': 2, 'cos': 3, 'alt': 4}
    phase_modes = [4, 0]  # alt, sine（ratio≈1 对照基线）
    phase_names = {v: k for k, v in phase_mode_encoding.items()}

    f0_nominal_list = np.geomspace(f0_min, f0_max, f0_n)

    # ±4% log-uniform 浮动变体（含中心值；奇数个点 → 中心恰为标称 F0）
    jitter_ratio = 1.0 + jitter_pct / 100.0
    f0_variants = np.geomspace(f0_nominal_list / jitter_ratio,
                               f0_nominal_list * jitter_ratio, n_jitter).T

    N = f0_n * len(band_list) * len(phase_modes) * n_jitter
    print(f"[B2] Total stimuli: {N:,}")

    # Josh Exp B 版滤波器：无 0dB 重缩放（跟代码，不跟论文）
    # fh = fs/2 时低通为无操作；数值上略退 1 Hz 以便 scipy 设计
    list_filters = []
    for (fl, fh) in band_list:
        fh_eff = min(fh, fs / 2 - 1.0)
        freq_response = get_bandpass_filter_frequency_response(
            fl, fh_eff, fs=fs, order=filter_order, rescale_max=False)
        list_filters.append({'fl': fl, 'fh': fh, 'freq_response': freq_response})

    config_dict = {
        'config_tone/fs': fs, 'config_tone/dur': dur,
        'config_tone/filter_order': filter_order,
        'config_tone/passband_component_dBSPL': passband_component_dBSPL,
        'config_tone/noise_dBHzSPL': noise_dBHzSPL,
        'config_tone/noise_attenuation_start': noise_attenuation_start,
        'config_tone/noise_attenuation_slope': noise_attenuation_slope,
        'config_tone/f0_min': f0_min, 'config_tone/f0_max': f0_max,
        'config_tone/f0_n': f0_n,
        'config_tone/n_jitter': n_jitter,
        'config_tone/jitter_pct': jitter_pct,
    }
    config_key_pair_list = [(k, k) for k in config_dict.keys()]

    itrN = 0
    for filter_dict in list_filters:
        freq_response = filter_dict['freq_response']
        for i_f0, f0_nominal in enumerate(f0_nominal_list):
            for f0 in f0_variants[i_f0]:
                for ph in phase_modes:
                    harmonic_freqs = np.arange(f0, fs/2, f0)
                    harmonic_dBSPL = passband_component_dBSPL + freq_response(harmonic_freqs)
                    amplitudes = 20e-6 * np.power(10.0, (harmonic_dBSPL / 20.0))
                    signal = complex_tone(f0, fs, dur,
                                         harmonic_numbers=harmonic_freqs / f0,
                                         amplitudes=amplitudes,
                                         phase_mode=phase_names[ph])
                    noise = modified_uniform_masking_noise(
                        fs, dur, dBHzSPL=noise_dBHzSPL,
                        attenuation_start=noise_attenuation_start,
                        attenuation_slope=noise_attenuation_slope)
                    tone_in_noise = (signal + noise).astype(np.float32)

                    data_dict = {
                        'tone_in_noise': tone_in_noise,
                        'f0': float(f0),
                        'f0_nominal': float(f0_nominal),
                        'phase_mode': int(ph),
                        'filter_fl': float(filter_dict['fl']),
                        'filter_fh': float(filter_dict['fh']),
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
                        print(f'  [{itrN}/{N}] band=[{filter_dict["fl"]:.0f},'
                              f'{filter_dict["fh"]:.0f}], f0_nominal={f0_nominal:.1f}')
                    itrN += 1

    hdf5_f.close()
    print(f"[B2] Done: {hdf5_filename}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_ERB_ExpB_doubling.py <hdf5_filename>")
        sys.exit(1)
    generate_ExpB_ERB_doubling_dataset(sys.argv[1])
