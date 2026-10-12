#!/usr/bin/env python3
"""
策略 B3: Exp C 完整版 — 连续 spectral envelope Pitch Shift 扫参刺激生成。

参数空间（README_strategy_B.md §5.3 B3）:
  45 f0（共享网格，80–1000 Hz）
  × centered_harmonic N = {2,3,4,5,6,8,10,12,15,18,22,26,30}（对数间隔）
    + 梯形补全 {35,45,60,80,110,150,190}（仅低 F0 行生效）
    ∩ {(N+b/2)×F0 ≤ 16000}（包络平带区完整落入 16 kHz；N=1 已去除）
  × 7 f0_shift（{0,4,8,12,16,20,24}%F0）
  × 5 个 ±4% log-uniform 浮动变体（含中心值，锚点聚合）
  = 22,960 条

关键口径:
  - 带宽分配规则：N∈{2..6} → b=3 谐波宽（镜像 low envelope）；
    N≥8 → b=5 谐波宽（镜像 mid/high）——仅用既有带宽值 {3,5}
  - 帽按标称 F0 计算；±4% 浮动仅使平带区上缘越界 <0.5 个谐波间距，可忽略
  - cos 相位、70 dB SPL 整体电平、无噪声（Exp C 原设计，原样）

f0_label_* 列由 stimuli_f0_labels.py 生成后统一后处理。

用法:
    python generate_ERB_ExpC_shift.py data/stim_ERB_B3.hdf5
"""

import sys
import os
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_stimulus_utils import (
    initialize_hdf5_file, write_example_to_hdf5,
    get_MooreMoore2003_spectral_envelope,
    get_MooreMoore2003_complex_tone,
)

N_LIST_BASE = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30]
N_LIST_SPARSE = [35, 45, 60, 80, 110, 150, 190]  # 梯形补全（仅低 F0 行生效）
SHIFT_LIST = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24]  # %F0


def bandwidth_for_N(N):
    """N∈{2..6} → 3 谐波宽；N≥8 → 5 谐波宽（镜像 Josh 既有带宽值）."""
    return 3 if N <= 6 else 5


def generate_ExpC_ERB_shift_dataset(
    hdf5_filename,
    fs=32000, dur=0.150,
    f0_min=80.0, f0_max=1000.0, f0_n=45,
    n_list_base=None, n_list_sparse=None, shift_list=None,
    n_jitter=5, jitter_pct=4.0,
    dBSPL=70.0, phase_mode='cos',
    disp_step=500
):
    """生成连续 spectral envelope 扫参刺激集（策略 B3）."""
    if n_list_base is None:
        n_list_base = N_LIST_BASE
    if n_list_sparse is None:
        n_list_sparse = N_LIST_SPARSE
    if shift_list is None:
        shift_list = SHIFT_LIST

    f0_nominal_list = np.geomspace(f0_min, f0_max, f0_n)

    # 每行有效的 N 集：(N+b/2)×F0 ≤ 16000（帽按标称 F0 计算）
    valid_n_lists = {}
    for f0_nominal in f0_nominal_list:
        valid_n = []
        for n in n_list_base + n_list_sparse:
            b = bandwidth_for_N(n)
            if (n + b / 2.0) * f0_nominal <= fs / 2:
                valid_n.append(n)
        valid_n_lists[f0_nominal] = np.array(valid_n, dtype=int)

    n_conditions = sum(len(v) for v in valid_n_lists.values())
    N = n_conditions * len(shift_list) * n_jitter
    print(f"[B3] Total stimuli: {N:,} ({(f0_n, n_conditions, len(shift_list), n_jitter)})")

    # ±4% log-uniform 浮动变体（含中心值）
    jitter_ratio = 1.0 + jitter_pct / 100.0
    f0_variants = np.geomspace(f0_nominal_list / jitter_ratio,
                               f0_nominal_list * jitter_ratio, n_jitter).T

    phase_mode_encoding = {'sine': 0, 'rand': 1, 'sch': 2, 'cos': 3, 'alt': 4}

    config_dict = {
        'config_tone/fs': fs, 'config_tone/dur': dur,
        'config_tone/dBSPL': dBSPL,
        'config_tone/phase_mode': phase_mode_encoding[phase_mode],
        'config_tone/f0_min': f0_min, 'config_tone/f0_max': f0_max,
        'config_tone/f0_n': f0_n,
        'config_tone/n_jitter': n_jitter,
        'config_tone/jitter_pct': jitter_pct,
    }
    config_key_pair_list = [(k, k) for k in config_dict.keys()]

    itrN = 0
    for i_f0, f0_nominal in enumerate(f0_nominal_list):
        for harm in valid_n_lists[f0_nominal]:
            b = bandwidth_for_N(int(harm))
            for f0 in f0_variants[i_f0]:
                # 包络随 (实际 f0, N) 变化；7 个 shift 共用一个包络（Josh 原样）
                f_center = f0 * harm
                f_bandwidth = f0 * b
                spectral_envelope = get_MooreMoore2003_spectral_envelope(
                    f0, f_center, f_bandwidth)

                for f0_shift in shift_list:
                    y = get_MooreMoore2003_complex_tone(
                        f0, f0_shift=f0_shift, spectral_envelope=spectral_envelope,
                        fs=fs, dur=dur, dBSPL=dBSPL, phase_mode=phase_mode)

                    data_dict = {
                        'stimuli/signal': y.astype(np.float32),
                        'f0': float(f0),
                        'f0_nominal': float(f0_nominal),
                        'f0_shift': float(f0_shift),
                        'spectral_envelope_centered_harmonic': int(harm),
                        'spectral_envelope_bandwidth_in_harmonics': int(b),
                        'spectral_envelope_f_center': float(f_center),
                        'spectral_envelope_f_bandwidth': float(f_bandwidth),
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
                              f'N={harm}, shift={f0_shift}')
                    itrN += 1

    hdf5_f.close()
    print(f"[B3] Done: {hdf5_filename}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_ERB_ExpC_shift.py <hdf5_filename>")
        sys.exit(1)
    generate_ExpC_ERB_shift_dataset(sys.argv[1])
