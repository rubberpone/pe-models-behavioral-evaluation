#!/usr/bin/env python3
"""
共享工具函数 — 感知 ERB 测量项目的基础模块。

包含:
  - 人类耳蜗 ERB 公式 (Glasberg & Moore 1990)
  - 10 个 PE 模型的关键参数
  - 文件路径解析与模型列表
  - 感知 ERB 数据结构定义
  - 各实验现有数据路径映射

Usage:
    from erb_utils import (
        human_erb, model_list, get_exp_prediction_dir,
        EXISTING_EXP_A_PARAMS, EXISTING_EXP_B_PARAMS, EXISTING_EXP_C_PARAMS
    )
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional

# ============================================================
#  人类耳蜗 ERB (Glasberg & Moore 1990)
# ============================================================

def human_erb(f: np.ndarray) -> np.ndarray:
    """人类耳蜗 ERB (Hz).

    Glasberg & Moore (1990, Hearing Research) Eq. 4.
    ERB(f) = 24.7 * (4.37 * f / 1000 + 1)

    Args:
        f: 频率 (Hz), 标量或数组
    Returns:
        ERB (Hz), 与 f 同形
    """
    f = np.asarray(f, dtype=float)
    return 24.7 * (4.37 * f / 1000.0 + 1.0)


def human_erb_number(f: np.ndarray) -> np.ndarray:
    """人类耳蜗 ERB-number 尺度 (Cams).

    Glasberg & Moore (1990) Eq. 5.
    ERB_number(f) = 21.4 * log10(0.00437 * f + 1)
    """
    f = np.asarray(f, dtype=float)
    return 21.4 * np.log10(0.00437 * f + 1.0)


# ============================================================
#  F0-N 平面上的人类 ERB 分界线
# ============================================================

# RES/URS 边界常用 α 值 (α = F0 / ERB(N·F0))
ALPHA_VALUES = {
    0.55: {'label': 'α=0.55 (RES boundary)',  'ls': '--', 'color': '#2166ac', 'lw': 1.5},
    0.90: {'label': 'α=0.90 (RES boundary)',   'ls': '-.', 'color': '#92c5de', 'lw': 1.2},
    1.00: {'label': 'α=1.00 (ERB classic)',   'ls': '-',  'color': 'black',   'lw': 2.0},
    1.25: {'label': 'α=1.25 (URS boundary)',  'ls': '--', 'color': '#b2182b', 'lw': 1.5},
}

# 实验拟合的感知 ERB 线样式
FITTED_ERB_STYLE = {
    'ls': '-', 'color': '#f4a582', 'lw': 2.5,
    'label': 'Fitted perceptual ERB',
}


def human_erb_boundary_on_F0N(
    f0_values: np.ndarray,
    alpha: float = 1.0,
    f_min: float = 20.0,
) -> tuple:
    """计算人类耳蜗 ERB 在 F0-N 平面上的分界线.

    对每个 F0 值, 求解 N 使得 F0 = alpha * ERB(N * F0).

    推导:
      F0 = α * 24.7 * (4.37 * N * F0 / 1000 + 1)
      F0 / (α * 24.7) = 4.37 * N * F0 / 1000 + 1
      N = (F0 / (α * 24.7) - 1) * 1000 / (4.37 * F0)

    Args:
        f0_values: F0 值数组 (Hz)
        alpha: α 阈值 (默认 1.0 = 经典 ERB 边界)
        f_min: 最小有效频率 (Hz), 低于此值时 ERB 公式不适用
    Returns:
        (valid_f0s, N_values) — 分界线上的点, 仅包含有效 F0 (F0 > α*24.7)
    """
    f0_values = np.asarray(f0_values, dtype=float)
    # 仅保留 F0 > α*24.7 的点 (否则 N 为负)
    valid_mask = f0_values > alpha * 24.7
    valid_f0s = f0_values[valid_mask]

    N_values = (valid_f0s / (alpha * 24.7) - 1.0) * 1000.0 / (4.37 * valid_f0s)
    return valid_f0s, N_values


def human_erb_boundary_lines(
    f0_range: tuple = (25, 500),
    n_f0_points: int = 200,
    alphas: list = None,
) -> dict:
    """计算多个人类 ERB 分界线 (用于绘图).

    Args:
        f0_range: (f0_min, f0_max) — F0 范围 (Hz)
        n_f0_points: F0 采样点数
        alphas: α 值列表, 默认使用 ALPHA_VALUES 的键
    Returns:
        {alpha: {'f0': array, 'N': array, 'style': dict}, ...}
    """
    if alphas is None:
        alphas = sorted(ALPHA_VALUES.keys())

    f0_values = np.logspace(np.log10(f0_range[0]), np.log10(f0_range[1]), n_f0_points)
    lines = {}

    for alpha in alphas:
        valid_f0s, N_vals = human_erb_boundary_on_F0N(f0_values, alpha=alpha)
        style = ALPHA_VALUES.get(alpha, {'label': f'α={alpha}', 'ls': '-',
                                          'color': 'gray', 'lw': 1.0})
        lines[alpha] = {'f0': valid_f0s, 'N': N_vals, 'style': dict(style)}

    return lines


# ============================================================
#  频率 → 音名转换
# ============================================================

# MIDI 音名 (A4 = MIDI 69 = 440 Hz)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note_name(freq_hz: float) -> str:
    """将频率 (Hz) 转换为音名 + 八度编号.

    Args:
        freq_hz: 频率 (Hz)
    Returns:
        字符串, 如 "440.0 Hz (A4)"
    """
    if freq_hz <= 0:
        return "—"
    midi_num = 12.0 * np.log2(freq_hz / 440.0) + 69.0
    midi_rounded = int(round(midi_num))
    octave = (midi_rounded // 12) - 1
    note_idx = midi_rounded % 12
    note_name = NOTE_NAMES[note_idx]
    return f"{freq_hz:.0f} Hz ({note_name}{octave})"


def get_f0_tick_labels(f0_values: list) -> list:
    """为 F0 值列表生成双重标注 (Hz + 音名).

    Args:
        f0_values: F0 值列表 (Hz), 如 [50, 100, 200, 400]
    Returns:
        标签列表, 如 ["50 Hz (G1)", "100 Hz (G2)", ...]
    """
    return [freq_to_note_name(f) for f in f0_values]


# ============================================================
#  F0 bin 定义 (与 Josh 2021 行为实验兼容)
# ============================================================

# 标准 700 类 F0 grid: 80–1000 Hz, 6.25 cents/bin (= 192 bins/octave)
F0_MIN = 80.0
F0_MAX = 1000.0
BINS_PER_OCTAVE = 192
CENTS_PER_BIN = 1200.0 / BINS_PER_OCTAVE  # 6.25 cents
N_BINS_700 = 700

def get_f0_bins_700() -> np.ndarray:
    """返回 700 类 F0 bin 的中心频率 (Hz)."""
    max_octave = np.log2(F0_MAX / F0_MIN)
    bin_indices = np.arange(0, N_BINS_700)
    bins = F0_MIN * np.power(2.0, bin_indices * (max_octave / (N_BINS_700 - 1)))
    return bins


# ============================================================
#  10 个 PE 模型清单与关键参数
# ============================================================

MODEL_LIST = [
    'CREPE', 'FCN', 'JDC', 'FTATTENTION', 'HARMOF0',
    'RMVPE', 'FCPE', 'PENN_FCN', 'SWIFTF0', 'PESTO'
]

# 各模型的项目目录 (已有的行为实验预测数据)
MODEL_PSYCHOPHYSICS_DIRS = {
    'CREPE':       '/data/zyf/CREPE_psychophysics',
    'FCN':         '/data/zyf/FCN_psychophysics',
    'JDC':         '/data/zyf/JDC_psychophysics',
    'FTATTENTION': '/data/zyf/FTATTENTION_psychophysics',
    'HARMOF0':     '/data/zyf/HARMOF0_psychophysics',
    'RMVPE':       '/data/zyf/RMVPE_psychophysics',
    'FCPE':        '/data/zyf/FCPE_psychophysics',
    'PENN_FCN':    '/data/zyf/PENN_psychophysics',
    'SWIFTF0':     '/data/zyf/SWIFTF0_psychophysics',
    'PESTO':       '/data/zyf/PESTO_psychophysics',
}

# 各模型的原生参数
MODEL_PARAMS = {
    'CREPE':       {'native_sr': 16000, 'bin_spacing_cents': 20,   'frontend': 'raw_waveform'},
    'FCN':         {'native_sr': 8000,  'bin_spacing_cents': 12.5, 'frontend': 'raw_waveform'},
    'JDC':         {'native_sr': 8000,  'bin_spacing_cents': 6.25, 'frontend': 'dB_spectrogram'},
    'FTATTENTION': {'native_sr': 8000,  'bin_spacing_cents': 20,   'frontend': 'CFP_3ch'},
    'HARMOF0':     {'native_sr': 16000, 'bin_spacing_cents': 25,   'frontend': 'log_spectrogram'},
    'RMVPE':       {'native_sr': 16000, 'bin_spacing_cents': 20,   'frontend': 'internal_mel'},
    'FCPE':        {'native_sr': 16000, 'bin_spacing_cents': 20,   'frontend': 'internal_mel'},
    'PENN_FCN':    {'native_sr': 8000,  'bin_spacing_cents': 5,    'frontend': 'raw_waveform'},
    'SWIFTF0':     {'native_sr': 16000, 'bin_spacing_cents': 33,   'frontend': 'internal_stft'},
    'PESTO':       {'native_sr': 'any', 'bin_spacing_cents': None,  'frontend': 'VQT'},
}


# ============================================================
#  Josh 2021 现有刺激 HDF5 数据集路径
# ============================================================

JOSH_HDF5_DIR = '/data/zyf/pitchnet/data/dataset_psychophysics'

EXISTING_HDF5_FILES = {
    'A': os.path.join(JOSH_HDF5_DIR, 'stim_BernsteinOxenham2005.hdf5'),
    'B': os.path.join(JOSH_HDF5_DIR, 'stim_ShackletonCarlyon1994.hdf5'),
    'C': os.path.join(JOSH_HDF5_DIR, 'stim_MooreMoore2003.hdf5'),
    'D': os.path.join(JOSH_HDF5_DIR, 'stim_MooreEtAl1985.hdf5'),
    'E': os.path.join(JOSH_HDF5_DIR, 'stim_OxenhamEtAl2004.hdf5'),
}


# ============================================================
#  现有实验数据的参数空间 (来自 feedback_8_4.md 附录E 审查)
# ============================================================

# Exp A: 10 个 base_f0 (100–300 Hz, 对数均匀)
EXISTING_EXP_A_PARAMS = {
    'base_f0_list': [
        100.00, 112.98, 127.65, 144.22, 162.95,
        184.11, 208.01, 235.01, 265.53, 300.00
    ],
    'low_harm_range': (1, 30),      # 最低谐波编号
    'phase_modes': [0, 1],           # 0=sine, 1=rand
    'delta_f0_range': (0.94, 1.06),  # F0 变异因子
    'delta_f0_count': 121,
    'total_stimuli': 72600,
}

# Exp B: 密集 F0 grid (80–320 Hz, 6144 个) + 3 个 filter bands
# step=1/(192*16) octaves ≈ 0.39 cents, 每个 192-bin/octave 标签 bin 细分为 16 子步
_EXP_B_F0_BINS_N = 48  # v2: 48 个对数均匀代表 F0 (原来 6)
EXISTING_EXP_B_PARAMS = {
    'f0_min': 80.0,
    'f0_max': 320.0,
    'f0_count': 6144,               # log-uniform, step=1/(192*16) octaves
    'filter_bands': [
        {'name': 'Low',  'fl': 125.0,  'fh': 625.0},
        {'name': 'Mid',  'fl': 1375.0, 'fh': 1875.0},
        {'name': 'High', 'fl': 3900.0, 'fh': 5400.0},
    ],
    'phase_modes': [0, 4],           # 0=sine, 4=alt
    'total_stimuli': 36864,
    # 原分析使用的 f0_bin_centers (仅 125 Hz!)
    'original_f0_bin_centers': [125],
    # v2: 48 个对数均匀 F0 bins (80–320 Hz)
    'expanded_f0_bin_centers': list(np.logspace(np.log10(80), np.log10(320), _EXP_B_F0_BINS_N)),
}

# Exp C: 3971 个 F0 (80–480 Hz) + 3 个 spectral envelopes
# step=1/(192*8) octaves ≈ 0.78 cents, 每个 192-bin/octave 标签 bin 细分为 8 子步
_EXP_C_F0_BINS_N = 48  # v2: 48 个对数均匀代表 F0 (原来 5)
EXISTING_EXP_C_PARAMS = {
    'f0_min': 80.0,
    'f0_max': 480.0,
    'f0_count': 3971,               # log-uniform, step=1/(192*8) octaves
    'spectral_envelopes': [
        # harm_range: 通带内谐波成分 [center - bw//2, center + bw//2]
        {'centered_harmonic': 5,  'bandwidth_harmonics': 3,  'harm_range': (4, 6)},
        {'centered_harmonic': 11, 'bandwidth_harmonics': 5,  'harm_range': (9, 13)},
        {'centered_harmonic': 16, 'bandwidth_harmonics': 5,  'harm_range': (14, 18)},
    ],
    'f0_shifts': [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24],  # %F0
    'phase_mode': 3,                 # cos
    'total_stimuli': 83391,
    # v2: 48 个对数均匀代表 F0 (80–480 Hz)
    'representative_f0s': list(np.logspace(np.log10(80), np.log10(480), _EXP_C_F0_BINS_N)),
}

# Exp D: 534 F0s (3 标称值 × 178) + 12 个失谐谐波
EXISTING_EXP_D_PARAMS = {
    'f0_ref_list': [100.0, 200.0, 400.0],  # 标称 F0
    'f0_ref_width': 0.04,                  # ±4%
    'mistuned_harmonics': list(range(1, 13)),  # H1–H12 (连续全覆盖, 论文仅 H1-H6+H12)
    'mistuned_pcts': [-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8],
    'total_stimuli': 83304,
}

# Exp E: 6144 F0s (80-320 Hz) + 4 载波条件
_EXP_E_F0_BINS_N = 48
EXISTING_EXP_E_PARAMS = {
    'f0_min': 80.0,
    'f0_max': 320.0,
    'f0_count': 6144,               # log-uniform, step=1/(192*16) octaves (同 Exp B)
    'carriers': [0, 4000, 6350, 10080],  # 0=纯音, 其余=转置音载波
    'carrier_labels': ['pure_tone', 'T4k', 'T6.35k', 'T10k'],
    'total_stimuli': 24576,
    'representative_f0s': list(np.logspace(np.log10(80), np.log10(320), _EXP_E_F0_BINS_N)),
}


# ============================================================
#  Josh 心理物理学分析脚本路径
# ============================================================

JOSH_PSYCHOPHYSICS_DIR = '/data/zyf/pitchnet/assets_psychophysics'

JOSH_ANALYSIS_SCRIPTS = {
    'exp_a': os.path.join(JOSH_PSYCHOPHYSICS_DIR, 'f0dl_bernox.py'),
    'exp_b': os.path.join(JOSH_PSYCHOPHYSICS_DIR, 'f0experiment_alt_phase.py'),
    'exp_c': os.path.join(JOSH_PSYCHOPHYSICS_DIR, 'f0experiment_freq_shifted.py'),
    'exp_d': os.path.join(JOSH_PSYCHOPHYSICS_DIR, 'f0experiment_mistuned_harmonics.py'),
    'exp_e': os.path.join(JOSH_PSYCHOPHYSICS_DIR, 'f0dl_transposed_tones.py'),
    'util_compare': os.path.join(JOSH_PSYCHOPHYSICS_DIR, 'util_human_model_comparison.py'),
    'util_figures': os.path.join(JOSH_PSYCHOPHYSICS_DIR, 'util_figures_psychophysics.py'),
}


# ============================================================
#  感知 ERB 数据结构
# ============================================================

def create_erb_result_dict(model_name: str, method: str) -> dict:
    """创建标准化的感知 ERB 测量结果字典.

    Args:
        model_name: 模型名 (如 'CREPE')
        method: 方法标识 (如 'A1_ExpB_doubling', 'A2_ExpA_knee', 'A3_ExpC_slope')
    Returns:
        标准结果字典
    """
    return {
        'model': model_name,
        'method': method,
        'erb_points': [],       # List of (f_harmonic_Hz, erb_Hz)
        'erb_fit': {            # ERB(f) = a*f + b 拟合结果
            'a': None, 'b': None, 'r_squared': None
        },
        'erb_similarity_vs_human': None,  # Pearson r with log10(human_erb)
        'n_points_used': 0,
        'bootstrap_ci_95': [],   # 95% CI for each erb point
        'metadata': {},
    }


def erb_points_to_function(erb_points: List[Tuple[float, float]],
                            interp_kind: str = 'linear') -> callable:
    """将散点 ERB 估计转换为可调用的 ERB(f) 函数 (通过插值).

    Args:
        erb_points: [(f_harmonic, erb_value), ...]
        interp_kind: scipy.interpolate.interp1d 的 kind 参数
    Returns:
        erb_function: f(Hz) -> ERB(Hz) 的可调用函数
    """
    from scipy.interpolate import interp1d
    freqs = np.array([p[0] for p in erb_points])
    erbs = np.array([p[1] for p in erb_points])
    sort_idx = np.argsort(freqs)
    return interp1d(
        freqs[sort_idx], erbs[sort_idx],
        kind=interp_kind,
        bounds_error=False,
        fill_value=(erbs[sort_idx][0], erbs[sort_idx][-1])
    )


# ============================================================
#  ERB 相似度评分
# ============================================================

def compute_erb_similarity(erb_points: List[Tuple[float, float]]) -> dict:
    """计算模型的感知 ERB 与人类耳蜗 ERB 之间的相似度.

    Args:
        erb_points: [(f_harmonic, erb_value), ...] 模型的 ERB 散点
    Returns:
        {'pearson_r': ..., 'mae_hz': ..., 'mae_log10': ...}
    """
    from scipy.stats import pearsonr
    freqs = np.array([p[0] for p in erb_points])
    model_erb = np.array([p[1] for p in erb_points])
    # Clamp: ERB 值必须 >0 才能取 log10
    valid = (model_erb > 0) & (freqs > 0)
    if np.sum(valid) < 3:
        return {'pearson_r': float('nan'), 'pearson_pval': float('nan'),
                'mae_hz': float('nan'), 'mae_log10': float('nan')}
    freqs = freqs[valid]
    model_erb = model_erb[valid]
    human_erb_vals = human_erb(freqs)

    r, pval = pearsonr(np.log10(model_erb), np.log10(human_erb_vals))
    mae_hz = np.mean(np.abs(model_erb - human_erb_vals))
    mae_log10 = np.mean(np.abs(np.log10(model_erb) - np.log10(human_erb_vals)))

    return {
        'pearson_r': float(r),
        'pearson_pval': float(pval),
        'mae_hz': float(mae_hz),
        'mae_log10': float(mae_log10),
    }


# ============================================================
#  Per-F0 分析辅助函数
# ============================================================

def bin_f0_values(f0_values: np.ndarray,
                  bin_centers: List[float],
                  bin_width: float = 0.10) -> Dict[float, np.ndarray]:
    """将 F0 值分入以 bin_centers 为中心的 bin.

    Args:
        f0_values: 所有刺激的 F0 值 (Hz)
        bin_centers: bin 中心列表 (Hz), 如 [62, 88, 125, 177, 250, 354]
        bin_width: bin 半宽比例 (如 0.10 = ±10%)
    Returns:
        {center: mask array}, 每个 bin 中心的布尔索引
    """
    masks = {}
    for center in bin_centers:
        lo = center * (1 - bin_width)
        hi = center * (1 + bin_width)
        masks[center] = (f0_values >= lo) & (f0_values <= hi)
    return masks


def sigmoid(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """Sigmoid 函数: y = 1 / (1 + exp(-k * (x - x0)))."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def fit_sigmoid_transition(x: np.ndarray, y: np.ndarray,
                           p0: List[float] = None) -> Tuple[float, float]:
    """拟合 sigmoid 找过渡点.

    Args:
        x: 自变量 (如 harmonic number N 或 log10(F0))
        y: 因变量 (如 F0DL 或 normalized pitch doubling ratio), 范围 [0, 1]
        p0: 初始猜测 [k, x0]
    Returns:
        (k, x0) — x0 为过渡点
    """
    from scipy.optimize import curve_fit
    if p0 is None:
        p0 = [1.0, np.median(x)]
    popt, _ = curve_fit(sigmoid, x, y, p0=p0, maxfev=5000)
    return popt[0], popt[1]


def exp_decay(x: np.ndarray, N_decay: float, k: float) -> np.ndarray:
    """指数衰减: y = exp(-(x / N_decay)^k)."""
    return np.exp(-(x / N_decay) ** k)


def fit_exp_decay(x: np.ndarray, y: np.ndarray,
                  p0: List[float] = None) -> Tuple[float, float]:
    """拟合指数衰减.

    Args:
        x: 自变量 (如 harmonic number N)
        y: 因变量 (如 normalized pitch shift slope), 范围 [0, 1]
        p0: 初始猜测 [N_decay, k]
    Returns:
        (N_decay, k)
    """
    from scipy.optimize import curve_fit
    if p0 is None:
        p0 = [np.median(x), 1.5]
    popt, _ = curve_fit(exp_decay, x, y, p0=p0, maxfev=5000)
    return popt[0], popt[1]


# ============================================================
#  项目内路径
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
RESULTS_STRATEGY_A_DIR = os.path.join(RESULTS_DIR, 'data', 'strategy_A')
RESULTS_STRATEGY_B_DIR = os.path.join(RESULTS_DIR, 'data', 'strategy_B')
CONSENSUS_DIR = os.path.join(RESULTS_DIR, 'data', 'consensus')
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, 'predictions')
PREDICTIONS_STRATEGY_A_DIR = os.path.join(PREDICTIONS_DIR, 'strategy_A')
PREDICTIONS_STRATEGY_B_DIR = os.path.join(PREDICTIONS_DIR, 'strategy_B')


def get_model_prediction_dir(model: str, exp_label: str) -> str:
    """获取某模型在某实验 ERB 变体上的预测输出目录.

    Args:
        model: 模型名 (如 'CREPE')
        exp_label: 实验标识 (如 'ERB_F0N_sweep', 'ERB_ExpB_doubling')
    Returns:
        目录路径
    """
    d = os.path.join(PREDICTIONS_DIR, model, exp_label)
    os.makedirs(d, exist_ok=True)
    return d


print(f"[erb_utils] Project root: {PROJECT_ROOT}")
print(f"[erb_utils] {len(MODEL_LIST)} models: {MODEL_LIST}")
