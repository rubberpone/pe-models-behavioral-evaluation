#!/usr/bin/env python3
"""
Pitch Shift 斜率分析模块 — 从 Exp C (Moore & Moore 2003) 数据提取感知 ERB.

核心逻辑复制自 /data/zyf/pitchnet/assets_psychophysics/f0experiment_freq_shifted.py
的 compute_f0_shift_curve().

提供:
  - compute_f0_shift_curve(): 对单个 (spectral_envelope) 条件计算 pitch shift 曲线
  - compute_slope_vs_harmonic(): 计算 pitch shift slope 作为中心谐波编号的函数
  - extract_erb_from_pitch_shift(): 从 slope 衰减提取感知 ERB 散点
  - run_strategy_A3_pipeline(): 策略 A3 的完整分析流水线
"""

import sys
import os
import json
import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Tuple, Optional
from erb_utils import (
    EXISTING_EXP_C_PARAMS, exp_decay, fit_exp_decay,
    create_erb_result_dict, MODEL_LIST, get_model_prediction_dir
)


# ============================================================
#  从 Josh 项目复制的核心函数
#  来源: /data/zyf/pitchnet/assets_psychophysics/f0experiment_freq_shifted.py
# ============================================================

def compute_f0_shift_curve(
    expt_dict: dict,
    filter_key: str = 'spectral_envelope_centered_harmonic',
    filter_value: int = None,
    f0_min: float = 80.0,
    f0_max: float = 1000.0,
) -> dict:
    """对单个 spectral envelope 条件计算 pitch shift 曲线.

    对指定的谱包络条件, 聚合所有在 f0 范围内的刺激,
    计算 f0_pred_shift = (f0_pred - f0) / f0 作为 f0_shift 的函数.

    Args:
        expt_dict: 包含 f0, f0_pred, f0_shift, spectral_envelope_centered_harmonic 的字典
        filter_key: 用于选择谱包络条件的键
        filter_value: 谱包络条件值 (如 5, 11, 16)
        f0_min, f0_max: F0 过滤范围
    Returns:
        {'f0_shift': [...], 'f0_pred_shift_mean': [...],
         'f0_pred_shift_median': [...], 'f0_pred_shift_stddev': [...]}
    """
    idx = np.array(expt_dict[filter_key]) == filter_value
    idx &= (np.array(expt_dict['f0']) >= f0_min)
    idx &= (np.array(expt_dict['f0']) <= f0_max)

    f0_shift = np.array(expt_dict['f0_shift'])[idx]
    f0_pred = np.array(expt_dict['f0_pred'])[idx]
    f0_true = np.array(expt_dict['f0'])[idx]
    f0_pred_shift = (f0_pred - f0_true) / f0_true

    f0_shift_unique = np.unique(f0_shift)
    f0_pred_shift_mean = np.zeros_like(f0_shift_unique)
    f0_pred_shift_median = np.zeros_like(f0_shift_unique)
    f0_pred_shift_stddev = np.zeros_like(f0_shift_unique)

    for j, fs_val in enumerate(f0_shift_unique):
        curr = f0_shift == fs_val
        f0_pred_shift_mean[j] = np.mean(f0_pred_shift[curr])
        f0_pred_shift_median[j] = np.median(f0_pred_shift[curr])
        f0_pred_shift_stddev[j] = np.std(f0_pred_shift[curr])

    return {
        'f0_shift': (100.0 * f0_shift_unique).tolist(),  # → %F0
        'f0_pred_shift_mean': (100.0 * f0_pred_shift_mean).tolist(),
        'f0_pred_shift_median': (100.0 * f0_pred_shift_median).tolist(),
        'f0_pred_shift_stddev': (100.0 * f0_pred_shift_stddev).tolist(),
    }


# ============================================================
#  感知 ERB 提取 (策略 A3 核心)
# ============================================================

def compute_slope_vs_harmonic(
    expt_dict: dict,
    f0_target: float,
    f0_tolerance: float = 0.05,  # ±5%
    harmonic_numbers: List[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """对单个目标 F0, 计算 pitch shift slope 作为中心谐波编号 N 的函数.

    策略 A3 的核心: 对每个代表性 F0, 在已有数据中取该 F0 附近的刺激,
    对每个 spectral envelope 条件 (N=5/11/16) 计算 pitch shift slope.

    Args:
        expt_dict: Exp C 预测数据
        f0_target: 目标 F0 (Hz), 如 125
        f0_tolerance: F0 容差比例
        harmonic_numbers: 要分析的中心谐波编号列表 (默认使用 Exp C 已有的 3 个)
    Returns:
        (harmonic_numbers, slopes) — 每个 N 的 |pitch_shift_slope|
    """
    if harmonic_numbers is None:
        harmonic_numbers = [
            env['centered_harmonic']
            for env in EXISTING_EXP_C_PARAMS['spectral_envelopes']
        ]

    slopes = []
    valid_harmonics = []

    f0_lo = f0_target * (1 - f0_tolerance)
    f0_hi = f0_target * (1 + f0_tolerance)

    for harm in harmonic_numbers:
        shift_curve = compute_f0_shift_curve(
            expt_dict,
            filter_key='spectral_envelope_centered_harmonic',
            filter_value=int(harm),
            f0_min=f0_lo,
            f0_max=f0_hi,
        )

        f0_shifts = np.array(shift_curve['f0_shift'])
        pred_shifts = np.array(shift_curve['f0_pred_shift_median'])

        if len(f0_shifts) >= 3:
            slope, _, r_value, _, _ = linregress(f0_shifts, pred_shifts)
            slopes.append(abs(slope))
            valid_harmonics.append(harm)

    return np.array(valid_harmonics), np.array(slopes)


def extract_erb_from_pitch_shift(
    expt_dict: dict,
    representative_f0s: List[float] = None,
    f0_tolerance: float = 0.05,
) -> List[Tuple[float, float]]:
    """从 Pitch Shift 斜率衰减提取感知 ERB 散点 (策略 A3).

    核心逻辑:
      对每个代表性 F0:
        计算 pitch shift slope 作为中心谐波编号 N 的函数
        低 N → slope > 0 (RES)
        高 N → slope ≈ 0 (URS)
        拟合指数衰减 exp(-(N/N_decay)^k) → N_decay
        感知 ERB(N_decay × F0) ≈ F0

    Args:
        expt_dict: Exp C 预测数据
        representative_f0s: 代表性 F0 值列表 (Hz)
        f0_tolerance: F0 容差比例
    Returns:
        [(f_harmonic_Hz, ERB_Hz), ...] — 感知 ERB 散点
    """
    if representative_f0s is None:
        representative_f0s = EXISTING_EXP_C_PARAMS['representative_f0s']

    erb_points = []

    for f0 in representative_f0s:
        harmonics, slopes = compute_slope_vs_harmonic(
            expt_dict, f0_target=f0, f0_tolerance=f0_tolerance
        )

        if len(harmonics) < 3:
            continue

        # 归一化 slope
        if np.max(slopes) < 0.01:
            continue
        slope_norm = slopes / np.max(slopes)

        # 拟合指数衰减
        try:
            N_decay, k = fit_exp_decay(harmonics, slope_norm, p0=[8.0, 1.5])
            erb_points.append((N_decay * f0, f0))
        except Exception:
            continue

    return erb_points


# ============================================================
#  策略 A3 完整流水线
# ============================================================

def run_strategy_A3_pipeline(
    model_name: str,
    predictions_json_path: str,
    output_dir: str = None,
) -> dict:
    """策略 A3: 对单个模型的完整 Exp C per-F0 Pitch Shift 重分析.

    Args:
        model_name: 模型名
        predictions_json_path: Exp C 预测 JSON 文件路径
        output_dir: 输出目录
    Returns:
        erb_results: 标准感知 ERB 结果字典
    """
    print(f"\n{'='*60}")
    print(f"[Strategy A3] {model_name}: Exp C Per-F0 Pitch Shift Analysis")
    print(f"{'='*60}")

    if output_dir is None:
        output_dir = get_model_prediction_dir(model_name, 'ExpC_reanalysis')

    # 1. 加载预测数据
    print(f"  Loading predictions from: {predictions_json_path}")
    with open(predictions_json_path, 'r') as f:
        json_dict = json.load(f)

    expt_dict = _build_expt_dict_from_json(json_dict)
    print(f"  Loaded {len(expt_dict['f0'])} stimuli")

    # 2. Per-F0 分析
    rep_f0s = EXISTING_EXP_C_PARAMS['representative_f0s']
    print(f"  Representative F0s: {rep_f0s}")

    # 3. 提取感知 ERB 散点
    erb_points = extract_erb_from_pitch_shift(
        expt_dict,
        representative_f0s=rep_f0s,
        f0_tolerance=0.05
    )

    print(f"  Extracted {len(erb_points)} ERB points:")
    for f_harm, erb_val in erb_points:
        print(f"    ERB({f_harm:.0f} Hz) ≈ {erb_val:.1f} Hz")

    # 4. 构建结果
    from erb_utils import create_erb_result_dict, compute_erb_similarity
    results = create_erb_result_dict(model_name, 'A3_ExpC_slope')
    results['erb_points'] = erb_points
    if len(erb_points) >= 3:
        results['erb_similarity_vs_human'] = compute_erb_similarity(erb_points)
        from scipy.stats import linregress
        freqs = np.array([p[0] for p in erb_points])
        erbs = np.array([p[1] for p in erb_points])
        slope, intercept, r, _, _ = linregress(freqs, erbs)
        results['erb_fit'] = {'a': float(slope), 'b': float(intercept),
                             'r_squared': float(r**2)}
    results['n_points_used'] = len(erb_points)

    # 5. 保存
    output_fn = os.path.join(output_dir, f'{model_name}_A3_erb_results.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(output_fn, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {output_fn}")

    return results


def _build_expt_dict_from_json(json_dict: dict) -> dict:
    """从 Josh-format 预测 JSON 构建 expt_dict."""
    expt_dict = {}

    for key in ['f0', 'f0_pred', 'f0_shift',
                'spectral_envelope_centered_harmonic',
                'spectral_envelope_bandwidth_in_harmonics']:
        if key in json_dict:
            expt_dict[key] = np.array(json_dict[key])

    return expt_dict


if __name__ == '__main__':
    print("[compute_pitch_shift_slope] Module loaded.")
    print("  Functions: compute_f0_shift_curve(), extract_erb_from_pitch_shift()")
    print("  Pipeline: run_strategy_A3_pipeline()")
