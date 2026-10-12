#!/usr/bin/env python3
"""
Pitch Doubling 比率分析模块 — 从 Exp B (Shackleton & Carlyon 1994) 数据提取感知 ERB.

核心逻辑复制自 /data/zyf/pitchnet/assets_psychophysics/f0experiment_alt_phase.py
的 compute_f0_pred_ratio() 和 compute_f0_2x_pref().

提供:
  - compute_f0_pred_ratio(): 计算 f0_pred / f0 比率 (按 filter_band × F0_bin)
  - compute_f0_2x_pref(): 二分类判断模型偏好 1×F0 还是 2×F0
  - extract_erb_from_pitch_doubling(): 从比率-F0 曲线提取感知 ERB 散点
  - run_strategy_A1_pipeline(): 策略 A1 的完整分析流水线
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from erb_utils import (
    EXISTING_EXP_B_PARAMS, sigmoid, fit_sigmoid_transition,
    bin_f0_values, create_erb_result_dict, MODEL_LIST,
    get_model_prediction_dir
)


# ============================================================
#  从 Josh 项目复制的核心函数
#  来源: /data/zyf/pitchnet/assets_psychophysics/f0experiment_alt_phase.py
# ============================================================

def compute_f0_pred_ratio(
    expt_dict_list: List[dict],
    phase_mode: int = 4,  # ALT
    f0_bin_centers: List[float] = None,
    f0_bin_width: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
    """计算不同 F0 和 filter 条件下的 f0_pred_ratio = f0_pred / f0.

    策略 A1 的关键修改: f0_bin_centers 从 [125] 扩展为 [62, 88, 125, 177, 250, 354]，
    以利用 Exp B 现有数据中已有的密集 F0 grid (80–320 Hz, 6144 个值).

    Args:
        expt_dict_list: 预测数据字典列表 (每个 dict 包含 f0, f0_pred, filter_fl, phase_mode)
        phase_mode: 目标相位模式 (4 = ALT)
        f0_bin_centers: F0 bin 中心列表 (Hz). 默认使用扩展后的 6 个 bin
        f0_bin_width: bin 半宽比例 (如 0.10 = ±10%)
    Returns:
        (filter_condition_list, f0_condition_list, f0_pred_ratio_list)
        - filter_condition_list: 每个 bin 的 filter_fl 值
        - f0_condition_list: 每个 bin 的 f0_center 值
        - f0_pred_ratio_list: 每个 bin 的 f0_pred_ratio 值列表
    """
    if f0_bin_centers is None:
        f0_bin_centers = EXISTING_EXP_B_PARAMS['expanded_f0_bin_centers']

    if not isinstance(expt_dict_list, list):
        expt_dict_list = [expt_dict_list]

    # 获取所有唯一的 filter_fl
    filter_conditions = np.unique(expt_dict_list[0]['filter_fl'])

    f0_pred_ratio_list = []
    f0_condition_list = []
    filter_condition_list = []

    for filt_cond in filter_conditions:
        for f0_center in f0_bin_centers:
            f0_range = [f0_center * (1 - f0_bin_width),
                       f0_center * (1 + f0_bin_width)]
            f0_pred_ratio_sublist = []

            for expt_dict in expt_dict_list:
                # 使用 filter_expt_dict 逻辑 (手动实现以避免循环导入)
                keep = np.ones(len(expt_dict['f0']), dtype=bool)
                keep &= (np.array(expt_dict['filter_fl']) == filt_cond)
                keep &= (np.array(expt_dict['f0']) >= f0_range[0])
                keep &= (np.array(expt_dict['f0']) <= f0_range[1])
                keep &= (np.array(expt_dict['phase_mode']) == phase_mode)

                f0_pred = np.array(expt_dict['f0_pred'])[keep]
                f0_true = np.array(expt_dict['f0'])[keep]

                if len(f0_pred) > 0:
                    ratios = f0_pred / f0_true
                    f0_pred_ratio_sublist.extend(ratios.tolist())

            if len(f0_pred_ratio_sublist) > 0:
                f0_pred_ratio_list.append(f0_pred_ratio_sublist)
                filter_condition_list.append(filt_cond)
                f0_condition_list.append(f0_center)

    return (np.array(filter_condition_list),
            np.array(f0_condition_list),
            f0_pred_ratio_list)


def compute_f0_2x_pref(expt_dict: dict,
                       use_log_scale: bool = True) -> dict:
    """判断模型在每个刺激上的 2×F0 偏好.

    +1 = 预测 F0 更接近 2×F0 (pitch doubling)
    -1 = 预测 F0 更接近 1×F0 (正常音高)

    Args:
        expt_dict: 包含 f0 和 f0_pred 的字典
        use_log_scale: 若 True, 在对数空间计算中点
    Returns:
        expt_dict, 添加了 'f0_2x_pref' 字段
    """
    f0_true = np.array(expt_dict['f0'])
    f0_true_2x = 2.0 * f0_true
    f0_pred = np.array(expt_dict['f0_pred'])

    if use_log_scale:
        f0_mid = np.exp((np.log(f0_true) + np.log(f0_true_2x)) / 2.0)
    else:
        f0_mid = (f0_true + f0_true_2x) / 2.0

    f0_2x_pref = np.zeros_like(f0_pred)
    f0_2x_pref[f0_pred > f0_mid] = 1
    f0_2x_pref[f0_pred <= f0_mid] = -1
    expt_dict['f0_2x_pref'] = f0_2x_pref
    return expt_dict


# ============================================================
#  感知 ERB 提取 (策略 A1 核心)
# ============================================================

def compute_pitch_doubling_ratio_curve(
    expt_dict_list: List[dict],
    filter_fl: float,
    f0_bin_centers: List[float] = None,
    f0_bin_width: float = 0.10,
    phase_mode: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """对单个 filter_band, 计算 pitch doubling ratio 作为 F0 的函数.

    Args:
        expt_dict_list: 预测数据列表
        filter_fl: 滤波频段低频截止 (Hz)
        f0_bin_centers: F0 bin 中心列表
        f0_bin_width: bin 半宽
        phase_mode: 4 = ALT
    Returns:
        (f0_centers, median_ratios) — 每个 F0 bin 的 median(f0_pred_ratio)
    """
    if f0_bin_centers is None:
        f0_bin_centers = EXISTING_EXP_B_PARAMS['expanded_f0_bin_centers']

    median_ratios = []
    valid_centers = []

    for f0_center in f0_bin_centers:
        f0_range = [f0_center * (1 - f0_bin_width),
                   f0_center * (1 + f0_bin_width)]
        ratios = []

        for expt_dict in expt_dict_list:
            keep = np.ones(len(expt_dict['f0']), dtype=bool)
            keep &= (np.array(expt_dict['filter_fl']) == filter_fl)
            keep &= (np.array(expt_dict['f0']) >= f0_range[0])
            keep &= (np.array(expt_dict['f0']) <= f0_range[1])
            keep &= (np.array(expt_dict['phase_mode']) == phase_mode)

            if np.sum(keep) > 0:
                f0_pred = np.array(expt_dict['f0_pred'])[keep]
                f0_true = np.array(expt_dict['f0'])[keep]
                ratios.extend((f0_pred / f0_true).tolist())

        if len(ratios) > 0:
            valid_centers.append(f0_center)
            median_ratios.append(np.median(ratios))

    return np.array(valid_centers), np.array(median_ratios)


def extract_erb_from_pitch_doubling(
    expt_dict_list: List[dict],
    f0_bin_centers: List[float] = None,
    f0_bin_width: float = 0.10,
    phase_mode: int = 4,
) -> List[Tuple[float, float]]:
    """从 Pitch Doubling 比率曲线提取感知 ERB 散点 (策略 A1).

    核心逻辑:
      对于每个 filter_band (最低可听谐波频率 ≈ fl):
        跨 F0 变化绘制 median(f0_pred_ratio) vs F0
        低 F0 → ratio ≈ 1.0 (RES)
        高 F0 → ratio ≈ 2.0 (URS)
        Sigmoid 拟合找到过渡 F0_trans → 感知 ERB(fl) ≈ F0_trans

    Args:
        expt_dict_list: Exp B 预测数据列表
        f0_bin_centers: F0 bin 中心列表
        f0_bin_width: bin 半宽
        phase_mode: ALT 的编码 (4)
    Returns:
        [(f_harmonic_Hz, ERB_Hz), ...] — 感知 ERB 散点
    """
    if f0_bin_centers is None:
        f0_bin_centers = EXISTING_EXP_B_PARAMS['expanded_f0_bin_centers']

    erb_points = []

    for band in EXISTING_EXP_B_PARAMS['filter_bands']:
        fl = band['fl']

        f0s, ratios = compute_pitch_doubling_ratio_curve(
            expt_dict_list, filter_fl=fl,
            f0_bin_centers=f0_bin_centers,
            f0_bin_width=f0_bin_width,
            phase_mode=phase_mode
        )

        if len(f0s) < 4:
            continue  # 不足以拟合 sigmoid

        # 归一化 ratio: 1.0 → 0, 2.0 → 1
        ratio_norm = np.clip((ratios - 1.0), 0.0, 1.0)

        # 如果所有 ratio ≈ 1 (全 RES) 或所有 ratio ≈ 2 (全 URS), 跳过
        if np.std(ratio_norm) < 0.05:
            continue

        try:
            k, log10_f0_trans = fit_sigmoid_transition(
                np.log10(f0s), ratio_norm, p0=[10.0, np.log10(np.median(f0s))]
            )
            f0_trans = 10.0 ** log10_f0_trans
            erb_points.append((fl, f0_trans))
        except Exception:
            continue

    return erb_points


# ============================================================
#  策略 A1 完整流水线
# ============================================================

def run_strategy_A1_pipeline(
    model_name: str,
    predictions_json_path: str,
    output_dir: str = None,
) -> dict:
    """策略 A1: 对单个模型的完整 Exp B per-F0 Pitch Doubling 重分析.

    Args:
        model_name: 模型名
        predictions_json_path: Exp B 预测 JSON 文件路径
        output_dir: 输出目录 (默认: predictions/{model}/ExpB_reanalysis/)
    Returns:
        erb_results: 标准感知 ERB 结果字典
    """
    print(f"\n{'='*60}")
    print(f"[Strategy A1] {model_name}: Exp B Per-F0 Pitch Doubling Analysis")
    print(f"{'='*60}")

    if output_dir is None:
        output_dir = get_model_prediction_dir(model_name, 'ExpB_reanalysis')

    # 1. 加载预测数据
    print(f"  Loading predictions from: {predictions_json_path}")
    with open(predictions_json_path, 'r') as f:
        json_dict = json.load(f)

    # 构建 expt_dict (与 Josh 的 load_f0_expt_dict_from_json 兼容)
    expt_dict = _build_expt_dict_from_json(json_dict)
    print(f"  Loaded {len(expt_dict['f0'])} stimuli")

    # 2. 使用扩展的 f0_bin_centers 做 per-F0 分析
    expanded_centers = EXISTING_EXP_B_PARAMS['expanded_f0_bin_centers']
    print(f"  Using expanded F0 bin centers: {expanded_centers}")

    # 3. 提取感知 ERB 散点
    erb_points = extract_erb_from_pitch_doubling(
        [expt_dict],
        f0_bin_centers=expanded_centers,
        f0_bin_width=0.10,
        phase_mode=4  # ALT
    )

    print(f"  Extracted {len(erb_points)} ERB points:")
    for f_harm, erb_val in erb_points:
        print(f"    ERB({f_harm:.0f} Hz) ≈ {erb_val:.1f} Hz")

    # 4. 构建结果
    from erb_utils import create_erb_result_dict, compute_erb_similarity
    results = create_erb_result_dict(model_name, 'A1_ExpB_doubling')
    results['erb_points'] = erb_points
    if len(erb_points) >= 3:
        results['erb_similarity_vs_human'] = compute_erb_similarity(erb_points)
        # 拟合 ERB(f) = a*f + b
        from scipy.stats import linregress
        freqs = np.array([p[0] for p in erb_points])
        erbs = np.array([p[1] for p in erb_points])
        slope, intercept, r, _, _ = linregress(freqs, erbs)
        results['erb_fit'] = {'a': float(slope), 'b': float(intercept),
                             'r_squared': float(r**2)}
    results['n_points_used'] = len(erb_points)

    # 5. 保存结果
    output_fn = os.path.join(output_dir, f'{model_name}_A1_erb_results.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(output_fn, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {output_fn}")

    return results


def _build_expt_dict_from_json(json_dict: dict) -> dict:
    """从 Josh-format 预测 JSON 构建 expt_dict.

    处理与 Josh 的 load_f0_expt_dict_from_json 兼容的数据格式.
    """
    expt_dict = {}

    # F0 真值
    if 'f0' in json_dict:
        expt_dict['f0'] = np.array(json_dict['f0'])
    elif 'f0_label:labels_true' in json_dict:
        # 需要从 labels 解码 → 简化: 假设数据中有直接的 f0 字段
        expt_dict['f0'] = np.array(json_dict.get('f0', []))

    # F0 预测值
    if 'f0_pred' in json_dict:
        expt_dict['f0_pred'] = np.array(json_dict['f0_pred'])
    elif 'f0_label:labels_pred' in json_dict:
        expt_dict['f0_pred'] = np.array(json_dict.get('f0_pred', []))

    # 元数据
    for key in ['phase_mode', 'filter_fl', 'filter_fh', 'base_f0', 'low_harm']:
        if key in json_dict:
            expt_dict[key] = np.array(json_dict[key])

    return expt_dict


if __name__ == '__main__':
    print("[compute_pitch_doubling_ratio] Module loaded.")
    print("  Functions: compute_f0_pred_ratio(), extract_erb_from_pitch_doubling()")
    print("  Pipeline: run_strategy_A1_pipeline()")
