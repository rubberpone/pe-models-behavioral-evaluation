#!/usr/bin/env python3
"""
策略 A2: Exp A 现有数据 Per-F0 F0DL 膝点重分析.

利用 Exp A 现有预测数据中的 10 个 base_f0 (100–300 Hz, 对数均匀)
和 30 个最低谐波编号, 对每个 base_f0 独立绘制 F0DL vs N 曲线,
拟合 sigmoid 找膝点 N_knee → 感知 ERB(N_knee × F0) ≈ F0.

重分析改动 (相对于 Josh 原有分析):
  1. 对 10 个 base_f0 做 per-F0 分层 (而非聚合)
  2. 对每个 F0 的 F0DL-vs-N 曲线做 sigmoid 拟合 → 提取 N_knee

用法:
    python reanalyze_ExpA_per_F0.py --model CREPE
    python reanalyze_ExpA_per_F0.py --all
"""

import sys
import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import (
    MODEL_LIST, MODEL_PSYCHOPHYSICS_DIRS, EXISTING_EXP_A_PARAMS,
    create_erb_result_dict, compute_erb_similarity,
    get_model_prediction_dir, sigmoid, fit_sigmoid_transition
)
from compute_F0DL import filter_expt_dict


def find_exp_a_results(model_name: str) -> str:
    """查找某模型的 Exp A 结果 JSON (包含已有 per-condition F0DL)."""
    base_dir = MODEL_PSYCHOPHYSICS_DIRS.get(model_name)
    if not base_dir:
        raise ValueError(f"Unknown model: {model_name}")

    import glob
    candidates = [
        os.path.join(base_dir, 'josh_results', 'exp_a', '*results_dict*.json'),
        os.path.join(base_dir, 'results', 'ExpA', '*results_dict*.json'),
        os.path.join(base_dir, 'EVAL_SOFTMAX_lowharm_v01_bestckpt_results_dict.json'),
    ]
    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find Exp A results for {model_name}")


def extract_f0dl_by_base_f0(results_dict: dict) -> Dict[float, dict]:
    """从 Josh-format Exp A 结果中提取 per-base_f0 的 F0DL 数据.

    Josh 的 results_dict 包含:
      - 'base_f0' (list): 每个条件的 base_f0
      - 'low_harm' (list): 每个条件的 low_harm
      - 'phase_mode' (list): 每个条件的 phase_mode
      - 'f0dl' (list): 每个条件的 F0DL

    Returns:
        {base_f0: {'low_harm': [...], 'f0dl': [...], 'phase_mode': [...]}}
    """
    base_f0s = np.array(results_dict.get('base_f0', results_dict.get('f0_ref', [])))
    low_harms = np.array(results_dict.get('low_harm', []))
    phase_modes = np.array(results_dict.get('phase_mode', []))
    f0dls = np.array(results_dict.get('f0dl', []))

    # 转换列表元素为数值 (处理可能的 None)
    f0dls = np.array([float(f) if f is not None else np.nan for f in f0dls])

    per_f0 = {}
    for bf0 in np.unique(base_f0s):
        mask = base_f0s == bf0
        per_f0[float(bf0)] = {
            'low_harm': low_harms[mask].tolist(),
            'f0dl': f0dls[mask].tolist(),
            'phase_mode': phase_modes[mask].tolist(),
        }
    return per_f0


def detect_f0dl_knee(low_harms: np.ndarray, f0dls: np.ndarray,
                     phase_mode: int = 0) -> float:
    """对单个 F0 的 F0DL-vs-N 曲线检测膝点.

    膝点定义: F0DL 从低→高转变的 N 值 (sigmoid 拟合的过渡点).

    Args:
        low_harms: 最低谐波编号 (1D)
        f0dls: F0DL 值 (%F0, 1D)
        phase_mode: 相位模式 (0=sine, 1=rand). 如果为 None, 使用全部数据
    Returns:
        N_knee: 膝点谐波编号. 若无法拟合则返回 np.nan
    """
    # 过滤 NaN 和极端值
    valid = ~np.isnan(f0dls) & (f0dls > 0.1) & (f0dls < 100.0)
    if np.sum(valid) < 5:
        return np.nan

    x = low_harms[valid]
    y = f0dls[valid]

    # 按 low_harm 排序
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # 检查是否有足够的变异来检测膝点
    y_range = np.max(y) - np.min(y)
    if y_range < 1.0:  # F0DL 变化 < 1% — 没有明显的 RES→URS 过渡
        return np.nan

    # 归一化 y 到 [0, 1]
    y_norm = (y - np.min(y)) / y_range

    try:
        k, N_knee = fit_sigmoid_transition(x, y_norm, p0=[0.5, np.median(x)])
        return N_knee
    except Exception:
        return np.nan


def extract_erb_from_exp_a_knees(per_f0_data: Dict[float, dict]) -> List[Tuple[float, float]]:
    """从 per-F0 膝点检测结果提取感知 ERB 散点 (策略 A2).

    Args:
        per_f0_data: {base_f0: {'low_harm': [...], 'f0dl': [...]}}
    Returns:
        [(f_harmonic, ERB), ...] — 感知 ERB 散点
    """
    erb_points = []

    for base_f0, data in sorted(per_f0_data.items()):
        low_harms = np.array(data['low_harm'])
        f0dls = np.array(data['f0dl'])

        if len(low_harms) < 5:
            continue

        # 对 sine phase (0) 和 overall 分别检测
        N_knee = detect_f0dl_knee(low_harms, f0dls)
        if not np.isnan(N_knee) and N_knee > 0:
            erb_points.append((N_knee * base_f0, base_f0))

    return erb_points


def analyze_model(model_name: str,
                  override_results_path: str = None) -> dict:
    """对单个模型运行策略 A2 分析."""
    print(f"\n{'='*60}")
    print(f"[A2] {model_name}")
    print(f"{'='*60}")

    if override_results_path:
        results_path = override_results_path
    else:
        results_path = find_exp_a_results(model_name)
    print(f"  Results: {results_path}")

    with open(results_path, 'r') as f:
        results_dict = json.load(f)

    # 提取 per-F0 数据
    per_f0 = extract_f0dl_by_base_f0(results_dict)
    print(f"  Base F0s: {sorted(per_f0.keys())}")

    # 检测膝点
    erb_points = extract_erb_from_exp_a_knees(per_f0)

    # 构建结果
    results = create_erb_result_dict(model_name, 'A2_ExpA_knee')
    results['erb_points'] = [(float(f), float(e)) for f, e in erb_points]
    results['n_points_used'] = len(erb_points)

    if len(erb_points) >= 3:
        results['erb_similarity_vs_human'] = compute_erb_similarity(erb_points)
        from scipy.stats import linregress
        freqs = np.array([p[0] for p in erb_points])
        erbs = np.array([p[1] for p in erb_points])
        slope, intercept, r, _, _ = linregress(freqs, erbs)
        results['erb_fit'] = {
            'a': float(slope), 'b': float(intercept),
            'r_squared': float(r**2)
        }

    # 保存
    output_dir = get_model_prediction_dir(model_name, 'ExpA_reanalysis')
    output_fn = os.path.join(output_dir, f'{model_name}_A2_erb_results.json')
    with open(output_fn, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  ERB points: {len(erb_points)}")
    for f_harm, erb_val in erb_points:
        print(f"    ERB({f_harm:.0f} Hz) ≈ {erb_val:.1f} Hz")
    print(f"  Saved: {output_fn}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='策略 A2: Exp A 现有数据 Per-F0 F0DL 膝点重分析'
    )
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--results-path', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.all:
        models = MODEL_LIST
    elif args.model:
        models = [args.model]
    else:
        parser.print_help()
        return

    all_results = {}
    for model in models:
        try:
            r = analyze_model(model, args.results_path)
            all_results[model] = r
        except FileNotFoundError as e:
            print(f"  [SKIP] {model}: {e}")
        except Exception as e:
            print(f"  [ERROR] {model}: {e}")
            import traceback
            traceback.print_exc()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'results', 'data', 'strategy_A', 'A2_ExpA_all_models.json'
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[{len(all_results)}/{len(models)} models completed]")


if __name__ == '__main__':
    main()
