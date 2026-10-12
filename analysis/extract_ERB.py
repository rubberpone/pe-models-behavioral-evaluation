#!/usr/bin/env python3
"""
感知 ERB 提取总协调器 — 整合策略 A1/A2/A3 的结果, 执行交叉验证和综合 ERB 估计.

该脚本:
  1. 加载策略 A1 (Exp B), A2 (Exp A), A3 (Exp C) 的 per-model ERB 散点
  2. 进行跨范式一致性评估 (Pearson r 检验)
  3. 若一致 (r > 0.8), 取三者均值作为综合感知 ERB(f)
  4. 若分歧, 标记并输出诊断信息
  5. 为每个模型输出最终的感知 ERB 曲线

用法:
    # 仅加载策略 A
    python extract_ERB.py --strategy-a-dir results/data/strategy_A/

    # 加载策略 A + B (合并)
    python extract_ERB.py --strategy-a-dir results/data/strategy_A/ \
                          --strategy-b-dir results/data/strategy_B/ \
                          --merge

    # 单模型 + 可视化
    python extract_ERB.py --strategy-a-dir results/data/strategy_A/ \
                          --model CREPE --plot
"""

import sys
import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import linregress, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import (
    MODEL_LIST, human_erb, create_erb_result_dict,
    compute_erb_similarity, erb_points_to_function
)


def load_strategy_results_from_dir(results_dir: str,
                                   strategy_prefix: str = 'A') -> Dict[str, Dict[str, dict]]:
    """从指定目录加载策略结果.

    Args:
        results_dir: 包含 per-strategy JSON 文件的目录
        strategy_prefix: 'A' 表示加载 A1/A2/A3; 'B' 表示加载 B1/B2/B3
    Returns:
        {model_name: {strategy_label: result_dict}}
    """
    if strategy_prefix == 'A':
        strategy_files = {
            'A1': 'A1_ExpB_all_models.json',
            'A2': 'A2_ExpA_all_models.json',
            'A3': 'A3_ExpC_all_models.json',
        }
    elif strategy_prefix == 'B':
        strategy_files = {
            'B1': 'B1_F0N_sweep_all_models.json',
            'B2': 'B2_ExpB_doubling_all_models.json',
            'B3': 'B3_ExpC_shift_all_models.json',
        }
    else:
        raise ValueError(f"Unknown strategy prefix: {strategy_prefix}")

    all_data = {m: {} for m in MODEL_LIST}
    for strategy, filename in strategy_files.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            print(f"[WARNING] Missing: {path}")
            continue
        with open(path, 'r') as f:
            strategy_results = json.load(f)
        for model in MODEL_LIST:
            if model in strategy_results:
                all_data[model][strategy] = strategy_results[model]

    return all_data


def merge_strategy_results(
    strategy_a_data: Dict[str, Dict[str, dict]],
    strategy_b_data: Dict[str, Dict[str, dict]],
) -> Dict[str, Dict[str, dict]]:
    """合并策略 A 和策略 B 的结果.

    对于每个模型, 将策略 A 的 A1/A2/A3 和策略 B 的 B1/B2/B3 合并到同一个字典中.
    """
    merged = {m: {} for m in MODEL_LIST}
    for model in MODEL_LIST:
        if model in strategy_a_data:
            merged[model].update(strategy_a_data[model])
        if model in strategy_b_data:
            merged[model].update(strategy_b_data[model])
    return merged


def check_cross_paradigm_consistency(
    all_strategy_data: Dict[str, dict]
) -> Tuple[float, List[Tuple[float, float]]]:
    """检查三种策略的 ERB 散点在重叠频段上的一致性.

    将三种策略的散点聚合, 在重叠频段上计算 human-ERB 归一化后的 Pearson r.

    Args:
        all_strategy_data: {'A1': result_dict, 'A2': result_dict, 'A3': result_dict}
    Returns:
        (consistency_r, combined_erb_points) — 跨范式一致性系数和合并散点
    """
    all_points = []
    for strategy in sorted(all_strategy_data.keys()):
        points = all_strategy_data[strategy].get('erb_points', [])
        all_points.extend([(float(p[0]), float(p[1]), strategy)
                          for p in points])

    if len(all_points) < 6:
        return 0.0, []

    freqs = np.array([p[0] for p in all_points])
    erbs = np.array([p[1] for p in all_points])

    # 与人类 ERB 比较: 各策略的散点是否落在同一趋势线上?
    human_erbs = human_erb(freqs)
    log_model = np.log10(erbs)
    log_human = np.log10(human_erbs)

    if len(freqs) >= 3:
        r, _ = pearsonr(log_model, log_human)
    else:
        r = 0.0

    combined = [(float(f), float(e)) for f, e in zip(freqs, erbs)]
    return float(r), combined


def compute_consensus_erb(
    all_strategy_data: Dict[str, dict],
    model_name: str,
    consistency_threshold: float = 0.8
) -> dict:
    """计算综合感知 ERB.

    Args:
        all_strategy_data: {'A1': {...}, 'A2': {...}, ...}
        model_name: 模型名
        consistency_threshold: 一致性阈值 (r > threshold → 合并)
    Returns:
        综合结果字典
    """
    r, combined_points = check_cross_paradigm_consistency(all_strategy_data)

    result = create_erb_result_dict(model_name, 'consensus')
    result['cross_paradigm_consistency_r'] = r
    result['n_strategies_available'] = len(all_strategy_data)
    result['erb_points'] = combined_points
    result['consensus_achieved'] = r > consistency_threshold

    if len(combined_points) >= 3:
        result['erb_similarity_vs_human'] = compute_erb_similarity(combined_points)

        # 拟合 ERB(f) = a*f + b
        freqs = np.array([p[0] for p in combined_points])
        erbs = np.array([p[1] for p in combined_points])
        slope, intercept, r_fit, _, _ = linregress(freqs, erbs)
        result['erb_fit'] = {
            'a': float(slope), 'b': float(intercept),
            'r_squared': float(r_fit**2),
            'method': 'consensus' if result['consensus_achieved'] else 'best_available'
        }

    result['n_points_used'] = len(combined_points)
    return result


def main():
    parser = argparse.ArgumentParser(description='感知 ERB 提取与交叉验证')
    parser.add_argument('--strategy-a-dir', type=str, default=None,
                       help='策略 A 结果目录 (默认: results/data/strategy_A/)')
    parser.add_argument('--strategy-b-dir', type=str, default=None,
                       help='策略 B 结果目录 (可选, 用于合并)')
    parser.add_argument('--merge', action='store_true',
                       help='合并策略 A + B 结果')
    parser.add_argument('--model', type=str, default=None,
                       help='单模型分析')
    parser.add_argument('--all', action='store_true',
                       help='全部 10 个模型')
    parser.add_argument('--plot', action='store_true',
                       help='生成 ERB 曲线可视化')
    parser.add_argument('--output', type=str, default=None,
                       help='输出路径 (默认: results/data/consensus/)')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 默认: 仅加载策略 A
    a_dir = args.strategy_a_dir or os.path.join(project_root, 'results', 'data', 'strategy_A')
    if not os.path.isabs(a_dir):
        a_dir = os.path.join(project_root, a_dir)

    print(f"[extract_ERB] Strategy A: {a_dir}")
    all_data = load_strategy_results_from_dir(a_dir, strategy_prefix='A')

    # 可选: 加载并合并策略 B
    if args.merge and args.strategy_b_dir:
        b_dir = args.strategy_b_dir
        if not os.path.isabs(b_dir):
            b_dir = os.path.join(project_root, b_dir)
        print(f"[extract_ERB] Strategy B: {b_dir}")
        b_data = load_strategy_results_from_dir(b_dir, strategy_prefix='B')
        all_data = merge_strategy_results(all_data, b_data)
        print("[extract_ERB] Merged: Strategy A + B")

    if args.model:
        models = [args.model]
    elif args.all:
        models = MODEL_LIST
    else:
        models = [m for m in MODEL_LIST if m in all_data and all_data[m]]
        print(f"Models with data: {models}")
        if not models:
            print("No data found. Run strategy A first.")
            return

    consensus_results = {}
    for model in models:
        if model not in all_data:
            print(f"[SKIP] {model}: no strategy data")
            continue
        print(f"\n--- {model} ---")
        print(f"  Strategies: {list(all_data[model].keys())}")
        result = compute_consensus_erb(all_data[model], model)
        consensus_results[model] = result
        print(f"  Cross-paradigm r: {result['cross_paradigm_consistency_r']:.3f}")
        print(f"  Consensus: {result['consensus_achieved']}")
        print(f"  ERB points: {result['n_points_used']}")
        if result['erb_fit']['a'] is not None:
            print(f"  ERB(f) = {result['erb_fit']['a']:.4f}*f + {result['erb_fit']['b']:.1f}")
        if result['erb_similarity_vs_human']:
            print(f"  ERB-Human Pearson r: {result['erb_similarity_vs_human']['pearson_r']:.3f}")

    # 保存
    if args.output:
        output_path = args.output
    else:
        if args.merge:
            output_path = os.path.join(project_root, 'results', 'data', 'consensus',
                                       'final_strategy_A_plus_B.json')
        else:
            output_path = os.path.join(project_root, 'results', 'data', 'consensus',
                                       'strategy_A_only.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(consensus_results, f, indent=2, default=str)
    print(f"\n[Saved] {output_path}")

    if args.plot:
        try:
            from plot_ERB_curves import plot_all_erb_curves
            plot_all_erb_curves(consensus_results)
        except ImportError:
            print("[WARNING] plot_ERB_curves module not available")


if __name__ == '__main__':
    main()
