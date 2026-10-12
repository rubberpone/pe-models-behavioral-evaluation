#!/usr/bin/env python3
"""
策略 A1: Exp B 现有数据 Per-F0 Pitch Doubling 重分析.

利用 Exp B 现有预测数据中的密集 F0 grid (80–320 Hz, 6144 个 F0 值)
和 3 个 filter bands, 通过扩展 F0 bin centers + 重新启用 Mid band 来提取
每个模型的感知 ERB 散点。

重分析改动 (相对于 Josh 原有分析):
  1. f0_bin_centers: [125] → [62, 88, 125, 177, 250, 354]
  2. 重新启用 Mid band (1375–1875 Hz)
  3. Per-(F0, filter_band) 独立分析

用法:
    python reanalyze_ExpB_per_F0.py --model CREPE
    python reanalyze_ExpB_per_F0.py --all  # 对全部 10 个模型
"""

import sys
import os
import json
import argparse
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import (
    MODEL_LIST, MODEL_PSYCHOPHYSICS_DIRS, EXISTING_EXP_B_PARAMS,
    create_erb_result_dict, compute_erb_similarity,
    get_model_prediction_dir
)
from compute_pitch_doubling_ratio import (
    extract_erb_from_pitch_doubling, _build_expt_dict_from_json
)
from data_loader import load_prediction_json, build_F0N_data_expB, expB_to_F0N_heatmap_data


# ============================================================
#  Exp B 预测数据路径解析
# ============================================================

def find_exp_b_predictions(model_name: str) -> str:
    """查找某个模型的 Exp B 预测 JSON 文件."""
    pred = load_prediction_json(model_name, 'B')
    if pred is not None:
        base_dir = MODEL_PSYCHOPHYSICS_DIRS.get(model_name, '')
        import glob
        matches = glob.glob(os.path.join(base_dir, '**', '*ShackletonCarlyon1994*.json'), recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find Exp B predictions for {model_name}")


# ============================================================
#  主分析流程
# ============================================================

def analyze_model(model_name: str,
                  override_prediction_path: str = None) -> dict:
    """对单个模型运行策略 A1 分析.

    Args:
        model_name: 模型名
        override_prediction_path: 若提供, 使用此路径而非自动查找
    Returns:
        感知 ERB 结果字典
    """
    print(f"\n{'='*60}")
    print(f"[A1] {model_name}")
    print(f"{'='*60}")

    # 使用 data_loader 加载数据
    raw_data = build_F0N_data_expB(model_name)
    if len(raw_data.get('f0', [])) == 0:
        print(f"  [WARNING] Empty prediction data for {model_name}")
        return create_erb_result_dict(model_name, 'A1_ExpB_doubling')

    # 构建 expt_dict 格式 (兼容 extract_erb_from_pitch_doubling)
    expt_dict = {
        'f0': raw_data['f0'],
        'f0_pred': raw_data['f0_pred'],
        'filter_fl': raw_data['filter_fl'],
        'filter_fh': raw_data['filter_fh'],
        'phase_mode': raw_data['phase_mode'],
    }
    print(f"  Loaded {len(expt_dict['f0'])} stimuli")
    erb_points = extract_erb_from_pitch_doubling(
        [expt_dict],
        f0_bin_centers=EXISTING_EXP_B_PARAMS['expanded_f0_bin_centers'],
        f0_bin_width=0.10,
        phase_mode=4
    )

    # 5. 构建结果
    results = create_erb_result_dict(model_name, 'A1_ExpB_doubling')
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

    # 6. 保存
    output_dir = get_model_prediction_dir(model_name, 'ExpB_reanalysis')
    output_fn = os.path.join(output_dir, f'{model_name}_A1_erb_results.json')
    with open(output_fn, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  ERB points: {len(erb_points)}")
    for f_harm, erb_val in erb_points:
        print(f"    ERB({f_harm:.0f} Hz) ≈ {erb_val:.1f} Hz")
    print(f"  Saved: {output_fn}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='策略 A1: Exp B 现有数据 Per-F0 Pitch Doubling 重分析'
    )
    parser.add_argument('--model', type=str, default=None,
                       help='单个模型名 (如 CREPE)')
    parser.add_argument('--all', action='store_true',
                       help='对全部 10 个模型运行')
    parser.add_argument('--prediction-path', type=str, default=None,
                       help='手动指定预测文件路径 (覆盖自动查找)')
    parser.add_argument('--output', type=str, default=None,
                       help='汇总结果输出路径')
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
            r = analyze_model(model, args.prediction_path)
            all_results[model] = r
        except FileNotFoundError as e:
            print(f"  [SKIP] {model}: {e}")
        except Exception as e:
            print(f"  [ERROR] {model}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总保存
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'results', 'data', 'strategy_A', 'A1_ExpB_all_models.json'
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[{len(all_results)}/{len(models)} models completed]")
    print(f"Summary saved: {args.output}")


if __name__ == '__main__':
    main()
