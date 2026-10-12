#!/usr/bin/env python3
"""
策略 A3: Exp C 现有数据 Per-F0 Pitch Shift 斜率衰减重分析.

利用 Exp C 现有预测数据中的 3971 个 F0 (80–480 Hz) 和
3 个 spectral envelopes, 对代表性 F0 独立计算 pitch shift slope
作为中心谐波编号的函数, 检测 slope 衰减的 N_decay → 感知 ERB.

用法:
    python reanalyze_ExpC_per_F0.py --model CREPE
    python reanalyze_ExpC_per_F0.py --all
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import (
    MODEL_LIST, MODEL_PSYCHOPHYSICS_DIRS, EXISTING_EXP_C_PARAMS,
    create_erb_result_dict, compute_erb_similarity,
    get_model_prediction_dir
)
from compute_pitch_shift_slope import (
    extract_erb_from_pitch_shift, _build_expt_dict_from_json
)


def find_exp_c_predictions(model_name: str) -> str:
    """查找 Exp C 预测 JSON."""
    base_dir = MODEL_PSYCHOPHYSICS_DIRS.get(model_name)
    if not base_dir:
        raise ValueError(f"Unknown model: {model_name}")
    import glob
    candidates = [
        os.path.join(base_dir, 'josh_results', 'exp_c', '*results*.json'),
        os.path.join(base_dir, 'EVAL_SOFTMAX_freqshifted_v01_bestckpt.json'),
        os.path.join(base_dir, 'EVAL_SOFTMAX_freqshifted_v01_bestckpt_results_dict.json'),
    ]
    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find Exp C predictions for {model_name}")


def analyze_model(model_name: str,
                  override_prediction_path: str = None) -> dict:
    """对单个模型运行策略 A3 分析."""
    print(f"\n{'='*60}")
    print(f"[A3] {model_name}")
    print(f"{'='*60}")

    if override_prediction_path:
        pred_path = override_prediction_path
    else:
        pred_path = find_exp_c_predictions(model_name)
    print(f"  Predictions: {pred_path}")

    with open(pred_path, 'r') as f:
        json_dict = json.load(f)
    expt_dict = _build_expt_dict_from_json(json_dict)

    erb_points = extract_erb_from_pitch_shift(
        expt_dict,
        representative_f0s=EXISTING_EXP_C_PARAMS['representative_f0s'],
        f0_tolerance=0.05
    )

    results = create_erb_result_dict(model_name, 'A3_ExpC_slope')
    results['erb_points'] = [(float(f), float(e)) for f, e in erb_points]
    results['n_points_used'] = len(erb_points)

    if len(erb_points) >= 3:
        results['erb_similarity_vs_human'] = compute_erb_similarity(erb_points)
        from scipy.stats import linregress
        freqs = np.array([p[0] for p in erb_points])
        erbs = np.array([p[1] for p in erb_points])
        slope, intercept, r, _, _ = linregress(freqs, erbs)
        results['erb_fit'] = {'a': float(slope), 'b': float(intercept),
                             'r_squared': float(r**2)}

    output_dir = get_model_prediction_dir(model_name, 'ExpC_reanalysis')
    output_fn = os.path.join(output_dir, f'{model_name}_A3_erb_results.json')
    with open(output_fn, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  ERB points: {len(erb_points)}")
    for f_harm, erb_val in erb_points:
        print(f"    ERB({f_harm:.0f} Hz) ≈ {erb_val:.1f} Hz")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='策略 A3: Exp C 现有数据 Per-F0 Pitch Shift 重分析'
    )
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--prediction-path', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    models = MODEL_LIST if args.all else ([args.model] if args.model else [])
    if not models:
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

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'results', 'data', 'strategy_A', 'A3_ExpC_all_models.json'
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[{len(all_results)}/{len(models)} models completed]")


if __name__ == '__main__':
    main()
