#!/usr/bin/env python3
"""
模型特异性 RES/URS 重定义 — Phase 5.

基于每个模型自身的感知 ERB(f), 重新计算 α(F0, N) 值,
替换 Song2016 实验中默认的人类 α_low=0.55 / α_high=1.0.

核心逻辑:
  α_model(F0, N) = F0 / ERB_model(N × F0)
  若 α_model < α_low → RES
  若 α_model > α_high → URS
"""

import sys, os, json, numpy as np
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import human_erb, erb_points_to_function, MODEL_LIST


# Song2016 实验使用的 F0
SONG2016_F0S = [110, 155, 220, 311, 440, 622, 880]


def compute_model_alpha(
    f0: float, harmonic_n: int,
    erb_function: callable
) -> float:
    """计算模型特异性的 α = F0 / ERB(N×F0)."""
    f_harmonic = harmonic_n * f0
    erb_val = erb_function(np.array([f_harmonic]))[0]
    return f0 / max(erb_val, 1e-6)


def classify_harmonic_res_urs(
    f0: float, harmonic_n: int,
    erb_function: callable,
    alpha_low: float = 0.55,
    alpha_high: float = 1.0
) -> str:
    """使用模型自身 ERB 分类谐波为 RES / transitional / URS.

    Returns:
        'RES', 'transitional', 或 'URS'
    """
    alpha = compute_model_alpha(f0, harmonic_n, erb_function)
    if alpha < alpha_low:
        return 'RES'
    elif alpha > alpha_high:
        return 'URS'
    else:
        return 'transitional'


def compute_human_classification(f0: float, harmonic_n: int,
                                 alpha_low: float = 0.55,
                                 alpha_high: float = 1.0) -> str:
    """使用人类耳蜗 ERB 分类谐波 (原始 Song2016 方法)."""
    return classify_harmonic_res_urs(
        f0, harmonic_n, human_erb, alpha_low, alpha_high
    )


def compare_classifications(
    erb_results: dict,
    f0s: list = None,
    harmonic_range: Tuple[int, int] = (1, 40),
    output_path: str = None
) -> dict:
    """比较人类 ERB vs 模型自身 ERB 定义下的 RES/URS 分类差异.

    Args:
        erb_results: {model: consensus_erb_dict}
        f0s: F0 值列表
        harmonic_range: (N_min, N_max)
    Returns:
        每个模型的分类比较结果
    """
    if f0s is None:
        f0s = SONG2016_F0S

    comparison = {}
    for model in MODEL_LIST:
        if model not in erb_results:
            continue

        result = erb_results[model]
        erb_points = result.get('erb_points', [])
        if len(erb_points) < 2:
            continue

        # 构建模型自身的 ERB(f) 函数
        erb_func = erb_points_to_function(erb_points, interp_kind='linear')

        n_diff = 0
        n_total = 0
        per_f0 = {}

        for f0 in f0s:
            per_f0[f0] = []
            for n in range(harmonic_range[0], harmonic_range[1] + 1):
                human_class = compute_human_classification(f0, n)
                model_class = classify_harmonic_res_urs(f0, n, erb_func)
                if human_class != model_class:
                    n_diff += 1
                n_total += 1
                per_f0[f0].append({
                    'harmonic': n, 'human': human_class, 'model': model_class,
                })

        comparison[model] = {
            'classification_agreement': float(1.0 - n_diff / n_total),
            'n_differences': n_diff, 'n_total': n_total,
            'per_f0': per_f0,
        }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)

    for model, comp in comparison.items():
        print(f"  {model}: agreement = {comp['classification_agreement']:.2%} "
              f"({comp['n_differences']}/{comp['n_total']} diffs)")

    return comparison


if __name__ == '__main__':
    print("[model_specific_RES_URS] Module loaded.")
