#!/usr/bin/env python3
"""
ERB-行为关联分析 — Phase 4 补充.

加载共识 ERB 结果和 Josh 2021 行为实验的人-模型相似度数据,
计算:
  1. ERB 相似度 vs Josh Exp A-E 各实验的人-模型 Spearman r
  2. ERB 相似度 vs Song2016 Ratio 指标

用法:
    python erb_behavior_correlation.py --erb-results results/data/consensus_erb_all_models.json
"""

import sys, os, json, numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import MODEL_LIST


def load_behavior_data(behavior_json_path: str) -> dict:
    """加载 Josh 行为实验的人-模型相似度数据.

    预期格式:
    {
      "CREPE": {
        "exp_a_human_model_r": 0.85,
        "exp_b_human_model_r": 0.72,
        ...
      },
      ...
    }
    """
    if not os.path.exists(behavior_json_path):
        return {}
    with open(behavior_json_path, 'r') as f:
        return json.load(f)


def erb_behavior_correlation(erb_results: dict, behavior_data: dict,
                              output_path: str = None) -> dict:
    """计算 ERB-行为相关性."""
    exp_keys = ['exp_a', 'exp_b', 'exp_c', 'exp_d', 'exp_e']
    correlations = {}

    for exp_key in exp_keys:
        erb_sims, behavior_vals, models_used = [], [], []

        for model in MODEL_LIST:
            if model not in erb_results or model not in behavior_data:
                continue
            erb_sim = erb_results[model].get('erb_similarity_vs_human', {})
            if isinstance(erb_sim, dict):
                erb_sim = erb_sim.get('pearson_r')
            beh_val = behavior_data[model].get(f'{exp_key}_human_model_r')
            if erb_sim is not None and beh_val is not None:
                erb_sims.append(erb_sim)
                behavior_vals.append(beh_val)
                models_used.append(model)

        if len(erb_sims) >= 5:
            r, p = spearmanr(erb_sims, behavior_vals)
        else:
            r, p = np.nan, np.nan

        correlations[exp_key] = {
            'spearman_r': float(r), 'spearman_p': float(p),
            'n': len(erb_sims), 'models': models_used,
        }

    results = {'correlations': correlations, 'n_models_total': len(MODEL_LIST)}

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    for exp, corr in correlations.items():
        print(f"  {exp}: r={corr['spearman_r']:.3f} (n={corr['n']})")

    return results


if __name__ == '__main__':
    print("[erb_behavior_correlation] Module loaded.")
