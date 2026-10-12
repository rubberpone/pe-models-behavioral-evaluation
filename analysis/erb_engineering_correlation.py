#!/usr/bin/env python3
"""
ERB-工程关联分析 — Phase 4 核心.

加载共识 ERB 结果和 QC 工程数据, 计算:
  1. ERB 相似度 vs 工程 HM (散点图 + Spearman r)
  2. ERB 相似度 vs 各 HM 子指标 (CA, OA, RPA, ...)
  3. 按 ERB 相似度分组 → 比较两组工程表现差异
  4. 按工程 HM 分组 → 比较两组 ERB 差异

两种叙事逻辑并行:
  叙事 1: 按工程 HM 分组 → 比较 ERB
  叙事 2: 按 ERB 相似度分组 → 解释工程差异 (导师建议优先)
"""

import sys, os, json, numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import MODEL_LIST


def load_qc_data(qc_json_path: str) -> dict:
    """加载 QC 工程评估结果.

    Args:
        qc_json_path: QC 工程结果 JSON 路径
    Returns:
        {model: {'HM': ..., 'RPA': ..., 'CA': ..., ...}}
    """
    if not os.path.exists(qc_json_path):
        print(f"[WARNING] QC data not found: {qc_json_path}")
        return {}

    with open(qc_json_path, 'r') as f:
        qc_data = json.load(f)
    return qc_data


def erb_engineering_correlation(
    erb_results: dict, qc_data: dict,
    output_path: str = None
) -> dict:
    """计算 ERB-工程相关性.

    Args:
        erb_results: {model: consensus_erb_dict}
        qc_data: {model: qc_metrics_dict}
    Returns:
        相关性分析结果字典
    """
    models = [m for m in MODEL_LIST if m in erb_results and m in qc_data]
    if len(models) < 3:
        return {'error': 'insufficient data'}

    erb_sims = []
    hms = []

    for model in models:
        erb_result = erb_results[model]
        sim = erb_result.get('erb_similarity_vs_human', {})
        if isinstance(sim, dict) and sim.get('pearson_r') is not None:
            erb_sims.append(sim['pearson_r'])
            hms.append(qc_data[model].get('HM', np.nan))

    erb_sims = np.array(erb_sims)
    hms = np.array(hms)

    # 过滤 NaN
    valid = ~np.isnan(hms)
    erb_sims = erb_sims[valid]
    hms = hms[valid]

    if len(erb_sims) < 3:
        return {'error': 'too few valid data points'}

    r_narrative1, p1 = spearmanr(hms, erb_sims)
    r_narrative2, p2 = pearsonr(hms, erb_sims)

    # 分组分析 (叙事 2: ERB 相似度中位分割)
    median_erb = np.median(erb_sims)
    erb_like_group = hms[erb_sims >= median_erb]
    erb_unlike_group = hms[erb_sims < median_erb]

    # Mann-Whitney U test
    if len(erb_like_group) >= 3 and len(erb_unlike_group) >= 3:
        u_stat, u_pval = mannwhitneyu(erb_like_group, erb_unlike_group)
    else:
        u_stat, u_pval = np.nan, np.nan

    results = {
        'n_models': int(len(models)),
        'n_valid': int(len(erb_sims)),
        'narrative_1': {
            'description': '工程 HM → ERB 相似度',
            'spearman_r': float(r_narrative1), 'spearman_p': float(p1),
        },
        'narrative_2': {
            'description': 'ERB 相似度 → 工程 HM',
            'pearson_r': float(r_narrative2), 'pearson_p': float(p2),
            'erb_like_mean_hm': float(np.mean(erb_like_group)),
            'erb_unlike_mean_hm': float(np.mean(erb_unlike_group)),
            'mann_whitney_u': float(u_stat), 'mann_whitney_p': float(u_pval),
        },
        'per_model': {m: {'erb_sim': float(erb_sims[i]), 'hm': float(hms[i])}
                     for i, m in enumerate(models) if i < len(erb_sims)},
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"[ERB-Engineering] Narrative 1: Spearman r={r_narrative1:.3f}, p={p1:.4f}")
    print(f"[ERB-Engineering] Narrative 2: Pearson r={r_narrative2:.3f}, p={p2:.4f}")
    print(f"[ERB-Engineering] ERB-like group HM={np.mean(erb_like_group):.3f} "
          f"vs unlike={np.mean(erb_unlike_group):.3f}")

    return results


if __name__ == '__main__':
    print("[erb_engineering_correlation] Module loaded.")
