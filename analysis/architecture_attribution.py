#!/usr/bin/env python3
"""
ERB 形成机制的架构归因分析 — Phase 6.

分析维度:
  1. 前端表示 → ERB 下界 (比较 VQT/log-spectrogram vs raw waveform 前端的 ERB)
  2. 架构家族 → ERB 分组特征
  3. 卷积核/感受野 → 频率平滑效应标记
  4. 训练数据/任务 → ERB 差异的潜在解释

用法:
    python architecture_attribution.py --erb-results results/data/consensus_erb_all_models.json
"""

import sys, os, json, numpy as np
from scipy.stats import kruskal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import MODEL_LIST, MODEL_PARAMS


# 模型架构家族分类
ARCHITECTURE_FAMILIES = {
    'pure_cnn':     ['CREPE', 'FCN', 'PENN_FCN', 'SWIFTF0'],
    'cnn_toeplitz': ['PESTO'],
    'crnn_lstm':    ['JDC'],
    'unet_attention': ['FTATTENTION'],
    'mrdc_conv':    ['HARMOF0'],
    'unet_gru':     ['RMVPE'],
    'transformer':  ['FCPE'],
}

# 前端类型分组
FRONTEND_GROUPS = {
    'raw_waveform':    ['CREPE', 'FCN', 'PENN_FCN'],
    'internal_mel/stft': ['RMVPE', 'FCPE', 'SWIFTF0'],
    'explicit_spectrogram': ['JDC', 'FTATTENTION', 'HARMOF0', 'PESTO'],
}

# 显式 Q 因子前端 (可计算 frontend ERB)
EXPLICIT_Q_MODELS = {
    'PESTO':    {'q_factor': 12,      'frontend_erb_factor': 0.058},   # VQT 12 bins/oct
    'HARMOF0':  {'q_factor': 48,      'frontend_erb_factor': 0.015},   # Log-spectrogram
}


def analyze_architecture_attribution(erb_results: dict, output_path: str = None) -> dict:
    """执行架构归因分析.

    Args:
        erb_results: {model: consensus_erb_dict}
    Returns:
        归因分析结果字典
    """
    # 按架构家族分组 ERB 相似度
    family_erb_sims = {}
    for family, models in ARCHITECTURE_FAMILIES.items():
        sims = []
        for m in models:
            if m in erb_results:
                s = erb_results[m].get('erb_similarity_vs_human', {})
                if isinstance(s, dict) and s.get('pearson_r') is not None:
                    sims.append(s['pearson_r'])
        if sims:
            family_erb_sims[family] = {
                'mean': float(np.mean(sims)), 'std': float(np.std(sims)),
                'n': len(sims), 'models': models,
            }

    # 前端类型分组
    frontend_erb_sims = {}
    for group, models in FRONTEND_GROUPS.items():
        sims = []
        for m in models:
            if m in erb_results:
                s = erb_results[m].get('erb_similarity_vs_human', {})
                if isinstance(s, dict) and s.get('pearson_r') is not None:
                    sims.append(s['pearson_r'])
        if sims:
            frontend_erb_sims[group] = {
                'mean': float(np.mean(sims)), 'std': float(np.std(sims)), 'n': len(sims),
            }

    # 前端 ERB vs 模型整体 ERB (对显式 Q 模型)
    frontend_vs_model = {}
    for model, params in EXPLICIT_Q_MODELS.items():
        if model in erb_results:
            erb_fit = erb_results[model].get('erb_fit', {})
            if erb_fit.get('a') is not None:
                model_erb_factor = erb_fit['a']  # ERB_model = a*f + b ≈ a*f
                frontend_vs_model[model] = {
                    'frontend_erb_factor': params['frontend_erb_factor'],
                    'model_erb_factor': float(model_erb_factor),
                    'ratio_model_vs_frontend': float(model_erb_factor / params['frontend_erb_factor']),
                }

    results = {
        'architecture_families': family_erb_sims,
        'frontend_groups': frontend_erb_sims,
        'frontend_vs_model': frontend_vs_model,
        'constraint_held': all(
            v['ratio_model_vs_frontend'] >= 1.0
            for v in frontend_vs_model.values()
        ),
    }

    print(f"[Architecture Attribution] Constraint 'ERB only widens' held: "
          f"{results['constraint_held']}")
    for family, data in family_erb_sims.items():
        print(f"  {family}: mean ERB sim = {data['mean']:.3f} ± {data['std']:.3f} (n={data['n']})")
    for model, data in frontend_vs_model.items():
        print(f"  {model}: frontend ERB/f = {data['frontend_erb_factor']:.4f}, "
              f"model ERB/f = {data['model_erb_factor']:.4f} "
              f"(ratio = {data['ratio_model_vs_frontend']:.2f}x)")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    print("[architecture_attribution] Module loaded.")
