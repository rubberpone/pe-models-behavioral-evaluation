#!/usr/bin/env python3
"""
感知 ERB 曲线可视化模块.

绘制:
  1. 全部 10 个模型的感知 ERB(f) 曲线 + 人类耳蜗 ERB 参照
  2. 跨范式 ERB 一致性散点图
  3. ERB 相似度 vs 工程 HM 散点图
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import human_erb, MODEL_LIST


def plot_all_erb_curves(consensus_results: dict,
                        output_path: str = None,
                        figsize: tuple = (10, 7)):
    """绘制所有模型的感知 ERB 曲线 (log-log).

    Args:
        consensus_results: {model_name: erb_result_dict}
        output_path: 图片保存路径
        figsize: 图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 人类耳蜗 ERB (黑色粗线, 参照)
    f_ref = np.logspace(np.log10(100), np.log10(4000), 100)
    erb_ref = human_erb(f_ref)
    ax.plot(f_ref, erb_ref, 'k-', lw=3, label='Human cochlear ERB (Glasberg & Moore 1990)',
            zorder=10, alpha=0.8)

    # 各模型的感知 ERB
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, model in enumerate(MODEL_LIST):
        if model not in consensus_results:
            continue
        result = consensus_results[model]
        points = result.get('erb_points', [])
        if len(points) < 2:
            continue

        freqs = np.array([p[0] for p in points])
        erbs = np.array([p[1] for p in points])

        # 散点
        ax.scatter(freqs, erbs, color=colors[idx], s=40, alpha=0.7,
                  edgecolors='k', linewidth=0.5)

        # 拟合线 (如果有)
        fit = result.get('erb_fit', {})
        if fit.get('a') is not None:
            f_line = np.logspace(np.log10(np.min(freqs)*0.8),
                                np.log10(np.max(freqs)*1.2), 50)
            erb_line = fit['a'] * f_line + fit['b']
            ax.plot(f_line, erb_line, color=colors[idx], lw=1.5, alpha=0.6,
                   label=f"{model} (r={result.get('erb_similarity_vs_human', {}).get('pearson_r', 0):.2f})")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=13)
    ax.set_ylabel('ERB (Hz)', fontsize=13)
    ax.set_title('Perceptual ERB: 10 PE Models vs Human Cochlear ERB', fontsize=14)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([80, 5000])
    ax.set_ylim([10, 1000])

    plt.tight_layout()

    if output_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(project_root, 'results', 'figures', 'ERB_curves_all_models.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot_ERB_curves] Saved: {output_path}")


def plot_cross_paradigm_consistency(consensus_results: dict,
                                     output_path: str = None,
                                     figsize: tuple = (12, 4)):
    """绘制跨范式 ERB 一致性: (A1 vs consensus), (A2 vs consensus), (A3 vs consensus).

    对于每个模型, 显示各策略的散点与综合 ERB 的拟合线.
    """
    n_models = len([m for m in MODEL_LIST if m in consensus_results])
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    strategy_labels = [
        ('A1', 'Exp B: Pitch Doubling'),
        ('A2', 'Exp A: F0DL knee'),
        ('A3', 'Exp C: Pitch Shift slope'),
    ]

    for ax_idx, (strategy, label) in enumerate(strategy_labels):
        ax = axes[ax_idx]
        all_freqs, all_erbs = [], []

        for model in MODEL_LIST:
            if model not in consensus_results:
                continue
            # 查找该策略的散点
            result = consensus_results[model]
            points = result.get('erb_points', [])
            freqs = np.array([p[0] for p in points])
            erbs = np.array([p[1] for p in points])
            ax.scatter(freqs, erbs, s=15, alpha=0.5)

        # 人类 ERB
        f_ref = np.logspace(np.log10(80), np.log10(4000), 50)
        ax.plot(f_ref, human_erb(f_ref), 'k--', lw=2, alpha=0.5, label='Human ERB')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Frequency (Hz)')
        if ax_idx == 0:
            ax.set_ylabel('ERB (Hz)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(project_root, 'results', 'figures',
                                   'cross_paradigm_consistency.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot_ERB_curves] Saved: {output_path}")


if __name__ == '__main__':
    print("[plot_ERB_curves] Module loaded.")
