#!/usr/bin/env python3
"""
F0DL 计算模块 — 从模型预测模拟 2AFC F0 辨别实验.

核心逻辑复制自 /data/zyf/pitchnet/assets_psychophysics/f0dl_bernox.py
的 run_f0dl_experiment().

提供:
  - compute_F0DL_for_condition(): 对单个 (base_f0, low_harm, phase_mode) 条件计算 F0DL
  - run_F0DL_experiment(): 批量运行所有条件 (并行化)
  - 辅助函数: load_expt_dict, add_f0_estimates, filter_expt_dict, fit_normcdf
"""

import sys
import os
import json
import numpy as np
import scipy.optimize
import scipy.stats
import itertools
import functools
import multiprocessing
from typing import Dict, List, Optional, Tuple

# ============================================================
#  从 Josh 项目复制的核心函数
#  来源: /data/zyf/pitchnet/assets_psychophysics/f0dl_bernox.py
# ============================================================

def fit_normcdf(xvals: np.ndarray, yvals: np.ndarray,
                mu: float = 0.0) -> Tuple[float, float]:
    """拟合 normal CDF (固定均值 mu).

    Args:
        xvals: 自变量 (%F0 difference)
        yvals: 因变量 (fraction correct)
        mu: 固定均值
    Returns:
        (sigma_opt, sigma_opt_cov)
    """
    normcdf = lambda x, sigma: scipy.stats.norm(mu, sigma).cdf(x)
    sigma_opt, sigma_opt_cov = scipy.optimize.curve_fit(normcdf, xvals, yvals)
    return float(np.squeeze(sigma_opt)), float(np.squeeze(sigma_opt_cov))


def get_empirical_psychometric_function(
    pct_diffs: np.ndarray,
    judgments: np.ndarray,
    bin_width: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """从 pairwise 比较数据构建经验心理测量函数.

    Args:
        pct_diffs: %F0 差异数组 (一维)
        judgments: 判断数组 (一维, 0 或 1)
        bin_width: bin 宽度 (%F0)
    Returns:
        (bins, bin_means)
    """
    bins = np.arange(np.min(pct_diffs), np.max(pct_diffs), bin_width)
    bin_idx = np.digitize(pct_diffs, bins)
    bin_means = np.zeros(len(bins))
    bin_counts = np.zeros(len(bins))
    for i, bi in enumerate(bin_idx):
        bin_means[bi - 1] += judgments[i]
        bin_counts[bi - 1] += 1
    keep = bin_counts > 0
    return bins[keep], bin_means[keep] / bin_counts[keep]


def add_f0_judgments_to_expt_dict(
    expt_dict: dict,
    f0_true_key: str = 'f0',
    f0_pred_key: str = 'f0_pred',
    max_pct_diff: float = 100.0 / 6,
    noise_stdev: float = 1e-12
) -> dict:
    """模拟 2AFC F0 辨别实验.

    从模型预测生成 pairwise 判断矩阵.

    Args:
        expt_dict: 包含 f0_true 和 f0_pred 的字典
        max_pct_diff: 仅比较 %F0 差异 <= max_pct_diff 的刺激对
        noise_stdev: 决策阶段高斯噪声标准差 (打破平局)
    Returns:
        expt_dict, 添加了 'pairwise_pct_diffs' 和 'pairwise_judgments'
    """
    sort_idx = np.argsort(expt_dict[f0_true_key])
    f0_true = np.array(expt_dict[f0_true_key])[sort_idx]
    f0_pred = np.array(expt_dict[f0_pred_key])[sort_idx]

    N = len(f0_true)
    pairwise_pct_diffs = np.full([N, N], np.nan, dtype=np.float32)
    pairwise_judgments = np.full([N, N], np.nan, dtype=np.float32)

    for idx_ref in range(N):
        f0_ref = f0_true[idx_ref]
        f0_pred_ref = f0_pred[idx_ref]

        pct_diffs = 100.0 * (f0_true - f0_ref) / f0_ref
        comparable = np.abs(pct_diffs) <= max_pct_diff
        pairwise_pct_diffs[idx_ref, comparable] = pct_diffs[comparable]

        pred_pct_diffs = 100.0 * (f0_pred[comparable] - f0_pred_ref) / f0_pred_ref
        noise = noise_stdev * np.random.randn(np.sum(comparable))
        pairwise_judgments[idx_ref, comparable] = \
            np.array(pred_pct_diffs > noise, dtype=np.float32)

    expt_dict['pairwise_pct_diffs'] = pairwise_pct_diffs
    expt_dict['pairwise_judgments'] = pairwise_judgments
    return expt_dict


def filter_expt_dict(expt_dict: dict,
                     filter_dict: dict = None) -> dict:
    """按条件过滤 expt_dict.

    Args:
        expt_dict: 实验数据字典
        filter_dict: {key: value} 或 {key: [min, max]} 的过滤条件
    Returns:
        过滤后的字典 (浅拷贝)
    """
    if filter_dict is None:
        return expt_dict

    keep_idx = np.ones(len(expt_dict[list(filter_dict.keys())[0]]), dtype=bool)
    for key, fvalue in filter_dict.items():
        val = np.array(expt_dict[key])
        if hasattr(fvalue, '__len__') and not isinstance(fvalue, str):
            keep_idx &= (val >= fvalue[0]) & (val <= fvalue[1])
        else:
            keep_idx &= (val == fvalue)

    filtered = {}
    for key in expt_dict:
        val = expt_dict[key]
        if isinstance(val, (np.ndarray, list)):
            val = np.array(val)
            if val.ndim == 1 and len(val) == len(keep_idx):
                filtered[key] = val[keep_idx]
            else:
                filtered[key] = val
        else:
            filtered[key] = val
    return filtered


# ============================================================
#  F0DL 计算核心
# ============================================================

def compute_F0DL_for_condition(
    predictions: dict,
    base_f0: float,
    max_pct_diff: float = 100.0 / 6,
    noise_stdev: float = 1e-12,
    bin_width: float = 0.05,
    threshold_value: float = 0.707,
    mu: float = 0.0,
    use_empirical_f0dl: bool = False,
    threshold_cap: float = 100.0
) -> float:
    """对单个实验条件计算 F0DL (%F0).

    模拟 2AFC 范式：对同一条件下的 121 个 delta_f0 刺激做 pairwise 比较.

    Args:
        predictions: 该条件下所有刺激的预测字典.
                     必须包含: 'f0' (真值 Hz), 'f0_pred' (预测 Hz)
        base_f0: 参考 F0 (Hz) — 用于 ±0.5 octave 先验
        max_pct_diff: 最大 %F0 差异阈值
        noise_stdev: 决策噪声标准差
        bin_width: 心理测量函数 bin 宽度
        threshold_value: F0DL 阈值 (0.707 = 70.7% correct for 2-down-1-up)
        mu: Normal CDF 均值
        use_empirical_f0dl: 若 True, 使用经验阈值而非拟合值
        threshold_cap: F0DL 上限 (%), 超过此值封顶
    Returns:
        F0DL (%F0)
    """
    # 1. Apply ±0.5 octave prior (using base_f0 as reference)
    f0_true = np.array(predictions['f0'])
    f0_pred_raw = np.array(predictions['f0_pred'])
    f0_bins = predictions.get('f0_bins')

    if f0_bins is not None and predictions.get('f0_label:probs_out') is not None:
        # Prior-constrained argmax from 700-class probability distribution
        probs = np.array(predictions['f0_label:probs_out'])
        f0_pred = np.zeros_like(f0_true)
        for i in range(len(f0_true)):
            prior_lo = base_f0 * np.power(2.0, -0.5)
            prior_hi = base_f0 * np.power(2.0, 0.5)
            mask = (f0_bins >= prior_lo) & (f0_bins <= prior_hi)
            masked_probs = probs[i] * mask.astype(float)
            f0_pred[i] = f0_bins[np.argmax(masked_probs)]
    else:
        f0_pred = f0_pred_raw

    # 2. Simulate 2AFC pairwise comparisons
    expt_dict = {
        'f0': f0_true.tolist(),
        'f0_pred': f0_pred.tolist(),
    }
    expt_dict = add_f0_judgments_to_expt_dict(
        expt_dict,
        f0_true_key='f0', f0_pred_key='f0_pred',
        max_pct_diff=max_pct_diff, noise_stdev=noise_stdev
    )

    # 3. Build empirical psychometric function
    pct_diffs = expt_dict['pairwise_pct_diffs'].reshape([-1])
    pct_diffs = pct_diffs[~np.isnan(pct_diffs)]
    judgments = expt_dict['pairwise_judgments'].reshape([-1])
    judgments = judgments[~np.isnan(judgments)]

    if len(pct_diffs) < 10:
        return threshold_cap  # insufficient data

    bins, bin_means = get_empirical_psychometric_function(
        pct_diffs, judgments, bin_width=bin_width
    )

    if len(bins) < 3:
        return threshold_cap

    # 4. Fit Normal CDF and extract threshold
    sigma_opt, _ = fit_normcdf(bins, bin_means, mu=mu)
    f0dl = scipy.stats.norm(mu, sigma_opt).ppf(threshold_value)

    # 5. Optionally replace with empirical value
    if use_empirical_f0dl and np.any(bins >= 0):
        above = (bins >= 0) & (bin_means > threshold_value)
        if np.any(above):
            f0dl = bins[above][0]

    return min(float(f0dl), threshold_cap)


def run_F0DL_experiment_parallel(
    predictions_by_condition: List[dict],
    max_processes: int = 60,
    **kwargs
) -> dict:
    """并行运行 F0DL 实验.

    Args:
        predictions_by_condition: 列表, 每个元素是一个条件的预测数据,
            包含 'base_f0', 'low_harm', 'phase_mode', 'f0', 'f0_pred' 等
        max_processes: 最大并行进程数
        **kwargs: 传递给 compute_F0DL_for_condition()
    Returns:
        results_dict, 包含 'base_f0', 'low_harm', 'phase_mode', 'f0dl' 列表
    """
    def _run_one(idx, preds, base_f0):
        f0dl = compute_F0DL_for_condition(preds, base_f0, **kwargs)
        return idx, {
            'base_f0': base_f0,
            'low_harm': preds.get('low_harm'),
            'phase_mode': preds.get('phase_mode'),
            'f0dl': f0dl,
        }

    tasks = []
    for i, preds in enumerate(predictions_by_condition):
        bf0 = preds['base_f0']
        if isinstance(bf0, (list, np.ndarray)):
            bf0 = float(bf0[0]) if len(np.atleast_1d(bf0)) > 0 else float(bf0)
        tasks.append((i, preds, float(bf0)))

    N = len(tasks)
    results = {'base_f0': [None]*N, 'low_harm': [None]*N,
               'phase_mode': [None]*N, 'f0dl': [None]*N}

    with multiprocessing.Pool(processes=min(N, max_processes)) as pool:
        parallel_results = pool.starmap(_run_one, tasks)
        for idx, sub in parallel_results:
            for k in sub:
                results[k][idx] = sub[k]

    return results


# ============================================================
#  命令行接口 (用于单个条件调试)
# ============================================================

if __name__ == '__main__':
    print("[compute_F0DL] Module loaded.")
    print("  Functions: compute_F0DL_for_condition(), run_F0DL_experiment_parallel()")
    print("  Usage: import this module and call compute_F0DL_for_condition()")
