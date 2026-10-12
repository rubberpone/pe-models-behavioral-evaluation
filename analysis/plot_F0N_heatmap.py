#!/usr/bin/env python3
"""
F0-N 二维彩色散点图 — 感知 ERB 的核心可视化模块.

三层视觉结构 (自下而上):
  Layer 1 (底层): 谐波族连线 — 每个 F0 下被测的 N 范围, 极细灰线水平连接
  Layer 2 (中层): 色点散点 — 每个 (N, F0) 条件 = 一个色点, 颜色编码行为指标
  Layer 3 (顶层): ERB 等高线 — 该实验策略拟合的感知 ERB + 人类耳蜗 ERB 参照

三个绘图函数:
  1. plot_F0N_data_only()        — 纯数据: 谐波族连线 + 散点
  2. plot_F0N_with_human_ERB()   — + 人类 α=1.0 ERB 参照线
  3. plot_F0N_with_strategy_ERB()— + 该实验自身的策略ERB散点+拟合 + 人类参照线
  4. plot_F0N_3strategy_summary()— 三策略ERB散点汇总 (不同颜色), 每模型一张
"""

import sys, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker, matplotlib.colors
from matplotlib.cm import ScalarMappable
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import (
    human_erb_boundary_lines, ALPHA_VALUES, FITTED_ERB_STYLE,
    freq_to_note_name, MODEL_LIST
)

CMAP_NAME = 'coolwarm_r'
CMAP_BAD_COLOR = '#eeeeee'
FAMILY_LINE_COLOR = '#aaaaaa'
FAMILY_LINE_ALPHA = 0.30
SCATTER_ALPHA_DATA_ONLY = 0.75
SCATTER_ALPHA_OVERLAY = 0.50
SCATTER_SIZE = 30

# 三策略汇总图用色
STRATEGY_COLORS = {'A1': '#e41a1c', 'A2': '#377eb8', 'A3': '#4daf4a'}
STRATEGY_LABELS = {'A1': 'Exp B: Pitch Doubling', 'A2': 'Exp A: F0DL knee',
                   'A3': 'Exp C: Pitch Shift slope'}

EXPERIMENT_COLOR_RANGES = {
    'A': {'vmin': 0.1, 'vmax': 50.0,  'label': 'F0DL (%F0)',      'cmap_norm': 'log'},
    'B': {'vmin': 0.8, 'vmax': 2.2,   'label': 'F0_pred / F0',    'cmap_norm': 'linear'},
    'C': {'vmin': 0.0, 'vmax': 1.0,   'label': '|Pitch shift slope|','cmap_norm': 'linear'},
    'D': {'vmin': 1e-5, 'vmax': 0.2,  'label': '|Mistuning sensitivity|','cmap_norm': 'log'},
    'E': {'vmin': 0.1, 'vmax': 50.0,  'label': 'F0DL (%F0)',      'cmap_norm': 'log'},
}

# ============================================================
def _setup_F0N_axes(ax, f0_range=(50, 1050), n_range=(0.5, 320)):
    ax.set_yscale('log'); ax.set_ylabel('F0 (Hz)', fontsize=12)
    ax.set_ylim(f0_range)
    f0_ticks = [t for t in [50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
                if f0_range[0] <= t <= f0_range[1]]
    ax.set_yticks(f0_ticks); ax.set_yticklabels([str(t) for t in f0_ticks], fontsize=9)
    ax2 = ax.twinx(); ax2.set_yscale('log'); ax2.set_ylim(f0_range)
    ax2.set_yticks(f0_ticks)
    ax2.set_yticklabels([freq_to_note_name(t) for t in f0_ticks],
                        fontsize=8, color='#666666')
    ax2.tick_params(axis='y', colors='#666666', length=0)
    ax.set_xscale('log')
    ax.set_xlabel('Harmonic number N', fontsize=12); ax.set_xlim(n_range)
    xticks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300]
    ax.set_xticks(xticks); ax.set_xticklabels([str(t) for t in xticks], fontsize=8)
    ax.tick_params(labelsize=9); ax.grid(True, alpha=0.15, linestyle='-')
    ax.set_axisbelow(True)
    return ax, ax2

def _add_harmonic_family_lines(ax, data_dict, experiment):
    N_vals = data_dict['N']; F0_vals = data_dict['F0']
    unique_f0s = np.unique(F0_vals)
    if experiment == 'B' or experiment == 'C':
        # Exp B/C: 为每个 (F0, condition) 画谐波范围短横线
        _add_expBC_harmonic_range_bars(ax, data_dict)
        return
    for f0 in unique_f0s:
        if experiment != 'A': mask = np.isclose(F0_vals, f0, rtol=0.05)
        else: mask = (F0_vals == f0)
        if not np.any(mask) or np.sum(mask) < 2: continue
        n_at_f0 = np.sort(N_vals[mask])
        ax.hlines(f0, n_at_f0[0], n_at_f0[-1], colors='#aaaaaa',
                  linewidths=0.6, alpha=0.30, zorder=0)

def _add_expBC_harmonic_range_bars(ax, data_dict):
    """Exp B/C v2: 每个 (F0, condition) 画短横线表示 [h_min, h_max]."""
    harm_min = data_dict.get('harm_min')
    harm_max = data_dict.get('harm_max')
    if harm_min is None or harm_max is None:
        return
    N_vals = data_dict['N']; F0_vals = data_dict['F0']
    color_vals = data_dict['color_val']
    # 按颜色值归一化获取色调
    norm = matplotlib.colors.Normalize(vmin=0.8, vmax=2.2)
    cmap = plt.cm.get_cmap(CMAP_NAME)
    for i in range(len(N_vals)):
        rgba = cmap(norm(color_vals[i]))
        ax.hlines(F0_vals[i], harm_min[i], harm_max[i],
                  colors=rgba, linewidths=2.5, alpha=0.25, zorder=1)

def _add_expBC_same_f0_binding(ax, data_dict):
    """Exp B/C v2: 同一 F0 在所有条件的谐波成分用统一低饱和度色调标记.

    对每个代表 F0, 收集其所有谐波成分, 用该 F0 上所有条件的平均 color_val
    对应的低饱和度色调绘制, 表示"这些谐波来自同一刺激".
    """
    harm_min = data_dict.get('harm_min')
    harm_max = data_dict.get('harm_max')
    if harm_min is None or harm_max is None:
        return
    N_vals = data_dict['N']; F0_vals = data_dict['F0']
    color_vals = data_dict['color_val']
    norm = matplotlib.colors.Normalize(vmin=0.8, vmax=2.2)
    cmap = plt.cm.get_cmap(CMAP_NAME)

    # 按 F0 分组
    unique_f0s = np.unique(F0_vals)
    for f0 in unique_f0s:
        mask = np.isclose(F0_vals, f0, rtol=0.01)
        if np.sum(mask) == 0:
            continue
        # 该 F0 在所有 band 的平均色调
        avg_color = np.mean(color_vals[mask])
        rgba = cmap(norm(avg_color))
        # 用低饱和度绘制该 F0 的所有谐波成分
        for i in np.where(mask)[0]:
            harmonics = np.arange(harm_min[i], harm_max[i] + 1)
            ax.scatter(harmonics, np.full_like(harmonics, f0, dtype=float),
                      s=4, c=[rgba], alpha=0.15, edgecolors='none', zorder=0)

def _add_expA_annotation(ax):
    """Exp A: 标注数据点独立性."""
    ax.text(0.02, 0.02, 'Each point = independent condition\n(harmonics from low_harm to ~30)',
            transform=ax.transAxes, fontsize=7, color='#555555',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                      edgecolor='#cccccc'))

def _add_expB_data_note(ax):
    """Exp B v2: 标注数据聚合方式."""
    note = (
        'Exp B dataset: 6,144 F0 values (80-320 Hz, log-uniform, step=0.39 cents)\n'
        'Plot: 48 representative F0s (log-uniform), color = median(f0$_{pred}$/f0)\n'
        'within $\pm$0.5% of each rep. F0, ALT phase only. '
        'Bars: [h$_{min}$, h$_{max}$] per band'
    )
    ax.text(0.02, 0.02, note,
            transform=ax.transAxes, fontsize=6.5, color='#555555',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                      edgecolor='#cccccc'))

def _add_expC_data_note(ax):
    """Exp C v2: 标注数据聚合方式."""
    note = (
        'Exp C dataset: 3,971 F0 values (80-480 Hz, log-uniform, step=0.78 cents)\n'
        'Plot: 48 representative F0s (log-uniform). N = centered harmonic (5/11/16).\n'
        'Color = |pitch shift slope|. Slope via linregress of median(pred_shift) vs shift,\n'
        'aggregated within $\pm$0.5% of each rep. F0. Bars: harmonic passband per envelope'
    )
    ax.text(0.02, 0.02, note,
            transform=ax.transAxes, fontsize=6.5, color='#555555',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                      edgecolor='#cccccc'))

def _add_expD_data_note(ax):
    """Exp D: 标注数据聚合方式."""
    note = (
        'Exp D dataset: 534 F0s near 100/200/400 Hz, 12 harmonics (H1-H12), 13 mistuning %.\n'
        'Plot: 3 nominal F0s. N = mistuned harmonic #. Color = |mistuning sensitivity|\n'
        '(slope of pred_shift vs mistuning %). Aggregated within $\pm$2% of nominal F0.\n'
        'NB: only 3 F0s → ERB fit fragile. Low sens. values (<0.001) common for most models.'
    )
    ax.text(0.02, 0.02, note,
            transform=ax.transAxes, fontsize=6.5, color='#555555',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                      edgecolor='#cccccc'))

def _add_expE_data_note(ax):
    """Exp E: 标注数据聚合方式."""
    note = (
        'Exp E dataset: 6,144 envelope F0s (80-320 Hz), 4 carrier conditions.\n'
        'Plot: 48 rep. F0s. Pure tones: N=1. Transposed: N = fc / f_env.\n'
        'Color = median(|F0DL|). ERB via per-carrier sigmoid knee (max 3 pts).\n'
        'NB: ERB mapping indirect (f_harm=fc), F0DL from simplified median approx.'
    )
    ax.text(0.02, 0.02, note,
            transform=ax.transAxes, fontsize=6.5, color='#555555',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                      edgecolor='#cccccc'))

def _add_expE_cluster_labels(ax, data_dict):
    """Exp E: 在数据簇旁标注载波条件."""
    N_vals = data_dict['N']; F0_vals = data_dict['F0']
    # 纯音簇: N≈1 (最左)
    pure_mask = N_vals < 1.1
    if np.any(pure_mask):
        f0_mid = np.median(F0_vals[pure_mask])
        ax.text(1.15, f0_mid, 'pure tone', fontsize=7, color='#333333',
                verticalalignment='center', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7,
                          edgecolor='none'))
    # 转置音: 按 fc 分簇 (N ≈ fc/F0)
    for fc, label, y_off in [(4000, 'fc=4.0 kHz', 0.85), (6350, 'fc=6.35 kHz', 1.0), (10080, 'fc=10.1 kHz', 1.15)]:
        fc_mask = np.abs(N_vals * F0_vals - fc) < fc * 0.01
        if np.sum(fc_mask) > 3:
            n_mid = np.median(N_vals[fc_mask])
            f0_mid = np.median(F0_vals[fc_mask])
            ax.text(n_mid * 1.08, f0_mid * y_off, label, fontsize=7, color='#333333',
                    verticalalignment='center', alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7,
                              edgecolor='none'))

def _add_human_upper_freq_boundary(ax, f0_range, freq_limit_hz=20000):
    """添加人类听觉上限边界线 (20 kHz).

    F0-N 平面上: N × F0 = freq_limit_hz → N = freq_limit_hz / F0.
    在对数坐标上是一条向下倾斜的直线.
    """
    f0_line = np.logspace(np.log10(f0_range[0]), np.log10(f0_range[1]), 200)
    N_line = freq_limit_hz / f0_line
    ax.plot(N_line, f0_line, ls='--', color='#444444', lw=1.2,
            alpha=0.6, zorder=20, label='Human upper freq. boundary (20 kHz)')

def _add_scatter(ax, data_dict, experiment, alpha):
    exp_cfg = EXPERIMENT_COLOR_RANGES.get(experiment, EXPERIMENT_COLOR_RANGES['A'])
    norm = (matplotlib.colors.LogNorm(vmin=exp_cfg['vmin'], vmax=exp_cfg['vmax'])
            if exp_cfg['cmap_norm']=='log'
            else matplotlib.colors.Normalize(vmin=exp_cfg['vmin'], vmax=exp_cfg['vmax']))
    cmap = plt.cm.get_cmap(CMAP_NAME); cmap.set_bad(CMAP_BAD_COLOR)
    sc = ax.scatter(data_dict['N'], data_dict['F0'], c=data_dict['color_val'],
                    cmap=cmap, norm=norm, s=SCATTER_SIZE, alpha=alpha,
                    edgecolors='none', zorder=10)
    return cmap, norm, exp_cfg['label']

def _add_colorbar(fig, cmap, norm, label):
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.25])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(label, fontsize=10, rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)

def _overlay_strategy_erb(ax, erb_points, f0_range, experiment=''):
    """叠加该策略自身的 ERB 散点 + 拟合线.

    修复:
      - 过滤 N 值极端/不合理的散点 (如 sigmoid cap 产生的 f0_trans=1000)
      - 扩展 F0 范围使拟合线覆盖更宽 N 范围
      - 仅绘制 ERB>0 且物理合理的部分
    """
    if not erb_points or len(erb_points) < 2: return
    from scipy.stats import linregress

    # 过滤不合理散点: N = f_harm / erb 需在合理范围 (0.05, 100)
    pts_freqs_raw = np.array([p[0] for p in erb_points])
    pts_erbs_raw  = np.array([p[1] for p in erb_points])
    pts_N_raw = pts_freqs_raw / np.where(pts_erbs_raw > 0, pts_erbs_raw, np.inf)
    reasonable = (pts_freqs_raw > 0) & (pts_erbs_raw > 0) & \
                 (pts_N_raw > 0.05) & (pts_N_raw < 100)
    if np.sum(reasonable) < 2: return

    pts_freqs = pts_freqs_raw[reasonable]
    pts_erbs  = pts_erbs_raw[reasonable]
    pts_N = pts_freqs / pts_erbs

    # 绘制散点: 仅 N 在可见范围附近的 (0.1, 80)
    visible = (pts_N > 0.1) & (pts_N < 80)
    if np.sum(visible) > 0:
        ax.scatter(pts_N[visible], pts_erbs[visible], s=60,
                   c=FITTED_ERB_STYLE['color'],
                   edgecolors='k', linewidth=0.8, zorder=30,
                   label='Strategy ERB points')

    # 线性拟合
    s, b, rv, _, _ = linregress(pts_freqs, pts_erbs)

    # 扩展 F0 范围以展示更多拟合曲线 (1 Hz 到 2000 Hz)
    # ERB 边界: F0 = ERB(N*F0) = a*(N*F0) + b → N = (F0 - b) / (a * F0)
    f0_line = np.logspace(np.log10(1), np.log10(2000), 400)
    erb_line = s * f0_line + b
    # 仅绘制 ERB>0 且 a>0 (物理合理的 ERB 随频率增加), 且在可见范围附近
    if s > 0:
        # 正确公式: N = (F0 - b) / (a * F0)
        N_fit = (f0_line - b) / (s * f0_line)
        F0_fit = f0_line
        visible_fit = (erb_line > 0) & (f0_line >= f0_range[0] * 0.3)
        in_view = (N_fit > 0.1) & (N_fit < 200) & visible_fit
        if np.sum(in_view) > 2:
            ax.plot(N_fit[in_view], F0_fit[in_view],
                    ls=FITTED_ERB_STYLE['ls'], color=FITTED_ERB_STYLE['color'],
                    lw=FITTED_ERB_STYLE['lw'],
                    label='Strategy ERB fit (r²={:.2f})'.format(rv**2),
                    alpha=0.9, zorder=22)

def _overlay_human_erb(ax, f0_range):
    lines = human_erb_boundary_lines(f0_range=f0_range, n_f0_points=300, alphas=[1.0])
    for _, ld in lines.items():
        ax.plot(ld['N'], ld['f0'], ls='--', color='black', lw=2.0,
                label='Human cochlear ERB (α=1.0)', alpha=0.8, zorder=25)

def _overlay_all_alphas(ax, f0_range, alphas=None):
    if alphas is None: alphas = sorted(ALPHA_VALUES.keys())
    lines = human_erb_boundary_lines(f0_range=f0_range, n_f0_points=300, alphas=alphas)
    for alpha, ld in lines.items():
        s = ld['style']
        ax.plot(ld['N'], ld['f0'], ls=s['ls'], color=s['color'],
                lw=s['lw'], label=s['label'], alpha=0.8, zorder=25)

# ============================================================
def _make_figure(data_dict, experiment, model_name, f0_range, n_range,
                 figsize, output_path, overlay_mode='none',
                 strategy_erb_points=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax, ax2 = _setup_F0N_axes(ax, f0_range, n_range)

    # Exp B/C v2: 先画同刺激绑定 (最底层)
    if experiment == 'B' or experiment == 'C':
        _add_expBC_same_f0_binding(ax, data_dict)

    _add_harmonic_family_lines(ax, data_dict, experiment)
    alpha_sc = SCATTER_ALPHA_OVERLAY if overlay_mode != 'none' else SCATTER_ALPHA_DATA_ONLY
    cmap, norm, cbar_label = _add_scatter(ax, data_dict, experiment, alpha=alpha_sc)

    if overlay_mode == 'human':
        _overlay_human_erb(ax, f0_range)
    elif overlay_mode == 'strategy':
        _overlay_human_erb(ax, f0_range)
        _overlay_strategy_erb(ax, strategy_erb_points, f0_range, experiment)
    elif overlay_mode == 'full':
        _overlay_all_alphas(ax, f0_range)
        _overlay_strategy_erb(ax, strategy_erb_points, f0_range, experiment)

    # 人类听觉上限 (20 kHz) — 所有实验公用
    _add_human_upper_freq_boundary(ax, f0_range)

    # Exp A: 数据点独立性标注
    if experiment == 'A':
        _add_expA_annotation(ax)
    # Exp B: 数据聚合方式标注
    if experiment == 'B':
        _add_expB_data_note(ax)
    # Exp C: 数据聚合方式标注
    if experiment == 'C':
        _add_expC_data_note(ax)
    # Exp D: 数据聚合方式标注
    if experiment == 'D':
        _add_expD_data_note(ax)
    # Exp E: 数据聚合方式标注 + 簇标签
    if experiment == 'E':
        _add_expE_data_note(ax)
        _add_expE_cluster_labels(ax, data_dict)

    _add_colorbar(fig, cmap, norm, cbar_label)
    if overlay_mode != 'none':
        ax.legend(loc='upper right', fontsize=7, frameon=True,
                  facecolor='white', edgecolor='#cccccc', framealpha=0.9)

    title = f"F0-N Plane: {experiment}"
    if model_name: title += f" — {model_name}"
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.subplots_adjust(right=0.88 if overlay_mode=='full' else 0.90)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig

def plot_F0N_data_only(data_dict, experiment='A', model_name=None,
                       f0_range=(50,1050), n_range=(1,45), figsize=(9,7),
                       output_path=None):
    return _make_figure(data_dict, experiment, model_name, f0_range, n_range,
                        figsize, output_path, overlay_mode='none')

def plot_F0N_with_human_ERB(data_dict, experiment='A', model_name=None,
                            f0_range=(50,1050), n_range=(1,45), figsize=(9,7),
                            output_path=None):
    return _make_figure(data_dict, experiment, model_name, f0_range, n_range,
                        figsize, output_path, overlay_mode='human')

def plot_F0N_with_strategy_ERB(data_dict, experiment='A', model_name=None,
                                strategy_erb_points=None,
                                f0_range=(50,1050), n_range=(1,45), figsize=(10,8),
                                output_path=None):
    """该实验自身的策略 ERB 散点 + 拟合线 + 人类参照."""
    return _make_figure(data_dict, experiment, model_name, f0_range, n_range,
                        figsize, output_path, overlay_mode='strategy',
                        strategy_erb_points=strategy_erb_points)

def plot_F0N_full_overlay(data_dict, experiment='A', model_name=None,
                          strategy_erb_points=None,
                          f0_range=(50,1050), n_range=(1,45), figsize=(10,8),
                          output_path=None):
    """完整叠加: 多α线 + 该策略自身的 ERB 散点."""
    return _make_figure(data_dict, experiment, model_name, f0_range, n_range,
                        figsize, output_path, overlay_mode='full',
                        strategy_erb_points=strategy_erb_points)

# ============================================================
def plot_F0N_3strategy_summary(model_name, a1_pts, a2_pts, a3_pts,
                                f0_range=(50,1050), n_range=(1,45),
                                figsize=(10,8), output_path=None):
    """三策略 ERB 散点汇总图: 不同颜色, 每模型一张.

    Args:
        a1_pts, a2_pts, a3_pts: [(f_harm, erb), ...] 各策略的 ERB 散点
    """
    from scipy.stats import linregress
    fig, ax = plt.subplots(figsize=figsize)
    ax, ax2 = _setup_F0N_axes(ax, f0_range, n_range)

    # 人类 ERB 参照
    _overlay_human_erb(ax, f0_range)

    # 各策略散点 + 拟合线
    strategy_data = [('A1', a1_pts), ('A2', a2_pts), ('A3', a3_pts)]
    for label, pts in strategy_data:
        if not pts or len(pts) < 2: continue
        pf = np.array([p[0] for p in pts]); pe = np.array([p[1] for p in pts])
        v = (pf > 0) & (pe > 0)
        if np.sum(v) < 2: continue
        pf, pe = pf[v], pe[v]
        color = STRATEGY_COLORS[label]
        ax.scatter(pf/pe, pe, s=50, c=color, edgecolors='k', linewidth=0.6,
                  zorder=30, alpha=0.8, label=f'{STRATEGY_LABELS[label]} ({len(pf)} pts)')
        s, b, rv, _, _ = linregress(pf, pe)
        fl = np.logspace(np.log10(f0_range[0]), np.log10(f0_range[1]), 200)
        el = s*fl + b
        v2 = el > 0
        ax.plot(fl[v2]/el[v2], fl[v2], color=color, lw=1.5, alpha=0.7, zorder=22,
               ls='-.')

    ax.legend(loc='upper right', fontsize=7, frameon=True,
              facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax.set_title(f'3-Strategy ERB Summary — {model_name}', fontsize=13, fontweight='bold')
    plt.subplots_adjust(right=0.90)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig

# ============================================================
#  Exp C 数据构建 (修正版)
# ============================================================
def build_expC_F0N_data_corrected(model_name, n_f0_samples=18):
    """修正版 Exp C F0-N 数据构建.

    修正点:
      1. 使用 3917 个 F0 中的 ~18 个代表性采样 (而非仅5个)
      2. 每个 (F0, envelope) 条件显示其全部谐波成分范围 (而非仅中心谐波)
      3. 同一条水平线上的谐波 = 同一刺激, 用灰线连接
    """
    from data_loader import build_F0N_data_expC
    from scipy.stats import linregress

    raw = build_F0N_data_expC(model_name)
    f0_all = raw['f0']; f0_pred = raw['f0_pred']
    f0_shift = raw['f0_shift']; centered = raw['spectral_envelope_centered_harmonic']
    bandwidth = raw.get('spectral_envelope_bandwidth_in_harmonics',
                         np.full_like(centered, 3))

    # 对数均匀采样 F0
    f0_samples = np.logspace(np.log10(80), np.log10(480), n_f0_samples)

    N_list, F0_list, slope_list = [], [], []
    for f0_tgt in f0_samples:
        for harm in np.unique(centered):
            bw = int(bandwidth[centered == harm][0]) if np.any(centered == harm) else 3
            mask = (centered == harm)
            mask &= (f0_all >= f0_tgt * 0.95)
            mask &= (f0_all <= f0_tgt * 1.05)
            if np.sum(mask) < 10: continue

            fps = f0_pred[mask]; fs = f0_all[mask]; shs = f0_shift[mask]
            pred_shift = (fps - fs) / fs
            unique_shifts = np.unique(shs)
            if len(unique_shifts) < 3: continue
            mean_ps = [np.median(pred_shift[shs==s]) for s in unique_shifts]
            slope, _, _, _, _ = linregress(unique_shifts, mean_ps)
            slope_abs = abs(slope)

            # 该刺激的谐波成分范围: [N_center - bw/2, N_center + bw/2]
            half_bw = bw / 2
            harmonics_in_stimulus = np.arange(
                max(1, int(harm - half_bw)),
                int(harm + half_bw) + 1
            )
            for n in harmonics_in_stimulus:
                N_list.append(n)
                F0_list.append(f0_tgt)
                slope_list.append(slope_abs)

    return {
        'N': np.array(N_list), 'F0': np.array(F0_list),
        'color_val': np.array(slope_list), 'experiment': 'C',
    }
