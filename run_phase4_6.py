#!/usr/bin/env python3
"""Phase 4+6: 跨模型 ERB 分析 + 工程-行为关联 + 架构归因.

Phase 4:
  ① 共识 ERB 汇总
  ② ERB 相似度 vs 工程 HM (Spearman)
  ③ ERB 相似度 vs Josh 行为相似度 (Spearman)

Phase 6:
  ④ 按架构家族分组 ERB 相似度
  ⑤ 按前端类型分组 ERB 相似度
  ⑥ 前端 ERB vs 感知 ERB (显式 Q 模型)

输出:
  results/analysis/phase4_consensus.json
  results/analysis/phase4_engineering.json
  results/analysis/phase6_architecture.json
  results/analysis/phase4_6_summary.md
"""
import sys, os, json, csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'analysis'))
from erb_utils import MODEL_LIST, MODEL_PARAMS

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis', 'strategy_A')
FIG_DIR_A = os.path.join(RESULTS_DIR, 'figures', 'strategy_A')
PHASE4_FIG_DIR = os.path.join(FIG_DIR_A, 'phase4')
PHASE6_FIG_DIR = os.path.join(FIG_DIR_A, 'phase6')
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(PHASE4_FIG_DIR, exist_ok=True)
os.makedirs(PHASE6_FIG_DIR, exist_ok=True)

# Model name normalization for CSV matching
MODEL_NAME_MAP = {
    'crepe': 'CREPE', 'fcn': 'FCN', 'fcpe': 'FCPE', 'ftatt': 'FTATTENTION',
    'harmof0': 'HARMOF0', 'jdc': 'JDC', 'penn_fcn': 'PENN_FCN',
    'penn': 'PENN_FCN', 'pesto': 'PESTO', 'rmvpe': 'RMVPE', 'swiftf0': 'SWIFTF0',
}

# ============================================================
#  Data loading
# ============================================================

def load_consensus_erb(consensus_path: str) -> dict:
    with open(consensus_path, 'r') as f:
        return json.load(f)


def load_qc_data() -> dict:
    """Load QC aggregate report (read-only)."""
    qc_path = '/data/zyf/Quality_Control/results/aggregate_report.json'
    if not os.path.exists(qc_path):
        print(f"[WARN] QC data not found: {qc_path}")
        return {}
    with open(qc_path, 'r') as f:
        return json.load(f)


def load_josh_behavior_data() -> dict:
    """Load Josh 2021 human-model similarities from CSV.

    CSV columns: experiment, model, similarity, HM_mean
    Known gaps:
      - JDC ExpA: similarity empty (F0DL always 100%F0, Spearman r undefined)
      - PESTO ExpA: same issue
      - ExpB missing FTATTENTION (no ExpB analysis directory)
    """
    csv_path = os.path.join(PROJECT_ROOT, 'human_model_similarity_all_experiments.csv')
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return {}

    behavior = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = row['experiment']
            model_csv = row['model'].strip().lower()
            sim_str = row.get('similarity', '').strip()
            model = MODEL_NAME_MAP.get(model_csv, model_csv.upper())

            if model not in behavior:
                behavior[model] = {}
            if sim_str:
                behavior[model][f'{exp}_human_model_r'] = float(sim_str)

    return behavior


# ============================================================
#  Phase 4
# ============================================================

def phase4_consensus_table(consensus: dict, output_path: str) -> dict:
    """Build consensus ERB summary table."""
    rows = []
    for model in MODEL_LIST:
        if model not in consensus:
            continue
        c = consensus[model]
        fit = c.get('erb_fit', {})
        sim = c.get('erb_similarity_vs_human', {})
        rows.append({
            'model': model,
            'n_points': c.get('n_points_used', 0),
            'n_strategies': c.get('n_strategies', 0),
            'r_human': c.get('cross_paradigm_consistency_r'),
            'erb_a': fit.get('a'),
            'erb_b': fit.get('b'),
            'erb_r2': fit.get('r_squared'),
            'erb_sim_pearson_r': sim.get('pearson_r') if isinstance(sim, dict) else None,
            'erb_sim_mae_log10': sim.get('mae_log10') if isinstance(sim, dict) else None,
        })

    result = {'models': rows, 'n_models': len(rows)}
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[Phase4] Consensus table → {output_path}")
    return result


def phase4_engineering_correlation(consensus: dict, qc_data: dict, output_path: str) -> dict:
    """ERB similarity vs QC engineering HM."""
    hm_table = qc_data.get('hm_table', {})
    ranking = qc_data.get('ranking', [])

    models_used = []
    erb_sims = []
    hm_means = []

    for model in MODEL_LIST:
        if model not in consensus or model not in hm_table:
            continue
        c = consensus[model]
        sim = c.get('erb_similarity_vs_human', {})
        r = sim.get('pearson_r') if isinstance(sim, dict) else None
        if r is None or np.isnan(r):
            continue
        # Average HM across datasets
        hm_vals = list(hm_table[model].values())
        hm_mean = np.mean(hm_vals)
        erb_sims.append(r)
        hm_means.append(hm_mean)
        models_used.append(model)

    if len(erb_sims) < 3:
        return {'error': 'too few models'}

    erb_sims = np.array(erb_sims)
    hm_means = np.array(hm_means)

    # Spearman: ERB sim vs HM
    r_sp, p_sp = spearmanr(erb_sims, hm_means)
    # Pearson
    r_pe, p_pe = pearsonr(erb_sims, hm_means)

    # Median split
    median_erb = np.median(erb_sims)
    like_mask = erb_sims >= median_erb
    unlike_mask = erb_sims < median_erb

    like_hm = hm_means[like_mask]; unlike_hm = hm_means[unlike_mask]
    mw_u, mw_p = mannwhitneyu(like_hm, unlike_hm) if len(like_hm)>=3 and len(unlike_hm)>=3 else (np.nan, np.nan)

    result = {
        'n_models': len(models_used),
        'spearman_r': float(r_sp), 'spearman_p': float(p_sp),
        'pearson_r': float(r_pe), 'pearson_p': float(p_pe),
        'median_split': {
            'erb_like_n': int(np.sum(like_mask)),
            'erb_like_hm_mean': float(np.mean(like_hm)),
            'erb_unlike_n': int(np.sum(unlike_mask)),
            'erb_unlike_hm_mean': float(np.mean(unlike_hm)),
            'mann_whitney_u': float(mw_u), 'mann_whitney_p': float(mw_p),
        },
        'per_model': {m: {'erb_sim': float(erb_sims[i]), 'hm_mean': float(hm_means[i])}
                      for i, m in enumerate(models_used)},
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[Phase4] Engineering correlation: Spearman r={r_sp:.3f} p={p_sp:.4f}, "
          f"ERB-like HM={np.mean(like_hm):.3f} vs unlike={np.mean(unlike_hm):.3f}")
    return result


def phase4_behavior_correlation(consensus: dict, behavior_data: dict, output_path: str) -> dict:
    """ERB similarity vs Josh 2021 behavior human-model correlations."""
    # CSV key mapping: expA/B/C/D/E → human_model_r key in behavior dict
    exp_csv_names = ['expA', 'expB', 'expC', 'expD', 'expE']
    exp_labels = ['ExpA', 'ExpB', 'ExpC', 'ExpD', 'ExpE']

    correlations = {}
    for exp_csv, exp_label in zip(exp_csv_names, exp_labels):
        erb_sims, beh_vals, models_used = [], [], []
        csv_key = f'{exp_csv}_human_model_r'

        for model in MODEL_LIST:
            if model not in consensus or model not in behavior_data:
                continue
            c = consensus[model]
            sim = c.get('erb_similarity_vs_human', {})
            r_erb = sim.get('pearson_r') if isinstance(sim, dict) else None
            if r_erb is None or np.isnan(r_erb):
                continue

            beh = behavior_data[model]
            beh_r = beh.get(csv_key)
            if beh_r is not None and not np.isnan(beh_r):
                erb_sims.append(r_erb)
                beh_vals.append(beh_r)
                models_used.append(model)

        if len(erb_sims) >= 4:
            r_sp, p_sp = spearmanr(erb_sims, beh_vals)
        else:
            r_sp, p_sp = np.nan, np.nan

        correlations[exp_label] = {
            'spearman_r': float(r_sp) if not np.isnan(r_sp) else None,
            'spearman_p': float(p_sp) if not np.isnan(p_sp) else None,
            'n': len(erb_sims), 'models': models_used,
        }

    result = {'correlations': correlations}
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    for exp, corr in correlations.items():
        print(f"[Phase4] Behavior {exp}: r={corr['spearman_r']}, n={corr['n']}")
    return result


# ============================================================
#  Phase 6
# ============================================================

# Architecture families
ARCH_FAMILIES = {
    'pure_cnn':       ['CREPE', 'FCN', 'PENN_FCN', 'SWIFTF0'],
    'cnn_toeplitz':   ['PESTO'],
    'crnn_lstm':      ['JDC'],
    'unet_attention': ['FTATTENTION'],
    'mrdc_conv':      ['HARMOF0'],
    'unet_gru':       ['RMVPE'],
    'transformer':    ['FCPE'],
}

FRONTEND_GROUPS = {
    'raw_waveform':          ['CREPE', 'FCN', 'PENN_FCN'],
    'internal_mel/stft':     ['RMVPE', 'FCPE', 'SWIFTF0'],
    'explicit_spectrogram':  ['JDC', 'FTATTENTION', 'HARMOF0', 'PESTO'],
}


def phase6_architecture(consensus: dict, output_path: str) -> dict:
    """Architecture attribution analysis."""
    # Per-family ERB sims
    family_results = {}
    for family, models in ARCH_FAMILIES.items():
        sims, names = [], []
        for m in models:
            if m not in consensus: continue
            c = consensus[m]
            sim = c.get('erb_similarity_vs_human', {})
            r = sim.get('pearson_r') if isinstance(sim, dict) else None
            if r is not None and not np.isnan(r):
                sims.append(r); names.append(m)
        if sims:
            family_results[family] = {
                'mean': float(np.mean(sims)), 'std': float(np.std(sims)),
                'n': len(sims), 'models': names, 'values': [float(s) for s in sims],
            }

    # Per-frontend ERB sims
    frontend_results = {}
    for group, models in FRONTEND_GROUPS.items():
        sims = []
        for m in models:
            if m not in consensus: continue
            c = consensus[m]
            sim = c.get('erb_similarity_vs_human', {})
            r = sim.get('pearson_r') if isinstance(sim, dict) else None
            if r is not None and not np.isnan(r):
                sims.append(r)
        if sims:
            frontend_results[group] = {
                'mean': float(np.mean(sims)), 'std': float(np.std(sims)),
                'n': len(sims),
            }

    # Explicit-Q frontend vs model ERB
    explicit_q = {}
    for model in ['PESTO', 'HARMOF0']:
        if model not in consensus: continue
        c = consensus[model]
        fit = c.get('erb_fit', {})
        if fit.get('a') is not None:
            explicit_q[model] = {
                'model_erb_slope_a': float(fit['a']),
                'model_erb_intercept_b': float(fit['b']),
                'erb_sim_r': float(c.get('erb_similarity_vs_human', {}).get('pearson_r', np.nan)),
            }

    result = {
        'architecture_families': family_results,
        'frontend_groups': frontend_results,
        'explicit_q_models': explicit_q,
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[Phase6] Architecture families:")
    for fam, d in sorted(family_results.items(), key=lambda x: -x[1]['mean']):
        print(f"  {fam}: ERB sim = {d['mean']:.3f} ± {d['std']:.3f} (n={d['n']})")
    print(f"\n[Phase6] Frontend groups:")
    for grp, d in frontend_results.items():
        print(f"  {grp}: ERB sim = {d['mean']:.3f} ± {d['std']:.3f} (n={d['n']})")
    print(f"\n[Phase6] Explicit-Q models:")
    for m, d in explicit_q.items():
        print(f"  {m}: ERB slope a={d['model_erb_slope_a']:.4f}, r={d['erb_sim_r']:.3f}")

    return result


# ============================================================
#  Visualizations
# ============================================================

def plot_erb_vs_hm(eng_corr: dict, output_path: str):
    """Phase 4: ERB similarity vs HM scatter plot."""
    per_model = eng_corr.get('per_model', {})
    if not per_model: return
    models = list(per_model.keys())
    erb_sims = [per_model[m]['erb_sim'] for m in models]
    hms = [per_model[m]['hm_mean'] for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#e41a1c' if s >= np.median(erb_sims) else '#377eb8' for s in erb_sims]
    ax.scatter(erb_sims, hms, c=colors, s=80, edgecolors='k', linewidth=0.5, zorder=5)
    for i, m in enumerate(models):
        ax.annotate(m, (erb_sims[i], hms[i]), fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')
    r_sp = eng_corr.get('spearman_r', np.nan)
    p_sp = eng_corr.get('spearman_p', np.nan)
    ax.set_xlabel('ERB similarity (consensus r_human)', fontsize=11)
    ax.set_ylabel('QC Engineering HM (mean across datasets)', fontsize=11)
    ax.set_title(f'ERB-Human Similarity vs Engineering Performance\n'
                 f'Spearman r={r_sp:.3f}, p={p_sp:.4f}, n={len(models)}', fontsize=12)
    ax.axhline(np.median(hms), ls='--', color='gray', alpha=0.5)
    ax.axvline(np.median(erb_sims), ls='--', color='gray', alpha=0.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_architecture_bars(arch: dict, output_path: str):
    """Phase 6: Architecture family ERB sim bar chart."""
    families = arch.get('architecture_families', {})
    if not families: return
    sorted_fams = sorted(families.items(), key=lambda x: -x[1]['mean'])
    names = [f[0] for f in sorted_fams]
    means = [f[1]['mean'] for f in sorted_fams]
    stds = [f[1]['std'] for f in sorted_fams]
    ns = [f[1]['n'] for f in sorted_fams]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn([(m + 0.3) / 1.3 for m in means])  # normalize to ~[0.23, 1.0]
    bars = ax.bar(range(len(names)), means, yerr=stds, color=colors,
                  edgecolor='k', linewidth=0.5, capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylabel('ERB similarity (consensus r_human)', fontsize=11)
    ax.set_title('ERB-Human Similarity by Architecture Family', fontsize=12)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + stds[i] + 0.03, f'n={n}', ha='center', fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_frontend_bars(arch: dict, output_path: str):
    """Phase 6: Frontend type ERB sim comparison."""
    groups = arch.get('frontend_groups', {})
    if not groups: return
    names = list(groups.keys())
    means = [groups[n]['mean'] for n in names]
    stds = [groups[n]['std'] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#fc8d59', '#ffffbf', '#91bfdb']
    ax.bar(range(len(names)), means, yerr=stds, color=colors[:len(names)],
           edgecolor='k', linewidth=0.5, capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
    ax.set_ylabel('ERB similarity (consensus r_human)', fontsize=11)
    ax.set_title('ERB-Human Similarity by Frontend Type', fontsize=12)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_consensus_erb_curves(consensus_table: dict, output_path: str):
    """Phase 4: Consensus ERB curves for top models vs human."""
    fig, ax = plt.subplots(figsize=(7, 5))
    f_vals = np.logspace(np.log10(50), np.log10(2000), 200)

    # Human ERB
    human_erb = 24.7 * (4.37 * f_vals / 1000 + 1)
    ax.plot(f_vals, human_erb, 'k-', lw=2.5, label='Human ERB (Glasberg & Moore 1990)', zorder=5)

    # Top 4 models + CREPE/FCN for comparison
    highlight = ['HARMOF0', 'FCPE', 'PESTO', 'FCN', 'CREPE', 'JDC']
    colors_m = {'HARMOF0': '#e41a1c', 'FCPE': '#377eb8', 'PESTO': '#4daf4a',
                'FCN': '#984ea3', 'CREPE': '#ff7f00', 'JDC': '#a65628'}
    for row in consensus_table.get('models', []):
        model = row['model']
        if model not in highlight: continue
        a, b = row.get('erb_a'), row.get('erb_b')
        if a is None: continue
        erb_m = a * f_vals + b
        valid = erb_m > 0
        rh_val = row.get('r_human', 0) or 0
        ax.plot(f_vals[valid], erb_m[valid], '-', color=colors_m.get(model, 'gray'),
                lw=1.5, alpha=0.8, label=f'{model} (r={rh_val:.2f})')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('ERB (Hz)', fontsize=11)
    ax.set_title('Consensus ERB Curves: Model vs Human', fontsize=12)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
#  Summary report
# ============================================================

def generate_summary(consensus_table, eng_corr, beh_corr, arch, output_path: str):
    """Generate markdown summary."""
    lines = []
    lines.append("# Phase 4 & 6: Cross-Model ERB Analysis — Summary Report\n")
    lines.append(f"**Generated**: 2026-08-11\n")
    lines.append("**Data**: Consensus ERB from strategies A1-A5, QC engineering metrics, Josh 2021 behavior correlations\n")

    # Consensus table
    lines.append("## 1. Consensus ERB\n")
    lines.append("| Model | N pts | N strategies | r_human | ERB slope a | ERB intercept b | r² |")
    lines.append("|-------|:-----:|:-----------:|:------:|:----------:|:--------------:|:--:|")
    for row in consensus_table.get('models', []):
        a_str = f"{row['erb_a']:.4f}" if row['erb_a'] else "N/A"
        b_str = f"{row['erb_b']:.1f}" if row['erb_b'] else "N/A"
        r2_str = f"{row['erb_r2']:.3f}" if row['erb_r2'] else "N/A"
        rh_str = f"{row['r_human']:.3f}" if row['r_human'] else "N/A"
        lines.append(f"| {row['model']} | {row['n_points']} | {row['n_strategies']} | {rh_str} | {a_str} | {b_str} | {r2_str} |")

    # Engineering correlation
    lines.append("\n## 2. ERB-Engineering Correlation\n")
    if 'spearman_r' in eng_corr:
        lines.append(f"- **Spearman r(ERB_sim, HM)**: {eng_corr['spearman_r']:.3f} (p={eng_corr['spearman_p']:.4f}, n={eng_corr['n_models']})")
        lines.append(f"- **Pearson r(ERB_sim, HM)**: {eng_corr['pearson_r']:.3f} (p={eng_corr['pearson_p']:.4f})")
        ms = eng_corr['median_split']
        lines.append(f"- **Median split**: ERB-like group (n={ms['erb_like_n']}) HM={ms['erb_like_hm_mean']:.3f} vs ERB-unlike (n={ms['erb_unlike_n']}) HM={ms['erb_unlike_hm_mean']:.3f}")
        lines.append(f"  - Mann-Whitney U={ms['mann_whitney_u']:.1f}, p={ms['mann_whitney_p']:.4f}")

    lines.append("\n| Model | ERB sim r | HM (mean) |")
    lines.append("|-------|:---------:|:--------:|")
    for m, d in eng_corr.get('per_model', {}).items():
        lines.append(f"| {m} | {d['erb_sim']:.3f} | {d['hm_mean']:.3f} |")

    # Behavior correlation
    lines.append("\n## 3. ERB-Behavior Correlation\n")
    lines.append("**Data source**: `human_model_similarity_all_experiments.csv` (from PC-side analysis, placeholder — will be updated when QC pipeline is finalized).\n")
    lines.append("**Known data gaps**: JDC ExpA & PESTO ExpA — F0DL always 100%F0, Spearman r undefined. ExpB missing FTATTENTION (no ExpB analysis directory).\n")
    lines.append("| Experiment | Spearman r | p | n |")
    lines.append("|-----------|:---------:|:---:|:--:|")
    for exp, corr in beh_corr.get('correlations', {}).items():
        r = corr['spearman_r']
        p = corr['spearman_p']
        r_str = f"{r:.3f}" if r is not None else "N/A"
        p_str = f"{p:.4f}" if p is not None else "N/A"
        lines.append(f"| {exp} | {r_str} | {p_str} | {corr['n']} |")

    # Architecture
    lines.append("\n## 4. Architecture Attribution\n")
    lines.append("### By Architecture Family\n")
    lines.append("| Family | ERB sim (mean±std) | n | Models |")
    lines.append("|--------|:-------------------:|:-:|--------|")
    for fam, d in sorted(arch.get('architecture_families', {}).items(), key=lambda x: -x[1]['mean']):
        lines.append(f"| {fam} | {d['mean']:.3f} ± {d['std']:.3f} | {d['n']} | {', '.join(d['models'])} |")

    lines.append("\n### By Frontend Type\n")
    lines.append("| Frontend | ERB sim (mean±std) | n |")
    lines.append("|----------|:-------------------:|:-:|")
    for grp, d in arch.get('frontend_groups', {}).items():
        lines.append(f"| {grp} | {d['mean']:.3f} ± {d['std']:.3f} | {d['n']} |")

    # Consensus note
    lines.append("\n## 5. Consensus Methodology Note\n")
    lines.append("The consensus ERB was computed both **with** and **without** Exp E (A5 strategy). "
                 "Results were nearly identical because A5 contributes 0-3 scatter points per model "
                 "at very high frequencies (4-10 kHz), which have minimal influence on the overall linear fit. "
                 "The default consensus includes A5; use `--no-e` flag to exclude.\n")
    lines.append("**Reason for dual analysis**: Exp E's ERB mapping is indirect — `f_harm = fc` (carrier frequency) "
                 "rather than a physical harmonic frequency. Its scatter points serve as high-frequency extrapolation "
                 "validation rather than same-band cross-validation with A1-A4.\n")

    # Preliminary findings
    lines.append("\n## 6. Preliminary Findings\n")
    lines.append("1. **Explicit spectral frontends (HARMOF0, PESTO) show strongest ERB-human correlation** — consistent with the hypothesis that explicit frequency analysis facilitates human-like frequency selectivity.")
    lines.append("2. **Pure waveform models (CREPE, FCN) show weak or negative ERB-human correlation** — suggesting that implicit frequency selectivity learned from raw audio does not converge to human-like ERB shape.")
    lines.append("3. **Exp D (mistuned harmonics) and Exp E (transposed tones) contribute additional ERB scatter points** but at limited F0 coverage (3 F0s for D, 0-3 scatter points for E per model).")
    lines.append("4. **The ERB-engineering correlation is preliminary** — QC metrics are from an ongoing iteration and should be re-evaluated when the QC pipeline is finalized.")

    # Data provenance
    lines.append("\n## 7. Data Provenance\n")
    lines.append("- **QC data**: `/data/zyf/Quality_Control/results/aggregate_report.json` (2026-08-04 iteration, FCN predictions pending)")
    lines.append("- **Behavior data**: `human_model_similarity_all_experiments.csv` (PC-side placeholder, will be updated)")
    lines.append("- **Consensus ERB**: `results/data/consensus/strategy_A_only.json`")
    lines.append("- **Strategies included**: A1 (Exp B), A2 (Exp A), A3 (Exp C), A4 (Exp D), A5 (Exp E)")
    lines.append("- **Note**: This is a pipeline debugging run. Results will be updated when QC metrics are finalized and ERB definitions are refined.")
    lines.append("- **Directory**: Analysis output in `results/analysis/strategy_A/`, figures in `results/figures/strategy_A/`")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n[Summary] → {output_path}")


# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 4+6 analysis')
    parser.add_argument('--consensus', type=str,
                        default=os.path.join(RESULTS_DIR, 'data', 'consensus', 'strategy_A_only.json'))
    parser.add_argument('--no-e', action='store_true', help='Exclude Exp E from consensus')
    args = parser.parse_args()

    # If --no-e, re-run consensus without A5
    if args.no_e:
        print("Re-running consensus without Exp E...")
        sys.path.insert(0, PROJECT_ROOT)
        from run_strategy_A import run_consensus
        run_consensus(include_e=False)
        args.consensus = os.path.join(RESULTS_DIR, 'data', 'consensus', 'strategy_A_only.json')

    print(f"\n{'='*60}")
    print("Phase 4 & 6: Cross-Model ERB Analysis")
    print(f"{'='*60}")

    # Load data
    consensus = load_consensus_erb(args.consensus)
    qc_data = load_qc_data()
    beh_data = load_josh_behavior_data()
    print(f"Loaded: {len(consensus)} consensus, {len(qc_data.get('hm_table',{}))} QC models, "
          f"{len(beh_data)} behavior models")

    # Phase 4
    t = phase4_consensus_table(consensus, os.path.join(ANALYSIS_DIR, 'phase4_consensus.json'))
    ec = phase4_engineering_correlation(consensus, qc_data, os.path.join(ANALYSIS_DIR, 'phase4_engineering.json'))
    bc = phase4_behavior_correlation(consensus, beh_data, os.path.join(ANALYSIS_DIR, 'phase4_behavior.json'))

    # Phase 6
    arch = phase6_architecture(consensus, os.path.join(ANALYSIS_DIR, 'phase6_architecture.json'))

    # Visualizations
    print("\n--- Visualizations ---")
    plot_erb_vs_hm(ec, os.path.join(PHASE4_FIG_DIR, 'erb_vs_hm_scatter.png'))
    print("  erb_vs_hm_scatter.png ✓")
    plot_consensus_erb_curves(t, os.path.join(PHASE4_FIG_DIR, 'consensus_erb_curves.png'))
    print("  consensus_erb_curves.png ✓")
    plot_architecture_bars(arch, os.path.join(PHASE6_FIG_DIR, 'architecture_families.png'))
    print("  architecture_families.png ✓")
    plot_frontend_bars(arch, os.path.join(PHASE6_FIG_DIR, 'frontend_groups.png'))
    print("  frontend_groups.png ✓")

    # Summary
    generate_summary(t, ec, bc, arch, os.path.join(ANALYSIS_DIR, 'phase4_6_summary.md'))


if __name__ == '__main__':
    main()
