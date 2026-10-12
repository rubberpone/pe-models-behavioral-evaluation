#!/usr/bin/env python3
"""
策略 A 总驱动脚本 — 零新数据成本的感知 ERB 测量.

对 10 个 PE 模型, 利用已有的 Josh 2021 行为实验预测数据,
通过 per-F0 分层重分析提取每个模型的感知 ERB 散点,
然后进行跨范式交叉验证.

执行:
  python run_strategy_A.py --all
  python run_strategy_A.py --model CREPE
"""

import sys, os, json, argparse, time, numpy as np
from scipy.stats import linregress

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'analysis'))

from erb_utils import (
    MODEL_LIST, MODEL_PSYCHOPHYSICS_DIRS,
    EXISTING_EXP_B_PARAMS, EXISTING_EXP_C_PARAMS, EXISTING_EXP_A_PARAMS,
    EXISTING_EXP_D_PARAMS, EXISTING_EXP_E_PARAMS,
    create_erb_result_dict, compute_erb_similarity,
    sigmoid, fit_sigmoid_transition, exp_decay, fit_exp_decay
)
from data_loader import (
    build_F0N_data_expA, build_F0N_data_expB, build_F0N_data_expC,
    build_F0N_data_expD, build_F0N_data_expE,
    expA_to_F0N_heatmap_data, expB_to_F0N_heatmap_data, expC_to_F0N_heatmap_data,
    expD_to_F0N_heatmap_data, expE_to_F0N_heatmap_data
)

STRATEGY_A_DIR = os.path.join(PROJECT_ROOT, 'results', 'data', 'strategy_A')
CONSENSUS_DIR = os.path.join(PROJECT_ROOT, 'results', 'data', 'consensus')
os.makedirs(STRATEGY_A_DIR, exist_ok=True)
os.makedirs(CONSENSUS_DIR, exist_ok=True)


# ============================================================
def run_a1_exp_b(models):
    """策略 A1 v3: Exp B per-F0 ratio crossing → ERB 散点.

    改进 (v2→v3):
      - v2: 每个 filter_band 跨 48 F0 拟合 sigmoid → 最多 3 个 ERB 点
      - v3: 每个 F0 独立检测 3 个 band 中的 ratio 跨越 1.5 位置 → 最多 48 个 ERB 点
      - 跳过 ratio 从未跨越 1.5 的 F0 (无 pitch doubling 行为)
      - 在对数 N 空间插值以获得更精确的过渡点
    """
    print("\n" + "=" * 70)
    print("A1 v3: Exp B — per-F0 ratio crossing detection")
    print("=" * 70)
    results = {}
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model}")
        try:
            raw = build_F0N_data_expB(model)
            hm = expB_to_F0N_heatmap_data(raw)  # 48 F0s × 3 bands, v2 数据

            f0_vals = hm['F0']
            n_vals = hm['N']
            ratio_vals = hm['color_val']
            unique_f0s = np.unique(f0_vals)

            erb_pts = []
            n_skipped_all_res = 0; n_skipped_all_urs = 0; n_skipped_no_cross = 0
            n_skipped_few_bands = 0

            for f0_rep in unique_f0s:
                mask = np.isclose(f0_vals, f0_rep, rtol=0.01)
                if np.sum(mask) < 2:
                    n_skipped_few_bands += 1
                    continue

                Ns = n_vals[mask]; ratios = ratio_vals[mask]
                # 按 N 排序 (低 N → RES 区, 高 N → URS 区)
                sort_idx = np.argsort(Ns); Ns = Ns[sort_idx]; ratios = ratios[sort_idx]

                all_res  = np.all(ratios < 1.2)   # 全部 ≈1, 无 pitch doubling
                all_urs  = np.all(ratios > 1.8)   # 全部 ≈2, 已在 URS
                if all_res:
                    n_skipped_all_res += 1; continue
                if all_urs:
                    n_skipped_all_urs += 1; continue

                # 寻找 ratio 跨越 1.5 的相邻 band 对
                cross_found = False
                for j in range(len(ratios) - 1):
                    r1, r2 = ratios[j], ratios[j+1]
                    if (r1 - 1.5) * (r2 - 1.5) < 0:  # 符号相反 → 跨越 1.5
                        # 在对数 N 空间插值
                        log_N1, log_N2 = np.log(Ns[j]), np.log(Ns[j+1])
                        log_N_trans = log_N1 + (log_N2 - log_N1) * (1.5 - r1) / (r2 - r1)
                        N_trans = np.exp(log_N_trans)
                        erb_pts.append((N_trans * f0_rep, f0_rep))
                        cross_found = True
                        break

                if not cross_found:
                    n_skipped_no_cross += 1

            r = create_erb_result_dict(model, 'A1_ExpB_doubling_v3')
            r['erb_points'] = [(float(f), float(e)) for f, e in erb_pts]
            r['n_points_used'] = len(erb_pts)
            if len(erb_pts) >= 3:
                r['erb_similarity_vs_human'] = compute_erb_similarity(erb_pts)
                freqs = np.array([p[0] for p in erb_pts])
                erbs  = np.array([p[1] for p in erb_pts])
                s, b, rv, _, _ = linregress(freqs, erbs)
                r['erb_fit'] = {'a':float(s), 'b':float(b), 'r_squared':float(rv**2)}
            results[model] = r
            print(f"  {len(erb_pts)} ERB points (skipped: {n_skipped_all_res} all-RES, "
                  f"{n_skipped_all_urs} all-URS, {n_skipped_no_cross} no-cross, "
                  f"{n_skipped_few_bands} few-bands)")
        except Exception as e:
            print(f"  [FAILED] {e}")
            import traceback; traceback.print_exc()

    out = os.path.join(STRATEGY_A_DIR, 'A1_ExpB_all_models.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[A1 v3] {len(results)}/{len(models)} → {out}")
    return results


# ============================================================
def run_a2_exp_a(models):
    """策略 A2 v2: Exp A per-F0 F0DL 膝点检测 → ERB 散点.

    v2 改进: 拟合前检查 F0DL 曲线形状
      - 单调性: mean(F0DL[N=1:5]) < mean(F0DL[N=25:30]) (低 N 必须低于高 N)
      - 过渡强度: 高低差 >= 3.0 %F0 (过渡有实质意义)
      - 排除非单调/尖峰后降/近乎平坦的无效曲线
    """
    print("\n" + "=" * 70)
    print("A2 v2: Exp A — F0DL knee-point with monotonicity check")
    print("=" * 70)
    results = {}
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model}")
        try:
            raw = build_F0N_data_expA(model)
            hm_data = expA_to_F0N_heatmap_data(raw)
            erb_pts = []
            unique_f0s = np.unique(hm_data['F0'])
            n_skipped_range = 0; n_skipped_mono = 0; n_skipped_fit = 0
            for bf0 in unique_f0s:
                mask = hm_data['F0'] == bf0
                Ns = hm_data['N'][mask]; f0dls = hm_data['color_val'][mask]
                sort = np.argsort(Ns); Ns = Ns[sort]; f0dls = f0dls[sort]
                if len(Ns) < 5: continue

                # ① F0DL 范围检查
                yrange = np.max(f0dls) - np.min(f0dls)
                if yrange < 2.0:
                    n_skipped_range += 1; continue

                # ② 单调性检查: 低 N 的 F0DL 必须低于高 N
                f0dl_early = np.mean(f0dls[:5])    # N=1-5 (RES 端)
                f0dl_late  = np.mean(f0dls[-5:])   # N=26-30 (URS 端)
                if f0dl_early >= f0dl_late:
                    n_skipped_mono += 1; continue

                # ③ 过渡强度: 至少 3%F0 的提升
                if f0dl_late - f0dl_early < 3.0:
                    n_skipped_mono += 1; continue

                y_norm = (f0dls - np.min(f0dls)) / yrange
                try:
                    k, Nk = fit_sigmoid_transition(Ns, y_norm, p0=[0.5, np.median(Ns)])
                    if Nk > 0:
                        erb_pts.append((Nk * bf0, bf0))
                except:
                    n_skipped_fit += 1

            r = create_erb_result_dict(model, 'A2_ExpA_knee_v2')
            r['erb_points'] = [(float(f), float(e)) for f, e in erb_pts]
            r['n_points_used'] = len(erb_pts)
            if len(erb_pts) >= 3:
                r['erb_similarity_vs_human'] = compute_erb_similarity(erb_pts)
                freqs = np.array([p[0] for p in erb_pts])
                erbs  = np.array([p[1] for p in erb_pts])
                s, b, rv, _, _ = linregress(freqs, erbs)
                r['erb_fit'] = {'a':float(s), 'b':float(b), 'r_squared':float(rv**2)}
            results[model] = r
            print(f"  {len(erb_pts)} ERB points (skipped: {n_skipped_range} low-range, "
                  f"{n_skipped_mono} non-monotonic/weak, {n_skipped_fit} fit-fail)")
        except Exception as e:
            print(f"  [FAILED] {e}")
            import traceback; traceback.print_exc()

    out = os.path.join(STRATEGY_A_DIR, 'A2_ExpA_all_models.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[A2 v2] {len(results)}/{len(models)} → {out}")
    return results


# ============================================================
def run_a3_exp_c(models):
    """策略 A3: Exp C per-F0 Pitch Shift 斜率衰减 → ERB 散点 (v2: 48 F0s)."""
    print("\n" + "=" * 70)
    print("A3: Exp C — Pitch Shift slope decay per-F0 reanalysis (v2: 48 F0s)")
    print("=" * 70)
    results = {}
    rep_f0s = EXISTING_EXP_C_PARAMS['representative_f0s']
    f0_tol = 0.005  # v2: ±0.5% tolerance
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model}")
        try:
            raw = build_F0N_data_expC(model)
            erb_pts = []
            for f0_tgt in rep_f0s:
                slopes_by_harm = {}
                centered = raw['spectral_envelope_centered_harmonic']
                f0s = raw['f0']; f0ps = raw['f0_pred']; shifts = raw['f0_shift']
                unique_harms = np.unique(centered)
                for harm in unique_harms:
                    mask = (centered == harm)
                    mask &= (f0s >= f0_tgt * (1 - f0_tol))
                    mask &= (f0s <= f0_tgt * (1 + f0_tol))
                    if np.sum(mask) < 5: continue
                    pred_shift = (f0ps[mask] - f0s[mask]) / f0s[mask]
                    unique_shifts = np.unique(shifts[mask])
                    if len(unique_shifts) < 3: continue
                    mean_pred_shift = [np.median(pred_shift[shifts[mask]==s]) for s in unique_shifts]
                    slope, *_ = linregress(unique_shifts, mean_pred_shift)
                    slopes_by_harm[harm] = abs(slope)
                if len(slopes_by_harm) < 3: continue
                harms = np.array(sorted(slopes_by_harm.keys()))
                slps = np.array([slopes_by_harm[h] for h in harms])
                if np.max(slps) < 0.01: continue
                slp_norm = slps / np.max(slps)
                try:
                    Nd, k = fit_exp_decay(harms, slp_norm, p0=[np.median(harms), 1.5])
                    # 过滤不合理的衰减常数 (溢出或无意义的极大值)
                    if Nd > 0 and Nd < 1000:
                        erb_pts.append((Nd * f0_tgt, f0_tgt))
                except: pass
            r = create_erb_result_dict(model, 'A3_ExpC_slope')
            r['erb_points'] = [(float(f), float(e)) for f, e in erb_pts]
            r['n_points_used'] = len(erb_pts)
            if len(erb_pts) >= 3:
                r['erb_similarity_vs_human'] = compute_erb_similarity(erb_pts)
                freqs = np.array([p[0] for p in erb_pts])
                erbs  = np.array([p[1] for p in erb_pts])
                s, b, rv, _, _ = linregress(freqs, erbs)
                r['erb_fit'] = {'a':float(s), 'b':float(b), 'r_squared':float(rv**2)}
            results[model] = r
            print(f"  {len(erb_pts)} ERB points from {len(rep_f0s)} representative F0s")
        except Exception as e:
            print(f"  [FAILED] {e}")
            import traceback; traceback.print_exc()

    out = os.path.join(STRATEGY_A_DIR, 'A3_ExpC_all_models.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[A3] {len(results)}/{len(models)} → {out}")
    return results


# ============================================================
def run_a4_exp_d(models):
    """策略 A4: Exp D per-F0 mistuning sensitivity decay → ERB 散点.

    对每个标称 F0 (100/200/400 Hz):
      - 对每个失谐谐波 H (1-12), 计算 |mistuning_sensitivity|
      - 拟合指数衰减 exp(-(H/H_decay)^k) → H_decay
      - ERB 散点: (H_decay × F0, F0)
    """
    print("\n" + "=" * 70)
    print("A4: Exp D — Mistuning sensitivity decay per-F0")
    print("=" * 70)
    results = {}
    f0_tol = 0.02  # ±2% tolerance (F0 集中在标称值 ±4% 内)
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model}")
        try:
            raw = build_F0N_data_expD(model)
            erb_pts = []
            for f0_tgt in EXISTING_EXP_D_PARAMS['f0_ref_list']:
                sens_by_harm = {}
                f0_all = raw['f0']; f0_pred = raw['f0_pred']
                harm_all = raw['mistuned_harm']; pct_all = raw['mistuned_pct']
                for harm in EXISTING_EXP_D_PARAMS['mistuned_harmonics']:
                    mask = (harm_all == harm)
                    mask &= (f0_all >= f0_tgt * (1 - f0_tol))
                    mask &= (f0_all <= f0_tgt * (1 + f0_tol))
                    if np.sum(mask) < 10: continue
                    f0s = f0_all[mask]; f0ps = f0_pred[mask]; pcts = pct_all[mask]
                    pred_shift = (f0ps - f0s) / f0s
                    unique_pcts = np.unique(pcts)
                    if len(unique_pcts) < 3: continue
                    mean_ps = [np.median(pred_shift[pcts == mp]) for mp in unique_pcts]
                    slope, *_ = linregress(unique_pcts, mean_ps)
                    sens_by_harm[harm] = abs(slope)
                if len(sens_by_harm) < 4: continue
                harms = np.array(sorted(sens_by_harm.keys()))
                senses = np.array([sens_by_harm[h] for h in harms])
                if np.max(senses) < 0.0001: continue
                snorm = senses / np.max(senses)
                try:
                    Hd, k = fit_exp_decay(harms, snorm, p0=[np.median(harms), 1.5])
                    if Hd > 0 and Hd < 1000:
                        erb_pts.append((Hd * f0_tgt, f0_tgt))
                except: pass
            r = create_erb_result_dict(model, 'A4_ExpD_mistuning')
            r['erb_points'] = [(float(f), float(e)) for f, e in erb_pts]
            r['n_points_used'] = len(erb_pts)
            if len(erb_pts) >= 3:
                r['erb_similarity_vs_human'] = compute_erb_similarity(erb_pts)
                freqs = np.array([p[0] for p in erb_pts])
                erbs  = np.array([p[1] for p in erb_pts])
                s, b, rv, _, _ = linregress(freqs, erbs)
                r['erb_fit'] = {'a':float(s), 'b':float(b), 'r_squared':float(rv**2)}
            results[model] = r
            print(f"  {len(erb_pts)} ERB points from {len(EXISTING_EXP_D_PARAMS['f0_ref_list'])} nominal F0s")
        except Exception as e:
            print(f"  [FAILED] {e}")
            import traceback; traceback.print_exc()
    out = os.path.join(STRATEGY_A_DIR, 'A4_ExpD_all_models.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[A4] {len(results)}/{len(models)} → {out}")
    return results


# ============================================================
def run_a5_exp_e(models):
    """策略 A5: Exp E per-carrier F0DL knee → ERB 散点.

    对每个转置音载波 fc (4000/6350/10080):
      - 取 48 个 F0 bins 的 F0DL 值
      - 拟合 sigmoid: F0DL_norm vs log(F0) → F0_knee
      - ERB 散点: (f_harm = fc, erb = F0_knee)
      物理含义: "在载波频率 fc Hz 处, ERB ≈ F0_knee Hz"
      (包络能否被解析取决于听觉滤波器在 fc 处的带宽)

    纯音 (fc=0) 不参与 ERB 提取 (N=1 固定, 无 F0DL 随 N 的过渡).
    """
    print("\n" + "=" * 70)
    print("A5: Exp E — per-carrier F0DL knee detection")
    print("=" * 70)
    results = {}
    for i, model in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {model}")
        try:
            raw = build_F0N_data_expE(model)
            hm = expE_to_F0N_heatmap_data(raw)

            # ERB 提取: 每个转置音载波独立
            erb_pts = []
            carriers_tr = [4000.0, 6350.0, 10080.0]
            for fc in carriers_tr:
                mask = (hm['N'] > 1.0)  # 排除纯音
                # 找出最接近该载波的 N 对应的点 (N ≈ fc/F0)
                # hm 中每个点有 N = fc/F0, 对给定 fc, 按 F0 排序
                f0s_for_carrier = []; f0dls_for_carrier = []
                N_vals = hm['N']; F0_vals = hm['F0']; f0dl_vals = hm['color_val']
                # 通过 N 范围反推: 对给定 fc, N ∈ [fc/320, fc/80]
                for j in range(len(N_vals)):
                    # 仅取该载波附近的点: N × F0 ≈ fc
                    if abs(N_vals[j] * F0_vals[j] - fc) < fc * 0.001:
                        f0s_for_carrier.append(F0_vals[j])
                        f0dls_for_carrier.append(f0dl_vals[j])

                if len(f0s_for_carrier) < 6:
                    continue

                f0s = np.array(f0s_for_carrier)
                f0dls = np.array(f0dls_for_carrier)
                sort_idx = np.argsort(f0s); f0s = f0s[sort_idx]; f0dls = f0dls[sort_idx]

                yrange = np.max(f0dls) - np.min(f0dls)
                if yrange < 2.0:
                    continue

                # 单调性检查 (Exp E 方向反转: F0DL ↓ as F0 ↑)
                early = np.mean(f0dls[:5]); late = np.mean(f0dls[-5:])
                if early <= late or (early - late) < 1.0:
                    continue

                # 反转归一化方向以匹配 sigmoid (使 y 仍然从 0→1)
                f0dl_norm = 1.0 - (f0dls - np.min(f0dls)) / yrange
                try:
                    k, F0k = fit_sigmoid_transition(np.log10(f0s), f0dl_norm,
                                                     p0=[5.0, np.log10(np.median(f0s))])
                    if F0k > 0:
                        erb_pts.append((fc, 10**F0k))
                except:
                    pass

            r = create_erb_result_dict(model, 'A5_ExpE_transposed')
            r['erb_points'] = [(float(f), float(e)) for f, e in erb_pts]
            r['n_points_used'] = len(erb_pts)
            if len(erb_pts) >= 3:
                r['erb_similarity_vs_human'] = compute_erb_similarity(erb_pts)
                freqs = np.array([p[0] for p in erb_pts])
                erbs  = np.array([p[1] for p in erb_pts])
                s, b, rv, _, _ = linregress(freqs, erbs)
                r['erb_fit'] = {'a':float(s), 'b':float(b), 'r_squared':float(rv**2)}
            results[model] = r
            print(f"  {len(erb_pts)} ERB points from {len(carriers_tr)} carriers")
        except Exception as e:
            print(f"  [FAILED] {e}")
            import traceback; traceback.print_exc()
    out = os.path.join(STRATEGY_A_DIR, 'A5_ExpE_all_models.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[A5] {len(results)}/{len(models)} → {out}")
    return results


# ============================================================
def run_consensus(include_e: bool = True):
    """加载 A1/A2/A3/A4(/A5) 结果, 跨范式交叉验证.

    Args:
        include_e: 是否包含 Exp E (A5) 的 ERB 散点. 默认 True.
                   Exp E 的 ERB 映射是间接的 (f_harm=fc, 通过包络推断),
                   可比性弱于 A1-A4. 设为 False 可排除.
    """
    print("\n" + "=" * 70)
    print("Consensus: Cross-paradigm ERB")
    print(f"  (include A5/ExpE = {include_e})")
    print("=" * 70)
    from extract_ERB import load_strategy_results_from_dir, merge_strategy_results

    all_data = load_strategy_results_from_dir(STRATEGY_A_DIR, strategy_prefix='A')
    results = {}
    for model in MODEL_LIST:
        if model not in all_data or not all_data[model]:
            print(f"  [SKIP] {model}: no data"); continue
        all_pts = []
        for strategy in sorted(all_data[model].keys()):
            if not include_e and 'A5' in strategy:
                continue  # 跳过 Exp E
            pts = all_data[model][strategy].get('erb_points', [])
            all_pts.extend([(float(p[0]), float(p[1])) for p in pts])
        if len(all_pts) < 3:
            print(f"  [SKIP] {model}: {len(all_pts)} points (need >=3)"); continue
        freqs = np.array([p[0] for p in all_pts])
        erbs  = np.array([p[1] for p in all_pts])
        from scipy.stats import pearsonr
        if len(freqs) >= 3:
            r_human, _ = pearsonr(np.log10(erbs), np.log10(24.7*(4.37*freqs/1000+1)))
        else:
            r_human = 0.0
        r_dict = create_erb_result_dict(model, 'consensus_A')
        r_dict['erb_points'] = [(float(f), float(e)) for f, e in zip(freqs, erbs)]
        r_dict['n_points_used'] = len(all_pts)
        r_dict['n_strategies'] = len(all_data[model])
        r_dict['cross_paradigm_consistency_r'] = float(r_human)
        if len(all_pts) >= 3:
            r_dict['erb_similarity_vs_human'] = compute_erb_similarity(all_pts)
            s, b, rv, _, _ = linregress(freqs, erbs)
            r_dict['erb_fit'] = {'a':float(s), 'b':float(b), 'r_squared':float(rv**2)}
        results[model] = r_dict
        print(f"  {model}: {len(all_pts)} pts, {len(all_data[model])} strategies, r_human={r_human:.3f}")

    out = os.path.join(CONSENSUS_DIR, 'strategy_A_only.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n[{len(results)} models] → {out}")
    return results


# ============================================================
def main():
    parser = argparse.ArgumentParser(description='策略 A: 感知 ERB 测量')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--steps', type=str, default='A1,A2,A3,consensus')
    parser.add_argument('--no-e', action='store_true',
                       help='Exclude Exp E (A5) from consensus')
    args = parser.parse_args()
    models = MODEL_LIST if args.all else ([args.model] if args.model else [])
    if not models: parser.print_help(); return
    steps = [s.strip() for s in args.steps.split(',')]
    print(f"Models: {models}\nSteps: {steps}")
    t0 = time.time()
    for s in steps:
        {'A1': lambda: run_a1_exp_b(models),
         'A2': lambda: run_a2_exp_a(models),
         'A3': lambda: run_a3_exp_c(models),
         'A4': lambda: run_a4_exp_d(models),
         'A5': lambda: run_a5_exp_e(models),
         'consensus': lambda: run_consensus(include_e=not args.no_e)}.get(s, lambda: None)()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
