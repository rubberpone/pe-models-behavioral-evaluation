#!/usr/bin/env python3
"""Exp A & B 绘图驱动脚本 (v2).

- Exp A: 添加数据点独立性标注
- Exp B: 48 F0 bins + 精确 N 计算 + 谐波范围可视化 + 同刺激绑定
"""
import sys, os, json, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'analysis'))

from erb_utils import MODEL_LIST, EXISTING_EXP_B_PARAMS
from data_loader import (
    build_F0N_data_expA, build_F0N_data_expB, build_F0N_data_expC,
    build_F0N_data_expD, build_F0N_data_expE,
    expA_to_F0N_heatmap_data, expB_to_F0N_heatmap_data, expC_to_F0N_heatmap_data,
    expD_to_F0N_heatmap_data, expE_to_F0N_heatmap_data
)
from plot_F0N_heatmap import (
    plot_F0N_data_only, plot_F0N_with_strategy_ERB,
    plot_F0N_full_overlay, plot_F0N_3strategy_summary
)

FIG_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures', 'strategy_A', 'F0N_heatmaps')
DATA_DIR = os.path.join(PROJECT_ROOT, 'results', 'data', 'strategy_A')
os.makedirs(FIG_DIR, exist_ok=True)


def load_strategy_erb(model_name, strategy):
    """加载策略 ERB 散点."""
    strategy_files = {
        'A1': 'A1_ExpB_all_models.json',
        'A2': 'A2_ExpA_all_models.json',
        'A3': 'A3_ExpC_all_models.json',
        'A4': 'A4_ExpD_all_models.json',
        'A5': 'A5_ExpE_all_models.json',
    }
    fpath = os.path.join(DATA_DIR, strategy_files.get(strategy, ''))
    if not os.path.exists(fpath):
        return []
    with open(fpath, 'r') as f:
        data = json.load(f)
    model_data = data.get(model_name, {})
    return [(p[0], p[1]) for p in model_data.get('erb_points', [])]


def generate_expA_plots(model_name):
    """Exp A: 3 种图, 带数据点独立性标注."""
    try:
        raw = build_F0N_data_expA(model_name)
        hm = expA_to_F0N_heatmap_data(raw)
        a2_pts = load_strategy_erb(model_name, 'A2')

        for plot_type, func, suffix in [
            ('data', lambda: plot_F0N_data_only(hm, 'A', model_name,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_A_data_only.png')),
             'data_only'),
            ('strategy', lambda: plot_F0N_with_strategy_ERB(hm, 'A', model_name,
                      strategy_erb_points=a2_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_A_strategy_ERB.png')),
             'strategy_ERB'),
            ('full', lambda: plot_F0N_full_overlay(hm, 'A', model_name,
                      strategy_erb_points=a2_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_A_full_overlay.png')),
             'full_overlay'),
        ]:
            func()
            print(f"    {suffix} ✓")
        return True
    except Exception as e:
        print(f"    [FAILED] {e}")
        return False


def generate_expB_plots(model_name):
    """Exp B v2: 48 F0 bins + 精确 N 计算 + 谐波范围可视化."""
    try:
        raw = build_F0N_data_expB(model_name)
        hm = expB_to_F0N_heatmap_data(raw)  # v2: 48 bins + harm_min/max
        a1_pts = load_strategy_erb(model_name, 'A1')

        print(f"    Exp B data: {len(hm['N'])} points, "
              f"N range=[{np.min(hm['N']):.1f}, {np.max(hm['N']):.1f}], "
              f"F0 bins={len(np.unique(hm['F0']))}")

        for plot_type, func, suffix in [
            ('data', lambda: plot_F0N_data_only(hm, 'B', model_name,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_B_data_only.png')),
             'data_only'),
            ('strategy', lambda: plot_F0N_with_strategy_ERB(hm, 'B', model_name,
                      strategy_erb_points=a1_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_B_strategy_ERB.png')),
             'strategy_ERB'),
            ('full', lambda: plot_F0N_full_overlay(hm, 'B', model_name,
                      strategy_erb_points=a1_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_B_full_overlay.png')),
             'full_overlay'),
        ]:
            func()
            print(f"    {suffix} ✓")
        return True
    except Exception as e:
        print(f"    [FAILED] {e}")
        import traceback; traceback.print_exc()
        return False


def generate_expD_plots(model_name):
    """Exp D: Mistuning sensitivity decay — 3 F0s × 12 harmonics."""
    try:
        raw = build_F0N_data_expD(model_name)
        hm = expD_to_F0N_heatmap_data(raw)
        a4_pts = load_strategy_erb(model_name, 'A4')
        print(f"    Exp D data: {len(hm['N'])} points, "
              f"F0 bins={len(np.unique(hm['F0']))}, N range=[{np.min(hm['N']):.0f}, {np.max(hm['N']):.0f}]")
        for plot_type, func, suffix in [
            ('data', lambda: plot_F0N_data_only(hm, 'D', model_name, n_range=(0.5, 20),
                      output_path=os.path.join(FIG_DIR, f'{model_name}_D_data_only.png')),
             'data_only'),
            ('strategy', lambda: plot_F0N_with_strategy_ERB(hm, 'D', model_name, n_range=(0.5, 20),
                      strategy_erb_points=a4_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_D_strategy_ERB.png')),
             'strategy_ERB'),
            ('full', lambda: plot_F0N_full_overlay(hm, 'D', model_name, n_range=(0.5, 20),
                      strategy_erb_points=a4_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_D_full_overlay.png')),
             'full_overlay'),
        ]:
            func()
            print(f"    {suffix} ✓")
        return True
    except Exception as e:
        print(f"    [FAILED] {e}")
        import traceback; traceback.print_exc()
        return False


def generate_expE_plots(model_name):
    """Exp E: F0DL for pure tones + transposed tones — full overlay set."""
    try:
        raw = build_F0N_data_expE(model_name)
        hm = expE_to_F0N_heatmap_data(raw)
        a5_pts = load_strategy_erb(model_name, 'A5')
        print(f"    Exp E data: {len(hm['N'])} points, "
              f"N range=[{np.min(hm['N']):.1f}, {np.max(hm['N']):.1f}], "
              f"F0 bins={len(np.unique(hm['F0']))}")
        for plot_type, func, suffix in [
            ('data', lambda: plot_F0N_data_only(hm, 'E', model_name,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_E_data_only.png')),
             'data_only'),
            ('strategy', lambda: plot_F0N_with_strategy_ERB(hm, 'E', model_name,
                      strategy_erb_points=a5_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_E_strategy_ERB.png')),
             'strategy_ERB'),
            ('full', lambda: plot_F0N_full_overlay(hm, 'E', model_name,
                      strategy_erb_points=a5_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_E_full_overlay.png')),
             'full_overlay'),
        ]:
            func()
            print(f"    {suffix} ✓")
        return True
    except Exception as e:
        print(f"    [FAILED] {e}")
        import traceback; traceback.print_exc()
        return False


def generate_expC_plots(model_name):
    """Exp C: Pitch Shift slope 数据图."""
    try:
        raw = build_F0N_data_expC(model_name)
        hm = expC_to_F0N_heatmap_data(raw)
        a3_pts = load_strategy_erb(model_name, 'A3')

        print(f"    Exp C data: {len(hm['N'])} points")

        for plot_type, func, suffix in [
            ('data', lambda: plot_F0N_data_only(hm, 'C', model_name,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_C_data_only.png')),
             'data_only'),
            ('strategy', lambda: plot_F0N_with_strategy_ERB(hm, 'C', model_name,
                      strategy_erb_points=a3_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_C_strategy_ERB.png')),
             'strategy_ERB'),
            ('full', lambda: plot_F0N_full_overlay(hm, 'C', model_name,
                      strategy_erb_points=a3_pts,
                      output_path=os.path.join(FIG_DIR, f'{model_name}_C_full_overlay.png')),
             'full_overlay'),
        ]:
            func()
            print(f"    {suffix} ✓")
        return True
    except Exception as e:
        print(f"    [FAILED] {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--experiment', type=str, default='A,B')
    args = parser.parse_args()

    models = MODEL_LIST if args.all else ([args.model] if args.model else [])
    if not models:
        parser.print_help()
        return

    exps = [e.strip() for e in args.experiment.split(',')]

    for model in models:
        print(f"\n{'='*50}\n{model}\n{'='*50}")
        if 'A' in exps:
            generate_expA_plots(model)
        if 'B' in exps:
            generate_expB_plots(model)
        if 'C' in exps:
            generate_expC_plots(model)
        if 'D' in exps:
            generate_expD_plots(model)
        if 'E' in exps:
            generate_expE_plots(model)

    print(f"\nDone. Figures saved to: {FIG_DIR}")


if __name__ == '__main__':
    main()
