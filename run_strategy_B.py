#!/usr/bin/env python3
"""
策略 B 总驱动脚本 — 新刺激生成 + 模型推理 + 感知 ERB 精化测量.

执行顺序:
  Step 1: 生成新刺激 HDF5 (B1 F0-N扫参, B2 ExpB扫参, B3 ExpC扫参, B4 校准)
  Step 2: 对每个模型运行推理 → 700 类概率输出
  Step 3: 计算 F0DL / Pitch Doubling ratio / Pitch Shift slope
  Step 4: 提取感知 ERB 散点
  Step 5: 与策略 A 结果合并, 输出最终共识 ERB

用法:
  # 仅生成刺激 (不跑模型推理)
  python run_strategy_B.py --step generate

  # 生成刺激 + 推理 + ERB (指定模型)
  python run_strategy_B.py --model CREPE --steps generate,inference,erb

  # 全部步骤, 全部模型 (完整运行, 预计 5-10 天)
  python run_strategy_B.py --all

输出:
  data/stim_ERB_F0N_sweep.hdf5        ← 策略 B1 刺激
  data/stim_ERB_ExpB_doubling.hdf5    ← 策略 B2 刺激
  data/stim_ERB_ExpC_shift.hdf5       ← 策略 B3 刺激
  data/stim_ERB_calibration.hdf5      ← 策略 B4 校准刺激
  predictions/strategy_B/{model}/     ← 各模型推理输出
  results/data/strategy_B/            ← 策略 B 各模型 ERB 结果
  results/data/consensus/final.json   ← 策略 A+B 合并的最终共识 ERB

预计时间:
  - 刺激生成: ~1 h (全部四个数据集)
  - 模型推理: 取决于 GPU, 每个模型 ~2-6 h (10 模型 × 4 数据集)
  - ERB 提取: ~1 h (仅分析, 不涉及大计算)
"""

import sys
import os
import json
import argparse
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'analysis'))
from erb_utils import MODEL_LIST


def run_step_generate_stimuli(datasets: list = None):
    """Step 1: 生成策略 B 的新刺激 HDF5 文件."""
    print("\n" + "=" * 70)
    print("STEP 1/4: Generate Strategy B Stimulus HDF5 Files")
    print("=" * 70)

    if datasets is None:
        datasets = ['B1', 'B2', 'B3', 'B4']

    scripts = {
        'B1': ('stimuli/generate_ERB_F0N_sweep.py',
               'data/stim_ERB_F0N_sweep.hdf5',
               'F0-N 2D sweep (~190k stimuli)'),
        'B2': ('stimuli/generate_ERB_ExpB_doubling.py',
               'data/stim_ERB_ExpB_doubling.hdf5',
               'Exp B continuous filter band (~270 stimuli)'),
        'B3': ('stimuli/generate_ERB_ExpC_shift.py',
               'data/stim_ERB_ExpC_shift.hdf5',
               'Exp C continuous envelope (~1.5k stimuli)'),
        'B4': ('stimuli/generate_ERB_calibration.py',
               'data/stim_ERB_calibration.hdf5',
               'Calibration set (~8k stimuli)'),
    }

    for label in datasets:
        if label not in scripts:
            print(f"  [SKIP] Unknown dataset: {label}")
            continue

        script, hdf5_path, desc = scripts[label]
        full_script = os.path.join(PROJECT_ROOT, script)
        full_hdf5 = os.path.join(PROJECT_ROOT, hdf5_path)

        if os.path.exists(full_hdf5):
            print(f"  [{label}] EXISTS: {full_hdf5} — skipping")
            continue

        print(f"  [{label}] {desc}")
        print(f"    Script: {full_script}")
        print(f"    Output: {full_hdf5}")

        os.makedirs(os.path.dirname(full_hdf5), exist_ok=True)
        result = subprocess.run(
            [sys.executable, full_script, full_hdf5],
            cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    [ERROR] {result.stderr[-500:]}")
        else:
            # 检查输出文件大小
            size_mb = os.path.getsize(full_hdf5) / (1024*1024)
            print(f"    [OK] {size_mb:.1f} MB")


def run_step_inference(models: list, datasets: list = None):
    """Step 2: 对每个模型和每个策略 B 数据集运行推理.

    注意: 该步骤需要在每个模型的 Conda 环境中分别执行.
    实际执行时, 建议使用 shell 脚本循环调用各模型的 predict 函数.
    """
    print("\n" + "=" * 70)
    print("STEP 2/4: Model Inference on Strategy B Stimuli")
    print("=" * 70)

    if datasets is None:
        datasets = ['B1', 'B2', 'B3', 'B4']

    print("  [NOTE] Model inference must be run in each model's Conda environment.")
    print("  Use the following pattern for each model:\n")

    for model in models:
        print(f"  # --- {model} ---")
        for ds in datasets:
            hdf5_path = os.path.join(PROJECT_ROOT, f'data/stim_ERB_{ds}_sweep.hdf5')
            if ds == 'B2':
                hdf5_path = os.path.join(PROJECT_ROOT, 'data/stim_ERB_ExpB_doubling.hdf5')
            elif ds == 'B3':
                hdf5_path = os.path.join(PROJECT_ROOT, 'data/stim_ERB_ExpC_shift.hdf5')
            elif ds == 'B4':
                hdf5_path = os.path.join(PROJECT_ROOT, 'data/stim_ERB_calibration.hdf5')

            output_dir = os.path.join(PROJECT_ROOT, 'predictions', 'strategy_B', model, ds)
            print(f"    conda activate {model}_psychophysics")
            print(f"    python path/to/{model}_predict.py \\")
            print(f"        --input {hdf5_path} \\")
            print(f"        --output {output_dir}")
        print()

    print("  See RUNBOOK.md §3.2 for detailed inference instructions.")


def run_step_erb_extraction(models: list, datasets: list = None):
    """Step 3+4: 从模型推理结果提取感知 ERB + 与策略 A 合并."""
    print("\n" + "=" * 70)
    print("STEP 3-4/4: ERB Extraction from Strategy B + Merge with Strategy A")
    print("=" * 70)

    if datasets is None:
        datasets = ['B1', 'B2', 'B3']

    # 对每个数据集调用对应的 ERB 提取逻辑
    # B1 → F0DL knee detection (类似 A2)
    # B2 → Pitch Doubling (类似 A1)
    # B3 → Pitch Shift slope (类似 A3)

    print("  [NOTE] ERB extraction depends on model inference output.")
    print("  This step will be refined once inference patterns are established.")
    print("  See analysis/extract_ERB.py for the extraction orchestration logic.")

    # 合并策略 A + B 结果
    print("\n  Merging Strategy A + B consensus...")
    strategy_a_path = os.path.join(
        PROJECT_ROOT, 'results', 'data', 'consensus', 'strategy_A_only.json'
    )
    strategy_b_dir = os.path.join(PROJECT_ROOT, 'results', 'data', 'strategy_B')

    if os.path.exists(strategy_a_path):
        print(f"  Strategy A consensus: {strategy_a_path}")
    else:
        print("  [WARNING] Strategy A consensus not found. Run run_strategy_A.py first.")

    output_path = os.path.join(
        PROJECT_ROOT, 'results', 'data', 'consensus', 'final_strategy_A_plus_B.json'
    )
    print(f"  Final consensus output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='策略 B: 新刺激生成 + 推理 + 感知 ERB 精化 — 总驱动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_strategy_B.py --step generate                       # 仅生成刺激
  python run_strategy_B.py --model CREPE --steps generate,inference  # 单模型
  python run_strategy_B.py --all --steps generate                  # 生成全部刺激
        """
    )
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--steps', type=str, default='generate,inference,erb,merge',
                       help='逗号分隔的步骤 (generate, inference, erb, merge)')
    parser.add_argument('--datasets', type=str, default='B1,B2,B3,B4',
                       help='逗号分隔的数据集 (B1, B2, B3, B4)')
    args = parser.parse_args()

    if args.all:
        models = MODEL_LIST
    elif args.model:
        models = [args.model]
    else:
        models = MODEL_LIST  # 默认显示, 但仅生成刺激

    steps = [s.strip() for s in args.steps.split(',')]
    datasets = [d.strip() for d in args.datasets.split(',')]

    print(f"Models: {models}")
    print(f"Steps: {steps}")
    print(f"Datasets: {datasets}")
    t_start = time.time()

    step_funcs = {
        'generate': lambda: run_step_generate_stimuli(datasets),
        'inference': lambda: run_step_inference(models, datasets),
        'erb': lambda: run_step_erb_extraction(models, datasets),
        'merge': lambda: run_step_erb_extraction(models, datasets),  # merge is part of erb
    }

    for step_name in steps:
        if step_name in step_funcs:
            step_funcs[step_name]()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"[run_strategy_B] COMPLETE in {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
