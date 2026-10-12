#!/usr/bin/env python3
"""
数据加载模块 — 将 HDF5 刺激元数据与预测 JSON 对齐, 构建 F0-N 平面数据.

核心逻辑:
  - 预测 JSON: {f0_label:labels_pred, f0_label:probs_out}
  - HDF5: {f0, base_f0, low_harm, phase_mode, filter_fl, f0_shift, ...}
  - 二者按行索引对齐, 合并为统一的 DataFrame-like 结构
"""

import os
import json
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erb_utils import (
    get_f0_bins_700, JOSH_HDF5_DIR,
    EXISTING_HDF5_FILES, EXISTING_EXP_A_PARAMS,
    EXISTING_EXP_B_PARAMS, EXISTING_EXP_C_PARAMS,
    EXISTING_EXP_D_PARAMS, EXISTING_EXP_E_PARAMS,
    MODEL_PSYCHOPHYSICS_DIRS, MODEL_LIST
)


def load_prediction_json(model_name: str, experiment: str) -> Optional[dict]:
    """加载某个模型在某个实验上的预测 JSON.

    Args:
        model_name: 'CREPE', 'FCN', ...
        experiment: 'A'/'B'/'C'/'D'/'E'
    Returns:
        dict with keys f0_label:labels_pred, f0_label:probs_out
    """
    exp_names = {
        'A': 'BernsteinOxenham2005',
        'B': 'ShackletonCarlyon1994',
        'C': 'MooreMoore2003',
        'D': 'MooreEtAl1985',
        'E': 'OxenhamEtAl2004',
    }
    exp_name = exp_names.get(experiment, experiment)

    base_dir = MODEL_PSYCHOPHYSICS_DIRS.get(model_name)
    if not base_dir:
        return None

    import glob
    # 自动发现预测目录和文件: {MODEL}_psy_predictions/predictions_*.json
    # 支持带后缀的目录名, 如 FTATTENTION_psy_predictions_7_14
    pred_dirs = sorted(glob.glob(os.path.join(base_dir, '*_psy_predictions*')))
    for pred_dir in pred_dirs:
        pattern = os.path.join(pred_dir, f'predictions_*_{exp_name}.json')
        matches = sorted(glob.glob(pattern))
        if matches:
            with open(matches[0], 'r') as f:
                return json.load(f)

    # fallback: direct glob without subdirectory
    pattern = os.path.join(base_dir, f'predictions_*_{exp_name}.json')
    matches = sorted(glob.glob(pattern))
    if matches:
        with open(matches[0], 'r') as f:
            return json.load(f)

    return None


def load_hdf5_metadata(hdf5_path: str) -> dict:
    """从 HDF5 文件加载刺激元数据 (所有列).

    Args:
        hdf5_path: HDF5 文件路径
    Returns:
        dict, 每个键 → numpy 数组 (N_trials,)
    """
    metadata = {}
    with h5py.File(hdf5_path, 'r') as f:
        for key in f.keys():
            if key.startswith('config_'):
                continue
            ds = f[key]
            if not hasattr(ds, 'shape'):
                continue
            # 仅加载一维标量元数据, 跳过大型多维数据集 (波形等)
            if len(ds.shape) != 1:
                continue
            size_mb = ds.size * ds.dtype.itemsize / 1e6
            if size_mb > 50:  # >50 MB, 跳过
                continue
            metadata[key] = ds[:]
    return metadata


def build_F0N_data_expA(model_name: str,
                         hdf5_path: str = None,
                         prediction_json_path: str = None) -> Dict[str, np.ndarray]:
    """构建 Exp A 的 F0-N 平面数据.

    策略 A2 的核心数据源:
      从 HDF5 取 base_f0, low_harm, f0, phase_mode
      从预测 JSON 取 f0_label:labels_pred, f0_label:probs_out
      解码 → f0_pred
      按 (base_f0, low_harm) 分组 → 计算 per-condition F0DL

    Returns:
        {'N': array, 'F0': array, 'f0dl': array, 'f0': array, 'f0_pred': array}
    """
    if hdf5_path is None:
        hdf5_path = EXISTING_HDF5_FILES['A']
    if prediction_json_path is None:
        pred_json = load_prediction_json(model_name, 'A')
    else:
        with open(prediction_json_path, 'r') as f:
            pred_json = json.load(f)

    if pred_json is None:
        raise FileNotFoundError(f"No Exp A predictions for {model_name}")

    # HDF5 元数据
    meta = load_hdf5_metadata(hdf5_path)
    base_f0 = meta.get('base_f0', meta.get('f0'))
    low_harm = meta.get('low_harm')
    f0_true = meta.get('f0')
    phase_mode = meta.get('phase_mode')

    # 预测 → Hz
    f0_bins = get_f0_bins_700()
    labels_pred = np.array(pred_json.get('f0_label:labels_pred', []))
    probs_out = pred_json.get('f0_label:probs_out', [])

    # 先验约束 argmax: 对每个刺激, 在 base_f0 ±0.5 octave 范围内选最大值
    f0_pred = np.zeros(len(labels_pred))
    for i in range(len(labels_pred)):
        bf0 = base_f0[i] if base_f0 is not None else f0_true[i]
        prior_lo = bf0 * np.power(2.0, -0.5)
        prior_hi = bf0 * np.power(2.0, 0.5)
        mask = (f0_bins >= prior_lo) & (f0_bins <= prior_hi)
        if probs_out and i < len(probs_out) and probs_out[i] is not None:
            masked = np.array(probs_out[i]) * mask.astype(float)
            f0_pred[i] = f0_bins[np.argmax(masked)]
        else:
            # fallback: direct label
            lbl = labels_pred[i]
            if lbl is not None and 0 <= lbl < len(f0_bins):
                f0_pred[i] = f0_bins[int(lbl)]

    return {
        'N': low_harm,
        'F0': base_f0,
        'f0': f0_true,
        'f0_pred': f0_pred,
        'phase_mode': phase_mode,
    }


def build_F0N_data_expB(model_name: str,
                         hdf5_path: str = None,
                         prediction_json_path: str = None) -> Dict[str, np.ndarray]:
    """构建 Exp B 的 F0-N 平面数据.

    策略 A1 的核心数据源.
    从 HDF5 取 f0, filter_fl, filter_fh, phase_mode
    从预测 JSON 取 f0_label:labels_pred, f0_label:probs_out
    """
    if hdf5_path is None:
        hdf5_path = EXISTING_HDF5_FILES['B']
    if prediction_json_path is None:
        pred_json = load_prediction_json(model_name, 'B')
    else:
        with open(prediction_json_path, 'r') as f:
            pred_json = json.load(f)

    if pred_json is None:
        raise FileNotFoundError(f"No Exp B predictions for {model_name}")

    meta = load_hdf5_metadata(hdf5_path)
    f0_true = meta.get('f0')
    filter_fl = meta.get('filter_fl')
    filter_fh = meta.get('filter_fh')
    phase_mode = meta.get('phase_mode')

    f0_bins = get_f0_bins_700()
    labels_pred = np.array(pred_json.get('f0_label:labels_pred', []))
    probs_out = pred_json.get('f0_label:probs_out', [])

    f0_pred = np.zeros(len(labels_pred))
    for i in range(len(labels_pred)):
        lbl = labels_pred[i]
        if lbl is not None and 0 <= lbl < len(f0_bins):
            f0_pred[i] = f0_bins[int(lbl)]

    return {
        'f0': f0_true,
        'f0_pred': f0_pred,
        'filter_fl': filter_fl,
        'filter_fh': filter_fh,
        'phase_mode': phase_mode,
    }


def build_F0N_data_expC(model_name: str,
                         hdf5_path: str = None,
                         prediction_json_path: str = None) -> Dict[str, np.ndarray]:
    """构建 Exp C 的 F0-N 平面数据.

    策略 A3 的核心数据源.
    """
    if hdf5_path is None:
        hdf5_path = EXISTING_HDF5_FILES['C']
    if prediction_json_path is None:
        pred_json = load_prediction_json(model_name, 'C')
    else:
        with open(prediction_json_path, 'r') as f:
            pred_json = json.load(f)

    if pred_json is None:
        raise FileNotFoundError(f"No Exp C predictions for {model_name}")

    meta = load_hdf5_metadata(hdf5_path)
    f0_true = meta.get('f0')
    f0_shift = meta.get('f0_shift')
    centered_harmonic = meta.get('spectral_envelope_centered_harmonic')
    bandwidth_harmonic = meta.get('spectral_envelope_bandwidth_in_harmonics')

    f0_bins = get_f0_bins_700()
    labels_pred = np.array(pred_json.get('f0_label:labels_pred', []))
    probs_out = pred_json.get('f0_label:probs_out', [])

    f0_pred = np.zeros(len(labels_pred))
    for i in range(len(labels_pred)):
        lbl = labels_pred[i]
        if lbl is not None and 0 <= lbl < len(f0_bins):
            f0_pred[i] = f0_bins[int(lbl)]

    return {
        'f0': f0_true,
        'f0_pred': f0_pred,
        'f0_shift': f0_shift,
        'spectral_envelope_centered_harmonic': centered_harmonic,
        'spectral_envelope_bandwidth_in_harmonics': bandwidth_harmonic,
    }


# ============================================================
#  便捷: 构建 F0-N 可视化数据 (直接来自 data_loader 的输出)
# ============================================================

def expA_to_F0N_heatmap_data(expA_data: dict,
                              model_name: str = None,
                              max_pct_diff: float = 100/6,
                              noise_stdev: float = 1e-12) -> dict:
    """从 Exp A 原始数据构建可传递给 plot_F0N_heatmap 的数据.

    对每个 (base_f0, low_harm) 条件计算 F0DL, 返回 F0-N 三元组.
    """
    from compute_F0DL import compute_F0DL_for_condition, filter_expt_dict

    N_list, F0_list, F0DL_list = [], [], []

    unique_base_f0s = np.unique(expA_data['F0'])
    unique_low_harms = np.unique(expA_data['N'])

    for bf0 in unique_base_f0s:
        for lh in unique_low_harms:
            mask = (expA_data['F0'] == bf0) & (expA_data['N'] == lh)
            n_stimuli = np.sum(mask)
            if n_stimuli < 10:
                continue

            # 提取该条件的所有刺激
            f0_true = expA_data['f0'][mask]
            f0_pred = expA_data['f0_pred'][mask]

            if len(f0_true) < 10:
                continue

            # 简化: 不跑完整 2AFC, 直接用预测误差的 CV 作为 F0DL 近似
            # 精确 F0DL 需要 pairwise 比较 (计算量太大)
            pred_errors = np.abs(f0_pred - f0_true) / f0_true * 100.0  # %F0
            f0dl_approx = np.median(pred_errors)

            N_list.append(lh)
            F0_list.append(bf0)
            F0DL_list.append(f0dl_approx)

    return {
        'N': np.array(N_list),
        'F0': np.array(F0_list),
        'color_val': np.array(F0DL_list),
        'experiment': 'A',
    }


def expB_to_F0N_heatmap_data(expB_data: dict,
                              f0_bin_centers: list = None,
                              f0_tolerance: float = 0.005,
                              phase_mode: int = 4) -> dict:
    """从 Exp B 原始数据构建 F0-N 热力图数据 (v2 改进版).

    改进点:
      - 48 个对数均匀代表 F0 (原 6 个)
      - N 的精确计算: h_min=ceil(fl/F0), h_max=floor(fh/F0), N_rep=sqrt(h_min*h_max)
      - 返回谐波范围信息用于可视化

    对于每个 (filter_fl, f0_representative):
      - 取 F0 ± tolerance 范围内的 ALT 相位 stimuli
      - 计算 median(f0_pred/f0) 作为颜色值
      - 计算精确的谐波覆盖范围 [h_min, h_max]
      - 代表点 N = sqrt(h_min * h_max) (几何平均)
    """
    if f0_bin_centers is None:
        f0_bin_centers = EXISTING_EXP_B_PARAMS['expanded_f0_bin_centers']

    N_list, F0_list, Ratio_list = [], [], []
    harm_min_list, harm_max_list = [], []

    f0_all = expB_data['f0']
    f0_pred = expB_data['f0_pred']
    filter_fl_all = expB_data['filter_fl']
    filter_fh_all = expB_data['filter_fh']
    phase_all = expB_data['phase_mode']

    for fl, fh in [(125.0, 625.0), (1375.0, 1875.0), (3900.0, 5400.0)]:
        for f0_rep in f0_bin_centers:
            # 精确匹配: F0 在代表值 ±tolerance 范围内
            mask = (filter_fl_all == fl)
            mask &= (f0_all >= f0_rep * (1 - f0_tolerance))
            mask &= (f0_all <= f0_rep * (1 + f0_tolerance))
            mask &= (phase_all == phase_mode)

            n_stim = np.sum(mask)
            if n_stim == 0:
                continue

            # 计算该 (f0_rep, filter_band) 的精确谐波覆盖范围
            h_min = int(np.ceil(fl / f0_rep))
            h_max = int(np.floor(fh / f0_rep))

            if h_max < h_min:
                continue

            # 代表点 N = 几何平均 (在对数 N 轴上更合适)
            N_rep = np.sqrt(h_min * h_max)

            ratios = f0_pred[mask] / f0_all[mask]
            N_list.append(N_rep)
            F0_list.append(f0_rep)
            Ratio_list.append(np.median(ratios))
            harm_min_list.append(h_min)
            harm_max_list.append(h_max)

    return {
        'N': np.array(N_list),
        'F0': np.array(F0_list),
        'color_val': np.array(Ratio_list),
        'experiment': 'B',
        'harm_min': np.array(harm_min_list),
        'harm_max': np.array(harm_max_list),
    }


def expC_to_F0N_heatmap_data(expC_data: dict,
                              f0_targets: list = None,
                              f0_tolerance: float = 0.005) -> dict:
    """从 Exp C 原始数据构建 F0-N 热力图数据 (v2 改进版).

    改进点:
      - 48 个对数均匀代表 F0 (原 5 个)
      - 画每个 envelope 通带内的全部谐波成分 (低饱和度绑定)
      - N 值 = centered_harmonic (5/11/16), 不变化
      - 对每个 (F0_rep, centered_harmonic), 在 ±tolerance 内取所有 shift 的
        stimulus, 按 shift 分组取 median(pred_shift) 后做线性回归求 slope
    """
    from scipy.stats import linregress

    if f0_targets is None:
        f0_targets = EXISTING_EXP_C_PARAMS['representative_f0s']

    N_list, F0_list, Slope_list = [], [], []
    harm_min_list, harm_max_list = [], []

    f0_all = expC_data['f0']
    f0_pred = expC_data['f0_pred']
    f0_shift = expC_data['f0_shift']
    centered = expC_data['spectral_envelope_centered_harmonic']

    # 建立 centered_harmonic → harm_range 映射
    env_harm_ranges = {}
    for env in EXISTING_EXP_C_PARAMS['spectral_envelopes']:
        h_center = env['centered_harmonic']
        h_min, h_max = env['harm_range']
        env_harm_ranges[h_center] = (h_min, h_max)

    for f0_tgt in f0_targets:
        for harm in np.unique(centered):
            mask = (centered == harm)
            mask &= (f0_all >= f0_tgt * (1 - f0_tolerance))
            mask &= (f0_all <= f0_tgt * (1 + f0_tolerance))

            if np.sum(mask) < 10:
                continue

            f0s = f0_all[mask]
            f0ps = f0_pred[mask]
            shifts = f0_shift[mask]

            pred_shift = (f0ps - f0s) / f0s
            unique_shifts = np.unique(shifts)

            if len(unique_shifts) < 3:
                continue

            # 按 shift 分组取 median, 然后线性回归
            mean_pred_shift = [np.median(pred_shift[shifts == s]) for s in unique_shifts]
            slope, _, _, _, _ = linregress(unique_shifts, mean_pred_shift)

            N_list.append(harm)
            F0_list.append(f0_tgt)
            Slope_list.append(abs(slope))

            # 谐波范围: 从 envelope 定义中获取
            h_range = env_harm_ranges.get(harm, (harm, harm))
            harm_min_list.append(h_range[0])
            harm_max_list.append(h_range[1])

    return {
        'N': np.array(N_list),
        'F0': np.array(F0_list),
        'color_val': np.array(Slope_list),
        'experiment': 'C',
        'harm_min': np.array(harm_min_list),
        'harm_max': np.array(harm_max_list),
    }


# ============================================================
#  Exp D: Mistuned Harmonics
# ============================================================

def build_F0N_data_expD(model_name: str,
                         hdf5_path: str = None,
                         prediction_json_path: str = None) -> Dict[str, np.ndarray]:
    """构建 Exp D 的 F0-N 平面数据.

    HDF5 元数据: f0, mistuned_harm, mistuned_pct
    预测 JSON: f0_label:labels_pred, f0_label:probs_out
    """
    if hdf5_path is None:
        hdf5_path = EXISTING_HDF5_FILES['D']
    if prediction_json_path is None:
        pred_json = load_prediction_json(model_name, 'D')
    else:
        with open(prediction_json_path, 'r') as f:
            pred_json = json.load(f)

    if pred_json is None:
        raise FileNotFoundError(f"No Exp D predictions for {model_name}")

    meta = load_hdf5_metadata(hdf5_path)
    f0_true = meta.get('f0')
    mistuned_harm = meta.get('mistuned_harm')
    mistuned_pct = meta.get('mistuned_pct')

    f0_bins = get_f0_bins_700()
    labels_pred = np.array(pred_json.get('f0_label:labels_pred', []))
    probs_out = pred_json.get('f0_label:probs_out', [])

    f0_pred = np.zeros(len(labels_pred))
    for i in range(len(labels_pred)):
        lbl = labels_pred[i]
        if lbl is not None and 0 <= lbl < len(f0_bins):
            f0_pred[i] = f0_bins[int(lbl)]

    return {
        'f0': f0_true,
        'f0_pred': f0_pred,
        'mistuned_harm': mistuned_harm,
        'mistuned_pct': mistuned_pct,
    }


def expD_to_F0N_heatmap_data(expD_data: dict,
                              f0_targets: list = None,
                              f0_tolerance: float = 0.02) -> dict:
    """从 Exp D 原始数据构建 F0-N 热力图数据.

    对每个 (标称F0, 失谐谐波 H):
      - 聚合该 F0 ± tolerance 内的 stimuli (178 个变体)
      - 按 mistuning_pct 分组取 median(pred_shift)
      - linregress(mistuning_pct, median_pred_shift) → |sensitivity|
      - N = H (失谐谐波编号), F0 = 标称F0, color = |sensitivity|
    """
    from scipy.stats import linregress

    if f0_targets is None:
        f0_targets = EXISTING_EXP_D_PARAMS['f0_ref_list']

    N_list, F0_list, Sens_list = [], [], []

    f0_all = expD_data['f0']
    f0_pred = expD_data['f0_pred']
    mistuned_harm = expD_data['mistuned_harm']
    mistuned_pct = expD_data['mistuned_pct']

    for f0_tgt in f0_targets:
        for harm in EXISTING_EXP_D_PARAMS['mistuned_harmonics']:
            # 筛选: 标称 F0 ± tolerance + 指定失谐谐波
            mask = (mistuned_harm == harm)
            mask &= (f0_all >= f0_tgt * (1 - f0_tolerance))
            mask &= (f0_all <= f0_tgt * (1 + f0_tolerance))

            if np.sum(mask) < 10:
                continue

            f0s = f0_all[mask]; f0ps = f0_pred[mask]; pcts = mistuned_pct[mask]
            pred_shift = (f0ps - f0s) / f0s  # %F0
            unique_pcts = np.unique(pcts)

            if len(unique_pcts) < 3:
                continue

            # 按 mistuning_pct 分组取 median, 线性回归求斜率
            mean_ps = [np.median(pred_shift[pcts == mp]) for mp in unique_pcts]
            slope, _, _, _, _ = linregress(unique_pcts, mean_ps)

            N_list.append(harm)
            F0_list.append(f0_tgt)
            Sens_list.append(abs(slope))

    return {
        'N': np.array(N_list),
        'F0': np.array(F0_list),
        'color_val': np.array(Sens_list),
        'experiment': 'D',
    }


# ============================================================
#  Exp E: Transposed Tones
# ============================================================

def build_F0N_data_expE(model_name: str,
                         hdf5_path: str = None,
                         prediction_json_path: str = None) -> Dict[str, np.ndarray]:
    """构建 Exp E 的 F0-N 平面数据.

    HDF5 元数据: f_envelope, f_carrier
    预测 JSON: f0_label:labels_pred, f0_label:probs_out
    """
    if hdf5_path is None:
        hdf5_path = EXISTING_HDF5_FILES['E']
    if prediction_json_path is None:
        pred_json = load_prediction_json(model_name, 'E')
    else:
        with open(prediction_json_path, 'r') as f:
            pred_json = json.load(f)

    if pred_json is None:
        raise FileNotFoundError(f"No Exp E predictions for {model_name}")

    meta = load_hdf5_metadata(hdf5_path)
    f_envelope = meta.get('f_envelope')
    f_carrier = meta.get('f_carrier')

    f0_bins = get_f0_bins_700()
    labels_pred = np.array(pred_json.get('f0_label:labels_pred', []))
    probs_out = pred_json.get('f0_label:probs_out', [])

    f0_pred = np.zeros(len(labels_pred))
    for i in range(len(labels_pred)):
        lbl = labels_pred[i]
        if lbl is not None and 0 <= lbl < len(f0_bins):
            f0_pred[i] = f0_bins[int(lbl)]

    return {
        'f_envelope': f_envelope,
        'f_carrier': f_carrier,
        'f0_pred': f0_pred,
    }


def expE_to_F0N_heatmap_data(expE_data: dict,
                              f0_targets: list = None,
                              f0_tolerance: float = 0.005) -> dict:
    """从 Exp E 原始数据构建 F0-N 热力图数据.

    对每个 (代表F0, 载波条件):
      - 纯音 (fc=0): N=1, 固定
      - 转置音 (fc>0): N = fc / f_env
      - F0DL 近似: median(|f0_pred - f_envelope| / f_envelope × 100)
      - 在 ±tolerance 范围内聚合取 median
    """
    if f0_targets is None:
        f0_targets = EXISTING_EXP_E_PARAMS['representative_f0s']

    N_list, F0_list, F0DL_list = [], [], []

    f_env_all = expE_data['f_envelope']
    f_carrier_all = expE_data['f_carrier']
    f0_pred = expE_data['f0_pred']

    unique_carriers = np.unique(f_carrier_all)

    for fc in unique_carriers:
        for f0_rep in f0_targets:
            mask = (f_carrier_all == fc)
            mask &= (f_env_all >= f0_rep * (1 - f0_tolerance))
            mask &= (f_env_all <= f0_rep * (1 + f0_tolerance))

            if np.sum(mask) == 0:
                continue

            f_envs = f_env_all[mask]
            preds = f0_pred[mask]

            # F0DL 近似: median(|pred_error| %)
            pred_errors = np.abs(preds - f_envs) / f_envs * 100.0
            f0dl = np.median(pred_errors)

            # N = fc / f_env (转置音) 或 N = 1 (纯音)
            if fc == 0:
                N_rep = 1.0
            else:
                N_rep = fc / f0_rep

            N_list.append(N_rep)
            F0_list.append(f0_rep)
            F0DL_list.append(f0dl)

    return {
        'N': np.array(N_list),
        'F0': np.array(F0_list),
        'color_val': np.array(F0DL_list),
        'experiment': 'E',
    }


print("[data_loader] Module loaded.")
print("  build_F0N_data_expA/B/C/D/E() — load raw data from HDF5 + predictions")
print("  expA/B/C/D/E_to_F0N_heatmap_data() — convert to F0-N heatmap format")
