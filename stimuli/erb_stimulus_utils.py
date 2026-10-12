#!/usr/bin/env python3
"""
ERB 刺激生成共享工具 — 从 Josh 项目复制的核心函数.

复制来源:
  - /data/zyf/pitchnet/assets_datasets/stimuli_generate_BernsteinOxenhamFixedFilter.py
  - /data/zyf/pitchnet/packages/msutil/util_stimuli.py
  - /data/zyf/pitchnet/assets_datasets/dataset_util.py

该文件为策略 B1–B4 的刺激生成脚本提供共享基础函数,
避免修改原 pitchnet 项目.
"""

import os
import sys
import h5py
import numpy as np
import scipy.signal
import warnings

# ============================================================
#  滤波器函数 (来自 stimuli_generate_BernsteinOxenhamFixedFilter.py)
# ============================================================

def get_bandpass_filter_frequency_response(fl, fh, fs=32000, order=4,
                                           rescale_max=True):
    """返回以 [fl, fh] 为 −3dB 截频的带通滤波器频率响应函数 (dB).

    Josh 代码里存在两个版本（逐函数核对发现，2026-08-13）:
      - BernsteinOxenhamFixedFilter 版（Exp A/B1/B6）: 重缩放使双滤波
        操作的最大响应恰为 0 dB（rescale_max=True, 默认）
      - AltPhaseHarmonics 版（Exp B/B2）: 无重缩放，直接返回两滤波器
        乘积响应（rescale_max=False）——B2 跟代码（与 Josh Exp B
        实际行为一致），不跟论文

    Args:
        fl: 高通 −3dB 截频 (Hz)
        fh: 低通 −3dB 截频 (Hz)
        fs: 采样率 (Hz)
        order: Butterworth 滤波器阶数
        rescale_max: True=Bernstein 版（0dB 最大响应重缩放）；
            False=AltPhase 版（无重缩放）
    Returns:
        frequency_response_in_dB(freqs) — 可调用函数
    """
    assert fl <= fh
    Bl, Al = scipy.signal.butter(order, fh / (fs / 2), btype='low')
    Bh, Ah = scipy.signal.butter(order, fl / (fs / 2), btype='high')
    if rescale_max:
        # Bernstein 版: 计算双滤波操作的频率响应并重缩放
        w, Hl = scipy.signal.freqz(Bl, a=Al, worN=int(fs), whole=False)
        w, Hh = scipy.signal.freqz(Bh, a=Ah, worN=int(fs), whole=False)
        H = Hl * Hh
        Bl = Bl / np.max(np.abs(H))  # 重缩放使最大响应为 0 dB

    def frequency_response_in_dB(freqs):
        _, Hh_r = scipy.signal.freqz(Bh, a=Ah, worN=2*np.pi*freqs/fs, whole=False)
        _, Hl_r = scipy.signal.freqz(Bl, a=Al, worN=2*np.pi*freqs/fs, whole=False)
        return 20.0 * np.log10(np.abs(Hl_r * Hh_r))
    return frequency_response_in_dB


def shift_bandpass_filter_frequency_response(
    desired_fl, desired_fl_gain_in_dB, fs=32000,
    unshifted_passband=None, frequency_response_in_dB=None
):
    """将滤波器频率响应平移到新位置.

    Args:
        desired_fl: 想要的低频截频 (Hz)
        desired_fl_gain_in_dB: 在 desired_fl 处的增益 (dB)
        frequency_response_in_dB: 未平移滤波器的频率响应函数
    Returns:
        shifted_frequency_response_in_dB(freqs)
    """
    assert frequency_response_in_dB is not None

    if unshifted_passband is None:
        freq_vector = np.arange(0, fs/2)
        gain_vector = frequency_response_in_dB(freq_vector)
        passed = freq_vector[gain_vector > desired_fl_gain_in_dB]
        unshifted_passband = [np.min(passed), np.max(passed)]

    fl_offset = desired_fl - unshifted_passband[0]

    def shifted_frequency_response_in_dB(freqs):
        shifted = freqs - fl_offset
        shifted[shifted < 0] = 0
        shifted[shifted > fs/2] = fs/2
        return frequency_response_in_dB(shifted)
    return shifted_frequency_response_in_dB


# ============================================================
#  信号生成 (来自 packages/msutil/util_stimuli.py)
# ============================================================

def rms(x):
    """均方根振幅."""
    out = np.sqrt(np.mean(np.square(x)))
    if np.isnan(out):
        raise ValueError('rms calculation resulted in NaN')
    return out


def set_dBSPL(x, dBSPL, mean_subtract=True):
    """将波形缩放到指定 dB SPL."""
    if mean_subtract:
        x = x - np.mean(x)
    rms_out = 20e-6 * np.power(10, dBSPL / 20)
    return rms_out * x / rms(x)


def complex_tone(f0, fs, dur, harmonic_numbers=None, frequencies=None,
                 amplitudes=None, phase_mode='sine',
                 offset_start=True, strict_nyquist=True):
    """生成谐波复合音.

    支持相位模式: 'sine', 'rand', 'sch', 'cos', 'alt'.
    """
    t = np.arange(0, dur, 1/fs)[0:int(dur*fs)]
    if offset_start:
        t = t + (1/f0) * np.random.rand()

    if frequencies is None:
        assert harmonic_numbers is not None
        harmonic_numbers = np.array(harmonic_numbers).reshape([-1])
        frequencies = harmonic_numbers * f0
    else:
        assert harmonic_numbers is None
        frequencies = np.array(frequencies).reshape([-1])

    if amplitudes is None:
        amplitudes = 1/len(frequencies) * np.ones_like(frequencies)
    else:
        # Josh 原版断言（防 zip 静默截断——失谐/包络类刺激的振幅错误会无声通过）
        assert_msg = "provided `amplitudes` must be same length as `frequencies`"
        assert len(amplitudes) == len(frequencies), assert_msg

    # 相位模式
    phase_mode_lower = phase_mode.lower()
    if phase_mode_lower == 'sine':
        phase_list = np.zeros(len(frequencies))
    elif phase_mode_lower in ('rand', 'random'):
        phase_list = 2 * np.pi * np.random.rand(len(frequencies))
    elif phase_mode_lower in ('sch', 'schroeder'):
        phase_list = np.pi/2 + (np.pi * np.square(frequencies) / len(frequencies))
    elif phase_mode_lower in ('cos', 'cosine'):
        phase_list = np.pi/2 * np.ones(len(frequencies))
    elif phase_mode_lower in ('alt', 'alternating'):
        phase_list = np.pi/2 * np.ones(len(frequencies))
        phase_list[::2] = 0
    else:
        raise ValueError(f'Unsupported phase_mode: {phase_mode}')

    signal = np.zeros_like(t)
    for f, amp, phase in zip(frequencies, amplitudes, phase_list):
        if f > fs/2:
            if strict_nyquist:
                raise ValueError('Nyquist frequency exceeded')
            else:
                break
        component = amp * np.sqrt(2) * np.sin(2*np.pi*f*t + phase)
        signal += component
    return signal


def modified_uniform_masking_noise(fs, dur, dBHzSPL=15.0,
                                   attenuation_start=600.0,
                                   attenuation_slope=2.0):
    """生成修正均匀掩蔽噪声 (Bernstein & Oxenham 2005)."""
    # 平坦谱噪声
    fxx = np.ones(int(dur*fs), dtype=np.complex128)
    freqs = np.fft.fftfreq(len(fxx), d=1/fs)
    pos_idx = np.argwhere(freqs > 0).reshape([-1])
    neg_idx = np.argwhere(freqs < 0).reshape([-1])
    if neg_idx.shape[0] > pos_idx.shape[0]:
        neg_idx = neg_idx[1:]
    phases = np.random.uniform(0, 2*np.pi, size=pos_idx.shape)
    phases = np.cos(phases) + 1j * np.sin(phases)
    fxx[pos_idx] = fxx[pos_idx] * phases
    fxx[neg_idx] = fxx[neg_idx] * np.flip(phases, axis=0)
    x = np.real(np.fft.ifft(fxx))

    PSD = np.power(10, (dBHzSPL/10)) * np.square(20e-6)
    A_rms = np.sqrt(PSD * fs/2)
    x = A_rms * x / rms(x)

    # 高频衰减
    fxx = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1/fs)
    dB_attenuation = np.zeros_like(freqs)
    nzidx = np.abs(freqs) > 0
    dB_attenuation[nzidx] = -attenuation_slope * np.log2(
        np.abs(freqs[nzidx]) / attenuation_start)
    dB_attenuation[dB_attenuation > 0] = 0
    amplitudes = np.power(10, (dB_attenuation/20))
    fxx = fxx * amplitudes
    return np.real(np.fft.ifft(fxx))


# ============================================================
#  HDF5 I/O (来自 assets_datasets/dataset_util.py)
# ============================================================

def initialize_hdf5_file(hdf5_filename, N, data_dict,
                         data_key_pair_list=None,
                         config_key_pair_list=None,
                         file_mode='w', fillvalue=-1):
    """创建 HDF5 文件并写入配置."""
    if data_key_pair_list is None:
        data_key_pair_list = []
    if config_key_pair_list is None:
        config_key_pair_list = []

    f = h5py.File(hdf5_filename, file_mode)

    for hdf5_key, data_key in data_key_pair_list:
        val = np.squeeze(np.array(data_dict[data_key]))
        shape = [N] + list(val.shape)
        f.create_dataset(hdf5_key, shape, dtype=val.dtype, fillvalue=fillvalue)

    for hdf5_key, config_key in config_key_pair_list:
        val = data_dict[config_key]
        if isinstance(val, str):
            f.create_dataset(hdf5_key, [1], dtype=h5py.special_dtype(vlen=str),
                           data=val)
        else:
            val = np.squeeze(np.array(val))
            f.create_dataset(hdf5_key, [1] + list(val.shape),
                           dtype=val.dtype, data=val)
    f.close()


def write_example_to_hdf5(hdf5_f, data_dict, idx, data_key_pair_list=None):
    """向打开的 HDF5 文件写入单个样本."""
    if data_key_pair_list is None:
        data_key_pair_list = []
    for hdf5_key, data_key in data_key_pair_list:
        try:
            hdf5_f[hdf5_key][idx] = np.squeeze(np.array(data_dict[data_key]))
        except Exception:
            warnings.warn(f"Failed to write key {hdf5_key} at index {idx}")


# ============================================================
#  F0 bins (来自 assets_datasets/dataset_util.py)
# ============================================================

def get_f0_bins(f0_min=80.0, f0_max=1000.0, binwidth_in_octaves=1/192):
    """生成对数间距 F0 bin."""
    max_octave = np.log2(f0_max / f0_min)
    bins = np.arange(0, max_octave, binwidth_in_octaves)
    return f0_min * np.power(2.0, bins)


def f0_to_label(f0_values, bins, right=False):
    """F0 (Hz) → 标签."""
    return np.digitize(f0_values, bins, right=right) - 1


def label_to_f0(f0_labels, bins):
    """标签 → F0 (Hz, bin 下界)."""
    return bins[np.array(f0_labels).astype(int)]


# ============================================================
#  Exp C 谱包络 (来自 stimuli_generate_FrequencyShiftedComplexes.py)
# ============================================================

def get_MooreMoore2003_spectral_envelope(f0, f_center, f_bandwidth):
    """返回 Moore & Moore (2003) 谱包络函数."""
    f_high_edge = f_center + f_bandwidth / 2
    f_low_edge = f_center - f_bandwidth / 2

    def sloping_region(f, f_edge, f0):
        x = 1.0 - np.abs((f - f_edge) / (1.5 * f0))
        x[x < 0] = 0
        return (np.power(10.0, x) - 1.0) / 9.0

    def spectral_envelope(f):
        envelope = np.ones_like(f, dtype=np.float32)
        envelope[f <= f_low_edge] = sloping_region(
            f[f <= f_low_edge], f_low_edge, f0)
        envelope[f >= f_high_edge] = sloping_region(
            f[f >= f_high_edge], f_high_edge, f0)
        envelope[envelope < 0] = 0
        return envelope
    return spectral_envelope


def get_MooreMoore2003_complex_tone(f0, f0_shift=0.0, spectral_envelope=None,
                                     fs=32000, dur=0.150, dBSPL=70.0,
                                     phase_mode='cos'):
    """生成频率偏移复合音."""
    assert spectral_envelope is not None
    frequencies = np.arange(f0, fs/2, f0, dtype=np.float32)
    frequencies = frequencies + f0 * f0_shift
    amplitudes = spectral_envelope(frequencies)
    y = complex_tone(f0, fs, dur, harmonic_numbers=None,
                     frequencies=frequencies, amplitudes=amplitudes,
                     phase_mode=phase_mode, offset_start=True,
                     strict_nyquist=False)
    y = set_dBSPL(y, dBSPL)
    return y


# ============================================================
#  Bernstein & Oxenham (2005) 滤波复合音
#  (来自 stimuli_generate_BernsteinOxenhamFixedFilter.py)
# ============================================================

def bernox2005_bandpass_complex_tone(f0,
                                     fs,
                                     dur,
                                     frequency_response_in_dB=None,
                                     threshold_dBSPL=33.3,
                                     component_dBSL=15.0,
                                     strict_low_harm=None,
                                     strict_audible_harm=False,
                                     **kwargs_complex_tone):
    """生成带通滤波复合音，谐波电平由 Bernstein & Oxenham (2005, JASA) 决定。

    Args:
        f0: 基频 (Hz)
        fs: 采样率 (Hz)
        dur: 时长 (s)
        frequency_response_in_dB: 见 get_bandpass_filter_frequency_response()
        threshold_dBSPL: 听阈 (dB re 20e-6 Pa)
        component_dBSL: 高于听阈的感觉级 (dB)
        strict_low_harm: 若指定，仅保留序号 >= strict_low_harm 的谐波
        strict_audible_harm: 若 True，仅保留电平 >= 听阈的谐波
        **kwargs_complex_tone: 直接传给 complex_tone()

    Returns:
        signal (np array): 波形 (Pa)
        audible_harmonic_numbers (np array): 电平 >= 听阈的谐波序号
    """
    assert frequency_response_in_dB is not None
    harmonic_freqs = np.arange(f0, fs/2, f0)
    harmonic_numbers = harmonic_freqs / f0
    if strict_low_harm is not None:
        harmonic_freqs = harmonic_freqs[harmonic_numbers >= strict_low_harm]
        harmonic_numbers = harmonic_numbers[harmonic_numbers >= strict_low_harm]
    harmonic_dBSPL = threshold_dBSPL + component_dBSL + frequency_response_in_dB(harmonic_freqs)
    if strict_audible_harm:
        harmonic_numbers = harmonic_numbers[harmonic_dBSPL >= threshold_dBSPL]
        harmonic_dBSPL = harmonic_dBSPL[harmonic_dBSPL >= threshold_dBSPL]
    amplitudes = 20e-6 * np.power(10, (harmonic_dBSPL/20))
    signal = complex_tone(f0, fs, dur, harmonic_numbers=harmonic_numbers,
                          amplitudes=amplitudes, **kwargs_complex_tone)
    audible_harmonic_numbers = harmonic_numbers[harmonic_dBSPL >= threshold_dBSPL]
    return signal, audible_harmonic_numbers


# ============================================================
#  Oxenham et al. (2004, PNAS) 移调音
#  (来自 stimuli_generate_TransposedTones.py)
# ============================================================

def get_Oxenham2004_transposed_tone(f_carrier,
                                    f_envelope,
                                    fs=32000,
                                    dur=0.150,
                                    buffer_dur=1.0,
                                    dBSPL=70.0,
                                    offset_start=True,
                                    lowpass_filter_envelope=True):
    """生成指定载波与包络频率的移调音（Oxenham et al. 2004, PNAS）。
    f_carrier=0 时返回频率为 f_envelope 的纯音。

    合成流程（Josh 原样）: 正弦包络 → 半波整流 → 0.2×f_carrier 低通
    (4 阶 Butterworth, filtfilt) → 与载波正弦相乘 → buffer 各半秒裁掉
    滤波边缘效应 → set_dBSPL。
    """
    t = np.arange(0, dur + buffer_dur, 1 / fs)
    if offset_start:
        t = t + np.random.rand() * dur
    envelope = np.sin(2 * np.pi * f_envelope * t)

    if f_carrier > 0:
        envelope[envelope < 0] = 0  # 半波整流
        if lowpass_filter_envelope:
            N = int(4 / 2)  # filtfilt 施加 4 阶低通（butter N=2 × filtfilt 双向 = 4 阶）
            Wn = 0.2 * f_carrier / (fs / 2)  # 截止频率 = 0.2 × f_carrier
            b, a = scipy.signal.butter(N, Wn, btype='low')
            envelope = scipy.signal.filtfilt(b, a, envelope)
        carrier = np.sin(2 * np.pi * f_carrier * t)
        signal = carrier * envelope
    else:
        # f_carrier = 0 → 纯音（Josh 原版的 print 已去除——11k 刺激会刷屏）
        signal = envelope
    start_index = int(buffer_dur/2 * fs)
    end_index = start_index + int(dur * fs)
    signal = signal[start_index:end_index]
    signal = set_dBSPL(signal, dBSPL)
    return signal
