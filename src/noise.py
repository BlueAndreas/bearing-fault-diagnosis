"""按逐窗口参考功率定标的人工噪声；不修改输入，不涉及标签或模型训练。"""
import numpy as np


def validate_windows(windows):
    raw = np.asarray(windows, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 4 or not np.isfinite(raw).all():
        raise ValueError('需要有限的二维振动窗口，每行至少四个点。')
    power = np.mean(raw * raw, axis=1)
    if np.any(power <= 0) or not np.isfinite(power).all():
        raise ValueError('参考窗口功率必须为正且有限，零信号不能按此方法定义 SNR。')
    return raw, power


def make_unit_noise(windows, seed):
    raw, _ = validate_windows(windows)
    gaussian = np.random.default_rng(seed).standard_normal(raw.shape)
    rms = np.sqrt(np.mean(gaussian * gaussian, axis=1, keepdims=True))
    return gaussian / rms


def add_noise_at_snr(windows, unit_noise, snr_db):
    """同一 seed 的噪声方向可复用于不同强度；返回实际噪声与逐窗实测 SNR。"""
    raw, signal_power = validate_windows(windows)
    unit = np.asarray(unit_noise, dtype=np.float64)
    if unit.shape != raw.shape or not np.isfinite(unit).all():
        raise ValueError('噪声模板形状或数值无效。')
    if not np.allclose(np.mean(unit * unit, axis=1), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError('噪声模板必须逐窗口具有单位 RMS。')
    if not np.isscalar(snr_db) or not np.isfinite(snr_db):
        raise ValueError('SNR 必须为有限标量。')
    factor = np.power(10.0, -float(snr_db) / 20.0)
    requested_noise = unit * (np.sqrt(signal_power) * factor)[:, None]
    noisy = raw + requested_noise
    actual_noise = noisy - raw
    noise_power = np.mean(actual_noise * actual_noise, axis=1)
    if not np.isfinite(noisy).all() or not np.isfinite(noise_power).all() or np.any(noise_power <= 0):
        raise ValueError('该噪声强度导致无效或无法数值表示的噪声。')
    measured_snr = 10.0 * np.log10(signal_power / noise_power)
    if not np.allclose(measured_snr, snr_db, rtol=0, atol=1e-8):
        raise AssertionError('逐窗口实测 SNR 未达到目标。')
    return noisy, {'noise': actual_noise, 'signal_power': signal_power,
                   'noise_power': noise_power, 'measured_snr_db': measured_snr}
