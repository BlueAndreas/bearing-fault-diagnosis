"""第一课：读取真实 CWRU 105 号记录，画波形、频谱与窗口切片。

运行：python -X utf8 scripts/lesson01_read_signal.py
尝试：python -X utf8 scripts/lesson01_read_signal.py --window-points 2048
原始 MAT 文件只读。采样率来自官方 12k 驱动端下载表，不从文件名猜测。
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.stats import kurtosis
import matplotlib
matplotlib.use('Agg')  # 保存图片，不依赖图形窗口。
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-points', type=int, default=1024)
    args = parser.parse_args()

    # 1. 找到你下载的文件；同时兼容官方数字文件名与现在的文件名。
    data_dir = ROOT / 'data/raw/CWRU/12k_DE'
    candidates = [data_dir / 'IR007_0.mat', data_dir / '105.mat']
    mat_path = next((p for p in candidates if p.is_file()), None)
    if mat_path is None:
        raise FileNotFoundError(f'请把 IR007_0.mat 或 105.mat 放到：{data_dir}')
    data = loadmat(mat_path)
    if 'X105_DE_time' not in data:
        raise ValueError('没有找到 X105_DE_time；请确认下载的是 105 号 IR007_0 记录。')

    # 2. MAT 是变量集合；选择驱动端，再把单列矩阵转成一维数组。
    x = np.asarray(data['X105_DE_time'], dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all() or len(x) < 12000:
        raise ValueError('数据存在非有限值，或长度不足一秒。')
    if args.window_points < 2 or 2 * args.window_points > len(x):
        raise ValueError('窗口长度需至少为 2，且两个完整窗口不能超过信号长度。')
    fs = 12000  # 官方 12k 驱动端数据；此设置不能直接套用正常基线记录。
    rpm = int(np.asarray(data['X105RPM']).item())
    t = np.arange(len(x)) / fs
    out = ROOT / 'outputs/lesson01'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 12})

    # 3. 先看全貌，再看前 0.1 秒。横轴使用秒，不是采样点编号。
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    ax[0].plot(t, x, linewidth=.45, color='#337EBB')
    ax[0].set_title(f'真实数据：IR007_0，驱动端，{rpm} rpm（全记录）')
    n = 1200
    ax[1].plot(t[:n], x[:n], linewidth=.9, color='#337EBB')
    ax[1].set_title('放大前 0.1 秒：1200 个采样点')
    for a in ax:
        a.set_xlabel('时间（秒）')
        a.set_ylabel('信号幅值（原始单位）')
        a.grid(alpha=.2)
    fig.savefig(out / '01-time-waveform.png', dpi=150)
    plt.close(fig)

    # 4. 频谱用前一秒。减均值去掉直流，加 Hann 窗减轻截断泄漏。
    # 单边幅度谱按窗的总和缩放；直流与 Nyquist 项不乘 2。
    segment = x[:fs]
    w = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft((segment-segment.mean()) * w)) / w.sum()
    spectrum[1:-1] *= 2
    frequencies = np.fft.rfftfreq(len(segment), d=1/fs)
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.plot(frequencies, spectrum, linewidth=.9, color='#248D77')
    ax.set(xlabel='频率（Hz）', ylabel='单边幅度（原始单位）', xlim=(0, fs/2),
           title='前 1 秒的频谱：横轴 0～6000 Hz；频率间隔 1 Hz')
    ax.grid(alpha=.2)
    fig.savefig(out / '02-frequency-spectrum.png', dpi=150)
    plt.close(fig)

    # 5. 演示两个不重叠窗口；这里只切片，不建立训练/测试集。
    win = args.window_points
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    for i, a in enumerate(ax):
        start, end = i*win, (i+1)*win
        a.plot(np.arange(start, end), x[start:end], color=['#337EBB', '#D58C27'][i])
        a.set(title=f'窗口 {i+1}：索引 {start}～{end-1}，{win} 点，覆盖 {win/fs:.4f} 秒',
              xlabel='原始采样点索引（从 0 开始）', ylabel='信号幅值（原始单位）')
        a.grid(alpha=.2)
    fig.savefig(out / '03-window-slicing.png', dpi=150)
    plt.close(fig)

    variables = [{'name': k, 'shape': list(np.asarray(v).shape),
                  'dtype': str(np.asarray(v).dtype)}
                 for k, v in data.items() if not k.startswith('__')]
    metrics = {'rms': float(np.sqrt(np.mean(x*x))),
               'absolute_peak': float(np.max(np.abs(x))),
               'pearson_kurtosis': float(kurtosis(x, fisher=False, bias=False))}
    summary = {'file': str(mat_path), 'sha256': hashlib.sha256(mat_path.read_bytes()).hexdigest(),
               'source': SOURCE, 'channel': 'X105_DE_time', 'samples': len(x),
               'sampling_rate_hz': fs, 'sampling_rate_source': 'CWRU 官方 12k 驱动端下载表',
               'duration_seconds': len(x)/fs, 'recorded_rpm': rpm,
               'label_from_download_table': 'inner race fault, 0.007 inch, 0 HP',
               'variables': variables, 'first_10_values': x[:10].tolist(), 'metrics': metrics,
               'window_points': win, 'window_seconds': win/fs,
               'full_nonoverlapping_windows': len(x)//win, 'remaining_points': len(x)%win,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__}}
    (out / 'data-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = '\n'.join(f"| `{v['name']}` | `{v['shape']}` | `{v['dtype']}` |" for v in variables)
    images = '\n\n'.join(f'![{title}]({(out/name).as_posix()})' for name, title in [
        ('01-time-waveform.png', '真实振动波形'), ('02-frequency-spectrum.png', '真实信号频谱'),
        ('03-window-slicing.png', '两个不重叠窗口')])
    report = f'''# 第一课：真实数据读取结果

这个报告由读取脚本生成。故障标签来自官方下载表，并非模型预测；本次没有训练分类模型。

## 文件与变量

- 文件：`{mat_path.name}`，记录号 105。
- 驱动端信号：`X105_DE_time`，共 {len(x):,} 点。
- 采样率：12000 Hz，来自[官方 12k 驱动端表]({SOURCE})。
- 记录覆盖时长：{len(x)/fs:.6f} 秒；最后一个采样点位于 {(len(x)-1)/fs:.6f} 秒。
- MAT 实际记录的转速：{rpm} rpm。

| 变量 | 原始形状 | 数据类型 |
| --- | --- | --- |
{rows}

`[121265, 1]` 表示一列 121265 个数；`time` 变量在这里存的是振动数值，不是时间戳。时间轴由采样率另行计算。

前 10 个驱动端数值：`{', '.join(f'{v:.6f}' for v in x[:10])}`。

## 全记录的三个统计量

| 指标 | 实际结果 | 含义 |
| --- | --- | --- |
| RMS | {metrics['rms']:.6f} | 均方根，用于描述信号幅值水平 |
| 绝对峰值 | {metrics['absolute_peak']:.6f} | 绝对值最大的采样点 |
| Pearson 峭度 | {metrics['pearson_kurtosis']:.6f} | 描述分布尾部/尖峰特征；这里不是减 3 后的超额峭度 |

这些指标目前只用于认识一条信号，不能据此得出故障分类准确率。

## 三张图怎么读

1. 波形：先看全记录，再看前 0.1 秒。横轴是秒，纵轴是采样值，观察幅值和短时变化。
2. 频谱：取前 1 秒，减均值、加 Hann 窗并做单边 FFT。横轴是 Hz；最强峰不一定就是内圈故障特征频率。
3. 切片：窗口长度 {win} 点，步长同样为 {win} 点。每段覆盖 {win/fs:.6f} 秒；整条信号能得到 {len(x)//win} 个完整窗口，末尾剩余 {len(x)%win} 点。这里的窗口仍来自同一次记录。

{images}

## 你亲手完成的任务

- 自己再运行一次脚本，确认能重新生成上述结果。
- 把窗口长度从 1024 改成 2048 再运行，观察切片时长与完整窗口数量如何变化。
- 在学习笔记中写清 `DE`、`FE`、`BA`、`RPM` 各是什么，以及为何采样点数不等于秒数。

配置与库版本、文件 SHA-256 和完整数值保存在同目录 `data-summary.json`。
'''
    (out / '第一课-真实数据结果.md').write_text(report, encoding='utf-8')
    print('读取成功：', mat_path.name)
    print(f'驱动端：{len(x):,} 点；12000 Hz；约 {len(x)/fs:.4f} 秒；{rpm} rpm')
    for v in variables:
        print(v['name'], '形状=', v['shape'], '类型=', v['dtype'])
    print(f'窗口：{win} 点 = {win/fs:.6f} 秒；完整窗口 {len(x)//win} 个；剩余 {len(x)%win} 点')
    print('图片和报告已保存到：', out)


if __name__ == '__main__':
    main()
