"""第二课：在统一采样率、测点和时长后，比较四种已知轴承状态。

运行：python -X utf8 scripts/lesson02_compare_states.py
练习：python -X utf8 scripts/lesson02_compare_states.py --start-second 1.0
仅生成观察图和描述统计，不训练分类器，不修改原始数据。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.stats import kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FS = 12000
SECONDS = 4.0
FAULT_SOURCE = 'https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data'
NORMAL_SOURCE = 'https://engineering.case.edu/bearingdatacenter/normal-baseline-data'
RATE_SOURCE = 'https://www.researchgate.net/publication/276248554_Rolling_Element_Bearing_Diagnostics_Using_the_Case_Western_Reserve_University_Data_A_Benchmark_Study'
SPECS = [
    {'label': 0, 'name': '正常', 'id': 97, 'files': ['Normal Baseline Data/1797 rpm.mat', 'Normal Baseline Data/97.mat'], 'fs': 48000, 'color': '#337EBB'},
    {'label': 1, 'name': '内圈故障', 'id': 105, 'files': ['12k_DE/IR007_0.mat', '12k_DE/105.mat'], 'fs': 12000, 'color': '#248D77'},
    {'label': 2, 'name': '外圈故障', 'id': 130, 'files': ['12k_DE/OR007@6_0.mat', '12k_DE/130.mat'], 'fs': 12000, 'color': '#D34747'},
    {'label': 3, 'name': '滚动体故障', 'id': 118, 'files': ['12k_DE/B007_0.mat', '12k_DE/118.mat'], 'fs': 12000, 'color': '#BE841A'},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-second', type=float, default=.5)
    args = parser.parse_args()
    if not np.isfinite(args.start_second) or args.start_second < .1:
        raise ValueError('起点需为至少 0.1 秒的有限数值，以避开重采样边界。')
    start = round(args.start_second * FS)
    actual_start = start / FS
    count = int(SECONDS * FS)
    # 不同练习分开保存，方便比较，不覆盖默认 0.5 秒示例。
    out = ROOT / 'outputs/lesson02'
    if actual_start != .5:
        out = out / ('start-' + f'{actual_start:g}'.replace('.', 'p') + 's')
    records = []
    signals = []
    for spec in SPECS:
        options = [ROOT / 'data/raw/CWRU' / f for f in spec['files']]
        path = next((p for p in options if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"缺少 {spec['name']} 文件，请按第二课文件表准备；首次补数据可运行 scripts/prepare_lesson02_data.py。")
        mat = loadmat(path)
        key = f"X{spec['id']:03d}_DE_time"
        rpm_key = f"X{spec['id']:03d}RPM"
        if key not in mat or rpm_key not in mat:
            raise ValueError(f'{path.name} 缺少预期通道/转速变量；不能只凭文件名判断记录身份。')
        raw = np.asarray(mat[key], dtype=np.float64).reshape(-1)
        if not np.isfinite(raw).all():
            raise ValueError(f'{path.name} 包含非有限值。')
        # 正常基线是 48 kHz：先低通抗混叠，再降到 12 kHz。
        # 禁止只改 fs=12000；也不要用 raw[::4] 代替抗混叠滤波。
        aligned = resample_poly(raw, up=1, down=4) if spec['fs'] == 48000 else raw
        if start+count > len(aligned):
            raise ValueError(f'{path.name} 时长不足：无法从 {actual_start} 秒起取 4 秒。请使用 0.5 或 1.0 秒起点。')
        segment = aligned[start:start+count]
        assert len(segment) == count
        signals.append(segment)
        record = {'label': spec['label'], 'state': spec['name'], 'record_id': spec['id'],
                  'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'channel': key, 'original_sampling_rate_hz': spec['fs'], 'original_points': len(raw),
                  'original_duration_seconds': len(raw)/spec['fs'], 'aligned_sampling_rate_hz': FS,
                  'aligned_points': len(aligned), 'selected_start_second': actual_start,
                  'selected_duration_seconds': SECONDS, 'selected_points': count,
                  'recorded_rpm': int(np.asarray(mat[rpm_key]).item()), 'motor_load_hp': 0,
                  'fault_diameter_inch': None if spec['label']==0 else .007,
                  'outer_race_position': '6 oclock' if spec['label']==2 else None,
                  'source': NORMAL_SOURCE if spec['label']==0 else FAULT_SOURCE,
                  'sampling_rate_source': 'Smith & Randall (2015), Table A1' if spec['label']==0 else 'CWRU 12k Drive End table',
                  'sampling_rate_reference': RATE_SOURCE if spec['label']==0 else FAULT_SOURCE,
                  'resampling': 'scipy.signal.resample_poly(up=1,down=4), default Kaiser FIR' if spec['fs']==48000 else 'none',
                  'rms': float(np.sqrt(np.mean(segment**2))),
                  'absolute_peak': float(np.max(np.abs(segment))),
                  'pearson_kurtosis': float(kurtosis(segment, fisher=False, bias=False))}
        records.append(record)

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 12})
    # 图 1：四段同长波形，使用相同纵轴范围，保留原始幅值差异。
    fig, axes = plt.subplots(4,1,figsize=(12,10),sharex=True,sharey=True,constrained_layout=True)
    preview = int(.1*FS)
    times = actual_start + np.arange(preview)/FS
    bound = max(np.max(np.abs(x[:preview])) for x in signals)*1.08
    for spec,x,ax in zip(SPECS,signals,axes):
        ax.plot(times,x[:preview],color=spec['color'],linewidth=.9)
        ax.set(title=spec['name'],ylabel='信号幅值',ylim=(-bound,bound))
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('各自记录中的时间（秒）；这些记录不是同时采集的')
    fig.suptitle('四种状态的 0.1 秒波形｜同为 12 kHz，纵轴范围相同',fontsize=16)
    fig.savefig(out/'01-four-state-waveforms.png',dpi=150)
    plt.close(fig)

    # 图 2：每种状态取同一时长的一秒数据，使用相同 FFT 设置。
    spectra=[]
    freqs=np.fft.rfftfreq(FS,d=1/FS)
    w=np.hanning(FS)
    for x in signals:
        z=x[:FS]
        amplitude=np.abs(np.fft.rfft((z-z.mean())*w))/w.sum()
        amplitude[1:-1]*=2
        spectra.append(amplitude)
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,sharey=True,constrained_layout=True)
    maximum=max(a.max() for a in spectra)*1.5
    for spec,amplitude,ax in zip(SPECS,spectra,axes):
        ax.semilogy(freqs,np.maximum(amplitude,1e-12),color=spec['color'],linewidth=.8)
        ax.set(title=spec['name'],ylabel='单边幅度\n对数刻度',xlim=(0,FS/2),ylim=(1e-6,maximum))
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('频率（Hz）')
    fig.suptitle(f'各记录 {actual_start:g}～{actual_start+1:g} 秒的频谱｜相同坐标范围；最强峰不等于故障特征频率',fontsize=15)
    fig.savefig(out/'02-four-state-spectra.png',dpi=150)
    plt.close(fig)

    # 图 3：同样 4 秒、同样 48000 点的三个描述统计量。
    fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
    names=[s['name'] for s in SPECS]
    for ax,key,title in zip(axes,['rms','absolute_peak','pearson_kurtosis'],['RMS（均方根）','绝对峰值','Pearson 峭度（无量纲）']):
        values=[r[key] for r in records]
        bars=ax.bar(names,values,color=[s['color'] for s in SPECS],width=.65)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=4,fontsize=11)
        ax.set_title(title)
        ax.set_ylim(0,max(values)*1.2)
        ax.tick_params(axis='x',rotation=15)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'相同 4 秒片段（{actual_start:g}～{actual_start+4:g} 秒）的统计量｜只描述当前片段',fontsize=15)
    fig.savefig(out/'03-statistics-comparison.png',dpi=150)
    plt.close(fig)

    summary={'purpose':'descriptive comparison only; no classifier or train/test split',
             'sampling_rate_hz':FS,'start_second':actual_start,'duration_seconds':SECONDS,
             'fft_seconds':1.0,'waveform_seconds':.1,'normal_rate_reference':RATE_SOURCE,
             'records':records,'versions':{'python':platform.python_version(),'numpy':np.__version__,
                                          'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}}
    (out/'comparison-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'state-statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['label','state','record_id','channel','original_sampling_rate_hz','recorded_rpm',
                'selected_start_second','selected_duration_seconds','selected_points','rms','absolute_peak','pearson_kurtosis']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    table='\n'.join(f"| {r['label']} | {r['state']} | {r['record_id']} | {r['recorded_rpm']} | {r['rms']:.6f} | {r['absolute_peak']:.6f} | {r['pearson_kurtosis']:.6f} |" for r in records)
    file_table='\n'.join(f"| {r['state']} | `{Path(r['file']).name}` | `{r['channel']}` | {r['original_sampling_rate_hz']} | {r['original_points']:,} | {r['original_duration_seconds']:.4f} |" for r in records)
    rms_top=max(records,key=lambda r:r['rms'])['state']
    kurt_top=max(records,key=lambda r:r['pearson_kurtosis'])['state']
    images='\n\n'.join(f'![{title}]({(out/name).as_posix()})' for name,title in [
        ('01-four-state-waveforms.png','四种状态同尺度波形'),('02-four-state-spectra.png','四种状态频谱'),
        ('03-statistics-comparison.png','三种描述统计量对照')])
    report=f'''# 第二课：四种状态的真实数据对比

本报告由脚本生成。类别来自官方文件表，不是模型预测；每类只选一条记录，只作探索性观察。

## 1. 本次读了哪些数据

| 状态 | 本地文件 | 驱动端变量 | 原采样率 Hz | 原点数 | 原时长 秒 |
| --- | --- | --- | --- | --- | --- |
{file_table}

三份故障记录来自[官方 12k 驱动端表]({FAULT_SOURCE})，均为 0 HP、0.007 英寸局部缺陷。外圈选择 6 点钟方位；正常记录来自[官方正常基线]({NORMAL_SOURCE})。

正常记录 97 的原采样率 48 kHz 依据[Smith & Randall 基准研究（表 A1）]({RATE_SOURCE})，不是通过点数或文件名猜测。用 `resample_poly(raw, 1, 4)` 带低通滤波降为 12 kHz，然后统一取 {actual_start:g}～{actual_start+SECONDS:g} 秒，每类 {count:,} 点。原始文件保持不变。MAT 的转速实际为 1797 或 1796 rpm，官方表为近似 1797 rpm。

## 2. 实际统计结果

| 标签 | 状态 | 记录号 | 文件记录转速 rpm | RMS | 绝对峰值 | Pearson 峭度 |
| --- | --- | --- | --- | --- | --- | --- |
{table}

- 本次片段 RMS 最大的是 **{rms_top}**；Pearson 峭度最大的是 **{kurt_top}**。
- RMS 描述幅值水平，峭度描述分布尾部/尖峰特征；它们回答的问题不同。Pearson 峭度没有减 3。
- 排名只描述这一组片段，不代表故障严重程度排序；也不能拿它直接制定跨设备的故障阈值。

## 3. 依次看三张图

1. 波形采用同一纵轴，先观察相对幅值与短时变化。不同状态可能有差异，但本图不能证明任何新信号属于哪一类。
2. 频谱使用一秒片段、去均值、Hann 窗、单边 FFT。纵轴是对数刻度，便于看较小的成分；看横轴频带位置，并留意最强峰不一定是故障特征频率。
3. 统计图采用同长四秒片段。比较 RMS 与峭度排名，尝试解释为何可能不同。

{images}

## 4. 亲手做一个小实验

在项目 PowerShell 中运行 `python -X utf8 scripts/lesson02_compare_states.py --start-second 1.0`，会把 1～5 秒的结果单独保存到 `outputs/lesson02/start-1s`。比较两份统计表：数值是否变化？说明一个短片段的统计量不能代表全部工况。

我们尚未训练模型，尚未建立独立测试集，也没有故障识别准确率。下一课才整理数据、标签和划分方案。

详细配置、文件 SHA-256、重采样方法和库版本见 `comparison-summary.json`；可用 Excel 打开 `state-statistics.csv`。
'''
    (out/'第二课-四种状态对比结果.md').write_text(report,encoding='utf-8')
    for r in records:
        print(f"{r['state']}：记录 {r['record_id']}，原采样率 {r['original_sampling_rate_hz']} Hz → {FS} Hz，取 {count} 点；RMS={r['rms']:.6f}，峭度={r['pearson_kurtosis']:.6f}")
    print('四种状态比较完成；图片、统计表和报告：',out)


if __name__=='__main__':
    main()
