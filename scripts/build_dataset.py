"""数据构建阶段：按原始记录/负载分组，再切片，最后仅用训练数据拟合标准化参数。

运行：python -X utf8 scripts/build_dataset.py
指定窗口长度：python -X utf8 scripts/build_dataset.py --window-points 2048
仅生成数据集与检查报告，不训练模型；不修改 MAT 或已有输出。
"""
from pathlib import Path
from math import gcd
import argparse
import csv
import hashlib
import json
import platform

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import resample_poly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ['train','val','test']
TITLES = {'train':'训练集','val':'验证集','test':'测试集'}
COLORS = {'train':'#337EBB','val':'#DBA032','test':'#248D77'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-points',type=int)
    args = parser.parse_args()
    config_path = ROOT/'configs/dataset.json'
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    fs = cfg['sampling_rate_hz']
    win = args.window_points or cfg['window_points']
    count = round(cfg['segment_duration_second']*fs)
    start = round(cfg['segment_start_second']*fs)
    if args.window_points is not None and args.window_points < 2:
        raise ValueError('窗口长度必须至少为 2。')
    if win < 2 or win > count:
        raise ValueError('窗口长度需为 2～48000 点。')
    cfg['window_points'] = win
    cfg['stride_points'] = win  # 不重叠；本程序不使用重叠窗口。
    chunks = {s:[] for s in SPLITS}
    labels = {s:[] for s in SPLITS}
    origins = {s:[] for s in SPLITS}
    loads = {s:[] for s in SPLITS}
    records = []
    indices = []
    raw_folder = ROOT/'data/raw/CWRU'
    classes = cfg['classes']

    for cls in classes:
        for load in range(4):
            split = cfg['split_by_load'][str(load)]
            record_id = cls['base_record_id']+load
            nominal_rpm = cfg['nominal_rpm_by_load'][load]
            fmt = {'load':load,'record_id':record_id,'nominal_rpm':nominal_rpm}
            paths = [raw_folder/cls[key].format(**fmt) for key in ['file_pattern','fallback_pattern']]
            path = next((p for p in paths if p.is_file()),None)
            if path is None:
                raise FileNotFoundError(f'缺少记录 {record_id}；请先运行 scripts/download_data.py。')
            mat = loadmat(path)
            # 显式选择记录号。99.mat/1750 rpm.mat 中可能同时含 98 和 99，不能选第一个 DE。
            key = f'X{record_id:03d}_DE_time'
            if key not in mat:
                raise ValueError(f'{path.name} 不含指定通道 {key}。')
            raw = np.asarray(mat[key],dtype=np.float64).reshape(-1)
            if not np.isfinite(raw).all():
                raise ValueError(f'{path.name}:{key} 存在非有限值。')
            source_fs = cls['source_rate_hz']
            rate_gcd = gcd(source_fs,fs)
            up,down = fs//rate_gcd,source_fs//rate_gcd
            aligned = resample_poly(raw,up,down) if source_fs!=fs else raw
            if start+count > len(aligned):
                raise ValueError(f'记录 {record_id} 不足以截取指定的四秒。')
            segment = aligned[start:start+count]
            number = count//win
            # 先分组（split 已由 load 决定），再在此记录内部切片，不跨边界拼接。
            windows = segment[:number*win].reshape(number,win).astype(np.float32)
            chunks[split].append(windows)
            labels[split].append(np.full(number,cls['label'],dtype=np.int64))
            origins[split].append(np.full(number,record_id,dtype=np.int64))
            loads[split].append(np.full(number,load,dtype=np.int64))
            rpm_key = f'X{record_id:03d}RPM'
            rpm = int(np.asarray(mat[rpm_key]).item()) if rpm_key in mat else None
            present_keys = [k for k in mat if k.endswith('_DE_time')]
            records.append({'record_id':record_id,'label':cls['label'],'state':cls['name'],
                            'load_hp':load,'split':split,'file':str(path),'channel':key,
                            'fault_diameter_inch':None if cls['label']==0 else cfg['fault_diameter_inch'],
                            'outer_race_position':cfg['outer_race_position'] if cls['label']==2 else None,
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'original_sampling_rate_hz':source_fs,'original_points':len(raw),
                            'aligned_sampling_rate_hz':fs,'selected_points':count,
                            'selected_start_second':start/fs,'selected_duration_second':count/fs,
                            'up':up,'down':down,'windows':number,'discarded_tail_points':count%win,
                            'nominal_rpm_from_source_table':nominal_rpm,'recorded_rpm':rpm,
                            'other_DE_variables_in_file':[k for k in present_keys if k!=key]})
            offset = sum(len(a) for a in chunks[split][:-1])
            for i in range(number):
                lo = start+i*win
                indices.append({'split':split,'sample_index_in_split':offset+i,'label':cls['label'],
                                'state':cls['name'],'record_id':record_id,'load_hp':load,'file':str(path),
                                'channel':key,'aligned_start_index':lo,'aligned_end_index_exclusive':lo+win,
                                'start_second':lo/fs,'end_second_exclusive':(lo+win)/fs,
                                'window_points':win,'sampling_rate_hz':fs})

    arrays = {}
    for split in SPLITS:
        arrays[f'X_{split}_raw'] = np.concatenate(chunks[split])
        arrays[f'y_{split}'] = np.concatenate(labels[split])
        arrays[f'record_id_{split}'] = np.concatenate(origins[split])
        arrays[f'load_hp_{split}'] = np.concatenate(loads[split])
    # 只用实际保留的训练窗口，拟合一个共享均值/标准差，保留相对幅值信息。
    train = arrays['X_train_raw']
    mean = float(train.mean(dtype=np.float64))
    std = float(train.std(dtype=np.float64))
    if not np.isfinite(std) or std<=0:
        raise ValueError('训练标准差无效。')
    for split in SPLITS:
        arrays[f'X_{split}'] = ((arrays[f'X_{split}_raw'].astype(np.float64)-mean)/std).astype(np.float32)

    # 实际内容检查：组不交叉、窗口不交叉、三组不含完全相同的原始窗口。
    groups = {s:set(arrays[f'record_id_{s}'].tolist()) for s in SPLITS}
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if groups[a]&groups[b]:
                raise AssertionError('原始记录进入了两个不同集合。')
    owners = {}
    for row in indices:
        ident = row['record_id']
        previous = owners.get(ident)
        if previous and (previous['split']!=row['split'] or previous['aligned_end_index_exclusive']>row['aligned_start_index']):
            raise AssertionError('记录/窗口区间重叠。')
        owners[ident] = row
    hashes = {s:{hashlib.sha256(x.tobytes()).hexdigest() for x in arrays[f'X_{s}_raw']} for s in SPLITS}
    if any(hashes[a]&hashes[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]):
        raise AssertionError('不同集合存在完全相同的窗口。')
    if not np.isclose(arrays['X_train'].mean(dtype=np.float64),0,atol=1e-6) or not np.isclose(arrays['X_train'].std(dtype=np.float64),1,atol=1e-6):
        raise AssertionError('训练集标准化检查失败。')
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise AssertionError('生成数组存在非有限值。')

    processed = ROOT/'data/processed/dataset'
    out = ROOT/'outputs/dataset'
    if win!=1024:
        processed = processed/f'window-{win}'
        out = out/f'window-{win}'
    processed.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(processed/'dataset.npz',**arrays)
    with (processed/'window-index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(indices[0]))
        writer.writeheader();writer.writerows(indices)
    scaler={'method':cfg['normalization'],'fit_split':'train','mean':mean,'std':std,
            'fitted_scalar_count':int(train.size),'std_ddof':0}
    (processed/'normalization.json').write_text(json.dumps(scaler,ensure_ascii=False,indent=2),encoding='utf-8')
    (processed/'label-map.json').write_text(json.dumps({str(c['label']):c['name'] for c in classes},ensure_ascii=False,indent=2),encoding='utf-8')
    (processed/'build-config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    per_class={s:{c['name']:int(np.count_nonzero(arrays[f'y_{s}']==c['label'])) for c in classes} for s in SPLITS}
    summary={'config':cfg,'normalization':scaler,'records':records,'per_class_counts':per_class,
             'shapes':{k:list(v.shape) for k,v in arrays.items()},
             'checks':{'record_groups_disjoint':True,'windows_do_not_overlap':True,
                       'no_identical_raw_window_across_splits':True,'normalizer_fit_on_train_only':True},
             'dataset_sha256':hashlib.sha256((processed/'dataset.npz').read_bytes()).hexdigest(),
             'evaluation_scope':'held-out motor loads on the same CWRU rig; bearing identities may be shared across loads; not unseen-device validation',
             'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}}
    (out/'dataset-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':12})
    fig,ax=plt.subplots(figsize=(12,6),constrained_layout=True)
    ax.set(xlim=(-.65,3.65),ylim=(-.65,3.7),xticks=range(4),yticks=range(4),
           xticklabels=['0 HP：训练','1 HP：训练','2 HP：验证','3 HP：测试'],
           yticklabels=[c['name'] for c in classes],xlabel='先按原始记录的负载分组，再切窗口')
    ax.invert_yaxis()
    for r in records:
        x,y=r['load_hp'],r['label']
        ax.add_patch(Rectangle((x-.43,y-.36),.86,.72,facecolor=COLORS[r['split']],alpha=.9))
        ax.text(x,y,f"记录 {r['record_id']}\n{TITLES[r['split']]}",ha='center',va='center',color='white',fontsize=13)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('16 条原始记录 → 3 个集合｜一条记录只进入一个集合',fontsize=17,pad=18)
    fig.savefig(out/'01-record-split.png',dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    x=np.arange(4)
    for i,s in enumerate(SPLITS):
        vals=list(per_class[s].values())
        bars=ax.bar(x+(i-1)*.24,vals,width=.23,label=TITLES[s],color=COLORS[s])
        ax.bar_label(bars,padding=3)
    ax.set(xticks=x,xticklabels=[c['name'] for c in classes],ylabel='窗口样本数',
           title=f'{win} 点窗口的各类样本数｜样本多不等于独立试验多')
    ax.set_ylim(0,max(per_class['train'].values())*1.2)
    ax.grid(axis='y',alpha=.2);ax.legend()
    fig.savefig(out/'02-class-counts.png',dpi=150);plt.close(fig)

    example=int(np.flatnonzero(arrays['y_train']==1)[0])
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,constrained_layout=True)
    times=np.arange(win)/fs
    axes[0].plot(times,arrays['X_train_raw'][example],color=COLORS['train'])
    axes[0].set(title='一段训练内圈窗口：标准化前',ylabel='原始幅值单位')
    axes[1].plot(times,arrays['X_train'][example],color=COLORS['test'])
    axes[1].set(title=f'标准化后：所有集合共享训练均值 {mean:.6f} 与标准差 {std:.6f}',
                ylabel='标准化值（无量纲）',xlabel='窗口内相对时间（秒）')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'03-normalization.png',dpi=150);plt.close(fig)

    count_table='\n'.join(f"| {c['label']} | {c['name']} | {per_class['train'][c['name']]} | {per_class['val'][c['name']]} | {per_class['test'][c['name']]} |" for c in classes)
    shape_table='\n'.join(f"| {TITLES[s]} | `{tuple(arrays[f'X_{s}'].shape)}` | `{tuple(arrays[f'y_{s}'].shape)}` | {', '.join(map(str,sorted(groups[s])))} |" for s in SPLITS)
    image_block='\n\n'.join(f'![{title}]({(out/name).as_posix()})' for name,title in [
        ('01-record-split.png','原始记录分组图'),('02-class-counts.png','类别样本数量'),('03-normalization.png','共享训练参数的标准化示例')])
    report=f'''# 数据构建阶段：数据集构建结果

脚本已生成数组、标签、窗口索引和标准化参数；尚未训练分类模型，因此没有诊断准确率。

## 1. 本次配置

- 四类、16 条记录，统一驱动端与 12 kHz。正常原记录由 48 kHz 带滤波降采样。
- 每条记录取 0.5～4.5 秒，共 48000 点；窗口长度和步长均为 {win} 点。
- 每条得到 {count//win} 个完整窗口，丢弃末尾 {count%win} 点，不补零、不跨记录拼接。
- 0/1 HP 记录进入训练集，2 HP 进入验证集，3 HP 进入测试集。

## 2. 标签与样本数量

| 标签 | 状态 | 训练样本 | 验证样本 | 测试样本 |
| --- | --- | --- | --- | --- |
{count_table}

| 集合 | X 形状（样本数，窗口点数） | y 形状 | 原始记录号 |
| --- | --- | --- | --- |
{shape_table}

`X` 的一行是一段振动窗口，`y` 中相同位置的数是它的状态标签；0～3 不是故障程度。

## 3. 标准化与检查

共享训练均值：`{mean:.10f}`；共享训练标准差：`{std:.10f}`。

每个输入值按 `(x - mean) / std` 变换。参数只来自训练窗口，验证与测试不重新拟合；单个窗口或验证/测试集不必均值为 0、标准差为 1。该共享变换保留类间相对幅值，不等同于逐窗口标准化。

- 原始记录号在不同集合中不交叉。
- 窗口在各自记录内部不重叠，且未发现跨集合完全相同的窗口。
- 训练标准化数组整体均值约 0、标准差约 1。
- 全部数组为有限数值，标签与行数对应。

这些检查能发现指定的程序/划分问题，不能证明真实设备泛化能力。同一故障轴承可能在多个负载下重复使用；这里保留的是原始记录与负载，不是按独立轴承身份或独立设备划分。

## 4. 三张图

{image_block}

## 5. 文件在哪里

- 数组：[dataset.npz]({(processed/'dataset.npz').as_posix()})，数值数组可用 `np.load(..., allow_pickle=False)` 读取。
- 追溯索引：[window-index.csv]({(processed/'window-index.csv').as_posix()})，每个样本对应记录号、负载、标签、起止索引。
- 参数：[normalization.json]({(processed/'normalization.json').as_posix()})。
- 标签：[label-map.json]({(processed/'label-map.json').as_posix()})。
- 固定配置：[build-config.json]({(processed/'build-config.json').as_posix()})。

`X_train_raw / X_val_raw / X_test_raw` 保留未标准化窗口，供后续建模阶段计算 RMS 等特征；`X_train / X_val / X_test` 使用共享标准化，供后续建模。X 为 float32，y/记录号/负载为 int64。原始 MAT 不修改。

## 6. 当前能说明什么

能复现一个按负载保留评估记录的数据准备流程。后续模型只用训练集学习，用验证集选择方案，最后再评估冻结的测试集；不要根据测试结果反复调参。

来源：[CWRU 故障表]({cfg['fault_reference']})、[正常表]({cfg['normal_reference']})、[正常采样率参考：Smith & Randall 表 A1]({cfg['normal_rate_reference']})。
'''
    (out/'dataset-report.md').write_text(report,encoding='utf-8')
    for s in SPLITS:
        print(TITLES[s],f"X={arrays[f'X_{s}'].shape}, y={arrays[f'y_{s}'].shape}",f'每类={list(per_class[s].values())}')
    print('第一个训练样本：X_train[0] 是一段窗口，y_train[0]=',int(arrays['y_train'][0]),'（正常）')
    print('标准化前的前 5 点：',arrays['X_train_raw'][0,:5])
    print('标准化后的前 5 点：',arrays['X_train'][0,:5])
    print(f'训练标准化参数：mean={mean:.8f}, std={std:.8f}')
    print('检查通过：记录组不交叉、窗口不重叠、无跨集合完全相同窗口、仅训练拟合参数。')
    print('数据保存到：',processed)
    print('图片与报告：',out)


if __name__=='__main__':
    main()
