"""噪声评估阶段：冻结模型、预设强度与种子、共同噪声输入，比较人工扰动表现。

python -X utf8 scripts/evaluate_noise.py
不训练、去噪、拟合 scaler 或改动已生成的数据和模型。
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import sys

import joblib
import numpy as np
import torch
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/noise'
sys.path.insert(0, str(ROOT / 'src'))
from cnn import load_saved_cnn, predict_raw_windows
from noise import make_unit_noise, add_noise_at_snr
from train_baseline import predict_windows
from evaluate import metrics, draw_matrix, sha256, write_json, write_csv


def predict_pair(model, meta, bundle, raw, cfg):
    logits = np.concatenate([predict_raw_windows(model, meta, raw[i:i+cfg['batch_size']], cfg['sampling_rate_hz'])['logits']
                             for i in range(0, len(raw), cfg['batch_size'])])
    return logits, logits.argmax(axis=1), predict_windows(bundle, raw, cfg['sampling_rate_hz'])


def aggregate_trials(trials, cfg, names):
    rows = []
    for condition in ['original'] + [f'{s}dB' for s in cfg['snr_db_levels']]:
        selected = [t for t in trials if t['condition'] == condition]
        for key in ['cnn', 'svm']:
            row = {'condition': condition, 'model': key, 'noise_draws': len(selected),
                   'windows_per_draw': selected[0]['models'][key]['total']}
            for score in ['accuracy', 'macro_f1']:
                values = [t['models'][key][score] for t in selected]
                row.update({f'{score}_mean': float(np.mean(values)), f'{score}_std': float(np.std(values, ddof=0)),
                            f'{score}_min': float(np.min(values)), f'{score}_max': float(np.max(values))})
            for c, name in enumerate(names):
                row[f'recall_class_{c}_mean'] = float(np.mean([t['models'][key]['per_class'][c]['recall'] for t in selected]))
            rows.append(row)
    return rows


def make_figures(cfg, names, raw, index, bank, summary, selected_noisy, audit):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 12})
    example = cfg['waveform_example']
    i, seed = example['test_index'], example['seed']
    t = float(index[i]['start_second']) + np.arange(cfg['window_points']) / cfg['sampling_rate_hz']
    signals = [('原始窗口（未额外加噪）', raw[i])]
    for snr in example['snr_db_levels']:
        noisy, _ = add_noise_at_snr(raw, bank[seed], snr)
        signals.append((f'{snr} dB：噪声 RMS / 原窗口 RMS = {10**(-snr/20):.3f}', noisy[i]))
    bound = max(float(np.abs(x).max()) for _, x in signals) * 1.06
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True, sharex=True, sharey=True)
    for ax, (title, values) in zip(axes.flat, signals):
        if not np.array_equal(values, raw[i]):
            ax.plot(t, raw[i], color='#337EBB', alpha=.45, lw=.8, label='原始窗口')
        ax.plot(t, values, color='#C65C75' if not np.array_equal(values, raw[i]) else '#337EBB', lw=.8,
                label='加噪窗口' if not np.array_equal(values, raw[i]) else '原始窗口')
        ax.set(title=title, xlabel='记录内时间（秒）', ylabel='振动幅值（原数据尺度）', ylim=(-bound, bound))
        ax.grid(alpha=.2)
        ax.legend(loc='upper right', fontsize=10)
    fig.suptitle(f"同一内圈窗口｜测试索引 {i}，记录 {index[i]['record_id']}｜种子 {seed}｜统一纵轴尺度", fontsize=15)
    fig.savefig(OUT / '01-noise-waveforms.png', dpi=150)
    plt.close(fig)

    conditions = ['original'] + [f'{s}dB' for s in cfg['snr_db_levels']]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, score, title in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy', 'Macro-F1']):
        for key, name, color, marker in [('svm', 'SVM-C1', '#337EBB', 'o'), ('cnn', 'CNN', '#248D77', 's')]:
            rows = [next(r for r in summary['aggregate'] if r['condition'] == c and r['model'] == key) for c in conditions]
            ax.errorbar(np.arange(len(rows)), [r[f'{score}_mean'] for r in rows],
                        yerr=[r[f'{score}_std'] for r in rows], label=name, color=color,
                        marker=marker, capsize=4, lw=2)
        ax.set(xticks=np.arange(len(conditions)), xticklabels=['原始', '20', '10', '5', '0', '−5'],
               xlabel='SNR（dB）｜从左到右额外噪声增强', ylabel=title, ylim=(0, 1.05),
               title=f'{title}：三次噪声抽样均值 ± 标准差')
        ax.grid(alpha=.2)
        ax.legend(loc='lower left')
    fig.suptitle('冻结模型／相同加噪输入｜误差线只反映噪声抽样变化，不是置信区间', fontsize=14)
    fig.savefig(OUT / '02-noise-performance.png', dpi=150)
    plt.close(fig)

    selected = next(t for t in summary['trials'] if t['snr_db'] == cfg['detailed_case']['snr_db'] and t['seed'] == cfg['detailed_case']['seed'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, key, name in zip(axes, ['cnn', 'svm'], ['CNN', 'SVM-C1']):
        im = draw_matrix(ax, selected['models'][key]['confusion_matrix'], ['正常', '内圈', '外圈', '滚动体'],
                         f"{name}｜固定案例 {selected['snr_db']} dB，种子 {selected['seed']}")
        fig.colorbar(im, ax=ax, shrink=.75, label='窗口数')
    fig.savefig(OUT / '03-noisy-confusion-matrices.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, row in zip(axes.flat, audit):
        i = row['sample_index_in_split']
        t = float(index[i]['start_second']) + np.arange(cfg['window_points']) / cfg['sampling_rate_hz']
        ax.plot(t, raw[i], color='#337EBB', alpha=.55, lw=.8, label='原始')
        ax.plot(t, selected_noisy[i], color='#C65C75', lw=.7, alpha=.8, label='加噪')
        ax.set(title=f"索引 {i}／记录 {row['record_id']}｜实际：{row['true_state']}\nCNN：{row['cnn_state']}；SVM：{row['svm_state']}",
               xlabel='记录内时间（秒）', ylabel='振动幅值（原数据尺度）')
        ax.legend(fontsize=10)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(audit):]:
        ax.axis('off')
    fig.suptitle(f"固定案例：{selected['snr_db']} dB／种子 {selected['seed']}｜每类优先检查首个任一模型出错窗口", fontsize=14)
    fig.savefig(OUT / '04-noisy-sample-audit.png', dpi=150)
    plt.close(fig)


def main():
    cfg_path = ROOT / 'configs/noise.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    assert cfg['noise_split'] == 'test' and cfg['device'] == 'cpu'
    assert len(set(cfg['noise_seeds'])) == len(cfg['noise_seeds']) == 3
    assert len(set(cfg['snr_db_levels'])) == len(cfg['snr_db_levels']) == 5
    assert cfg['detailed_case']['seed'] in cfg['noise_seeds'] and cfg['detailed_case']['snr_db'] in cfg['snr_db_levels']
    dataset_path = ROOT / cfg['dataset']
    cnn_folder, svm_path = ROOT / cfg['cnn_folder'], ROOT / cfg['baseline_model']
    protected = [dataset_path, dataset_path.parent / 'normalization.json',
                 cnn_folder / 'best-cnn.pt', cnn_folder / 'model-metadata.json', cnn_folder / 'training-summary.json',
                 svm_path, ROOT / 'outputs/baseline/metrics.json', ROOT / 'outputs/baseline/selection-frozen.json',
                 ROOT / 'outputs/evaluation/metrics.json', ROOT / 'outputs/evaluation/test-output.npz']
    hashes = {p.as_posix(): sha256(p) for p in protected}
    model, meta = load_saved_cnn(cnn_folder)
    bundle = joblib.load(svm_path)
    selection = json.loads((ROOT / 'outputs/baseline/selection-frozen.json').read_text(encoding='utf-8'))
    names = meta['label_names']
    label_map = json.loads((dataset_path.parent / 'label-map.json').read_text(encoding='utf-8'))
    assert names == bundle['label_names'] == [label_map[str(c)] for c in cfg['labels']]
    assert sha256(dataset_path) == meta['dataset_sha256'] == selection['dataset_sha256']
    assert meta['sampling_rate_hz'] == cfg['sampling_rate_hz'] and meta['window_points'] == cfg['window_points']
    assert meta['normalization']['fit_split'] == 'train'
    torch.set_num_threads(cfg['torch_threads'])
    torch.use_deterministic_algorithms(True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {'frozen_before_prediction_utc': datetime.now(timezone.utc).isoformat(),
              'config': cfg, 'config_sha256': sha256(cfg_path), 'source_hashes': hashes,
              'cnn_epoch': meta['selected_epoch'], 'svm_model': selection['chosen_candidate']['name'],
              'training_performed': False, 'new_independent_data': False}
    write_json(OUT / 'experiment-frozen.json', frozen)
    print('实验方案已固定：原始＋20/10/5/0/−5 dB，每档三个种子；只加载冻结模型。', flush=True)
    with np.load(dataset_path, allow_pickle=False) as z:
        arrays = {f'{k}_{s}': z[f'X_{s}_raw' if k == 'X_raw' else f'{k}_{s}'].copy() for s in ['train', 'val', 'test']
                  for k in ['X', 'X_raw', 'y', 'record_id', 'load_hp']}
    raw, y = arrays['X_raw_test'].astype(np.float64), arrays['y_test']
    assert raw.shape == (184, cfg['window_points']) and np.all(arrays['load_hp_test'] == 3)
    groups = {s: set(arrays[f'record_id_{s}'].tolist()) for s in ['train', 'val', 'test']}
    assert not any(groups[a] & groups[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    indices = list(csv.DictReader((dataset_path.parent / 'window-index.csv').open(encoding='utf-8-sig')))
    index = sorted([r for r in indices if r['split'] == 'test'], key=lambda r: int(r['sample_index_in_split']))
    assert len(index) == len(y)
    for i, row in enumerate(index):
        assert int(row['sample_index_in_split']) == i and int(row['label']) == y[i]
        assert int(row['record_id']) == arrays['record_id_test'][i]
    original_logits, original_cnn, original_svm = predict_pair(model, meta, bundle, raw, cfg)
    with np.load(ROOT / 'outputs/evaluation/test-output.npz', allow_pickle=False) as z:
        assert np.array_equal(z['cnn_prediction'], original_cnn) and np.array_equal(z['svm_prediction'], original_svm)
        np.testing.assert_allclose(z['logits'], original_logits, rtol=1e-6, atol=1e-7)
        assert np.array_equal(z['y_true'], y) and np.array_equal(z['record_id'], arrays['record_id_test'])

    # 各负载原始窗口审视；训练／验证成绩明确保留其原数据角色。
    load_rows = []
    for load, split, role in [(0, 'train', '已参与训练'), (1, 'train', '已参与训练'),
                              (2, 'val', '已参与验证选择'), (3, 'test', '复用的测试划分')]:
        mask = arrays[f'load_hp_{split}'] == load
        _, cnn_p, svm_p = predict_pair(model, meta, bundle, arrays[f'X_raw_{split}'][mask], cfg)
        for key, p in [('cnn', cnn_p), ('svm', svm_p)]:
            m = metrics(arrays[f'y_{split}'][mask], p, names, cfg['labels'])
            load_rows.append({'load_hp': load, 'split': split, 'data_role': role, 'model': key,
                              'windows': m['total'], 'correct': m['correct'], 'accuracy': m['accuracy'],
                              'macro_f1': m['macro_f1'], 'new_independent_test': False})
    write_csv(OUT / 'clean-load-audit.csv', load_rows, list(load_rows[0]))

    bank = {seed: make_unit_noise(raw, seed) for seed in cfg['noise_seeds']}
    np.savez_compressed(OUT / 'noise-bank.npz', **{f'unit_noise_seed_{seed}': u for seed, u in bank.items()})
    trials, all_rows, predicted_logits, predicted_cnn, predicted_svm = [], [], [], [], []
    selected_noisy = None
    conditions = [(None, None)] + [(snr, seed) for snr in cfg['snr_db_levels'] for seed in cfg['noise_seeds']]
    for trial_id, (snr, seed) in enumerate(conditions):
        condition = 'original' if snr is None else f'{snr}dB'
        if snr is None:
            input_windows, power = raw, None
            logits, cnn_p, svm_p = original_logits, original_cnn, original_svm
        else:
            input_windows, power = add_noise_at_snr(raw, bank[seed], snr)
            logits, cnn_p, svm_p = predict_pair(model, meta, bundle, input_windows, cfg)
        scores = {'cnn': metrics(y, cnn_p, names, cfg['labels']), 'svm': metrics(y, svm_p, names, cfg['labels'])}
        trial = {'trial_id': trial_id, 'condition': condition, 'snr_db': snr, 'seed': seed, 'models': scores,
                 'input_sha256': hashlib.sha256(np.asarray(input_windows, dtype=np.float64).tobytes()).hexdigest(),
                 'measured_snr_range_db': None if power is None else [float(power['measured_snr_db'].min()), float(power['measured_snr_db'].max())]}
        trials.append(trial)
        predicted_logits.append(logits)
        predicted_cnn.append(cnn_p)
        predicted_svm.append(svm_p)
        for i, idx in enumerate(index):
            all_rows.append({'trial_id': trial_id, 'condition': condition, 'snr_db': '' if snr is None else snr,
                             'seed': '' if seed is None else seed, 'sample_index_in_split': i,
                             'record_id': int(arrays['record_id_test'][i]), 'load_hp': 3,
                             'true_label': int(y[i]), 'true_state': names[int(y[i])],
                             'cnn_label': int(cnn_p[i]), 'cnn_state': names[int(cnn_p[i])], 'cnn_correct': int(cnn_p[i] == y[i]),
                             'svm_label': int(svm_p[i]), 'svm_state': names[int(svm_p[i])], 'svm_correct': int(svm_p[i] == y[i]),
                             'reference_power': float(np.mean(raw[i] * raw[i])),
                             'added_noise_power': 0.0 if power is None else float(power['noise_power'][i]),
                             'measured_snr_db': '' if power is None else float(power['measured_snr_db'][i]),
                             'start_second': float(idx['start_second']), 'end_second_exclusive': float(idx['end_second_exclusive']),
                             'file': idx['file'], 'channel': idx['channel']})
        if snr == cfg['detailed_case']['snr_db'] and seed == cfg['detailed_case']['seed']:
            selected_noisy = input_windows.copy()
        print(f"{condition:>8}／seed={seed}｜CNN F1={scores['cnn']['macro_f1']:.4f}，错误 {scores['cnn']['errors']}"
              f"｜SVM F1={scores['svm']['macro_f1']:.4f}，错误 {scores['svm']['errors']}", flush=True)
    columns = list(all_rows[0])
    write_csv(OUT / 'all-predictions.csv', all_rows, columns)
    write_csv(OUT / 'errors-cnn.csv', [r for r in all_rows if not r['cnn_correct']], columns)
    write_csv(OUT / 'errors-svm.csv', [r for r in all_rows if not r['svm_correct']], columns)
    selected_trial = next(t for t in trials if t['snr_db'] == cfg['detailed_case']['snr_db'] and t['seed'] == cfg['detailed_case']['seed'])
    case_rows = [r for r in all_rows if r['trial_id'] == selected_trial['trial_id']]
    audit = []
    for c in cfg['labels']:
        class_rows = [r for r in case_rows if r['true_label'] == c]
        class_errors = [r for r in class_rows if not r['cnn_correct'] or not r['svm_correct']]
        audit.append((class_errors or class_rows)[0])
    write_csv(OUT / 'sample-audit.csv', audit, columns)
    np.savez_compressed(OUT / 'detailed-case.npz', noisy=selected_noisy,
                        noise=selected_noisy-raw, y_true=y, record_id=arrays['record_id_test'])
    np.savez_compressed(OUT / 'trial-output.npz', logits=np.stack(predicted_logits),
                        cnn_prediction=np.stack(predicted_cnn), svm_prediction=np.stack(predicted_svm),
                        y_true=y, record_id=arrays['record_id_test'])
    aggregate = aggregate_trials(trials, cfg, names)
    write_csv(OUT / 'noise-summary.csv', aggregate, list(aggregate[0]))
    for k, v in model.state_dict().items():
        assert torch.equal(before[k], v), k
    for path, expected in hashes.items():
        assert sha256(Path(path)) == expected, path
    result = {'config': cfg, 'label_names': names, 'trials': trials, 'aggregate': aggregate,
              'clean_load_audit': load_rows, 'detailed_case_trial_id': selected_trial['trial_id'],
              'unique_original_test_windows': len(y), 'original_test_records': len(groups['test']),
              'training_performed': False, 'new_independent_data': False,
              'checks': {'frozen_before_predictions': True, 'same_noisy_input_for_both_models': True,
                         'per_window_snr_verified': True, 'original_predictions_match_clean_evaluation': True,
                         'cnn_weights_unchanged': True, 'previous_artifacts_unchanged': True,
                         'train_val_test_record_groups_disjoint': True},
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                           'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__}}
    write_json(OUT / 'metrics.json', result)
    make_figures(cfg, names, raw, index, bank, result, selected_noisy, audit)
    table_rows = []
    for condition in ['original'] + [f'{s}dB' for s in cfg['snr_db_levels']]:
        c = next(r for r in aggregate if r['condition'] == condition and r['model'] == 'cnn')
        s = next(r for r in aggregate if r['condition'] == condition and r['model'] == 'svm')
        display = '未额外加噪' if condition == 'original' else condition
        table_rows.append(f"| {display} | {c['accuracy_mean']:.2%} ± {c['accuracy_std']:.2%} | {s['accuracy_mean']:.2%} ± {s['accuracy_std']:.2%} | {c['macro_f1_mean']:.4f} ± {c['macro_f1_std']:.4f} | {s['macro_f1_mean']:.4f} ± {s['macro_f1_std']:.4f} |")
    comparison_table = '\n'.join(table_rows)
    load_table = '\n'.join(f"| {hp} | {next(r['data_role'] for r in load_rows if r['load_hp']==hp)} | "
                           + ' | '.join(f"{next(r['macro_f1'] for r in load_rows if r['load_hp']==hp and r['model']==key):.4f}" for key in ['cnn', 'svm']) + ' |'
                           for hp in [0, 1, 2, 3])
    case_table = '\n'.join(f"| {name} | {m['correct']}/{m['total']} | {m['errors']} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {m['normal_vs_fault']['fn']} | {m['normal_vs_fault']['fp']} | {m['normal_vs_fault']['fault_type_confusions']} |"
                           for key, name in [('cnn', 'CNN'), ('svm', 'SVM-C1')] for m in [selected_trial['models'][key]])
    error_pairs = []
    for key, name in [('cnn', 'CNN'), ('svm', 'SVM-C1')]:
        cm = np.asarray(selected_trial['models'][key]['confusion_matrix'])
        pairs = [(int(cm[a,b]), a, b) for a in range(4) for b in range(4) if a != b and cm[a,b] > 0]
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        description = '；'.join(f'{names[a]} → {names[b]}：{count} 个' for count,a,b in pairs[:3]) or '没有误判'
        error_pairs.append(f'- {name}：{description}。')
    errors_note = '\n'.join(error_pairs)
    report = f'''# 噪声评估阶段：人工噪声实验与负载数据审视结果

由实际运行生成。仅使用冻结模型；本程序没有训练、去噪或重新拟合 scaler，没有新增独立设备数据。

## 1. 预先固定的实验条件

3 HP 的 184 个既有测试窗口，原始条件加上 20／10／5／0／−5 dB 五档人工噪声。每档三个种子 2026／2027／2028，全部保留；两模型使用同一加噪数组。同一 seed 在各档使用相同噪声方向，仅改变幅度。

噪声加在未标准化、已对齐采样率的窗口上，再走各模型已保存的处理。逐窗以 `mean(original**2)`（含 DC）为参考功率，标准正态抽样后按实际 RMS 定标。每窗实测 SNR 已核对；这是相对于原窗口和新增噪声的比值，不是从记录里估计出的现场 SNR。

## 2. 三次噪声抽样结果

| 条件 | CNN Accuracy 均值 ± 标准差 | SVM Accuracy 均值 ± 标准差 | CNN Macro-F1 均值 ± 标准差 | SVM Macro-F1 均值 ± 标准差 |
| --- | --- | --- | --- | --- |
{comparison_table}

均值是三次独立噪声抽样指标的平均，标准差使用 ddof=0。Accuracy 的标准差按百分数尺度展示，例如 0.77% 表示 0.77 个百分点。原始条件只评估一次、标准差为 0；没有额外加噪不表示原记录没有背景噪声。误差线不是置信区间。三次重复仍是同一 184 个窗口，不是三批独立设备。

![同一窗口随噪声变化]({(OUT / '01-noise-waveforms.png').as_posix()})

![人工噪声下两模型表现]({(OUT / '02-noise-performance.png').as_posix()})

## 3. 预先固定的详细案例：−5 dB，种子 2026

不是看过结果后挑最差或最好 seed。该条件已写入预测前的配置。

| 模型 | 正确／总数 | 错误数 | Accuracy | Macro-F1 | 故障判正常 | 正常判故障 | 故障类型混淆 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{case_table}

![固定案例的两模型矩阵]({(OUT / '03-noisy-confusion-matrices.png').as_posix()})

数量最多的前三种错分方向：

{errors_note}

![固定案例的真实加噪窗口]({(OUT / '04-noisy-sample-audit.png').as_posix()})

这些是程序生成的人工加噪输入及真实模型预测，不是现场采集。每类优先展示两个模型任一出错的首个索引；该类若没有错误则展示首个正确窗口。原始和加噪波形可对照，原因解释仍需额外验证，不能只凭形状断言模型学到了什么。

## 4. 0～3 HP 原始窗口审视

| 负载 HP | 既有数据角色 | CNN Macro-F1 | SVM Macro-F1 |
| --- | --- | --- | --- |
{load_table}

每个负载 184 个窗口，来自每类一条记录。0／1 HP 已参与拟合，2 HP 已参与验证选择；它们的成绩不能当作独立泛化成绩。3 HP 是之前已用的跨负载测试，不是新设备。这里没有重新做留一负载训练实验。

## 5. 文件、复现与检查

- [experiment-frozen.json]({(OUT / 'experiment-frozen.json').as_posix()})：预测前的条件、模型与数据哈希。
- [noise-summary.csv]({(OUT / 'noise-summary.csv').as_posix()})：各条件三次抽样的均值、标准差及范围。
- [metrics.json]({(OUT / 'metrics.json').as_posix()})：每个种子的完整四分类矩阵、逐类指标与计数。
- [all-predictions.csv]({(OUT / 'all-predictions.csv').as_posix()})：每个条件、种子、索引的两模型预测与源窗口信息。
- [errors-cnn.csv]({(OUT / 'errors-cnn.csv').as_posix()})、[errors-svm.csv]({(OUT / 'errors-svm.csv').as_posix()})：全部条件下的错误行，需按条件／种子筛选，不能当独立新窗口总数。
- [sample-audit.csv]({(OUT / 'sample-audit.csv').as_posix()})：详细案例图中的观察窗口。
- [clean-load-audit.csv]({(OUT / 'clean-load-audit.csv').as_posix()})：按负载和既有角色记录的原始窗口成绩。

保存的 noise-bank.npz 记录三个种子的单位 RMS 模板；trial-output.npz 按 trials 的 trial_id 保存全部预测；detailed-case.npz 保存详细案例加噪窗口和实际噪声。已检查原始条件预测与测试评估阶段一致，模型参数及既有实验文件未改变、逐窗 SNR 达标。

## 6. 当前结论的范围

只对这套 CWRU 窗口、逐窗相对功率定标的随机噪声与三个预定种子作结论。原窗口已经带有采集背景，模拟噪声不是实际电机噪声；逐窗按相同 SNR 定标使不同幅值窗口的绝对噪声强度不同。有限窗口的高斯抽样再定标，也不同于未经约束的独立高斯过程。

当前未据结果调参或宣称抗噪能力提高，没有固定绝对噪声、色噪声、冲击干扰、新轴承／新设备、噪声增强训练或滤波改进实验。重复噪声抽样不提供独立设备泛化证据。诊断界面将把已验证的数据处理与模型加载做成演示界面，展示输入要求和结果来源。
'''
    (OUT / 'noise-report.md').write_text(report, encoding='utf-8')
    print('检查通过：逐窗 SNR、共同噪声输入、原始预测一致、权重与既有实验文件未改变。', flush=True)
    print('报告：', OUT / 'noise-report.md', flush=True)


if __name__ == '__main__':
    main()
