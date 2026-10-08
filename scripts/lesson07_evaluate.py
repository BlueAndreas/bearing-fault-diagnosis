"""第七课：只加载冻结的 CNN／SVM，测试、指标解释与可追溯样本检查。

python -X utf8 scripts/lesson07_evaluate.py
不会训练、重新拟合 scaler、更新权重，或覆盖前六课产物。
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/lesson07'
sys.path.insert(0, str(ROOT / 'src'))
from lesson06_cnn import load_saved_cnn, predict_raw_windows
from lesson04_train_baseline import predict_windows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, predicted, names, labels):
    cm = confusion_matrix(y, predicted, labels=labels)
    detail = classification_report(y, predicted, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    total, correct = len(y), int(np.count_nonzero(y == predicted))
    assert int(cm.sum()) == total and int(cm.trace()) == correct
    classes = []
    for c, name in zip(labels, names):
        tp = int(cm[c, c])
        fp, fn = int(cm[:, c].sum() - tp), int(cm[c, :].sum() - tp)
        classes.append({'label': c, 'state': name, 'support': int(cm[c].sum()),
                        'tp': tp, 'fp': fp, 'fn': fn,
                        'precision': float(detail[name]['precision']),
                        'recall': float(detail[name]['recall']), 'f1': float(detail[name]['f1-score'])})
    # 从四分类结果合并成正常／故障，只作另一个观察口径，不重新训练。
    tn, false_alarm = int(cm[0, 0]), int(cm[0, 1:].sum())
    missed_fault, detected_fault = int(cm[1:, 0].sum()), int(cm[1:, 1:].sum())
    binary = {'tn': tn, 'fp': false_alarm, 'fn': missed_fault, 'tp': detected_fault,
              'fault_recall': detected_fault / (detected_fault + missed_fault),
              'normal_false_alarm_rate': false_alarm / (false_alarm + tn),
              'fault_type_confusions': int(cm[1:, 1:].sum() - np.trace(cm[1:, 1:]))}
    return {'accuracy': float(accuracy_score(y, predicted)),
            'macro_f1': float(f1_score(y, predicted, labels=labels, average='macro', zero_division=0)),
            'correct': correct, 'errors': total - correct, 'total': total,
            'confusion_matrix': cm.tolist(), 'per_class': classes,
            'normal_vs_fault': binary}


def draw_matrix(ax, matrix, names, title):
    cm = np.asarray(matrix)
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(1, int(cm.max())))
    ax.set(xticks=range(4), yticks=range(4), xticklabels=names, yticklabels=names,
           xlabel='预测类别（看列）', ylabel='实际类别（看行）', title=title)
    ax.tick_params(axis='x', labelrotation=15)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=18,
                    color='white' if cm[i, j] > cm.max() / 2 else '#233244')
    return im


def make_figures(cfg, names, result, raw, audit_rows):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 12})
    short = ['正常', '内圈', '外圈', '滚动体']
    demo = np.asarray(cfg['teaching_demo_confusion_matrix'])
    fig, (ax, note) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1]},
                                 constrained_layout=True)
    draw_matrix(ax, demo, short, '教学虚构数据｜不是本项目测试成绩')
    ax.add_patch(Rectangle((-.5, .5), 4, 1, fill=False, edgecolor='#D99F2F', linewidth=3))
    ax.add_patch(Rectangle((.5, -.5), 1, 4, fill=False, edgecolor='#248D77', linewidth=2))
    note.axis('off')
    note.text(.03, .96, '只看“内圈”这一类', fontsize=21, weight='bold', va='top')
    note.text(.03, .78, '实际内圈：看第 2 行，共 10 个\n预测内圈：看第 2 列，共 9 个\n其中正确：交叉位置，共 7 个', fontsize=15, va='top', linespacing=1.7)
    note.text(.03, .42, 'Precision = 7 / 9 ≈ 77.78%\nRecall = 7 / 10 = 70%\nF1 = 2×7 / (2×7+2+3) ≈ 0.7368', fontsize=15, va='top', linespacing=1.7)
    note.text(.03, .12, '总体 Accuracy = (8+7+9+7) / 40 = 77.5%\n每类 F1 算完，再平均才是 Macro-F1。', fontsize=12, va='top', linespacing=1.5)
    fig.savefig(OUT / '01-metrics-teaching-example.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, key, title in zip(axes, ['cnn', 'svm'], ['冻结 CNN', '冻结 SVM-C1']):
        im = draw_matrix(ax, result['models'][key]['confusion_matrix'], short, f'{title}｜3 HP 测试窗口')
        fig.colorbar(im, ax=ax, shrink=.75, label='窗口数')
    fig.savefig(OUT / '02-test-confusion-matrices.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(2)
    for offset, key, name, color in [(-.18, 'svm', 'SVM-C1', '#337EBB'), (.18, 'cnn', 'CNN', '#248D77')]:
        values = [result['models'][key]['accuracy'], result['models'][key]['macro_f1']]
        bars = ax.bar(x + offset, values, width=.35, label=name, color=color)
        ax.bar_label(bars, labels=[f'{v:.4f}' for v in values], padding=4)
    ax.set(xticks=x, xticklabels=['Accuracy（0～1）', 'Macro-F1（0～1）'], ylim=(0, 1.15),
           ylabel='测试分数', title='同一测试窗口与冻结模型｜同分不能证明 CNN 更优')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT / '03-test-comparison.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, row in zip(axes.flat, audit_rows):
        i = row['sample_index_in_split']
        t = row['start_second'] + np.arange(cfg['window_points']) / cfg['sampling_rate_hz']
        ax.plot(t, raw[i], linewidth=.7, color='#337EBB')
        ax.set(title=f"索引 {i}／记录 {row['record_id']}｜实际：{row['true_state']}\nCNN：{row['cnn_state']}；SVM：{row['svm_state']}",
               xlabel='记录内时间（秒）', ylabel='振动幅值（原始数据尺度）')
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(audit_rows):]:
        ax.axis('off')
    all_correct = result['models']['cnn']['errors'] == 0
    fig.suptitle('本次 CNN 无误判：每类选分数间隔较小的正确窗口供观察' if all_correct
                 else 'CNN 样本检查：优先展示错误窗口，未用检查结果修改模型', fontsize=15)
    fig.savefig(OUT / '04-window-audit.png', dpi=150)
    plt.close(fig)


def main():
    config_path = ROOT / 'configs/lesson07_evaluation.json'
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    dataset_path = ROOT / cfg['dataset']
    cnn_folder = ROOT / cfg['cnn_folder']
    svm_path = ROOT / cfg['baseline_model']
    protected = [dataset_path, cnn_folder / 'best-cnn.pt', cnn_folder / 'model-metadata.json',
                 cnn_folder / 'lesson06-summary.json', svm_path, ROOT / 'outputs/lesson04/metrics.json',
                 ROOT / 'outputs/lesson04/selection-frozen.json']
    hashes = {p.as_posix(): sha256(p) for p in protected}
    model, meta = load_saved_cnn(cnn_folder)
    baseline_selection = json.loads((ROOT / 'outputs/lesson04/selection-frozen.json').read_text(encoding='utf-8'))
    bundle = joblib.load(svm_path)
    mapping = json.loads((dataset_path.parent / 'label-map.json').read_text(encoding='utf-8'))
    names = [mapping[str(c)] for c in cfg['labels']]
    if names != meta['label_names'] or names != bundle['label_names']:
        raise ValueError('两种模型和数据的类别映射不一致。')
    if sha256(dataset_path) != meta['dataset_sha256'] or sha256(dataset_path) != baseline_selection['dataset_sha256']:
        raise ValueError('数据不是两个已冻结模型对应的第三课数据。')
    if meta['window_points'] != cfg['window_points'] or meta['sampling_rate_hz'] != cfg['sampling_rate_hz']:
        raise ValueError('CNN 输入要求与评估配置不一致。')
    if meta['normalization']['fit_split'] != 'train':
        raise ValueError('CNN 标准化必须来自训练集。')
    torch.set_num_threads(cfg['torch_threads'])
    torch.use_deterministic_algorithms(True)
    before_state = {k: v.clone() for k, v in model.state_dict().items()}
    OUT.mkdir(parents=True, exist_ok=True)
    # 先保存模型身份与固定评估规则，然后才读取测试输入并作预测。
    manifest = {'frozen_before_prediction_utc': datetime.now(timezone.utc).isoformat(),
                'config': cfg, 'config_sha256': sha256(config_path),
                'source_hashes': hashes, 'cnn_selected_epoch': meta['selected_epoch'],
                'svm_selected_model': baseline_selection['chosen_candidate']['name'],
                'training_performed': False, 'test_split_already_used_in_lesson04': True}
    write_json(OUT / 'evaluation-frozen.json', manifest)
    print('已记录冻结模型和评估配置；不训练、不调参、不重新拟合 scaler。', flush=True)
    with np.load(dataset_path, allow_pickle=False) as z:
        raw, normalized, y = z['X_test_raw'].copy(), z['X_test'].copy(), z['y_test'].copy()
        record, load = z['record_id_test'].copy(), z['load_hp_test'].copy()
        groups = {s: set(z[f'record_id_{s}'].tolist()) for s in ['train', 'val', 'test']}
    assert not any(groups[a] & groups[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    assert raw.shape == normalized.shape == (len(y), cfg['window_points'])
    assert set(np.unique(y).tolist()) == set(cfg['labels']) and np.all(load == 3)
    index_rows = [r for r in csv.DictReader((dataset_path.parent / 'window-index.csv').open(encoding='utf-8-sig'))
                  if r['split'] == 'test']
    index_rows.sort(key=lambda r: int(r['sample_index_in_split']))
    assert len(index_rows) == len(y)
    for i, row in enumerate(index_rows):
        assert int(row['sample_index_in_split']) == i and int(row['record_id']) == int(record[i])
        assert int(row['label']) == int(y[i]) and int(row['load_hp']) == int(load[i])
    outputs = []
    for start in range(0, len(y), cfg['batch_size']):
        outputs.append(predict_raw_windows(model, meta, raw[start:start+cfg['batch_size']], cfg['sampling_rate_hz'])['logits'])
    logits = np.concatenate(outputs)
    cnn = logits.argmax(axis=1)
    # 独立核对第三课已标准化数组，避免推理和训练预处理不一致。
    with torch.no_grad():
        normalized_logits = model(torch.from_numpy(normalized).unsqueeze(1)).numpy()
    np.testing.assert_allclose(logits, normalized_logits, rtol=1e-5, atol=1e-6)
    svm = predict_windows(bundle, raw, cfg['sampling_rate_hz'])
    for k, v in model.state_dict().items():
        assert torch.equal(before_state[k], v)
    old = json.loads((ROOT / 'outputs/lesson04/metrics.json').read_text(encoding='utf-8'))
    previous = list(csv.DictReader((ROOT / 'outputs/lesson04/test-predictions.csv').open(encoding='utf-8-sig')))
    assert old['selected_model'] == baseline_selection['chosen_candidate']['name'] == 'SVM-C1'
    assert len(previous) == len(y)
    for i, row in enumerate(previous):
        assert int(row['sample_index_in_split']) == i and int(row['record_id']) == int(record[i])
        assert int(row['true_label']) == int(y[i]) and int(row['predicted_label']) == int(svm[i])
    scores = {'cnn': metrics(y, cnn, names, cfg['labels']), 'svm': metrics(y, svm, names, cfg['labels'])}
    assert scores['svm']['confusion_matrix'] == old['test']['confusion_matrix']
    per_record = []
    for r in sorted(groups['test']):
        mask = record == r
        per_record.append({'record_id': r, 'state': names[int(y[mask][0])], 'load_hp': 3,
                           'windows': int(mask.sum()), 'cnn_errors': int(np.sum(cnn[mask] != y[mask])),
                           'svm_errors': int(np.sum(svm[mask] != y[mask]))})
    gaps = np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2]
    rows = []
    for i, idx in enumerate(index_rows):
        rows.append({'sample_index_in_split': i, 'record_id': int(record[i]), 'load_hp': int(load[i]),
                     'true_label': int(y[i]), 'true_state': names[int(y[i])],
                     'cnn_label': int(cnn[i]), 'cnn_state': names[int(cnn[i])], 'cnn_correct': int(cnn[i] == y[i]),
                     'svm_label': int(svm[i]), 'svm_state': names[int(svm[i])], 'svm_correct': int(svm[i] == y[i]),
                     'cnn_top2_logit_gap': float(gaps[i]),
                     'start_second': float(idx['start_second']), 'end_second_exclusive': float(idx['end_second_exclusive']),
                     'file': idx['file'], 'channel': idx['channel'],
                     **{f'cnn_logit_{c}': float(logits[i, c]) for c in cfg['labels']}})
    columns = list(rows[0])
    write_csv(OUT / 'test-predictions.csv', rows, columns)
    write_csv(OUT / 'errors-cnn.csv', [r for r in rows if not r['cnn_correct']], columns)
    write_csv(OUT / 'errors-svm.csv', [r for r in rows if not r['svm_correct']], columns)
    wrong = np.flatnonzero(cnn != y)
    if len(wrong):
        audit_ids = wrong[:4].tolist()
    else:
        audit_ids = [int(np.flatnonzero(y == c)[np.argmin(gaps[y == c])]) for c in cfg['labels']]
    audit_rows = [rows[i] for i in audit_ids]
    write_csv(OUT / 'window-audit.csv', audit_rows, columns)
    np.savez_compressed(OUT / 'test-output.npz', logits=logits, y_true=y, cnn_prediction=cnn,
                        svm_prediction=svm, record_id=record, load_hp=load)
    demo_cm = np.asarray(cfg['teaching_demo_confusion_matrix'], dtype=np.int64)
    demo_y, demo_pred = [], []
    for a in range(4):
        for b in range(4):
            demo_y.extend([a] * int(demo_cm[a, b]))
            demo_pred.extend([b] * int(demo_cm[a, b]))
    demo = {'is_fictional_teaching_data': True,
            **metrics(np.asarray(demo_y), np.asarray(demo_pred), names, cfg['labels'])}
    write_json(OUT / 'teaching-demo-metrics.json', demo)
    for path, expected in hashes.items():
        assert sha256(Path(path)) == expected, f'前课输入或模型被修改：{path}'
    result = {'models': scores, 'per_record': per_record, 'label_names': names,
              'cnn_selected_epoch': meta['selected_epoch'], 'model_hashes': hashes,
              'cnn_test_evaluation_done': True, 'test_split_already_used_in_lesson04': True,
              'evaluation_scope': 'CWRU 0/1 HP train, 2 HP validation, 3 HP test; four records, window-level metrics; not unseen bearing/device validation',
              'checks': {'frozen_before_prediction': True, 'no_parameter_update': True,
                         'original_artifacts_unchanged': True, 'record_groups_disjoint': True,
                         'same_test_windows_for_both_models': True, 'saved_preprocessing_consistent': True,
                         'svm_predictions_match_lesson04': True, 'counts_match_confusion_matrices': True},
              'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                           'numpy': np.__version__, 'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__}}
    write_json(OUT / 'metrics.json', result)
    make_figures(cfg, names, result, raw, audit_rows)
    cnn_m, svm_m = scores['cnn'], scores['svm']
    comparison = '\n'.join(f"| {name} | {m['correct']}/{m['total']} | {m['accuracy']:.2%} | {m['macro_f1']:.6f} | {m['errors']} |"
                           for name, m in [('统计特征＋SVM-C1', svm_m), ('振动序列＋CNN', cnn_m)])
    class_table = '\n'.join(f"| {r['state']} | {r['support']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.6f} |" for r in cnn_m['per_class'])
    record_table = '\n'.join(f"| {r['record_id']} | {r['state']} | {r['windows']} | {r['cnn_errors']} | {r['svm_errors']} |" for r in per_record)
    error_note = ('本次 CNN 没有误判，errors-cnn.csv 只有表头。样本图为每类分数间隔较小的正确窗口，不是错误样本。没有真实误判，不能编造误判原因；教学矩阵另外明确标记为虚构数据。'
                  if cnn_m['errors'] == 0 else f"本次 CNN 有 {cnn_m['errors']} 个误判，全部索引与来源保存在 errors-cnn.csv。样本图优先展示前四个误判，具体原因需结合数据检查，不凭一张图推定物理原因。")
    comparison_note = ('本次两模型 Accuracy 与 Macro-F1 相同；没有依据声称 CNN 提高准确率。'
                       if cnn_m['accuracy'] == svm_m['accuracy'] and cnn_m['macro_f1'] == svm_m['macro_f1']
                       else '本次分数有差异，结论只适用于当前划分；未据此重选 CNN 权重或 SVM 参数。')
    report = f'''# 第七课：冻结模型测试与指标分析结果

本报告来自实际运行；只加载第六课 CNN 和第四课 SVM，没有训练或重新拟合标准化器。先记录模型／数据哈希及评估规则，再对同一批测试窗口预测。

## 1. 测试条件与比较

训练为 0／1 HP、验证为 2 HP、测试为 3 HP。CNN 加载验证选中的第 {meta['selected_epoch']} 轮。测试 184 个窗口来自四条记录，每窗 1024 点、12 kHz。两模型使用相同窗口、标签与类别顺序，沿用各自已保存的预处理。

| 方法 | 正确／总数 | Accuracy | Macro-F1 | 错误数 |
| --- | --- | --- | --- | --- |
{comparison}

{comparison_note} 没有测试推理耗时或资源占用实验，不能由模型文件大小推断实际速度。

## 2. CNN 各类别指标

| 类别 | 实际窗口数 | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{class_table}

TP：这一类预测正确；FP：其他类被预测为这一类；FN：这一类被预测为其他类。多分类指标按每类对其余类别分别计算。

CNN 把故障预测为正常：{cnn_m['normal_vs_fault']['fn']} 个；正常预测为故障：{cnn_m['normal_vs_fault']['fp']} 个；故障类型之间混淆：{cnn_m['normal_vs_fault']['fault_type_confusions']} 个。
合并正常／故障后，故障召回率为 {cnn_m['normal_vs_fault']['fault_recall']:.2%}，正常误报率为 {cnn_m['normal_vs_fault']['normal_false_alarm_rate']:.2%}。这是另一个统计口径，不能替代四类识别指标，也不是现场告警实验。

## 3. 混淆矩阵与记录来源

![两模型实际测试混淆矩阵]({(OUT / '02-test-confusion-matrices.png').as_posix()})

纵轴为实际类别、横轴为预测类别。对角线是正确数，其余格子是误判数。原始计数行和表示实际该类窗口数；列和表示被预测为该类窗口数。

| 记录号 | 类别 | 窗口数 | CNN 错误数 | SVM 错误数 |
| --- | --- | --- | --- | --- |
{record_table}

![测试分数比较]({(OUT / '03-test-comparison.png').as_posix()})

## 4. 样本检查：如实记录有无错误

{error_note}

![实际窗口与两模型预测]({(OUT / '04-window-audit.png').as_posix()})

`window-audit.csv` 记录样本索引、原始文件、记录号、通道及时间范围。图中横轴是记录内时间。CNN 两个最大 logits 的差仅用于选择观察样本，不是校准概率，不能当作已验证的可靠性阈值。检查样本未用于改模型。

## 5. 指标教学示例：与真实测试分开

![虚构教学矩阵与内圈 Precision／Recall]({(OUT / '01-metrics-teaching-example.png').as_posix()})

教学例子内圈 TP=7、FP=2、FN=3，所以 Precision=7/9，Recall=7/10，F1=14/19。40 个例子总体正确 31 个，Accuracy=77.5%；完整各类结果见 teaching-demo-metrics.json。这些数字不是模型实测结果。

## 6. 文件与核对

- [metrics.json]({(OUT / 'metrics.json').as_posix()})：实际指标、类别及记录计数、版本、检查结果。
- [evaluation-frozen.json]({(OUT / 'evaluation-frozen.json').as_posix()})：预测前记录的配置、模型与数据身份。
- [test-predictions.csv]({(OUT / 'test-predictions.csv').as_posix()})：全部窗口真实标签、两模型预测与来源。
- [errors-cnn.csv]({(OUT / 'errors-cnn.csv').as_posix()})、[errors-svm.csv]({(OUT / 'errors-svm.csv').as_posix()})：仅错误窗口；没有错误时只有表头。
- [window-audit.csv]({(OUT / 'window-audit.csv').as_posix()})：图中观察样本与来源。

已核对训练／验证／测试记录不交叉、两模型测试窗口一致、CNN 参数及前课文件未改变、CNN 预处理一致、SVM 逐窗口预测与第四课一致、混淆矩阵和正确数一致。

## 7. 结论范围

这是同一 CWRU 试验台上的跨负载窗口分类。184 个窗口不是 184 次独立实验、184 个独立轴承或 184 台电机。第四课已使用这个测试划分；本课作冻结模型对比，没有把它称为全新的独立数据验证，也未根据测试结果调参。

零误判也不能证明总体错误率为零或现场可用。当前没有新设备、未知故障、真实噪声采集或实际部署验证。第八课可设计预先固定的噪声／工况实验，明确复用数据与人工加噪的边界。
'''
    (OUT / '第七课-测试评估与模型对比结果.md').write_text(report, encoding='utf-8')
    for name, m in [('CNN', cnn_m), ('SVM-C1', svm_m)]:
        print(f"{name}：正确 {m['correct']}/{m['total']}，错误 {m['errors']}，Accuracy={m['accuracy']:.2%}，Macro-F1={m['macro_f1']:.6f}", flush=True)
    print('检查通过：冻结模型、共同测试窗口、预处理一致、SVM 旧结果一致、来源及计数可追溯。', flush=True)
    print('报告：', OUT / '第七课-测试评估与模型对比结果.md', flush=True)


if __name__ == '__main__':
    main()
