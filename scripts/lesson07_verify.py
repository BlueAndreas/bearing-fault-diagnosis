"""独立核对第七课计数、非满分教学案例、来源和一键评估。"""
from pathlib import Path
import csv
import json
import re
import subprocess
import sys

import numpy as np
from lesson07_evaluate import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/lesson07'


def manual_scores(y, predicted):
    cm = np.zeros((4, 4), dtype=np.int64)
    for a, b in zip(y, predicted):
        cm[int(a), int(b)] += 1
    rows = []
    for c in range(4):
        tp = int(cm[c, c])
        fp, fn = int(cm[:, c].sum()) - tp, int(cm[c].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append((tp, fp, fn, precision, recall, f1))
    return cm, float(np.trace(cm) / cm.sum()), float(np.mean([r[-1] for r in rows])), rows


def check_metrics(y, predicted, result):
    cm, accuracy, macro_f1, rows = manual_scores(y, predicted)
    assert np.array_equal(cm, result['confusion_matrix'])
    assert np.isclose(accuracy, result['accuracy']) and np.isclose(macro_f1, result['macro_f1'])
    assert len(y) == result['total'] and int(np.trace(cm)) == result['correct']
    for actual, values in zip(result['per_class'], rows):
        for key, value in zip(['tp', 'fp', 'fn', 'precision', 'recall', 'f1'], values):
            assert np.isclose(actual[key], value), (key, actual, value)


def main():
    summary = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
    names = summary['label_names']
    checks = {}
    with np.load(OUT / 'test-output.npz', allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    with np.load(ROOT / 'data/processed/lesson03/dataset.npz', allow_pickle=False) as z:
        assert np.array_equal(arrays['y_true'], z['y_test'])
        assert np.array_equal(arrays['record_id'], z['record_id_test'])
        assert np.array_equal(arrays['load_hp'], z['load_hp_test'])
    assert np.array_equal(arrays['logits'].argmax(axis=1), arrays['cnn_prediction'])
    for key in ['cnn', 'svm']:
        check_metrics(arrays['y_true'], arrays[f'{key}_prediction'], summary['models'][key])
    checks['independent_real_metric_counts_match'] = True

    # 用非对称、非满分案例检查 FP／FN 的方向，不只检查真实满分矩阵。
    demo = json.loads((OUT / 'teaching-demo-metrics.json').read_text(encoding='utf-8'))
    demo_y, demo_p = [], []
    for a, row in enumerate(demo['confusion_matrix']):
        for b, n in enumerate(row):
            demo_y.extend([a] * n)
            demo_p.extend([b] * n)
    check_metrics(np.asarray(demo_y), np.asarray(demo_p), demo)
    assert demo['is_fictional_teaching_data'] is True
    inner = demo['per_class'][1]
    assert (inner['tp'], inner['fp'], inner['fn']) == (7, 2, 3)
    assert np.isclose(inner['f1'], 14 / 19)
    assert np.isclose(demo['macro_f1'], (0.8 + 14/19 + 18/22 + 14/19) / 4)
    assert demo['normal_vs_fault']['fn'] == 2 and demo['normal_vs_fault']['fp'] == 2
    assert demo['normal_vs_fault']['fault_type_confusions'] == 5
    synthetic_y = np.repeat(np.arange(4), 10)
    all_normal = np.zeros_like(synthetic_y)
    absent_predictions = metrics(synthetic_y, all_normal, names, [0, 1, 2, 3])
    check_metrics(synthetic_y, all_normal, absent_predictions)
    assert absent_predictions['per_class'][1]['precision'] == 0
    assert absent_predictions['normal_vs_fault']['fault_recall'] == 0
    type_confusion = synthetic_y.copy()
    type_confusion[type_confusion == 1] = 2
    type_metrics = metrics(synthetic_y, type_confusion, names, [0, 1, 2, 3])
    check_metrics(synthetic_y, type_confusion, type_metrics)
    assert type_metrics['per_class'][1]['fn'] == 10
    assert type_metrics['normal_vs_fault']['fn'] == 0
    assert type_metrics['normal_vs_fault']['fault_type_confusions'] == 10
    checks['nonperfect_demo_zero_prediction_and_error_definitions_verified'] = True

    predictions = list(csv.DictReader((OUT / 'test-predictions.csv').open(encoding='utf-8-sig')))
    assert len(predictions) == 184
    indices = list(csv.DictReader((ROOT / 'data/processed/lesson03/window-index.csv').open(encoding='utf-8-sig')))
    indices = {int(r['sample_index_in_split']): r for r in indices if r['split'] == 'test'}
    for i, row in enumerate(predictions):
        assert int(row['sample_index_in_split']) == i
        assert int(row['true_label']) == arrays['y_true'][i]
        for key in ['cnn', 'svm']:
            assert int(row[f'{key}_label']) == arrays[f'{key}_prediction'][i]
            assert int(row[f'{key}_correct']) == int(arrays[f'{key}_prediction'][i] == arrays['y_true'][i])
        for key in ['record_id', 'load_hp', 'file', 'channel', 'start_second', 'end_second_exclusive']:
            assert row[key] == indices[i][key]
        assert np.isclose(float(row['end_second_exclusive']) - float(row['start_second']), 1024/12000)
    for key in ['cnn', 'svm']:
        errors = list(csv.DictReader((OUT / f'errors-{key}.csv').open(encoding='utf-8-sig')))
        assert len(errors) == summary['models'][key]['errors']
        assert all(r[f'{key}_correct'] == '0' for r in errors)
    audits = list(csv.DictReader((OUT / 'window-audit.csv').open(encoding='utf-8-sig')))
    if summary['models']['cnn']['errors'] == 0:
        assert len(audits) == 4
        gaps = np.sort(arrays['logits'], axis=1)[:, -1] - np.sort(arrays['logits'], axis=1)[:, -2]
        for c, row in enumerate(audits):
            expected = np.flatnonzero(arrays['y_true'] == c)[np.argmin(gaps[arrays['y_true'] == c])]
            assert int(row['sample_index_in_split']) == expected and row['cnn_correct'] == '1'
    checks['prediction_rows_empty_error_files_and_sample_trace_valid'] = True

    doc_paths = [ROOT / '07-第七课-测试指标与混淆矩阵.md', ROOT / 'docs/第七课-我的学习记录.md',
                 ROOT / 'docs/第七课-面试重点问答.md', OUT / '第七课-测试评估与模型对比结果.md',
                 ROOT / '00-项目学习路线.md']
    for path in doc_paths:
        content = path.read_text(encoding='utf-8')
        assert len(re.findall(r'^```', content, flags=re.M)) % 2 == 0
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
            if re.match(r'^[A-Za-z]:/', target):
                assert Path(target).exists(), (path, target)
    faq = doc_paths[2].read_text(encoding='utf-8')
    questions = re.split(r'^### A\d+\.', faq, flags=re.M)[1:]
    assert len(questions) == 6 and all('**口述参考答案：**' in q for q in questions)
    checks['documents_links_images_and_all_opening_answers_valid'] = True

    if '--check-launcher' in sys.argv:
        proc = subprocess.run(['cmd.exe', '/d', '/c', '运行第七课.bat'], cwd=ROOT,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding='utf-8', errors='replace', timeout=60)
        assert proc.returncode == 0 and '运行成功' in proc.stdout, proc.stdout
        with np.load(OUT / 'test-output.npz', allow_pickle=False) as z:
            for key, expected in arrays.items():
                assert np.array_equal(z[key], expected), key
        checks['one_click_evaluation_and_repeated_outputs_match'] = True
    result = {'checks': checks, 'real_cnn_test_errors': summary['models']['cnn']['errors'],
              'test_split_already_used_in_lesson04': True}
    (OUT / 'verification-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
