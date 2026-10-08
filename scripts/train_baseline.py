"""基线训练阶段：原始窗口 -> 8 个统计特征 -> 验证选模型 -> 冻结后测试。

运行：python -X utf8 scripts/train_baseline.py
仅加载已保存模型作示例预测：加 --predict-example
原始 MAT 与已构建的数据集不会被修改。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import scipy
from scipy.stats import kurtosis, skew
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/baseline'
PROCESSED = ROOT / 'data/processed/features'
MODEL_PATH = OUT / 'baseline-model.joblib'
FEATURE_NAMES = ['rms', 'std', 'absolute_peak', 'peak_to_peak', 'mean_absolute',
                 'crest_factor', 'skewness', 'pearson_kurtosis']
FEATURE_ZH = ['RMS', '标准差', '绝对峰值', '峰峰值', '平均绝对值', '峰值因子', '偏度', 'Pearson 峭度']
LABELS = [0, 1, 2, 3]
COLORS = ['#337EBB', '#DBA032', '#248D77', '#C65C75']


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_features(windows):
    """输入未标准化窗口，每行一个样本；输出顺序固定的 8 个特征。

    std 使用 ddof=0；偏度与 Pearson 峭度使用 bias=False。
    本程序拒绝常量窗口，避免未定义的峰值因子、偏度和峭度被静默填零。
    """
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError('输入应为有限的二维窗口数组，每行至少 4 点。')
    rms = np.sqrt(np.mean(x * x, axis=1))
    std = np.std(x, axis=1, ddof=0)
    peak = np.max(np.abs(x), axis=1)
    if np.any(rms <= 0) or np.any(std <= 0):
        raise ValueError('遇到全零或常量窗口：请先检查传感器、数据与切片。')
    features = np.column_stack([
        rms, std, peak, np.ptp(x, axis=1), np.mean(np.abs(x), axis=1), peak / rms,
        skew(x, axis=1, bias=False), kurtosis(x, axis=1, fisher=False, bias=False),
    ])
    if not np.isfinite(features).all():
        raise ValueError('特征中存在非有限值。')
    return features


def make_model(candidate, seed):
    if candidate['family'] == 'svm':
        # 每一列特征独立缩放；fit 仅接收训练特征。
        return Pipeline([('scaler', StandardScaler()),
                         ('classifier', SVC(**candidate['parameters'], random_state=seed))])
    if candidate['family'] == 'random_forest':
        # 树按特征阈值分支，本程序不为随机森林添加标准化。
        return Pipeline([('classifier', RandomForestClassifier(
            **candidate['parameters'], random_state=seed, n_jobs=1))])
    raise ValueError(f"未知模型类型：{candidate['family']}")


def scores(y, prediction):
    return {'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, labels=LABELS, average='macro', zero_division=0))}


def predict_windows(bundle, windows, sampling_rate_hz):
    """预处理完成的原始幅值窗口 -> 特征 -> 已保存的模型；不重新 fit。"""
    x = np.asarray(windows)
    if bundle['feature_names'] != FEATURE_NAMES:
        raise ValueError('模型的特征顺序与当前代码不一致。')
    if sampling_rate_hz != bundle['sampling_rate_hz']:
        raise ValueError('采样率不匹配，请先带抗混叠处理重采样。')
    if x.ndim != 2 or x.shape[1] != bundle['window_points']:
        raise ValueError('窗口形状不匹配。')
    return bundle['pipeline'].predict(extract_features(x))


def load_inputs(cfg):
    path = ROOT / cfg['dataset']
    if not path.is_file():
        raise FileNotFoundError('缺少数据构建阶段数据，请先运行 scripts/build_dataset.py。')
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for s in ['train', 'val', 'test']
                  for k in [f'X_{s}_raw', f'y_{s}', f'record_id_{s}', f'load_hp_{s}']}
    mapping = json.loads((ROOT / cfg['label_map']).read_text(encoding='utf-8'))
    names = [mapping[str(i)] for i in LABELS]
    groups = {}
    for s in ['train', 'val', 'test']:
        x, y = arrays[f'X_{s}_raw'], arrays[f'y_{s}']
        if x.ndim != 2 or x.shape[1] != cfg['window_points'] or len(x) != len(y):
            raise ValueError(f'{s} 数组形状不符合基线训练阶段配置。')
        if set(np.unique(y).tolist()) != set(LABELS):
            raise ValueError(f'{s} 缺少四类标签。')
        if len(arrays[f'record_id_{s}']) != len(y) or len(arrays[f'load_hp_{s}']) != len(y):
            raise ValueError(f'{s} 来源信息行数不匹配。')
        groups[s] = set(arrays[f'record_id_{s}'].tolist())
    if any(groups[a] & groups[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('发现跨集合的原始记录号。')
    build_cfg = json.loads((path.parent / 'build-config.json').read_text(encoding='utf-8'))
    if build_cfg['sampling_rate_hz'] != cfg['sampling_rate_hz']:
        raise ValueError('数据构建阶段数据采样率与基线训练阶段配置不匹配。')
    return path, arrays, names


def example_prediction(cfg):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('请先运行基线训练阶段训练，生成 baseline-model.joblib。')
    _, arrays, names = load_inputs(cfg)
    bundle = joblib.load(MODEL_PATH)
    # 使用既有验证集演示加载与推理，不声称这是新设备数据。
    prediction = int(predict_windows(bundle, arrays['X_val_raw'][:1], cfg['sampling_rate_hz'])[0])
    print('只加载本程序保存的模型，不重新训练。')
    print('示例来自验证集第一个窗口，记录号：', int(arrays['record_id_val'][0]))
    print('预测：', names[prediction], '；已知标签：', names[int(arrays['y_val'][0])])


def make_figures(features, arrays, names, results, matrix, chosen_name):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 12})
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for label, name, color in zip(LABELS, names, COLORS):
        rows = arrays['y_train'] == label
        ax.scatter(features['train'][rows, 0], features['train'][rows, 7],
                   label=name, c=color, s=28, alpha=.7, edgecolors='none')
    ax.set(xlabel='RMS（未标准化窗口）', ylabel='Pearson 峭度（无量纲）',
           title='每个点是一段训练振动窗口｜只画 8 个特征中的 2 个')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(OUT / '01-feature-scatter.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    bars = ax.bar([r['name'] for r in results], [r['macro_f1'] for r in results], color=COLORS)
    ax.bar_label(bars, labels=[f"{r['macro_f1']:.4f}" for r in results], padding=4)
    ax.set(ylim=(0, 1.12), ylabel='验证集 Macro-F1（越大越好）',
           title='只用验证集比较 4 个预设方案；测试集不参与选择')
    ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT / '02-validation-comparison.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, int(matrix.max())))
    ax.set(xticks=range(4), yticks=range(4), xticklabels=names, yticklabels=names,
           xlabel='模型预测的类别', ylabel='实际类别', title=f'冻结方案 {chosen_name} 的测试混淆矩阵')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=18,
                    color='white' if matrix[i, j] > matrix.max() / 2 else '#233244')
    fig.colorbar(im, ax=ax, label='窗口数量')
    fig.savefig(OUT / '03-test-confusion-matrix.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict-example', action='store_true')
    args = parser.parse_args()
    cfg_path = ROOT / 'configs/baseline.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    if cfg['feature_names'] != FEATURE_NAMES:
        raise ValueError('配置的特征顺序与程序不一致。')
    if args.predict_example:
        example_prediction(cfg)
        return
    path, arrays, names = load_inputs(cfg)
    input_hash = sha256(path)
    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    # 固定特征公式逐窗口计算，没有从验证/测试拟合参数。
    features = {s: extract_features(arrays[f'X_{s}_raw']) for s in ['train', 'val', 'test']}
    np.savez_compressed(PROCESSED / 'features.npz',
                        **{f'F_{s}': features[s] for s in features},
                        **{k: v for k, v in arrays.items() if not k.startswith('X_')})
    with (PROCESSED / 'features-per-window.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'sample_index_in_split', 'record_id', 'load_hp', 'label', 'state'] + FEATURE_NAMES)
        for s in features:
            for i, values in enumerate(features[s]):
                label = int(arrays[f'y_{s}'][i])
                writer.writerow([s, i, int(arrays[f'record_id_{s}'][i]),
                                 int(arrays[f'load_hp_{s}'][i]), label, names[label]] + values.tolist())
    results, fitted, val_predictions = [], [], []
    print('窗口 -> 特征：', {s: features[s].shape for s in features})
    for candidate in cfg['candidates']:
        model = make_model(candidate, cfg['random_seed'])
        model.fit(features['train'], arrays['y_train'])
        prediction = model.predict(features['val'])
        row = {'name': candidate['name'], **scores(arrays['y_val'], prediction)}
        results.append(row)
        fitted.append(model)
        val_predictions.append(prediction)
        print(f"验证 {row['name']}：Accuracy={row['accuracy']:.4f}，Macro-F1={row['macro_f1']:.4f}")
    # 唯一选择依据是验证集；完全同分优先配置中排在前面的方案。
    selected = max(range(len(results)), key=lambda i: (results[i]['macro_f1'], results[i]['accuracy'], -i))
    candidate, model = cfg['candidates'][selected], fitted[selected]
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'sklearn': sklearn.__version__, 'joblib': joblib.__version__, 'matplotlib': matplotlib.__version__}
    frozen = {'chosen_candidate': candidate, 'selection_rule': cfg['selection_rule'],
              'validation_results': results, 'config': cfg, 'config_sha256': sha256(cfg_path),
              'dataset_sha256': input_hash, 'fitted_on': 'train only; no train+val refit',
              'feature_input': 'X_*_raw: unstandardized aligned 1024-point windows',
              'feature_formulas': {'std_ddof': 0, 'skew_bias': False,
                                   'kurtosis_fisher': False, 'kurtosis_bias': False},
              'versions': versions, 'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
    # 在任何测试预测之前写出冻结方案并保存训练完成的模型。
    write_json(OUT / 'selection-frozen.json', frozen)
    bundle = {'pipeline': model, 'feature_names': FEATURE_NAMES, 'label_names': names,
              'sampling_rate_hz': cfg['sampling_rate_hz'], 'window_points': cfg['window_points'],
              'metadata': frozen}
    joblib.dump(bundle, MODEL_PATH)
    loaded = joblib.load(MODEL_PATH)
    reload_prediction = loaded['pipeline'].predict(features['val'])
    if not np.array_equal(reload_prediction, val_predictions[selected]):
        raise AssertionError('模型保存与加载后的验证预测不一致。')
    if 'scaler' in loaded['pipeline'].named_steps:
        scaler = loaded['pipeline'].named_steps['scaler']
        if not np.allclose(scaler.mean_, features['train'].mean(axis=0), rtol=1e-12, atol=1e-12):
            raise AssertionError('特征标准化均值不等于训练特征的列均值。')
        if int(scaler.n_samples_seen_) != len(features['train']):
            raise AssertionError('特征标准化使用了非训练样本。')
    # 所有候选方案已经冻结：仅选中模型在此处预测测试集。
    test_prediction = loaded['pipeline'].predict(features['test'])
    metrics = scores(arrays['y_test'], test_prediction)
    matrix = confusion_matrix(arrays['y_test'], test_prediction, labels=LABELS)
    detail = classification_report(arrays['y_test'], test_prediction, labels=LABELS,
                                   target_names=names, output_dict=True, zero_division=0)
    correct = int(np.count_nonzero(test_prediction == arrays['y_test']))
    total = len(test_prediction)
    if int(matrix.sum()) != total or int(matrix.trace()) != correct:
        raise AssertionError('混淆矩阵与正确数量不一致。')
    if sha256(path) != input_hash:
        raise AssertionError('数据构建阶段输入文件发生改变。')
    with (OUT / 'test-predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index_in_split', 'record_id', 'load_hp', 'true_label', 'true_state',
                         'predicted_label', 'predicted_state', 'correct'])
        for i, pred in enumerate(test_prediction):
            true = int(arrays['y_test'][i])
            writer.writerow([i, int(arrays['record_id_test'][i]), int(arrays['load_hp_test'][i]),
                             true, names[true], int(pred), names[int(pred)], int(pred == true)])
    with (OUT / 'validation-scores.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'accuracy', 'macro_f1'])
        writer.writeheader()
        writer.writerows(results)
    checks = {'source_dataset_unchanged': True, 'record_groups_disjoint': True,
              'model_fit_on_train_only': True, 'model_selected_on_validation_only': True,
              'saved_model_reload_matches_validation': True, 'confusion_matrix_matches_counts': True}
    result = {'selected_model': candidate['name'], 'validation_results': results,
              'test': {**metrics, 'correct': correct, 'total': total, 'confusion_matrix': matrix.tolist(),
                       'classification_report': detail}, 'checks': checks,
              'shapes': {s: list(features[s].shape) for s in features}, 'versions': versions,
              'evaluation_scope': 'window classification on held-out load 3 HP on CWRU; not unseen-bearing or unseen-device validation'}
    write_json(OUT / 'metrics.json', result)
    make_figures(features, arrays, names, results, matrix, candidate['name'])
    validation_table = '\n'.join(f"| {r['name']} | {r['accuracy']:.4%} | {r['macro_f1']:.6f} |" for r in results)
    per_class_table = '\n'.join(
        f"| {name} | {int(detail[name]['support'])} | {detail[name]['precision']:.4%} | {detail[name]['recall']:.4%} | {detail[name]['f1-score']:.6f} |"
        for name in names)
    images = '\n\n'.join(f'![{title}]({(OUT / name).as_posix()})' for name, title in [
        ('01-feature-scatter.png', '训练窗口的 RMS 与峭度'),
        ('02-validation-comparison.png', '四个预设模型的验证表现'),
        ('03-test-confusion-matrix.png', '选中模型的测试混淆矩阵')])
    wrong_indices = np.flatnonzero(test_prediction != arrays['y_test'])
    if len(wrong_indices):
        i = int(wrong_indices[0])
        wrong_note = (f"第一个错误样本的测试索引为 {i}，来自记录 {int(arrays['record_id_test'][i])}；"
                      f"实际为{names[int(arrays['y_test'][i])]}，预测为{names[int(test_prediction[i])]}。"
                      '可用 test-predictions.csv 对照数据构建阶段 window-index.csv 追溯。')
    else:
        wrong_note = '本次测试窗口没有误判。这个结果只对应当前小规模划分，不能推出所有设备或所有工况都能正确诊断。'
    report = f'''# 基线训练阶段：统计特征分类基线运行结果

本报告由实际运行生成。已训练分类器；模型输入是每个振动窗口的 8 个统计特征。

## 1. 本次流程

- 沿用数据构建阶段默认数据：训练 368、验证 184、测试 184 个窗口，每窗 1024 点、12 kHz。
- 使用 X_*_raw 提取 RMS、标准差、绝对峰值、峰峰值、平均绝对值、峰值因子、偏度、Pearson 峭度。
- 训练输入从 `(368, 1024)` 变成 `(368, 8)`；标签不变。
- SVM 的各列特征由训练特征拟合 StandardScaler；随机森林不缩放特征。
- 4 个预设候选方案均只用训练集拟合，以验证 Macro-F1 选模型，再以验证 Accuracy 和固定顺序打破同分。
- 同分优先不代表优先方案理论上更强。候选配置见 [baseline.json]({cfg_path.as_posix()})。
- 选中后不合并训练与验证重训，只评估选中模型的测试表现。

## 2. 验证比较：这里用于选方案

| 预设方案 | 验证 Accuracy | 验证 Macro-F1 |
| --- | --- | --- |
{validation_table}

冻结方案：**{candidate['name']}**。选择依据与参数先保存至 [selection-frozen.json]({(OUT / 'selection-frozen.json').as_posix()})，然后才预测测试集。

## 3. 测试结果：这里不用于调参

- 正确 **{correct} / {total}** 个窗口，错误 **{total - correct}** 个。
- 测试 Accuracy：**{metrics['accuracy']:.4%}**。
- 测试 Macro-F1：**{metrics['macro_f1']:.6f}**。

| 实际类别 | 样本数 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
{per_class_table}

{wrong_note}

## 4. 三张图

{images}

混淆矩阵：纵轴是实际类别，横轴是预测类别；对角线为正确数，其他格子为误判数。每一行共 46 个测试窗口。

## 5. 保存与检查

- [baseline-model.joblib]({MODEL_PATH.as_posix()})：模型、特征顺序、标签映射与输入要求；SVM 还包含训练特征缩放器。
- [features-per-window.csv]({(PROCESSED / 'features-per-window.csv').as_posix()})：每个窗口的 8 个特征与来源。
- [test-predictions.csv]({(OUT / 'test-predictions.csv').as_posix()})：逐窗口真实标签与预测，包含记录号和索引。
- [metrics.json]({(OUT / 'metrics.json').as_posix()})：完整指标、混淆矩阵、实际依赖版本和检查结果。

已检查原始记录组不交叉、数据构建阶段数据哈希未改变、选中模型保存加载后验证预测一致，以及混淆矩阵与正确数一致。

## 6. 结果范围

这是 CWRU 公开数据在 0/1 HP 训练、2 HP 验证、3 HP 测试下的窗口分类结果。测试只有四条记录，同一轴承可能跨负载重复使用；184 个测试窗口不是 184 台独立电机。不代表新轴承、新设备或现场运行效果，也不是工业故障严重程度评分。

基线提供后续 CNN 比较的参照。不要根据这次测试分数继续挑参数；未来改变方案或扩展数据时，应重新明确评估设计。

实际版本：{json.dumps(versions, ensure_ascii=False)}。
'''
    (OUT / 'baseline-report.md').write_text(report, encoding='utf-8')
    print('冻结方案：', candidate['name'])
    print(f"测试：正确 {correct}/{total}，Accuracy={metrics['accuracy']:.4%}，Macro-F1={metrics['macro_f1']:.6f}")
    print('检查通过：训练拟合、验证选择、冻结后测试、保存加载一致、输入数据未修改。')
    print('结果报告：', OUT / 'baseline-report.md')


if __name__ == '__main__':
    main()
