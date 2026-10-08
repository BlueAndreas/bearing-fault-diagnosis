"""第六课：固定小型 CNN 配置，完整训练，按验证集选择权重，保存加载。

默认：python -X utf8 scripts/lesson06_train_cnn.py
只加载预测示例：加 --predict-example
本课不加载测试数组，不计算 CNN 测试分数，留给第七课。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lesson06_cnn import SmallCNN1D, load_saved_cnn, predict_raw_windows

OUT = ROOT / 'outputs/lesson06'
LABELS = [0, 1, 2, 3]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_loader(x, y, batch_size, shuffle=False, seed=42):
    dataset = TensorDataset(torch.from_numpy(x.copy()).unsqueeze(1), torch.from_numpy(y.copy()))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0, generator=generator)


def evaluate(model, loader):
    model.eval()
    loss_sum, total = 0.0, 0
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss_sum += float(nn.functional.cross_entropy(logits, y, reduction='sum'))
            total += len(y)
            all_logits.append(logits.numpy())
            all_y.append(y.numpy())
    logits = np.concatenate(all_logits)
    y = np.concatenate(all_y)
    predicted = logits.argmax(axis=1)
    result = {'loss': loss_sum / total, 'accuracy': float(accuracy_score(y, predicted)),
              'macro_f1': float(f1_score(y, predicted, labels=LABELS, average='macro', zero_division=0))}
    if not all(math.isfinite(v) for v in result.values()) or not np.isfinite(logits).all():
        raise ValueError('评估结果包含非有限值。')
    return result, logits, predicted


def make_figures(history, best_epoch, shapes):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.axis('off')
    ax.set(xlim=(0, 12), ylim=(0, 6))
    ax.text(6, 5.65, '卷积：一个小窗口沿时间移动，每次算一个加权和', ha='center', fontsize=18)
    ax.text(6, 4.95, '教学数字：输入 [1, 2, 4, 1, 0]；权重 [1, 0, -1]；步长 1、无填充、偏置 0', ha='center')
    cases = [('[1, 2, 4]', '1×1 + 2×0 + 4×(-1)', '-3'),
             ('[2, 4, 1]', '2×1 + 4×0 + 1×(-1)', '1'),
             ('[4, 1, 0]', '4×1 + 1×0 + 0×(-1)', '4')]
    for i, (window, calculation, output) in enumerate(cases):
        left = .2 + i * 4
        ax.add_patch(FancyBboxPatch((left, 1.9), 3.6, 2.3, boxstyle='round,pad=.08',
                                   facecolor=['#337EBB', '#248D77', '#DBA032'][i], edgecolor='none'))
        ax.text(left + 1.8, 3.68, f'第 {i + 1} 个位置：{window}', ha='center', color='white', fontsize=14)
        ax.text(left + 1.8, 2.95, calculation, ha='center', color='white', fontsize=12)
        ax.text(left + 1.8, 2.25, f'输出 {output}', ha='center', color='white', fontsize=17)
    ax.text(6, 1.1, '输出序列 [-3, 1, 4]；同一组权重在多个位置复用。', ha='center', fontsize=15)
    ax.text(6, .45, '这组权重只用于演示；实际 CNN 的卷积权重由训练更新，不能直接解释为某种物理故障。', ha='center')
    fig.savefig(OUT / '01-convolution-example.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    ax.axis('off')
    ax.set(xlim=(0, 15), ylim=(0, 4))
    cards = [('振动输入', '(B, 1, 1024)'), ('卷积＋ReLU＋池化', '(B, 16, 512)'),
             ('卷积＋ReLU＋池化', '(B, 32, 256)'), ('平均汇总＋展平', '(B, 32)'), ('分类层', '(B, 4)')]
    for i, (title, shape) in enumerate(cards):
        left = .1 + i * 3
        ax.add_patch(FancyBboxPatch((left, 1.1), 2.55, 1.8, boxstyle='round,pad=.06',
                                   facecolor=['#337EBB', '#248D77', '#248D77', '#DBA032', '#C65C75'][i], edgecolor='none'))
        ax.text(left + 1.275, 2.38, title, ha='center', color='white', fontsize=12)
        ax.text(left + 1.275, 1.62, shape, ha='center', color='white', fontsize=14)
        if i < 4:
            ax.annotate('', xy=(left + 2.9, 2), xytext=(left + 2.65, 2),
                        arrowprops={'arrowstyle': '->', 'color': '#526579', 'lw': 2})
    ax.text(7.5, 3.5, '本课 CNN 主线｜B 表示本批样本数', ha='center', fontsize=18)
    ax.text(7.5, .42, '16／32 是学出的特征通道，不是新的传感器；4 是输出类别数。', ha='center')
    fig.savefig(OUT / '02-cnn-architecture.png', dpi=150)
    plt.close(fig)

    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(epochs, [r['train_loss'] for r in history], label='训练集', color='#337EBB')
    axes[0].plot(epochs, [r['val_loss'] for r in history], label='验证集', color='#DBA032')
    axes[0].set(xlabel='训练轮数', ylabel='平均交叉熵 loss', title='每轮结束，用当时权重评估')
    axes[1].plot(epochs, [r['train_macro_f1'] for r in history], label='训练集', color='#337EBB')
    axes[1].plot(epochs, [r['val_macro_f1'] for r in history], label='验证集', color='#DBA032')
    axes[1].set(xlabel='训练轮数', ylabel='Macro-F1', ylim=(0, 1.06), title='保存依据：验证 F1，再比较验证 loss')
    for ax in axes:
        ax.axvline(best_epoch, color='#C65C75', linestyle='--', label=f'保存第 {best_epoch} 轮')
        ax.grid(alpha=.2)
        ax.legend(fontsize=10)
    fig.savefig(OUT / '03-training-curves.png', dpi=150)
    plt.close(fig)


def prediction_example():
    model, metadata = load_saved_cnn(OUT)
    path = ROOT / metadata['dataset_relative_path']
    if file_hash(path) != metadata['dataset_sha256']:
        raise ValueError('演示数据与本次训练来源不一致，请检查数据版本。')
    with np.load(path, allow_pickle=False) as z:
        raw = z['X_val_raw'][:1].copy()
        true = int(z['y_val'][0])
        record = int(z['record_id_val'][0])
    result = predict_raw_windows(model, metadata, raw, metadata['sampling_rate_hz'])
    predicted = int(result['labels'][0])
    example = {'source': 'first window of existing validation split; not a new device',
               'record_id': record, 'true_label': true, 'predicted_label': predicted,
               'true_state': metadata['label_names'][true], 'predicted_state': metadata['label_names'][predicted],
               'logits': result['logits'][0].tolist()}
    write_json(OUT / 'example-prediction.json', example)
    print('仅加载已保存 CNN，不重新训练；示例来自已有验证集。', flush=True)
    print('记录：', record, '；预测：', example['predicted_state'], '；已知：', example['true_state'], flush=True)
    return example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict-example', action='store_true')
    args = parser.parse_args()
    if args.predict_example:
        prediction_example()
        return
    config_path = ROOT / 'configs/lesson06_cnn.json'
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    if cfg['device'] != 'cpu' or cfg['num_workers'] != 0 or cfg['drop_last']:
        raise ValueError('第六课固定 CPU、主进程加载、保留最后一批。')
    if cfg['epochs'] < 1 or cfg['batch_size'] < 1 or cfg['learning_rate'] <= 0:
        raise ValueError('训练轮数、批次和学习率必须为正。')
    path = ROOT / cfg['dataset']
    if not path.is_file():
        raise FileNotFoundError('缺少第三课默认数据，请先运行第三课。')
    source_hash = file_hash(path)
    arrays = {}
    with np.load(path, allow_pickle=False) as z:
        for split in ['train', 'val']:  # 测试数组留到第七课。
            for key in ['X', 'y', 'record_id']:
                arrays[f'{key}_{split}'] = z[f'{key}_{split}'].copy()
    for split in ['train', 'val']:
        x, y = arrays[f'X_{split}'], arrays[f'y_{split}']
        if x.ndim != 2 or x.shape[1] != cfg['window_points'] or len(x) != len(y):
            raise ValueError('数据形状不匹配。')
        if x.dtype != np.float32 or y.dtype != np.int64 or not np.isfinite(x).all():
            raise ValueError('输入类型或数值不符合 PyTorch 训练要求。')
        if set(np.unique(y).tolist()) != set(LABELS):
            raise ValueError('数据需要覆盖四个类别。')
    if set(arrays['record_id_train']) & set(arrays['record_id_val']):
        raise ValueError('训练与验证包含同一原始记录。')
    data_cfg = json.loads((path.parent / 'build-config.json').read_text(encoding='utf-8'))
    if data_cfg['sampling_rate_hz'] != cfg['sampling_rate_hz']:
        raise ValueError('数据采样率与配置不一致。')
    normalizer = json.loads((path.parent / 'normalization.json').read_text(encoding='utf-8'))
    if normalizer['fit_split'] != 'train' or normalizer['std'] <= 0:
        raise ValueError('必须沿用仅从训练窗口拟合的预处理参数。')
    label_map = json.loads((path.parent / 'label-map.json').read_text(encoding='utf-8'))
    names = [label_map[str(i)] for i in LABELS]
    torch.set_num_threads(cfg['torch_threads'])
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    arch = cfg['architecture']
    model = SmallCNN1D(arch['channels'], arch['kernels'], cfg['window_points'])
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    shapes = model.layer_shapes(cfg['batch_size'])
    parameter_count = sum(p.numel() for p in model.parameters())
    train_loader = make_loader(arrays['X_train'], arrays['y_train'], cfg['batch_size'], True, cfg['seed'])
    train_eval_loader = make_loader(arrays['X_train'], arrays['y_train'], cfg['batch_size'])
    val_loader = make_loader(arrays['X_val'], arrays['y_val'], cfg['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'run-config.json', cfg)
    weights_path = OUT / 'best-cnn.pt'
    print(f'CPU 训练：{cfg["epochs"]} 轮，每轮 {len(train_loader)} 批，网络参数 {parameter_count} 个。', flush=True)
    history, best, best_key, best_logits = [], None, None, None
    total_steps = 0
    started = time.perf_counter()
    columns = ['epoch', 'optimizer_steps', 'train_loss', 'train_accuracy', 'train_macro_f1',
               'val_loss', 'val_accuracy', 'val_macro_f1', 'saved_as_best']
    with (OUT / 'training-history.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            seen = 0
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError('训练 loss 无效。')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('训练梯度缺失或无效。')
                optimizer.step()
                total_steps += 1
                seen += len(y)
            if seen != len(arrays['y_train']):
                raise AssertionError('训练一轮未覆盖完整训练集合。')
            train_scores, _, _ = evaluate(model, train_eval_loader)
            val_scores, logits, _ = evaluate(model, val_loader)
            key = (val_scores['macro_f1'], -val_scores['loss'], -epoch)
            saved = best_key is None or key > best_key
            row = {'epoch': epoch, 'optimizer_steps': total_steps,
                   **{f'train_{k}': v for k, v in train_scores.items()},
                   **{f'val_{k}': v for k, v in val_scores.items()}, 'saved_as_best': int(saved)}
            history.append(row)
            if saved:
                torch.save(model.state_dict(), weights_path)
                best, best_key, best_logits = row.copy(), key, logits.copy()
            writer.writerow(row)
            f.flush()
            print(f"第 {epoch:02d}/{cfg['epochs']} 轮｜训练 loss={train_scores['loss']:.4f}"
                  f"｜验证 loss={val_scores['loss']:.4f}，F1={val_scores['macro_f1']:.4f}"
                  f"，Accuracy={val_scores['accuracy']:.2%}" + ('｜保存权重' if saved else ''), flush=True)
    elapsed = time.perf_counter() - started
    if not any(not torch.equal(initial_state[k], v) for k, v in model.state_dict().items()):
        raise AssertionError('CNN 参数没有变化。')
    versions = {'python': platform.python_version(), 'torch': torch.__version__,
                'numpy': np.__version__, 'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__}
    metadata = {'architecture': arch, 'window_points': cfg['window_points'], 'sampling_rate_hz': cfg['sampling_rate_hz'],
                'label_names': names, 'normalization': normalizer, 'weights_file': weights_path.name,
                'weights_sha256': file_hash(weights_path), 'dataset_relative_path': cfg['dataset'],
                'dataset_sha256': source_hash, 'config': cfg, 'config_sha256': file_hash(config_path),
                'selected_epoch': best['epoch'], 'checkpoint_selection': cfg['checkpoint_selection'],
                'selected_validation': {k: best[f'val_{k}'] for k in ['loss', 'accuracy', 'macro_f1']},
                'parameter_count': parameter_count, 'versions': versions, 'test_evaluation_done': False,
                'preprocessing': '12kHz unstandardized aligned window -> saved shared training mean/std -> (B,1,1024)',
                'network_source': 'project teaching implementation; not upstream benchmark reproduction'}
    write_json(OUT / 'model-metadata.json', metadata)
    # 从磁盘重新构建网络，核对全部验证分数，而非仅看一个预测标签。
    loaded, saved_meta = load_saved_cnn(OUT)
    loaded_scores, loaded_logits, loaded_prediction = evaluate(loaded, val_loader)
    if not np.allclose(loaded_logits, best_logits, rtol=1e-6, atol=1e-7):
        raise AssertionError('保存加载后验证 logits 不一致。')
    if not all(math.isclose(loaded_scores[k], metadata['selected_validation'][k], rel_tol=1e-6, abs_tol=1e-7)
               for k in loaded_scores):
        raise AssertionError('保存加载后的指标不一致。')
    with np.load(path, allow_pickle=False) as z:
        raw_prediction = predict_raw_windows(loaded, saved_meta, z['X_val_raw'], cfg['sampling_rate_hz'])
    if not np.allclose(raw_prediction['logits'], loaded_logits, rtol=1e-5, atol=1e-6):
        raise AssertionError('原始窗口预处理后的推理与验证装载不一致。')
    if file_hash(path) != source_hash:
        raise AssertionError('第三课数据被修改。')
    np.savez_compressed(OUT / 'validation-output.npz', logits=loaded_logits, prediction=loaded_prediction,
                        y_true=arrays['y_val'], record_id=arrays['record_id_val'])
    with (OUT / 'validation-predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index_in_split', 'record_id', 'true_label', 'true_state', 'predicted_label', 'predicted_state'])
        for i, predicted in enumerate(loaded_prediction):
            true = int(arrays['y_val'][i])
            writer.writerow([i, int(arrays['record_id_val'][i]), true, names[true], int(predicted), names[int(predicted)]])
    baseline_path = ROOT / 'outputs/lesson04/metrics.json'
    baseline = None
    if baseline_path.is_file():
        old = json.loads(baseline_path.read_text(encoding='utf-8'))
        baseline = next(r for r in old['validation_results'] if r['name'] == old['selected_model'])
    result = {'config': cfg, 'selected_epoch': best['epoch'], 'selected_validation': loaded_scores,
              'train_at_selected_epoch': {k: best[f'train_{k}'] for k in ['loss', 'accuracy', 'macro_f1']},
              'total_optimizer_steps': total_steps, 'epochs_completed': len(history), 'parameter_count': parameter_count,
              'training_seconds': elapsed, 'baseline_validation': baseline, 'layer_shapes': shapes,
              'device': 'cpu', 'versions': versions, 'cnn_test_evaluation_done': False,
              'checks': {'record_groups_disjoint': True, 'train_epoch_coverage_complete': True,
                         'gradients_finite': True, 'parameters_updated': True, 'selected_on_validation_only': True,
                         'saved_logits_match_validation': True, 'raw_window_inference_matches': True,
                         'source_dataset_unchanged': True}}
    write_json(OUT / 'lesson06-summary.json', result)
    make_figures(history, best['epoch'], shapes)
    example = prediction_example()
    shape_table = '\n'.join(f"| {r['layer']} | `{tuple(r['shape'])}` |" for r in shapes)
    baseline_note = (f"| 统计特征 {baseline['name']} | {baseline['accuracy']:.4%} | {baseline['macro_f1']:.6f} |"
                     if baseline else '| 第四课基线 | 尚未找到已有报告 | — |')
    report = f'''# 第六课：一维 CNN 训练与验证结果

已完整训练 CNN，保存并重载最佳验证权重。**本课未评估 CNN 测试集，第七课再评估冻结模型。**

## 1. 本次配置与结果

- 输入：第三课已共享标准化的振动序列，每窗 1024 点、12 kHz；训练 368，验证 184 个。
- 模型：两组 Conv1d + ReLU + MaxPool1d，再平均汇总、展平、四分类；没有 Dropout／BatchNorm。
- 参数量：{parameter_count}；CPU，{cfg['epochs']} 轮，每轮 {len(train_loader)} 批，合计 {total_steps} 次更新。
- Adam 学习率 {cfg['learning_rate']}，weight_decay={cfg['weight_decay']}，固定种子 {cfg['seed']}。
- 保存依据：验证 Macro-F1 优先，同分时验证 loss 更低优先，再同分选更早轮数。
- 选中 **第 {best['epoch']} 轮**：验证 Accuracy **{loaded_scores['accuracy']:.4%}**，Macro-F1 **{loaded_scores['macro_f1']:.6f}**，loss **{loaded_scores['loss']:.6f}**。
- 本次训练循环耗时约 {elapsed:.2f} 秒，不包括 Python 导入和绘图；实际版本见 summary。

这是一套预先固定的教学配置，没有声称架构或参数最优。它是本项目新建的教学实现，不是原参考仓库的复现结果。

## 2. 卷积和网络形状

![小窗口加权示例]({(OUT / '01-convolution-example.png').as_posix()})

![CNN 主线]({(OUT / '02-cnn-architecture.png').as_posix()})

下表用 B={cfg['batch_size']} 实际前向计算获得；最后一批样本数不同，首维也随之变化。

| 层 | 输出形状 |
| --- | --- |
{shape_table}

## 3. 训练曲线

![训练／验证曲线]({(OUT / '03-training-curves.png').as_posix()})

训练与验证曲线均为每轮结束后，使用当时权重在 eval + no_grad 下对对应全体数据计算。loss 按样本数量平均，包含最后不足一批的样本；验证数据不执行 backward 或 step。

## 4. 与已有基线的验证对照

| 方法 | 验证 Accuracy | 验证 Macro-F1 |
| --- | --- | --- |
{baseline_note}
| 直接振动序列 CNN | {loaded_scores['accuracy']:.4%} | {loaded_scores['macro_f1']:.6f} |

这是验证指标对照，不能当作最终测试对比。分数相同不能证明 CNN 更好；分数不同也需要结合完整评估条件解释。

## 5. 模型文件与加载检查

- [best-cnn.pt]({weights_path.as_posix()})：最佳验证轮的 state_dict。
- [model-metadata.json]({(OUT / 'model-metadata.json').as_posix()})：网络结构、标签、采样率、窗口长度、训练标准化参数、来源与权重哈希。
- [training-history.csv]({(OUT / 'training-history.csv').as_posix()})：每轮训练／验证指标与是否保存。
- [validation-predictions.csv]({(OUT / 'validation-predictions.csv').as_posix()})：验证预测与记录号、窗口索引。
- [lesson06-summary.json]({(OUT / 'lesson06-summary.json').as_posix()})：配置、版本、层形状与检查结果。

已核对全部验证 logits 保存重载后保持一致，并核对从未标准化验证窗口应用保存的标准化规则后推理一致。没有重新拟合标准化参数。

加载预测示例来自已有验证集记录 {example['record_id']}，已知状态“{example['true_state']}”，预测“{example['predicted_state']}”。该示例不代表新设备测试。

## 6. 结果范围和下一步

数据仍来自 CWRU 同一试验台，同一故障轴承可能跨负载使用。不同记录划分不是独立新轴承／新设备验证。当前 CNN 分数来自参与权重选择的验证集，不是独立测试成绩。

第七课使用本课已冻结的模型，计算测试指标、混淆矩阵并分析误判。第四课已评估过同一测试划分，后续要如实说明复用条件，不能因反复查看该划分调参而把结果称为全新的独立验证。
'''
    (OUT / '第六课-CNN训练与验证结果.md').write_text(report, encoding='utf-8')
    print('检查通过：完整训练、验证选权重、保存加载 logits 一致、原始窗口推理一致、输入未改。', flush=True)
    print('最佳轮数：', best['epoch'], '；验证结果：', loaded_scores, flush=True)
    print('本课没有 CNN 测试成绩。报告：', OUT / '第六课-CNN训练与验证结果.md', flush=True)


if __name__ == '__main__':
    main()
