"""第五课：Tensor、Dataset、DataLoader 和一次训练参数更新。

默认：python -X utf8 scripts/lesson05_pytorch_basics.py
练习：加 --batch-size 64，输出另存 batch-64。
只使用第三课训练数据。完整遍历用于核对装载，只有第一批执行一次参数更新。
教学网络为 Flatten + Linear，并非 CNN，也不评估分类准确率。
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import platform
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


class VibrationDataset(Dataset):
    """每次取出一段振动及它的状态标签，始终保持配对。"""

    def __init__(self, signals, labels):
        if signals.ndim != 3 or signals.shape[1] != 1 or labels.ndim != 1:
            raise ValueError('需要 signals=(N,1,L)，labels=(N,)。')
        if len(signals) == 0 or len(signals) != len(labels):
            raise ValueError('样本与标签行数不匹配或为空。')
        if signals.dtype != torch.float32 or labels.dtype != torch.long:
            raise ValueError('振动值需要 float32，类别编号需要 long。')
        if not torch.isfinite(signals).all() or not ((labels >= 0) & (labels < 4)).all():
            raise ValueError('振动值必须有限，标签需要为 0～3。')
        self.x = signals
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def make_loader(dataset, batch_size, seed):
    if batch_size < 1:
        raise ValueError('batch_size 必须为正整数。')
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      num_workers=0, generator=generator)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sample_key(x, y):
    # 仅用于检查取样配对和覆盖情况，不输入模型。
    return (x.detach().cpu().contiguous().numpy().tobytes(), int(y))


def figures(out, batch_sizes, total, batch_size, before, after):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 12})
    first_size = batch_sizes[0]
    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    ax.set(xlim=(0, 13), ylim=(0, 4))
    ax.axis('off')
    cards = [
        (0.1, '#337EBB', 'Tensor：装数字', f'振动 ({total}, 1, 1024)\n标签 ({total},)'),
        (3.4, '#248D77', 'Dataset：取一对', '一段振动 (1, 1024)\n一个状态编号'),
        (6.7, '#DBA032', 'DataLoader：打包', f'一批 ({first_size}, 1, 1024)\n对应标签 ({first_size},)'),
        (10.0, '#C65C75', '教学模型：算分数', f'输出 ({first_size}, 4)\n每段得到 4 个类别分数'),
    ]
    for left, color, title, body in cards:
        ax.add_patch(FancyBboxPatch((left, 1.25), 2.8, 1.75, boxstyle='round,pad=0.06',
                                   facecolor=color, edgecolor='none', alpha=.95))
        ax.text(left + 1.4, 2.58, title, ha='center', va='center', color='white', fontsize=14)
        ax.text(left + 1.4, 1.96, body, ha='center', va='center', color='white', fontsize=12)
        if left < 10:
            ax.annotate('', xy=(left + 3.2, 2.1), xytext=(left + 2.9, 2.1),
                        arrowprops={'arrowstyle': '->', 'color': '#526579', 'lw': 2})
    ax.text(6.5, 3.55, '一段一段存好 → 一批一批取出 → 交给模型', ha='center', fontsize=18)
    ax.text(6.5, .55, '中间的 1 是一个振动通道；打包只增加批次维度，样本和答案仍配对。', ha='center')
    fig.savefig(out / '01-data-flow.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    bars = ax.bar(range(1, len(batch_sizes) + 1), batch_sizes, color='#337EBB')
    ax.bar_label(bars, padding=3)
    ax.set(xlabel='第几批', ylabel='本批样本数', xticks=range(1, len(batch_sizes) + 1),
           title=f'{total} 个训练样本｜batch_size={batch_size}｜最后不足一批也保留')
    ax.set_ylim(0, max(batch_sizes) * 1.2)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(out / '02-batch-sizes.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(['参数更新前', '参数更新后'], [before, after], color=['#DBA032', '#248D77'])
    ax.bar_label(bars, labels=[f'{before:.6f}', f'{after:.6f}'], padding=4)
    ax.set(ylabel='同一训练批的交叉熵 loss', title='只演示一次参数更新｜不代表验证／测试效果')
    ax.set_ylim(0, max(before, after) * 1.2)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(out / '03-one-step-loss.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/lesson05_pytorch.json').read_text(encoding='utf-8'))
    batch_size = cfg['batch_size'] if args.batch_size is None else args.batch_size
    if batch_size < 1 or batch_size > 368:
        raise ValueError('本课练习的 batch_size 请设为 1～368。')
    expected_settings = {'device': 'cpu', 'num_workers': 0, 'shuffle_train': True,
                         'drop_last': False, 'demo_optimizer_steps': 1}
    if any(cfg[k] != v for k, v in expected_settings.items()):
        raise ValueError('本课固定使用 CPU、单进程、训练集内打乱、保留最后批次及一步演示。')
    path = ROOT / cfg['dataset']
    if not path.is_file():
        raise FileNotFoundError('缺少第三课数据，请先运行“运行第三课.bat”。')
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    # 不加载验证／测试数组，只取已按训练参数标准化的训练振动。
    with np.load(path, allow_pickle=False) as z:
        signal_array = z['X_train'].copy()
        label_array = z['y_train'].copy()
        record_ids = z['record_id_train'].copy()
    data_cfg = json.loads((path.parent / 'build-config.json').read_text(encoding='utf-8'))
    if signal_array.shape != (368, cfg['window_points']) or cfg['window_points'] != 1024:
        raise ValueError('第五课沿用默认 368×1024 的第三课训练数据。')
    if data_cfg['sampling_rate_hz'] != cfg['sampling_rate_hz']:
        raise ValueError('采样率与第五课配置不一致。')
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    x = torch.from_numpy(signal_array.astype(np.float32, copy=True)).unsqueeze(1)
    y = torch.from_numpy(label_array.astype(np.int64, copy=True))
    if not np.array_equal(x[:, 0].numpy(), signal_array):
        raise AssertionError('添加通道维度后数值改变。')
    dataset = VibrationDataset(x, y)
    loader = make_loader(dataset, batch_size, cfg['seed'])

    expected = Counter(sample_key(*dataset[i]) for i in range(len(dataset)))
    source_index = {sample_key(*dataset[i]): i for i in range(len(dataset))}
    seen = Counter()
    batch_sizes = []
    batch_rows = []
    first_x = first_y = None
    for number, (xb, yb) in enumerate(loader, 1):
        if first_x is None:
            first_x, first_y = xb.clone(), yb.clone()
        if xb.shape != (len(yb), 1, cfg['window_points']):
            raise AssertionError('批次形状不正确。')
        seen.update(sample_key(a, b) for a, b in zip(xb, yb))
        batch_sizes.append(len(yb))
        counts = torch.bincount(yb, minlength=4).tolist()
        batch_rows.append({'batch_number': number, 'samples': len(yb),
                           **{f'label_{i}_count': counts[i] for i in range(4)}})
    if seen != expected or sum(batch_sizes) != len(dataset) or len(loader) != math.ceil(len(dataset) / batch_size):
        raise AssertionError('一次完整取样未覆盖全部训练样本，或振动与标签配对错误。')
    first_indices = np.array([source_index[sample_key(a, b)] for a, b in zip(first_x, first_y)], dtype=np.int64)
    out = ROOT / 'outputs/lesson05'
    if batch_size != cfg['batch_size']:
        out = out / f'batch-{batch_size}'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'batch-index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(batch_rows[0]))
        writer.writeheader()
        writer.writerows(batch_rows)

    # 一个独立的算术例子：旋钮 w=2，希望接近目标 5。
    knob = torch.tensor(2.0, requires_grad=True)
    toy_optimizer = torch.optim.SGD([knob], lr=.1)
    toy_optimizer.zero_grad(set_to_none=True)
    toy_loss = (knob - 5) ** 2
    toy_before = float(toy_loss.detach())
    toy_loss.backward()
    toy_gradient = float(knob.grad)
    toy_optimizer.step()
    toy = {'w_before': 2.0, 'target': 5.0, 'loss_before': toy_before, 'gradient': toy_gradient,
           'learning_rate': .1, 'w_after': float(knob.detach()), 'loss_after': float(((knob - 5) ** 2).detach())}

    # 只演示一次训练 step。这个简单线性网络不是下一课的 1D-CNN。
    torch.manual_seed(cfg['seed'])
    model = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(cfg['window_points'], 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['demo_learning_rate'])
    criterion = nn.CrossEntropyLoss()
    model.train()
    weight_before = model[1].weight.detach().clone()
    bias_before = model[1].bias.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(first_x)
    loss = criterion(logits, first_y)
    loss_before = float(loss.detach())
    loss.backward()
    gradient = model[1].weight.grad.detach().clone()
    bias_gradient = model[1].bias.grad.detach().clone()
    if not torch.isfinite(gradient).all() or not torch.isfinite(bias_gradient).all():
        raise AssertionError('梯度出现非有限值。')
    optimizer.step()
    if torch.equal(weight_before, model[1].weight.detach()):
        raise AssertionError('参数没有更新。')
    model.eval()
    # eval 改变模型模式，本身不会关闭梯度记录。
    eval_still_records_grad = model(first_x).requires_grad
    with torch.no_grad():
        logits_after = model(first_x)
        loss_after = float(criterion(logits_after, first_y))
    if not (math.isfinite(loss_before) and math.isfinite(loss_after)):
        raise AssertionError('loss 出现非有限值。')
    np.savez_compressed(out / 'one-step-demo.npz',
        batch_x=first_x.numpy(), batch_y=first_y.numpy(), source_indices=first_indices,
        logits_before=logits.detach().numpy(), logits_after=logits_after.numpy(),
        weight_before=weight_before.numpy(), weight_after=model[1].weight.detach().numpy(),
        bias_before=bias_before.numpy(), bias_after=model[1].bias.detach().numpy(),
        weight_gradient=gradient.numpy(), bias_gradient=bias_gradient.numpy())
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise AssertionError('第三课数据文件发生改变。')
    sample_x, sample_y = dataset[0]
    versions = {'python': platform.python_version(), 'torch': torch.__version__,
                'numpy': np.__version__, 'matplotlib': matplotlib.__version__}
    runtime = {'python_executable': sys.executable, 'versions': versions,
               'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
               'demo_device': str(first_x.device), 'torch_threads': torch.get_num_threads()}
    summary = {'runtime': runtime, 'source_dataset_sha256': source_hash,
               'config': {**cfg, 'batch_size': batch_size},
               'shapes': {'all_signals': list(x.shape), 'all_labels': list(y.shape),
                          'one_signal': list(sample_x.shape), 'one_label': list(sample_y.shape),
                          'first_batch_x': list(first_x.shape), 'first_batch_y': list(first_y.shape),
                          'logits': list(logits.shape)},
               'dtypes': {'x': str(x.dtype), 'y': str(y.dtype)},
               'batch_sizes': batch_sizes, 'total_samples': len(dataset),
               'first_batch_source_indices': first_indices.tolist(),
               'first_batch_record_ids': record_ids[first_indices].tolist(),
               'toy_autograd': toy,
               'one_step': {'network': 'Flatten + Linear(1024,4); not CNN',
                            'optimizer_steps': 1, 'full_epoch_training': False,
                            'loss_before': loss_before, 'loss_after_same_training_batch': loss_after,
                            'weight_gradient_l2': float(torch.linalg.vector_norm(gradient)),
                            'max_abs_weight_change': float(torch.max(torch.abs(model[1].weight.detach() - weight_before))),
                            'validation_or_test_evaluated': False},
               'mode_demo': {'eval_alone_records_grad': bool(eval_still_records_grad),
                             'no_grad_output_requires_grad': bool(logits_after.requires_grad)},
               'checks': {'tensor_values_preserved': True, 'all_training_samples_seen_once': True,
                          'signal_label_pairs_preserved': True, 'finite_gradients': True,
                          'model_parameters_changed': True, 'source_dataset_unchanged': True}}
    write_json(out / 'lesson05-summary.json', summary)
    figures(out, batch_sizes, len(dataset), batch_size, loss_before, loss_after)
    batch_table = '\n'.join(f"| {r['batch_number']} | {r['samples']} | {r['label_0_count']} | {r['label_1_count']} | {r['label_2_count']} | {r['label_3_count']} |" for r in batch_rows)
    report = f'''# 第五课：PyTorch 基础运行结果

本课用于理解数据装载与一次参数更新。没有训练完整 CNN，也没有验证／测试准确率。

## 1. 实际环境

- Python：{versions['python']}；PyTorch：{versions['torch']}。
- 本次使用：{runtime['demo_device']}；当前 PyTorch CUDA 可用：{runtime['cuda_available']}。
- CUDA 构建版本：{runtime['torch_cuda_build']}。这是当前软件环境信息，不表示电脑没有 NVIDIA 显卡。
- 环境版本与源数据哈希见 [lesson05-summary.json]({(out / 'lesson05-summary.json').as_posix()})。

## 2. 从整体到一批

| 内容 | 形状 | 含义 |
| --- | --- | --- |
| 所有训练振动 | `{tuple(x.shape)}` | 368 个窗口、1 个驱动端通道、每窗 1024 点 |
| 所有训练标签 | `{tuple(y.shape)}` | 每个窗口一个类别编号 |
| Dataset 中一个窗口 | `{tuple(sample_x.shape)}` | 1 个通道、1024 点 |
| 第一批振动 | `{tuple(first_x.shape)}` | 这一批的多个窗口 |
| 第一批标签 | `{tuple(first_y.shape)}` | 每个窗口一个答案 |
| 模型输出 logits | `{tuple(logits.shape)}` | 每个窗口 4 个未归一化类别分数，不是概率 |

振动类型 `{x.dtype}`，标签类型 `{y.dtype}`。新增通道维度没有改变振动数值。

![Tensor 到 Dataset 到 DataLoader]({(out / '01-data-flow.png').as_posix()})

## 3. 训练数据怎样打包

batch_size={batch_size}，训练集内打乱顺序，drop_last=False。完整取样 {len(batch_sizes)} 批，最后一批 {batch_sizes[-1]} 个，总计 368 个。检查每个振动窗口与标签配对，并且每个训练样本恰好取出一次。

| 批号 | 样本数 | 正常 | 内圈 | 外圈 | 滚动体 |
| --- | --- | --- | --- | --- | --- |
{batch_table}

![每批的样本数量]({(out / '02-batch-sizes.png').as_posix()})

这个遍历用于检查数据装载，**没有逐批更新模型**；本课只对第一批演示一次训练 step。

## 4. 一个独立算术例子：自动求导

旋钮 w 从 2 开始，目标为 5，误差设为 `(w-5)²`。更新前 loss={toy_before:.4f}，backward 算出的梯度为 {toy_gradient:.4f}。用学习率 0.1 做 `w = w - 0.1 × 梯度`，得到 w≈{toy['w_after']:.4f}，误差变为 {toy['loss_after']:.4f}。

这个算术例子使用平方误差，后面的四分类网络使用交叉熵，两个演示不要混淆。

## 5. 真实训练批的一次参数更新

教学网络是 Flatten + Linear(1024,4)，不含卷积。对第一批训练振动计算四类分数，CrossEntropyLoss 对照标签算 loss，backward 算梯度，SGD 的 step 更新参数。

- 更新前交叉熵：{loss_before:.6f}。
- 更新后在同一批上的交叉熵：{loss_after:.6f}。
- 权重梯度 L2 范数：{summary['one_step']['weight_gradient_l2']:.6f}。
- 最大绝对权重变化：{summary['one_step']['max_abs_weight_change']:.6f}。

![同一训练批更新前后的 loss]({(out / '03-one-step-loss.png').as_posix()})

这些数用于确认求导与更新流程能运行。训练批 loss 的变化不能证明泛化，也不保证任意学习率下每一步 loss 都下降。

## 6. 模式与检查

- 单独 model.eval() 后，输出仍记录梯度：{eval_still_records_grad}。
- 在 torch.no_grad() 内计算，输出 requires_grad：{logits_after.requires_grad}。
- 本模型只有 Flatten 与 Linear；没有 Dropout／BatchNorm，所以切换模式本身不改变这里的计算规则。
- 已检查新增维度数值不变、样本覆盖完整、标签配对不变、梯度有限、参数更新及源数据哈希不变。

原始 MAT、第三课数据、第四课基线结果和个人学习记录均不由本脚本修改。默认结果可重复运行；其他 batch_size 另存子文件夹。

教学批次与更新快照：[one-step-demo.npz]({(out / 'one-step-demo.npz').as_posix()})；批次数量表：[batch-index.csv]({(out / 'batch-index.csv').as_posix()})。
'''
    (out / '第五课-PyTorch基础运行结果.md').write_text(report, encoding='utf-8')
    print('实际环境：', runtime)
    print('全部训练：X=', tuple(x.shape), 'y=', tuple(y.shape))
    print('一个样本：X=', tuple(sample_x.shape), 'y=', int(sample_y))
    print('第一批：X=', tuple(first_x.shape), 'y=', tuple(first_y.shape), 'logits=', tuple(logits.shape))
    print(f'共 {len(batch_sizes)} 批，最后 {batch_sizes[-1]} 个，总共 {sum(batch_sizes)} 个训练样本。')
    print('自动求导小例子：w=2，梯度=', toy_gradient, '，更新到≈', toy['w_after'])
    print(f'一次 step：loss {loss_before:.6f} -> {loss_after:.6f}（同一训练批）')
    print('检查通过：数据覆盖完整、答案配对、梯度有限、参数更新、第三课数据未修改。')
    print('本课没有 CNN 准确率。报告：', out / '第五课-PyTorch基础运行结果.md')


if __name__ == '__main__':
    main()
