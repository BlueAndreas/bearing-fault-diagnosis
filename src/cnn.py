"""本项目的原型小型一维 CNN；不是原参考仓库训练结果的复现。"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn


class SmallCNN1D(nn.Module):
    def __init__(self, channels=(16, 32), kernels=(9, 5), window_points=1024):
        super().__init__()
        if len(channels) != 2 or len(kernels) != 2 or any(c < 1 for c in channels):
            raise ValueError('本网络需要两组正整数通道数和卷积核。')
        if any(k < 1 or k % 2 == 0 for k in kernels) or window_points < 4:
            raise ValueError('卷积核需为正奇数，窗口长度至少为 4。')
        self.window_points = window_points
        c1, c2 = channels
        k1, k2 = kernels
        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(1, c1, k1, padding=k1 // 2)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool1d(2)),
            ('conv2', nn.Conv1d(c1, c2, k2, padding=k2 // 2)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool1d(2)),
            ('average', nn.AdaptiveAvgPool1d(1)),
            ('flatten', nn.Flatten(start_dim=1)),
            ('classifier', nn.Linear(c2, 4)),
        ]))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != self.window_points:
            raise ValueError(f'模型要求 (B,1,{self.window_points}) 的输入。')
        return self.net(x)

    def layer_shapes(self, batch_size=32):
        rows = [{'layer': 'input', 'shape': [batch_size, 1, self.window_points]}]
        x = torch.zeros(batch_size, 1, self.window_points)
        with torch.no_grad():
            for name, layer in self.net.named_children():
                x = layer(x)
                rows.append({'layer': name, 'shape': list(x.shape)})
        return rows


def load_saved_cnn(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'model-metadata.json').read_text(encoding='utf-8'))
    weight_path = folder / metadata['weights_file']
    if hashlib.sha256(weight_path.read_bytes()).hexdigest() != metadata['weights_sha256']:
        raise ValueError('权重文件与元数据校验值不一致。')
    arch = metadata['architecture']
    model = SmallCNN1D(arch['channels'], arch['kernels'], metadata['window_points'])
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    model.eval()
    return model, metadata


def predict_raw_windows(model, metadata, windows, sampling_rate_hz):
    """输入已对齐采样率、未标准化的窗口；沿用已保存的训练参数。"""
    raw = np.asarray(windows, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2 or len(raw) == 0 or raw.shape[1] != metadata['window_points']:
        raise ValueError('输入需要一段或多段固定长度的振动窗口。')
    if sampling_rate_hz != metadata['sampling_rate_hz']:
        raise ValueError('采样率不匹配，应先执行带抗混叠处理的重采样。')
    if not np.isfinite(raw).all() or np.any(np.std(raw, axis=1) == 0):
        raise ValueError('存在无效或常量窗口，请先检查输入。')
    scaler = metadata['normalization']
    if not np.isfinite(scaler['std']) or scaler['std'] <= 0:
        raise ValueError('保存的训练标准差无效。')
    normalized = ((raw - scaler['mean']) / scaler['std']).astype(np.float32)
    x = torch.from_numpy(normalized).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        logits = model(x).numpy()
    if not np.isfinite(logits).all():
        raise ValueError('模型输出出现非有限值。')
    return {'logits': logits, 'labels': logits.argmax(axis=1)}
