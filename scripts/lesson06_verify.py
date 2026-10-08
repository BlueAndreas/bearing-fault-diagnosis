"""第六课独立核对：只检查训练／验证产物，不评估 CNN 测试集。"""
from pathlib import Path
import csv
import hashlib
import json
import re
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/lesson06'
sys.path.insert(0, str(ROOT / 'src'))
from lesson06_cnn import SmallCNN1D, load_saved_cnn, predict_raw_windows


def main():
    torch.set_num_threads(2)
    model, meta = load_saved_cnn(OUT)
    summary = json.loads((OUT / 'lesson06-summary.json').read_text(encoding='utf-8'))
    checks = {}
    with np.load(ROOT / meta['dataset_relative_path'], allow_pickle=False) as z:
        x, raw, y = z['X_val'], z['X_val_raw'], z['y_val']
        train_records, val_records = z['record_id_train'], z['record_id_val']
    with np.load(OUT / 'validation-output.npz', allow_pickle=False) as z:
        saved_logits, saved_labels = z['logits'], z['prediction']
        assert np.array_equal(y, z['y_true'])
    checks['train_val_records_disjoint'] = not bool(set(train_records) & set(val_records))
    assert checks['train_val_records_disjoint']
    assert meta['normalization']['fit_split'] == 'train'
    assert meta['normalization']['fitted_scalar_count'] == 368 * 1024
    assert hashlib.sha256((ROOT / meta['dataset_relative_path']).read_bytes()).hexdigest() == meta['dataset_sha256']
    normalized = ((raw.astype(np.float64) - meta['normalization']['mean']) / meta['normalization']['std']).astype(np.float32)
    np.testing.assert_allclose(normalized, x, rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).unsqueeze(1)).numpy()
        for b in [1, 16, 24, 32]:
            assert model(torch.from_numpy(x[:b]).unsqueeze(1)).shape == (b, 4)
    np.testing.assert_allclose(logits, saved_logits, rtol=1e-5, atol=1e-6)
    raw_result = predict_raw_windows(model, meta, raw, 12000)
    np.testing.assert_allclose(raw_result['logits'], saved_logits, rtol=1e-5, atol=1e-6)
    assert np.array_equal(saved_labels, logits.argmax(axis=1))
    checks['saved_logits_and_raw_preprocessing_match'] = True

    # 独立用 NumPy 稳定 log-sum-exp 核对平均交叉熵和四类 F1。
    scores = saved_logits.astype(np.float64)
    shifted = scores - scores.max(axis=1, keepdims=True)
    loss = (np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]).mean()
    accuracy = float(np.mean(saved_labels == y))
    per_class_f1 = []
    for c in range(4):
        tp = np.sum((saved_labels == c) & (y == c))
        fp = np.sum((saved_labels == c) & (y != c))
        fn = np.sum((saved_labels != c) & (y == c))
        per_class_f1.append(float(2 * tp / (2 * tp + fp + fn)))
    independent = {'loss': float(loss), 'accuracy': accuracy, 'macro_f1': float(np.mean(per_class_f1))}
    for name, value in independent.items():
        assert np.isclose(value, summary['selected_validation'][name], rtol=1e-5, atol=1e-7)
    rows = list(csv.DictReader((OUT / 'training-history.csv').open(encoding='utf-8-sig')))
    best = max(rows, key=lambda r: (float(r['val_macro_f1']), -float(r['val_loss']), -int(r['epoch'])))
    assert int(best['epoch']) == meta['selected_epoch'] == summary['selected_epoch']
    assert len(rows) == 30 and int(rows[-1]['optimizer_steps']) == 360
    checks['metrics_and_checkpoint_rule_independently_verified'] = True
    assert sum(p.numel() for p in model.parameters()) == 2884
    assert model.layer_shapes() == summary['layer_shapes']
    torch.manual_seed(meta['config']['seed'])
    initial = SmallCNN1D().state_dict()
    assert not torch.equal(initial['net.conv1.weight'], model.state_dict()['net.conv1.weight'])
    assert not torch.equal(initial['net.conv2.weight'], model.state_dict()['net.conv2.weight'])
    example = F.conv1d(torch.tensor([[[1., 2., 4., 1., 0.]]]), torch.tensor([[[1., 0., -1.]]]))
    assert torch.equal(example, torch.tensor([[[-3., 1., 4.]]]))
    checks['shapes_convolution_example_and_learned_kernels_verified'] = True
    for invalid, rate in [(raw[0, :-1], 12000), (raw[0], 48000), (np.zeros(1024), 12000), (np.full(1024, np.nan), 12000)]:
        try:
            predict_raw_windows(model, meta, invalid, rate)
        except ValueError:
            continue
        raise AssertionError('无效输入未被拒绝。')
    checks['invalid_inputs_rejected'] = True
    assert summary['cnn_test_evaluation_done'] is False and meta['test_evaluation_done'] is False

    docs = [ROOT / '06-第六课-一维CNN训练与模型保存.md',
            ROOT / 'docs/第六课-我的学习记录.md', ROOT / 'docs/第六课-面试重点问答.md',
            ROOT / '00-项目学习路线.md', OUT / '第六课-CNN训练与验证结果.md']
    for path in docs:
        content = path.read_text(encoding='utf-8')
        assert len(re.findall(r'^```', content, flags=re.M)) % 2 == 0, path
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
            if re.match(r'^[A-Za-z]:/', target):
                assert Path(target).exists(), (path, target)
    checks['markdown_links_images_and_fences_valid'] = True

    if '--check-launcher' in sys.argv:
        before_weights = {k: v.clone() for k, v in model.state_dict().items()}
        proc = subprocess.run(['cmd.exe', '/d', '/c', '运行第六课.bat'], cwd=ROOT,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding='utf-8', errors='replace', timeout=90)
        assert proc.returncode == 0 and '运行成功' in proc.stdout, proc.stdout
        reloaded, _ = load_saved_cnn(OUT)
        for name, value in reloaded.state_dict().items():
            assert torch.equal(before_weights[name], value), name
        checks['one_click_launcher_and_same_environment_reproducibility_verified'] = True
    proc = subprocess.run([sys.executable, '-X', 'utf8', 'scripts/lesson06_train_cnn.py', '--predict-example'],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert proc.returncode == 0 and '仅加载已保存 CNN' in proc.stdout, proc.stdout + proc.stderr
    checks['load_only_prediction_command_verified'] = True
    result = {'checks': checks, 'independent_validation_metrics': independent, 'cnn_test_evaluation_done': False}
    (OUT / 'verification-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
