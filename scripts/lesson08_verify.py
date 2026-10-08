"""第八课独立核对：功率定标、全部计数、成对输入、样本来源与一键运行。"""
from pathlib import Path
import csv
import hashlib
import json
import re
import subprocess
import sys

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/lesson08'
sys.path.insert(0, str(ROOT / 'src'))
from lesson08_noise import make_unit_noise, add_noise_at_snr
from lesson06_cnn import load_saved_cnn, predict_raw_windows
from lesson04_train_baseline import predict_windows
from lesson07_verify import check_metrics


def main():
    torch.set_num_threads(2)
    summary = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
    cfg = summary['config']
    checks = {}
    with np.load(ROOT / cfg['dataset'], allow_pickle=False) as z:
        raw, y = z['X_test_raw'].astype(np.float64), z['y_test'].copy()
    raw_snapshot = raw.copy()
    with np.load(OUT / 'noise-bank.npz', allow_pickle=False) as z:
        bank = {seed: z[f'unit_noise_seed_{seed}'].copy() for seed in cfg['noise_seeds']}
    with np.load(OUT / 'trial-output.npz', allow_pickle=False) as z:
        output = {k: z[k].copy() for k in z.files}
    assert output['logits'].shape == (16, 184, 4)
    assert output['cnn_prediction'].shape == output['svm_prediction'].shape == (16, 184)
    assert np.array_equal(output['y_true'], y)
    assert np.array_equal(output['logits'].argmax(axis=2), output['cnn_prediction'])
    for seed, unit in bank.items():
        assert np.array_equal(unit, make_unit_noise(raw, seed))
        np.testing.assert_allclose(np.mean(unit * unit, axis=1), 1.0, rtol=0, atol=1e-12)
    assert not np.array_equal(bank[2026], bank[2027])
    signal_power = np.mean(raw * raw, axis=1)
    for trial in summary['trials']:
        i, snr, seed = trial['trial_id'], trial['snr_db'], trial['seed']
        # 独立用功率比反推噪声尺度，不调用实验加噪函数。
        if snr is None:
            noisy = raw
        else:
            noise_power = signal_power / (10.0 ** (snr / 10.0))
            noisy = raw + bank[seed] * np.sqrt(noise_power)[:, None]
            actual_power = np.mean((noisy - raw) ** 2, axis=1)
            measured = 10 * np.log10(signal_power / actual_power)
            np.testing.assert_allclose(measured, snr, rtol=0, atol=1e-8)
            # 两种代数等价的浮点计算可能有微小差别，输入哈希按实验规则重建。
            generated, detail = add_noise_at_snr(raw, bank[seed], snr)
            np.testing.assert_allclose(generated, noisy, rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(detail['measured_snr_db'], snr, rtol=0, atol=1e-8)
            noisy = generated
        assert hashlib.sha256(noisy.tobytes()).hexdigest() == trial['input_sha256']
        for model in ['cnn', 'svm']:
            check_metrics(y, output[f'{model}_prediction'][i], trial['models'][model])
    assert np.array_equal(raw, raw_snapshot)
    with np.load(ROOT / 'outputs/lesson07/test-output.npz', allow_pickle=False) as z:
        assert np.array_equal(z['cnn_prediction'], output['cnn_prediction'][0])
        assert np.array_equal(z['svm_prediction'], output['svm_prediction'][0])
    checks['all_target_snr_template_reproducibility_and_original_inputs_verified'] = True
    checks['all_16_trial_metrics_independently_counted'] = True

    for row in summary['aggregate']:
        chosen = [t for t in summary['trials'] if t['condition'] == row['condition']]
        assert row['noise_draws'] == len(chosen) and row['windows_per_draw'] == 184
        for name in ['accuracy', 'macro_f1']:
            values = np.array([t['models'][row['model']][name] for t in chosen])
            assert np.isclose(row[f'{name}_mean'], values.sum() / len(values))
            assert np.isclose(row[f'{name}_std'], np.sqrt(np.mean((values - values.mean())**2)))
    checks['aggregation_mean_population_std_and_sample_counts_verified'] = True

    case_id = summary['detailed_case_trial_id']
    case = summary['trials'][case_id]
    noisy, _ = add_noise_at_snr(raw, bank[case['seed']], case['snr_db'])
    with np.load(OUT / 'detailed-case.npz', allow_pickle=False) as z:
        assert np.array_equal(z['noisy'], noisy) and np.array_equal(z['noise'], noisy - raw)
    model, meta = load_saved_cnn(ROOT / cfg['cnn_folder'])
    cnn = predict_raw_windows(model, meta, noisy, cfg['sampling_rate_hz'])
    np.testing.assert_allclose(cnn['logits'], output['logits'][case_id], rtol=1e-5, atol=1e-6)
    assert np.array_equal(cnn['labels'], output['cnn_prediction'][case_id])
    bundle = joblib.load(ROOT / cfg['baseline_model'])
    assert np.array_equal(predict_windows(bundle, noisy, 12000), output['svm_prediction'][case_id])
    checks['same_reconstructed_noisy_case_matches_both_saved_models'] = True

    predictions = list(csv.DictReader((OUT / 'all-predictions.csv').open(encoding='utf-8-sig')))
    assert len(predictions) == 16 * 184
    source_rows = list(csv.DictReader((ROOT / 'data/processed/lesson03/window-index.csv').open(encoding='utf-8-sig')))
    sources = {int(r['sample_index_in_split']): r for r in source_rows if r['split'] == 'test'}
    for row in predictions:
        t, i = int(row['trial_id']), int(row['sample_index_in_split'])
        assert int(row['true_label']) == y[i]
        for key in ['cnn', 'svm']:
            assert int(row[f'{key}_label']) == output[f'{key}_prediction'][t, i]
            assert int(row[f'{key}_correct']) == int(output[f'{key}_prediction'][t, i] == y[i])
        for key in ['record_id', 'load_hp', 'file', 'channel', 'start_second', 'end_second_exclusive']:
            assert row[key] == sources[i][key]
        if row['condition'] != 'original':
            assert np.isclose(float(row['measured_snr_db']), float(row['snr_db']), rtol=0, atol=1e-8)
    for key in ['cnn', 'svm']:
        errors = list(csv.DictReader((OUT / f'errors-{key}.csv').open(encoding='utf-8-sig')))
        assert len(errors) == sum(t['models'][key]['errors'] for t in summary['trials'])
        assert all(r[f'{key}_correct'] == '0' for r in errors)
    audits = list(csv.DictReader((OUT / 'sample-audit.csv').open(encoding='utf-8-sig')))
    for c, audit in enumerate(audits):
        candidates = [r for r in predictions if int(r['trial_id']) == case_id and int(r['true_label']) == c]
        wrong = [r for r in candidates if r['cnn_correct'] == '0' or r['svm_correct'] == '0']
        assert audit == (wrong or candidates)[0]
    assert len(audits) == 4 and summary['unique_original_test_windows'] == 184 and summary['original_test_records'] == 4
    assert all(r['new_independent_test'] is False for r in summary['clean_load_audit'])
    checks['all_prediction_error_rows_and_source_roles_verified'] = True

    # 零功率、非有限数、错误形状与无效 SNR 应拒绝，输入不能被改写。
    toy = np.array([[1., -1., 2., -2.], [2., 1., -1., -2.]])
    unit = make_unit_noise(toy, 99)
    altered, detail = add_noise_at_snr(toy, unit, 0)
    np.testing.assert_allclose(np.mean(detail['noise']**2, axis=1), np.mean(toy**2, axis=1))
    for invalid in [np.zeros((1, 4)), np.full((1, 4), np.nan), np.array([1., 2., 3., 4.])]:
        try:
            make_unit_noise(invalid, 99)
        except ValueError:
            continue
        raise AssertionError('无效参考信号未拒绝。')
    for bad_unit, bad_snr in [(unit[:, :-1], 0), (unit * 2, 0), (unit, np.nan)]:
        try:
            add_noise_at_snr(toy, bad_unit, bad_snr)
        except ValueError:
            continue
        raise AssertionError('无效噪声或 SNR 未拒绝。')
    checks['noise_input_guards_and_zero_db_power_equality_verified'] = True

    docs = [ROOT / '08-第八课-噪声实验与模型鲁棒性.md', ROOT / 'docs/第八课-我的学习记录.md',
            ROOT / 'docs/第八课-面试重点问答.md', OUT / '第八课-噪声实验与负载审视结果.md', ROOT / '00-项目学习路线.md']
    for path in docs:
        content = path.read_text(encoding='utf-8')
        assert len(re.findall(r'^```', content, re.M)) % 2 == 0
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
            if re.match(r'^[A-Za-z]:/', target):
                assert Path(target).exists(), (path, target)
    opening = re.split(r'^### A\d+\.', docs[2].read_text(encoding='utf-8'), flags=re.M)[1:]
    assert len(opening) == 6 and all('**口述参考答案：**' in q for q in opening)
    checks['documents_images_links_and_opening_answers_verified'] = True

    if '--check-launcher' in sys.argv:
        proc = subprocess.run(['cmd.exe', '/d', '/c', '运行第八课.bat'], cwd=ROOT,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding='utf-8', errors='replace', timeout=60)
        assert proc.returncode == 0 and '运行成功' in proc.stdout, proc.stdout
        with np.load(OUT / 'trial-output.npz', allow_pickle=False) as z:
            for key, value in output.items():
                assert np.array_equal(z[key], value), key
        checks['one_click_run_and_same_environment_repeat_match'] = True
    (OUT / 'verification-summary.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
