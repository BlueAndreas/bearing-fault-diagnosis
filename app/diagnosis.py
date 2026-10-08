"""诊断服务的输入检查、处理、冻结模型推理与本地报告。"""
from pathlib import Path
from datetime import datetime
from math import gcd
import base64
import csv
import hashlib
import html
import io
import json
import re
import sys
import threading
import uuid

import joblib
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import resample_poly
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from cnn import load_saved_cnn, predict_raw_windows
from train_baseline import predict_windows, extract_features, FEATURE_ZH
from noise import make_unit_noise, add_noise_at_snr


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')


def numeric_vector(values, limit):
    a = np.asarray(values)
    if a.dtype.kind not in 'fiu' or np.iscomplexobj(a) or a.ndim not in [1, 2] or (a.ndim == 2 and min(a.shape) != 1):
        raise ValueError('振动变量必须是一维实数序列或单行／单列数值，不能是多通道矩阵。')
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if len(a) < 4 or len(a) > limit or not np.isfinite(a).all():
        raise ValueError('信号过短、超过点数限制，或含 NaN／Inf；请检查文件。')
    if np.std(a) == 0:
        raise ValueError('信号是常量或全零，不能作为有效振动进行本程序诊断。')
    return a


def read_file(name, content, cfg):
    if not isinstance(name, str) or not isinstance(content, bytes) or not content or len(content) > cfg['max_file_bytes']:
        raise ValueError('请选择不超过 24 MB 的 MAT 或单列 CSV 文件。')
    suffix = Path(name).suffix.lower()
    if suffix == '.mat':
        try:
            data = loadmat(io.BytesIO(content))
        except Exception as e:
            raise ValueError('MAT 无法读取；本程序支持 SciPy 可读取的 MATLAB v4／v5 至 v7.2，不支持 v7.3 HDF5。') from e
        variables = {k: v for k, v in data.items() if not k.startswith('__') and k.endswith('_DE_time')}
        if not variables:
            raise ValueError('MAT 中没有 *_DE_time 驱动端变量；本程序上传入口限定这种 CWRU 变量格式。')
        signals = {k: numeric_vector(v, cfg['max_source_points']) for k, v in variables.items()}
        return signals
    if suffix == '.csv':
        try:
            text = content.decode('utf-8-sig')
        except UnicodeDecodeError as e:
            raise ValueError('CSV 请保存为 UTF-8 编码、单列振动数值。') from e
        rows = list(csv.reader(io.StringIO(text)))
        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows or any(len(r) != 1 for r in rows):
            raise ValueError('CSV 需要且只能有一列振动数值，不应包含时间列或多通道列。')
        try:
            float(rows[0][0])
        except ValueError:
            rows = rows[1:]  # 允许第一行是标题，其余行必须全是数字。
        try:
            values = np.array([float(r[0]) for r in rows], dtype=np.float64)
        except ValueError as e:
            raise ValueError('CSV 的数据行含非数值内容；只允许第一行是标题。') from e
        return {'signal': numeric_vector(values, cfg['max_source_points'])}
    raise ValueError('目前只接受 .mat 和 .csv。')


def decode_upload(payload, cfg):
    name, encoded = payload.get('filename'), payload.get('file_base64')
    if not isinstance(encoded, str) or len(encoded) > (cfg['max_file_bytes'] * 4 // 3 + 16):
        raise ValueError('上传内容为空或超过文件限制。')
    try:
        content = base64.b64decode(encoded, validate=True)
    except Exception as e:
        raise ValueError('上传内容编码无效，请重新选择文件。') from e
    return name, content, read_file(name, content, cfg)


def segment_windows(signal, source_rate, start_second, duration_second, cfg):
    if isinstance(source_rate, bool) or source_rate not in cfg['allowed_source_rates']:
        raise ValueError('请按数据来源确认采样率；本程序只支持 12000 或 48000 Hz。')
    if not np.isfinite(start_second) or not np.isfinite(duration_second) or start_second < 0:
        raise ValueError('起始秒数需为非负有限数。')
    if duration_second < cfg['window_points']/cfg['sampling_rate_hz'] or duration_second > cfg['max_duration_seconds']:
        raise ValueError('分析时长至少覆盖 1024 个对齐点，且不得超过 20 秒。')
    divisor = gcd(source_rate, cfg['sampling_rate_hz'])
    up, down = cfg['sampling_rate_hz']//divisor, source_rate//divisor
    aligned = resample_poly(signal, up, down) if source_rate != cfg['sampling_rate_hz'] else signal.copy()
    start, count = round(start_second*cfg['sampling_rate_hz']), round(duration_second*cfg['sampling_rate_hz'])
    if start + count > len(aligned):
        raise ValueError(f'截取范围超出信号：对齐后约 {len(aligned)/cfg["sampling_rate_hz"]:.3f} 秒；请减小起点或时长。')
    piece = aligned[start:start+count]
    windows_count = len(piece)//cfg['window_points']
    windows = piece[:windows_count*cfg['window_points']].reshape(windows_count, cfg['window_points']).astype(np.float32)
    if not np.isfinite(windows).all() or np.any(np.std(windows, axis=1) == 0):
        raise ValueError('选取片段中存在常量、全零或无效窗口，请重新检查信号。')
    return windows, {'original_sampling_rate_hz': source_rate, 'aligned_sampling_rate_hz': cfg['sampling_rate_hz'],
                     'source_points': len(signal), 'aligned_points': len(aligned), 'up': up, 'down': down,
                     'start_second': start/cfg['sampling_rate_hz'], 'requested_duration_second': duration_second,
                     'selected_points': count, 'discarded_tail_points': count % cfg['window_points'],
                     'windows': windows_count, 'window_points': cfg['window_points'],
                     'window_seconds': cfg['window_points']/cfg['sampling_rate_hz']}


def spectrum(values, fs):
    x = np.asarray(values, dtype=np.float64)
    taper = np.hanning(len(x))
    amplitude = np.abs(np.fft.rfft((x-x.mean())*taper))/taper.sum()
    if len(x) % 2 == 0:
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2
    return np.fft.rfftfreq(len(x), 1/fs), amplitude


class DiagnosisService:
    def __init__(self):
        self.cfg = json.loads((ROOT / 'configs/app.json').read_text(encoding='utf-8'))
        torch.set_num_threads(2)
        self.model, self.meta = load_saved_cnn(ROOT / self.cfg['cnn_folder'])
        self.svm = joblib.load(ROOT / self.cfg['svm_file'])
        if self.svm['label_names'] != self.meta['label_names']:
            raise ValueError('CNN 与 SVM 类别顺序不一致。')
        if self.meta['sampling_rate_hz'] != self.cfg['sampling_rate_hz'] or self.meta['window_points'] != self.cfg['window_points']:
            raise ValueError('界面配置与已保存模型输入不一致。')
        self.names = self.meta['label_names']
        self.model_hashes = {'cnn_weights': self.meta['weights_sha256'],
                             'svm_file': digest((ROOT / self.cfg['svm_file']).read_bytes()),
                             'cnn_metadata': digest((ROOT / self.cfg['cnn_folder'] / 'model-metadata.json').read_bytes())}
        self.lock = threading.RLock()
        self.runs = ROOT / self.cfg['saved_runs']
        self.runs.mkdir(parents=True, exist_ok=True)
        with np.load(ROOT / self.cfg['dataset'], allow_pickle=False) as z:
            self.test_raw, self.test_y = z['X_test_raw'].copy(), z['y_test'].copy()
        index = csv.DictReader((ROOT / 'data/processed/dataset/window-index.csv').open(encoding='utf-8-sig'))
        self.test_index = sorted([r for r in index if r['split']=='test'], key=lambda r:int(r['sample_index_in_split']))
        if digest((ROOT / self.cfg['dataset']).read_bytes()) != self.meta['dataset_sha256']:
            raise ValueError('示例数据与模型的记录身份不一致。')

    def catalog(self):
        return {'app': 'bearing-fault-diagnosis', 'project_identity': digest(str(ROOT).encode()),
                'examples': [{'id':str(c), 'name':f'{name} · 记录 {self.test_index[c*46]["record_id"]} · 3 HP',
                              'known_state':name} for c,name in enumerate(self.names)],
                'model_status': '已加载 CNN 与 SVM-C1 · CPU · 只做预测',
                'window_points':1024, 'aligned_rate':12000}

    def inspect(self, payload):
        name, content, signals = decode_upload(payload, self.cfg)
        return {'filename':Path(name).name, 'sha256':digest(content),
                'variables':[{'name':k,'points':len(v)} for k,v in signals.items()],
                'needs_explicit_channel':len(signals)>1, 'sampling_rate_inferred':False}

    def analyze(self, payload):
        with self.lock:
            source = payload.get('source')
            if source == 'example':
                example = payload.get('example_id')
                if example not in ['0','1','2','3']:
                    raise ValueError('请选择一个已有数据示例。')
                c = int(example)
                mask = self.test_y == c
                raw = self.test_raw[mask].copy()
                noise_level = payload.get('noise_level', 'original')
                if noise_level not in ['original','0','-5']:
                    raise ValueError('已有示例支持未加噪、0 dB 或 −5 dB 演示。')
                if noise_level != 'original':
                    all_noisy, _ = add_noise_at_snr(self.test_raw, make_unit_noise(self.test_raw, 2026), int(noise_level))
                    raw = all_noisy[mask]
                idx = [r for r in self.test_index if int(r['label']) == c]
                source_meta = {'kind':'example','filename':Path(idx[0]['file']).name, 'channel':idx[0]['channel'],
                               'record_id':int(idx[0]['record_id']), 'known_state':self.names[c],
                               'known_label_source':'数据来源清单，不是模型推断', 'data_role':'既有 3 HP 测试窗口，已在既有实验使用',
                               'noise_level':noise_level, 'noise_seed':None if noise_level=='original' else 2026,
                               'input_windows_sha256':digest(np.asarray(raw).tobytes())}
                prep = {'original_sampling_rate_hz':48000 if c==0 else 12000, 'aligned_sampling_rate_hz':12000,
                        'up':1,'down':4 if c==0 else 1, 'start_second':.5, 'requested_duration_second':4,
                        'selected_points':48000, 'discarded_tail_points':896, 'windows':46,
                        'window_points':1024,'window_seconds':1024/12000}
            elif source == 'upload':
                name, content, signals = decode_upload(payload, self.cfg)
                channel = payload.get('channel')
                if channel not in signals:
                    raise ValueError('请明确选择文件中的振动变量；多变量 MAT 不能默认取第一个。')
                try:
                    rate = int(payload.get('source_rate_hz'))
                    start, duration = float(payload.get('start_second')), float(payload.get('duration_second'))
                except (ValueError, TypeError) as e:
                    raise ValueError('请输入有效的采样率、起始秒数和分析时长。') from e
                if str(rate) != str(payload.get('source_rate_hz')):
                    raise ValueError('采样率必须为 12000 或 48000 整数。')
                raw, prep = segment_windows(signals[channel], rate, start, duration, self.cfg)
                source_meta = {'kind':'upload','filename':Path(name).name,'channel':channel, 'file_sha256':digest(content),
                               'known_state':None, 'data_role':'用户上传；没有已知真标签，不计算准确率',
                               'noise_level':'original', 'input_windows_sha256':digest(raw.tobytes())}
            else:
                raise ValueError('请选择已有示例或上传文件。')
            logits = np.concatenate([predict_raw_windows(self.model,self.meta,raw[i:i+32],12000)['logits']
                                     for i in range(0,len(raw),32)])
            cnn, svm = logits.argmax(axis=1), predict_windows(self.svm, raw, 12000)
            features = extract_features(raw)
            entries = [{'index':i, 'start_second':prep['start_second']+i*1024/12000,
                        'end_second_exclusive':prep['start_second']+(i+1)*1024/12000,
                        'cnn_label':int(cnn[i]), 'cnn_state':self.names[int(cnn[i])],
                        'svm_label':int(svm[i]),'svm_state':self.names[int(svm[i])],
                        'cnn_logits':logits[i].tolist(),'features':features[i].tolist()} for i in range(len(raw))]
            counts = {key:np.bincount(p,minlength=4).tolist() for key,p in [('cnn',cnn),('svm',svm)]}
            modes = {key:[self.names[i] for i,n in enumerate(counts[key]) if n==max(counts[key])] for key in counts}
            run_id = uuid.uuid4().hex[:16]
            report = {'run_id':run_id,'created_at':datetime.now().astimezone().isoformat(timespec='seconds'),
                      'source':source_meta,'preprocessing':prep,'label_names':self.names,'feature_names':FEATURE_ZH,
                      'model_identity':self.model_hashes,'cnn_selected_epoch':self.meta['selected_epoch'],
                      'training_performed':False,'window_votes':counts,'window_majority_classes':modes,
                      'model_agreement_windows':int(np.sum(cnn==svm)),'windows':entries,
                      'scope':'CWRU 四类离线原型；窗口多数类只作摘要，logits 不是置信度，不支持未知故障识别。',
                      'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__}}
            folder = self.runs/run_id
            folder.mkdir()
            np.savez_compressed(folder/'windows.npz', raw=raw)
            (folder/'report.json').write_bytes(json_bytes(report))
            with (folder/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['window_index','start_second','end_second_exclusive','cnn_state','svm_state'])
                writer.writerows([r['index'],r['start_second'],r['end_second_exclusive'],r['cnn_state'],r['svm_state']] for r in entries)
            self.write_html_report(report, raw, folder)
            return report

    def folder(self, run_id):
        if not re.fullmatch('[a-f0-9]{16}',run_id):
            raise ValueError('诊断记录编号无效。')
        folder=self.runs/run_id
        if not (folder/'report.json').is_file():
            raise ValueError('找不到这条诊断记录。')
        return folder

    def report(self, run_id):
        return json.loads((self.folder(run_id)/'report.json').read_text(encoding='utf-8'))

    def history(self):
        rows=[]
        for p in self.runs.glob('*/report.json'):
            try:
                r=json.loads(p.read_text(encoding='utf-8'))
                rows.append({'run_id':r['run_id'],'created_at':r['created_at'],'filename':r['source']['filename'],
                             'noise_level':r['source']['noise_level'],'windows':len(r['windows'])})
            except (ValueError,KeyError):
                continue
        return sorted(rows,key=lambda r:r['created_at'],reverse=True)[:20]

    def window(self, run_id, index):
        report=self.report(run_id)
        if isinstance(index,bool) or index<0 or index>=len(report['windows']):
            raise ValueError('窗口索引超出当前记录。')
        with np.load(self.folder(run_id)/'windows.npz',allow_pickle=False) as z:
            x=z['raw'][index].astype(np.float64)
        freq,amp=spectrum(x,12000)
        entry=report['windows'][index]
        return {'entry':entry, 'time_seconds':(entry['start_second']+np.arange(1024)/12000).tolist(),
                'waveform':x.tolist(),'frequency_hz':freq.tolist(),'amplitude':amp.tolist(),
                'frequency_spacing_hz':12000/1024}

    def write_html_report(self, report, raw, folder):
        plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
        x=raw[0].astype(np.float64)
        f,a=spectrum(x,12000)
        fig,axes=plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
        axes[0].plot(report['preprocessing']['start_second']+np.arange(1024)/12000,x,lw=.8)
        axes[0].set(title='第 0 个窗口的振动波形',xlabel='记录内时间（秒）',ylabel='原数据幅值')
        axes[1].plot(f,a,color='#248D77',lw=.8)
        axes[1].set(title='去均值、Hann 窗后的单边幅度谱',xlabel='频率（Hz）',ylabel='原数据幅度')
        fig.savefig(folder/'first-window.png',dpi=135)
        plt.close(fig)
        picture=base64.b64encode((folder/'first-window.png').read_bytes()).decode()
        esc=html.escape
        rows=''.join(f"<tr><td>{r['index']}</td><td>{r['start_second']:.6f}</td><td>{esc(r['cnn_state'])}</td><td>{esc(r['svm_state'])}</td></tr>" for r in report['windows'])
        meta=esc(json.dumps({'source':report['source'],'preprocessing':report['preprocessing'],
                             'model_identity':report['model_identity']},ensure_ascii=False,indent=2))
        content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>轴承诊断演示报告</title>
<style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;max-width:1000px;margin:40px auto;padding:20px;color:#243444}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd6df;padding:8px}}img{{width:100%}}pre{{white-space:pre-wrap;word-break:break-all;background:#f1f5f8;padding:18px}}</style>
<h1>轴承诊断演示报告</h1><p>{esc(report['created_at'])} · {esc(report['source']['filename'])}</p>
<p>已知状态：{esc(report['source']['known_state'] or '未知，未提供真标签')}。CNN 窗口多数类：{esc('／'.join(report['window_majority_classes']['cnn']))}；SVM 窗口多数类：{esc('／'.join(report['window_majority_classes']['svm']))}。</p>
<p>{esc(report['scope'])} 本次仅加载模型预测。上传数据没有已知标签时不计算准确率；已有示例及人工噪声也不代表新设备验证。</p>
<img alt="第0个窗口波形和频谱" src="data:image/png;base64,{picture}"><table><thead><tr><th>窗口索引</th><th>起始秒</th><th>CNN</th><th>SVM</th></tr></thead><tbody>{rows}</tbody></table>
<h2>输入与模型记录</h2><pre>{meta}</pre></html>'''
        (folder/'report.html').write_text(content,encoding='utf-8')
