"""可重复验证诊断 API；会生成少量真实诊断历史，不修改既有实验数据或模型。"""
from pathlib import Path
import argparse
import base64
import csv
import io
import json
import sys
import numpy as np
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from diagnosis import spectrum
URL='http://127.0.0.1:8879'

def get(path):
    r=requests.get(URL+path,timeout=30);r.raise_for_status();return r.json()

def post(path,payload,expected=200):
    r=requests.post(URL+path,json=payload,timeout=60)
    assert r.status_code==expected,(r.status_code,r.text[:500])
    return r.json()

def upload(path,**kw):
    return {'source':'upload','filename':path.name,'file_base64':base64.b64encode(path.read_bytes()).decode(),**kw}

def main():
    global URL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8879,help='本机诊断服务端口，默认 8879')
    args=parser.parse_args()
    if not 1<=args.port<=65535:
        parser.error('端口需在 1～65535 之间。')
    URL=f'http://127.0.0.1:{args.port}'
    old=np.load(ROOT/'outputs/evaluation/test-output.npz')
    ds=np.load(ROOT/'data/processed/dataset/dataset.npz')
    noise_rows=list(csv.DictReader((ROOT/'outputs/noise/all-predictions.csv').open(encoding='utf-8-sig')))
    cases=[]
    for condition in ['original','0','-5']:
        for c in range(4):
            r=post('/api/analyze',{'source':'example','example_id':str(c),'noise_level':condition})
            mask=ds['y_test']==c
            cnn=np.array([w['cnn_label'] for w in r['windows']]);svm=np.array([w['svm_label'] for w in r['windows']])
            if condition=='original':
                np.testing.assert_array_equal(cnn,old['cnn_prediction'][mask]);np.testing.assert_array_equal(svm,old['svm_prediction'][mask])
                np.testing.assert_allclose([w['cnn_logits'] for w in r['windows']],old['logits'][mask],rtol=1e-5,atol=1e-5)
            else:
                rows=[x for x in noise_rows if x['snr_db']==condition and x['seed']=='2026' and x['true_label']==str(c)]
                np.testing.assert_array_equal(cnn,[int(x['cnn_label']) for x in rows]);np.testing.assert_array_equal(svm,[int(x['svm_label']) for x in rows])
            assert len(cnn)==46 and not r['training_performed']
            for ext in ['html','json','csv']:
                d=requests.get(URL+f'/download/{r["run_id"]}/{ext}',timeout=20)
                assert d.status_code==200 and 'attachment' in d.headers['Content-Disposition']
                if ext=='html':assert 'data:image/png;base64,' in d.text and '<table>' in d.text
                if ext=='csv':assert len(list(csv.reader(io.StringIO(d.content.decode('utf-8-sig')))))==47
                if ext=='json':assert d.json()['run_id']==r['run_id']
            cases.append({'condition':condition,'class':c,'run_id':r['run_id'],'cnn_votes':r['window_votes']['cnn'],'svm_votes':r['window_votes']['svm']})
    normal_folder=ROOT/'data/raw/CWRU/Normal Baseline Data'
    mat=next((normal_folder/name for name in ['1750 rpm.mat','99.mat']
              if (normal_folder/name).is_file()),None)
    if mat is None:
        raise FileNotFoundError('缺少正常状态 2 HP 的 99 号 MAT 记录。')
    p=upload(mat,source_rate_hz=48000,start_second=.5,duration_second=4)
    info=post('/api/inspect',p)
    assert {x['name'] for x in info['variables']}=={'X098_DE_time','X099_DE_time'}
    before=len(list((ROOT/'outputs/diagnosis/runs').iterdir()))
    post('/api/analyze',p,400)
    p['channel']='X099_DE_time'
    r=post('/api/analyze',p)
    assert r['source']['known_state'] is None and r['preprocessing']['down']==4
    with np.load(ROOT/'outputs/diagnosis/runs'/r['run_id']/'windows.npz') as z:
        np.testing.assert_array_equal(z['raw'],ds['X_val_raw'][ds['record_id_val']==99])
    # CSV 数值取已有测试片段；检查相同输入的预测结果，且不给文件名赋真标签。
    raw=ds['X_test_raw'][46:92].reshape(-1)
    encoded=base64.b64encode(('amplitude\n'+'\n'.join(str(float(x)) for x in raw)).encode()).decode()
    p={'source':'upload','filename':'inner-known-name.csv','file_base64':encoded,'channel':'signal','source_rate_hz':12000,'start_second':0,'duration_second':len(raw)/12000}
    post('/api/inspect',p)
    r=post('/api/analyze',p)
    assert r['source']['known_state'] is None
    np.testing.assert_array_equal([w['cnn_label'] for w in r['windows']],old['cnn_prediction'][46:92])
    for content in ['t,x\n0,1\n1,2','1\n1\n1\n1','1\nNaN\n2\n3','1\nInf\n2\n3','head\n1\nx\n2']:
        bad={**p,'file_base64':base64.b64encode(content.encode()).decode()}
        post('/api/inspect',bad,400)
    for overrides in [{'source_rate_hz':44100},{'duration_second':.01},{'start_second':100},{'channel':'wrong'},{'duration_second':21}]:
        post('/api/analyze',{**p,**overrides},400)
    assert len(list((ROOT/'outputs/diagnosis/runs').iterdir()))==before+2
    window=get(f'/api/runs/{r["run_id"]}/window?index=4')
    assert len(window['waveform'])==1024 and len(window['frequency_hz'])==513
    assert window['entry']['index']==4 and window['frequency_spacing_hz']==11.71875
    assert requests.get(URL+f'/api/runs/{r["run_id"]}/window?index=10000').status_code==400
    assert requests.get(URL+'/../configs/app.json').status_code==404
    assert requests.get(URL+'/download/../json').status_code in [400,404]
    count=len(list((ROOT/'outputs/diagnosis/runs').iterdir()));get(f'/api/runs/{r["run_id"]}');get('/api/history')
    assert len(list((ROOT/'outputs/diagnosis/runs').iterdir()))==count
    f,a=spectrum(np.sin(2*np.pi*32*np.arange(1024)/1024),12000)
    assert f[a.argmax()]==375 and abs(a.max()-1)<.001
    summary={'status':'passed','example_cases':cases,'multi_variable_mat':'explicit X099; matched dataset validation raw exactly',
             'csv':'matched prior CNN predictions; unknown true label','validation':'invalid column/value/constant/rate/channel/range rejected',
             'exports':'HTML embedded PNG; CSV 46 rows; JSON run identity; all 12 examples verified',
             'spectrum':'375 Hz bin-aligned unit sine; peak amplitude within .001',
             'history':'reading report/window/history creates no new run','industrial_validation':False}
    (ROOT/'outputs/diagnosis/verification-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
