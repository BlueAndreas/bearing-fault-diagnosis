"""第九课本地 HTTP 服务：python -X utf8 app/server.py [--open-browser]。"""
from pathlib import Path
import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen
import webbrowser

from diagnosis import DiagnosisService, ROOT, json_bytes, digest

STATIC = Path(__file__).resolve().parent / 'static'


class Handler(BaseHTTPRequestHandler):
    server_version = 'BearingDemo/1.0'
    def log_message(self, format, *args):
        print(format % args, flush=True)

    def reply(self, body, content_type='application/json; charset=utf-8', status=200, filename=None):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        if filename:
            self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            parsed=urlparse(self.path)
            path=parsed.path
            service=self.server.service
            if path=='/api/catalog':
                return self.reply(json_bytes(service.catalog()))
            if path=='/api/history':
                return self.reply(json_bytes(service.history()))
            if path.startswith('/api/runs/'):
                parts=path.strip('/').split('/')
                if len(parts)==3:
                    return self.reply(json_bytes(service.report(parts[2])))
                if len(parts)==4 and parts[3]=='window':
                    index=int(parse_qs(parsed.query).get('index',['0'])[0])
                    return self.reply(json_bytes(service.window(parts[2],index)))
            if path.startswith('/download/'):
                parts=path.strip('/').split('/')
                names={'html':('report.html','text/html; charset=utf-8'),
                       'json':('report.json','application/json; charset=utf-8'),
                       'csv':('predictions.csv','text/csv; charset=utf-8')}
                if len(parts)==3 and parts[2] in names:
                    name,mime=names[parts[2]]
                    return self.reply((service.folder(parts[1])/name).read_bytes(),mime,filename=f'diagnosis-{parts[1]}.{parts[2]}')
            files={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),
                   '/style.css':('style.css','text/css; charset=utf-8')}
            if path in files:
                name,mime=files[path]
                return self.reply((STATIC/name).read_bytes(),mime)
            self.reply(json_bytes({'error':'没有这个页面或接口。'}),status=404)
        except (ValueError,FileNotFoundError) as e:
            self.reply(json_bytes({'error':str(e)}),status=400)
        except Exception:
            traceback.print_exc()
            self.reply(json_bytes({'error':'读取失败，请查看启动窗口中的报错信息。'}),status=500)

    def do_POST(self):
        try:
            if self.path not in ['/api/inspect','/api/analyze']:
                return self.reply(json_bytes({'error':'没有这个操作。'}),status=404)
            origin=self.headers.get('Origin')
            if origin and origin not in [f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}']:
                raise ValueError('请从本地诊断页面提交。')
            length=int(self.headers.get('Content-Length','0'))
            if length<=0 or length>36*1024*1024:
                raise ValueError('提交内容为空或超过大小限制。')
            payload=json.loads(self.rfile.read(length).decode('utf-8'))
            if not isinstance(payload,dict):
                raise ValueError('请求格式无效。')
            fn=self.server.service.inspect if self.path=='/api/inspect' else self.server.service.analyze
            self.reply(json_bytes(fn(payload)))
        except (ValueError,TypeError) as e:
            self.reply(json_bytes({'error':str(e)}),status=400)
        except Exception:
            traceback.print_exc()
            self.reply(json_bytes({'error':'诊断失败，请查看启动窗口中的报错；本次未获得有效结果。'}),status=500)


def main():
    args=argparse.ArgumentParser()
    args.add_argument('--port',type=int,default=8879)
    args.add_argument('--open-browser',action='store_true')
    opt=args.parse_args()
    url=f'http://127.0.0.1:{opt.port}'
    try:
        with urlopen(url+'/api/catalog',timeout=1) as r:
            old=json.load(r)
        if old.get('app')=='bearing-lesson09' and old.get('project_identity')==digest(str(ROOT).encode()):
            print('这个项目的界面已经运行：',url,flush=True)
            if opt.open_browser:
                webbrowser.open(url)
            return
    except Exception:
        pass
    service=DiagnosisService()
    try:
        server=ThreadingHTTPServer(('127.0.0.1',opt.port),Handler)
    except OSError as e:
        raise SystemExit(f'端口 {opt.port} 已被其他程序占用。可使用 --port 8880 启动。') from e
    server.service=service
    print('模型已加载，访问：',url,flush=True)
    print('保留此启动窗口；结束时按 Ctrl+C。仅监听本机，分析结果保存在 outputs/lesson09/runs。',flush=True)
    if opt.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('界面服务已停止。',flush=True)
    finally:
        server.server_close()


if __name__=='__main__':
    main()
