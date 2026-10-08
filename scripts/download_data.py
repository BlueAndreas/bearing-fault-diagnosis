"""补齐 0～3 HP 的四类记录；公开官方数据只在缺失时下载，不覆盖已有文件。"""
from pathlib import Path
from urllib.request import Request, urlopen
from io import BytesIO
import hashlib
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / 'data/raw/CWRU'
    specs = []
    for load, number in enumerate(range(97, 101)):
        rpm = [1797,1772,1750,1730][load]
        specs.append((number,raw/'Normal Baseline Data'/f'{rpm} rpm.mat',
                      raw/'Normal Baseline Data'/f'{number}.mat'))
    for prefix, first_id in [('IR007',105),('OR007@6',130),('B007',118)]:
        for load in range(4):
            number = first_id+load
            specs.append((number,raw/'12k_DE'/f'{prefix}_{load}.mat',raw/'12k_DE'/f'{number}.mat'))
    for number,preferred,alternative in specs:
        existing=next((p for p in [preferred,alternative] if p.is_file()),None)
        path=existing or preferred
        if existing:
            content=path.read_bytes()
        else:
            url=f'https://engineering.case.edu/sites/default/files/{number}.mat'
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as response:
                content=response.read()
            print('从官方获取：',url)
        data=loadmat(BytesIO(content))
        key=f'X{number:03d}_DE_time'
        if key not in data or data[key].size<12000:
            raise ValueError(f'{path.name} 缺少记录 {number} 的驱动端信号；未写入下载内容。')
        if not existing:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as output:
                output.write(content)
        print(number,path.name,'点数=',data[key].size,'SHA-256=',hashlib.sha256(content).hexdigest())
    print('数据构建阶段的 16 条记录已准备好。')


if __name__=='__main__':
    main()
