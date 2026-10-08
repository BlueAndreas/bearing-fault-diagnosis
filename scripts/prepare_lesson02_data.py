"""从 CWRU 官方补齐第二课的 118、130 号记录；不覆盖已有文件。"""
from pathlib import Path
from urllib.request import Request, urlopen
from io import BytesIO
from scipy.io import loadmat
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/raw/CWRU/12k_DE'
    folder.mkdir(parents=True, exist_ok=True)
    for number, name in [(118, 'B007_0.mat'), (130, 'OR007@6_0.mat')]:
        dest = folder / name
        if dest.exists():
            content = dest.read_bytes()
            print('使用已有文件：', name)
        else:
            url = f'https://engineering.case.edu/sites/default/files/{number}.mat'
            with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30) as response:
                content = response.read()
            print('从官方获取：', url)
        data = loadmat(BytesIO(content))
        key = f'X{number:03d}_DE_time'
        if key not in data or data[key].size < 12000:
            raise ValueError(f'{name} 缺少预期的振动通道 {key}。')
        if not dest.exists():
            with dest.open('xb') as output:
                output.write(content)
        print(name, '点数=', data[key].size, 'SHA-256=', hashlib.sha256(content).hexdigest())


if __name__ == '__main__':
    main()
