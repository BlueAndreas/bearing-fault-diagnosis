"""在仓库根目录依次复现数据处理、模型训练、测试与噪声实验。"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-download', action='store_true',
                        help='已有 16 条官方记录，跳过联网准备步骤')
    args = parser.parse_args()
    stages = [] if args.skip_download else ['download_data.py']
    stages += ['build_dataset.py', 'train_baseline.py',
               'train_cnn.py', 'evaluate.py',
               'evaluate_noise.py']
    for index, name in enumerate(stages, 1):
        print(f'\n[{index}/{len(stages)}] {name}', flush=True)
        subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts' / name)],
                       cwd=ROOT, check=True)
    print('\n流程完成。结果在 outputs/；启动界面：python -X utf8 app/server.py --open-browser',
          flush=True)


if __name__ == '__main__':
    main()
