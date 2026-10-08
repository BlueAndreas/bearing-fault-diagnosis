# 复现说明

## 1. 数据来源

- [正常记录](https://engineering.case.edu/bearingdatacenter/normal-baseline-data)：97、98、99、100。
- [12 kHz 驱动端故障](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data)：内圈 105～108；滚动体 118～121；外圈 6 点钟 130～133，均为 0.007 英寸。

下载脚本：`python -X utf8 scripts/prepare_lesson03_data.py`。脚本仅在文件缺失时从官方获取，检查驱动端变量，不覆盖已有文件；它不是绕过来源网站使用条件的工具。

官方网站不可访问时，可自行从上面的入口获取后保存为：

```text
data/raw/CWRU/
  Normal Baseline Data/97.mat ... 100.mat
  12k_DE/105.mat ... 108.mat
  12k_DE/118.mat ... 121.mat
  12k_DE/130.mat ... 133.mat
```

亦支持原始命名：正常为 `1797 rpm.mat`、`1772 rpm.mat`、`1750 rpm.mat`、`1730 rpm.mat`；故障为 `IR007_0.mat`～`IR007_3.mat`、`B007_0.mat`～`B007_3.mat`、`OR007@6_0.mat`～`OR007@6_3.mat`。不要只修改一个错误文件的名字来代替对应记录。

## 2. 按阶段运行

以下命令在仓库根目录、已安装依赖的 Python 环境中执行。Windows 使用 `python -X utf8` 可减少终端编码问题。

```powershell
python -X utf8 scripts/prepare_lesson03_data.py
python -X utf8 scripts/lesson03_build_dataset.py
python -X utf8 scripts/lesson04_train_baseline.py
python -X utf8 scripts/lesson06_train_cnn.py
python -X utf8 scripts/lesson07_evaluate.py
python -X utf8 scripts/lesson08_noise_experiment.py
python -X utf8 app/server.py --open-browser
```

或执行 `python -X utf8 scripts/run_pipeline.py`；准备好了 16 条记录，可加 `--skip-download`。第五课数据装载练习可在数据集构建后单独执行 `python -X utf8 scripts/lesson05_beginner_walkthrough.py`。

运行脚本保留原工程中的相对目录约定，无须安装到特定盘符。不要先更改窗口配置：训练、已保存模型及界面要求相同采样率和 1024 点窗口。

## 3. 校验

训练、评估和噪声实验完成后可执行已有检查：

```powershell
python -X utf8 scripts/lesson06_verify.py
python -X utf8 scripts/lesson07_verify.py
python -X utf8 scripts/lesson08_verify.py
```

启动网页服务后，可执行 `python -X utf8 scripts/verify_lesson09.py` 检查导入、推理、报告和异常输入等行为。使用其他端口时，服务和校验程序均可添加 `--port 8881`。校验程序会生成局部检查结果和演示历史，保存在被忽略的 `outputs/`。

主要产物：

| 阶段 | 文件 |
| --- | --- |
| 数据集 | `data/processed/lesson03/dataset.npz`、`window-index.csv`、`normalization.json` |
| 基线 | `outputs/lesson04/baseline-model.joblib`、`metrics.json` |
| CNN | `outputs/lesson06/best-cnn.pt`、`model-metadata.json`、`training-history.csv` |
| 测试 | `outputs/lesson07/metrics.json`、`test-predictions.csv` |
| 噪声 | `outputs/lesson08/noise-summary.csv`、`metrics.json` |
| 界面 | `outputs/lesson09/runs/` 下的每次诊断与导出报告 |

## 4. 复现边界

参考结果使用 Python 3.14.4、CPU PyTorch 2.12.0、NumPy 2.4.4、SciPy 1.17.1、scikit-learn 1.8.0。依赖版本、数据文件和配置变化可能造成数值或校验哈希变化；固定种子不保证跨硬件、跨版本完全一致。

中文结果图优先使用 Microsoft YaHei / SimHei；其他系统没有这些字体时会回退，中文可能缺字。公开副本已在 Windows 本机复核，尚未验证 Linux 或 macOS 的完整运行。

参考 JSON 中的原工程目录已替换为相对路径。文件哈希对应原实验文件，仍作为原结果的来源记录保留；不应拿这些历史值当成新一轮运行的期望哈希。
