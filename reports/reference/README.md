# 工程参考结果

这里保留公开工程版本在 2026 年 10 月 8 日完整运行的实际输出数值，方便核查 README 中的表格：

- `baseline-metrics.json`：验证集选择与基线测试。
- `test-metrics.json`：CNN / SVM 的测试指标、类别指标与记录来源。
- `noise-metrics.json`、`noise-summary.csv`：固定人工噪声条件下的实验。
- `training-history.csv`：CNN 的每轮训练和验证记录。

绝对目录已替换为相对路径；内部哈希描述的是本次实验产物。重新运行后的结果在本机 `outputs/` 下查看，脚本不会覆盖这里的参考结果。
