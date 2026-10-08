# 原工程参考结果

这里保留原学习工程的实际输出数值，方便核查 README 中的表格：

- `baseline-metrics.json`：验证集选择与基线测试。
- `test-metrics.json`：CNN / SVM 的测试指标、类别指标与记录来源。
- `noise-metrics.json`、`noise-summary.csv`：固定人工噪声条件下的实验。
- `training-history.csv`：原 CNN 的每轮训练和验证记录。

原工程的绝对目录已替换为相对路径。指标数值与实验范围保持原样；内部哈希描述的是原实验产物。复现后的新结果在本机 `outputs/` 下查看，这个目录不会被运行脚本覆盖。
