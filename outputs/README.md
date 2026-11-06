# 完整实验产物

本目录随小组仓库提交，包含模型权重、训练轨迹、预测结果、解释系数和运行元数据。

| 目录 | 用途 | 已有检查点 |
|---|---|---:|
| `five_run/` | 报告的正式五次 CPU 实验，3 个数据集 × 5 个模型 × 5 次训练 | 75 |
| `single_run/` | 较早的单次 GPU 实验，报告解释分析和全量测试审计的来源 | 15 |
| `incomplete_run/` | 早期未完成的实验，保留作追溯，不混入正式结果汇总 | 4 |

`five_run/verified_detail.csv` 和 `verified_summary.csv` 是最终统计，
其中的 `progress_*.csv` 是早期中间进度，不应替代 verified 表。
PM2.5 的目录名沿用上游的 `newrevision_pmc`。

原训练日志位于 `local_archive/logs/`，小型证据索引见
[results/README.md](../results/README.md)。新增实验请使用新的输出目录，
保留参数、种子、日志和权重，核对后一起提交。

服务器补充实验默认生成 `ablation/`（正式任务）和 `ablation-smoke/`（短测试），每个任务保留权重、预测、日志及尝试记录，报告在其 `report/` 下。两者均不被忽略；运行方法见 [补充实验全流程](../doc/ablation_workflow.md)。

`verification/` 及隐藏的环境安装验证目录仅用于本地核验与上传过程，
由 `.gitignore` 排除，不属于实验交付物。
