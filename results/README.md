# 报告实验结果与审计证据

本目录保存课程报告所用的历史模型成绩、图表原图、逐次明细、时间协议核查及 H100 正式时间回测。[当前 Word 报告](../doc/6013_Paper_public.docx) · [项目进度](../docs/project_process.md)

旧交付包标注的 2026-10-31 和 2026-11-01 是整理日期。历史环境与指标文件没有完整训练时间戳。[历史 Word 稿](../doc/DeLELSTM_Five_Run_Course_Paper.docx)

## 未来时段预测

Word 表 3 使用 `chronological/delelstm45/` 下的正式结果。服务器运行代码 `2984120`，只用物理 GPU 1。三个数据集各有三个时间折，每折计划五个种子。45 项中有 43 项得到测试成绩，PM2.5 第二折以 3 个成功种子计算各项均值及样本标准差。

- [summary.csv](chronological/delelstm45/summary.csv) 给出九组完整测试窗口的 RMSE、MAE、MAPE、样本标准差和成功数。
- [runs.csv](chronological/delelstm45/runs.csv) 保留服务器的全部 45 条记录，包括有效测试目标数、成功任务的耗时与失败原因。`artifact_dir` 相对于服务器运行目录。
- [campaign.json](chronological/delelstm45/campaign.json) 登记训练代码版本、环境、3 个复用任务和两个失败位置。各测试区间见[九折数据核查](audits/temporal_protocol.csv)。
- [inference_gpu1.json](chronological/delelstm45/inference_gpu1.json) 保存 Word 表 4 的独立 H100 推理测速，记录各数据集末折权重、单窗口 P50/P95、完整测试集批量吞吐率、PyTorch 峰值分配及预测一致性。

逐种子记录可用于复算表 3 的种子间统计。若需从单个预测值复核误差，服务器的 `outputs/h100_gpu1_delelstm45/jobs/<任务>/attempt_001/` 保存了成功任务的 `predictions.npz`、`split.json` 和 `best.pt`。同一位置保存失败任务的日志、触发批次和模型状态。

## 五次独立训练

`five_run/` 保留 75 份测试指标 CSV、15 份运行元数据、原始汇总与运行环境：

- [verified_detail.csv](five_run/verified_detail.csv)：3 个数据集 × 5 个模型 × 5 次训练，共 75 行。
- [verified_summary.csv](five_run/verified_summary.csv)：15 组均值和**样本标准差**（`ddof=1`）；沿用原文件的三行表头。
- [environment.json](five_run/environment.json)：实际运行于 macOS ARM64 **CPU**，Python 3.13.7、PyTorch 2.10.0。仓库提供的 Conda GPU 环境用于后续运行，不是这些结果的原始运行环境。
- `{dataset}/exp_555_{0..4}/{model}/0_Explain_test_results.csv`：直接保存的测试指标。PM2.5 沿用上游目录名 `newrevision_pmc`。
- `{dataset}/exp_555_{0..4}/run_metadata.json`：划分种子、训练种子、运行序号和重试序号。

所有运行的划分种子均为 `555`。Electricity 和 Exchange 的训练种子为 `555–559`，PM2.5 为 `556–560`。五次重复衡量固定划分上的训练随机性，不是五个独立数据划分。训练轮数按报告为 Electricity 350、PM2.5 和 Exchange 各 300；本目录不包含完整训练轨迹，不能从测试指标 CSV 单独恢复训练轮数或最优验证轮次。

| 数据集 | DeLELSTM RMSE（均值 ± 样本标准差，n = 5） |
|---|---:|
| Electricity | 1.7185 ± 0.0596 |
| PM2.5 | 26.3124 ± 0.4471 |
| Exchange | 0.002293 ± 0.000069 |

原测试 CSV 的 `mape_test` 是比例，`verified_detail.csv` 和 `verified_summary.csv` 的 MAPE 已乘以 100，单位为百分比。整理时已逐项核对 75 份原始测试指标与明细，并由明细重算汇总表的 90 个均值/标准差；数值一致。旧 `progress_*.csv` 仅含 70 次训练，其中 Exchange 每个模型只有 4 次，不用于报告的最终结果。

## 当前报告的三幅图

| 图 | Word 稿内原图 | 复算输入或明细 |
| --- | --- | --- |
| 图 1，各模型五次 RMSE | [PNG](figures/figure_1_rmse_five_runs.png) | [75 次训练明细](five_run/verified_detail.csv) |
| 图 2，PM2.5 瞬时占比 | [PNG](figures/figure_2_pm25_five_run.png) | [五次各 22 个位置](figures/pm25_instantaneous_share_five_runs.csv) |
| 图 3，完整及保留窗口的 RMSE | [PNG](figures/figure_3_test_windows_five_run.png) | [三个数据集的 15 组配对](figures/audit_full_test_five_runs.csv) |

准备好项目环境和数据后，执行 `python scripts/reproduce_paper_figures.py`。脚本从归档的检查点与系数重算两张明细表，逐次核对保留窗口 RMSE，并从原始数值重新绘图。输出在 `outputs/paper_figures/`。[复算脚本](../scripts/reproduce_paper_figures.py)

## 单次运行与解释结果

`single_run/` 来自较早的 GPU 单次实验，与上述 CPU 五次实验分开保留：

- [summary.md](single_run/summary.md) 和 [summary.csv](single_run/summary.csv)：15 组单次测试结果及当时的验证指标。
- `{dataset}/exp_0/Delelstm/0_Explain_test_weight.csv`：三个数据集的平均测试解释系数，是解释脚本的输入。
- 同目录的 `0_soft_testweight.csv`：原训练程序额外导出的系数文件，原样保留以供核对。
- `interpret/`：由上述单次结果恢复的瞬时占比、全局重要性及系数诊断表。

早期单次检查点用于复核历史单次分析。当前报告图 2 和图 3 采用上节列出的五次实验产物。历史解释系数已跨测试样本平均，横轴表示窗口内的位置。

## 审计日志

`audits/` 将原日志原文保存为 `.txt`，保留历史命令输出及报错上下文：

| 文件 | 内容 |
|---|---|
| [temporal_protocol.csv](audits/temporal_protocol.csv) | 三个数据集各三折的时间边界、跨集合重叠、训练集缩放和有效目标数 |
| [temporal_smoke.csv](audits/temporal_smoke.csv) | 本地 CPU 共完成 32 项小窗口测试、1 项全窗口 PM2.5 核验和全新 GitHub 副本上的 15 项预测流程测试。每项训练一轮，`validation_run` 标记各次执行 |
| [h100_forecast_smoke.json](audits/h100_forecast_smoke.json) | 组员回传的 H100 短测试摘要和软件版本。代码 `6a0e3a6`，15/15 成功，一张可见的 H100，CUDA 12.6 |
| [full_test.txt](audits/full_test.txt) | 相同单次检查点、相同划分下，保留尾批与丢弃尾批的比较；没有重训 |
| [exchange_overlap.txt](audits/exchange_overlap.txt) | Exchange 随机划分中测试窗口与训练窗口的重叠计数 |
| [pm_seed555_nan.txt](audits/pm_seed555_nan.txt) | 早期 PM2.5 种子 555 在 epoch 85 出现 NaN 的失败日志 |

PM2.5 的失败发生在正式五次结果之前，不能把它当作五次成功运行之一；正式运行的实际种子以各 `run_metadata.json` 为准。NaN 的具体原因未由现有日志确定。

完整测试集审计的 RMSE：

| 数据集 | 原保留窗口 | 完整测试集 | 原保留数 / 总数 |
|---|---:|---:|---:|
| Electricity | 1.673889 | 1.647470 | 64 / 92 |
| PM2.5 | 27.191929 | 26.370747 | 160 / 184 |
| Exchange | 0.002343 | 0.002318 | 640 / 757 |

这是三个单检查点的敏感性对照，不能替代五次重复实验。Exchange 审计显示全部 757 个测试窗口都与至少一个训练窗口重叠，其中 713 个有相邻一天的训练窗口；原 `drop_last=True` 保留的 640 个窗口中，相应计数为 640 和 603。

## 旧目录到本目录的映射

下表的旧路径相对于原交付包 `DS6013DeLELSTM复现_改进版/repro/`，用于对照原报告中的路径。复制到本目录的 **114 个文件均逐个通过 SHA-256 一致性核验**。

| 旧路径 | 当前证据路径 |
|---|---|
| `results_five_independent/verified_*.csv` | `five_run/verified_*.csv` |
| `results_five_independent/{dataset}/exp_555_*/{model}/*_Explain_test_results.csv` | `five_run/` 下相同子路径 |
| `results_five_independent/{dataset}/exp_555_*/run_metadata.json` | `five_run/` 下相同子路径 |
| `logs_five_independent/environment.json` | `five_run/environment.json` |
| `results/summary.csv`、`results/summary.md` | `single_run/summary.csv`、`single_run/summary.md` |
| `results/{dataset}/exp_0/Delelstm/*test*weight.csv` | `single_run/` 下对应解释系数文件 |
| `results/interpret/*.csv` | `single_run/interpret/*.csv` |
| `logs_five_macos/audit_full_test.log` | `audits/full_test.txt` |
| `logs_five_macos/audit_exchange_overlap.log` | `audits/exchange_overlap.txt` |
| `logs_five_macos/PM.log` | `audits/pm_seed555_nan.txt` |

完整权重、训练轨迹和预测/解释产出保存在 [outputs/](../outputs/README.md)，训练日志保存在 [local_archive/logs/](../local_archive/README.md)，均随小组仓库提交。这里的小型证据文件便于阅读报告、重算预测指标和解释表。重新执行检查点审计时，使用仓库内 `outputs/single_run/` 的对应权重，并先通过 `scripts/setup_upstream.py` 准备上游代码与输入数据。
