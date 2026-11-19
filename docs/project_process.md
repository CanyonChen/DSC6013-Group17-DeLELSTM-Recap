# DSC6013 第 17 组｜DeLELSTM 复现项目进度

更新于 **2026-11-19**。修正时间划分和输入归一化后，小组在一张 H100 上完成了 DeLELSTM 的正式未来时段回测。3 个数据集 × 3 个时间折 × 5 个种子共 45 项，43 项得到测试成绩，2 项在训练中失败。Word 报告第 6.1、6.3 节记录了泄漏证据和修正方法，第 6.4 节表 3 给出九组成绩、测试范围和有效目标数。[共享 Word 报告](../doc/6013_Paper_public.docx) · [九个划分的核查记录](../results/audits/temporal_protocol.csv)

## H100 正式回测结果

这次在代码 `2984120` 上运行，物理 GPU 1 是唯一可见设备，在进程内显示为 `cuda:0`。PyTorch 为 2.7.1+cu126，CUDA 为 12.6。Electricity 第一折的种子 555、556、557 从前一队列复用，本次新启动 42 项，其中 40 项成功。与五模型的 225 项计划相比，本阶段少安排 180 项训练，计划任务数减少 80%。此前的 15 项 H100 短测试也全部通过。[正式运行记录](../results/chronological/delelstm45/campaign.json) · [短测试环境](../results/audits/h100_forecast_smoke.json)

表内给出测试 RMSE 的均值 ± 样本标准差。历史列使用旧协议的五份 DeLELSTM 权重，在完整随机测试集上重新评分。后三列各自对应一个更晚的测试区间，按成功种子统计。除 PM2.5 第二折为 3/5 外，其余八组均为 5/5。

| 数据集 | 历史完整随机测试集 | 时间折 1 | 时间折 2 | 时间折 3 |
| --- | ---: | ---: | ---: | ---: |
| Electricity | 1.6825 ± 0.0357 | 1.6299 ± 0.0682 | 1.5753 ± 0.0613 | 1.9677 ± 0.0236 |
| PM2.5 | 25.6103 ± 0.4114 | 18.09 ± 0.24 | 23.01 ± 0.32 | 17.85 ± 0.05 |
| Exchange | 0.002273 ± 0.000060 | 0.025994 ± 0.007192 | 0.008203 ± 0.003832 | 0.005136 ± 0.001467 |

与历史参照相比，PM2.5 三折的 RMSE 都更低，Electricity 前两折更低、末折更高，Exchange 三折都更高。新旧协议使用的测试窗口不同，PM2.5 又改为仅对真实观测评分。这些差异支持描述各时段的预测表现，单项修正对误差的贡献仍需同测试段对照。已按 45 条逐种子记录复算九组 RMSE、MAE、MAPE 的均值和样本标准差，并核对每个成功任务的有效测试目标数。[逐折汇总](../results/chronological/delelstm45/summary.csv) · [逐种子明细](../results/chronological/delelstm45/runs.csv) · [历史完整测试集明细](../results/figures/audit_full_test_five_runs.csv)

两项失败均在 PM2.5 第二折（`fold=1`）。种子 556 的训练损失在 `epoch=70, batch=15` 出现非有限值，种子 559 的梯度在 `epoch=75, batch=32` 出现非有限值。轮次和批次均从零计数。该折的三个成功种子是 557、558、560，每个种子在 3,999 个实测目标上评分。失败种子保留在计划分母中。[失败记录](../results/chronological/delelstm45/campaign.json)

40 个新成功任务的记录耗时合计 **11 小时 55 分钟**。进度到 10/45 时，3 项来自复用，另外 7 项的用时合计 **1 小时 54 分钟**。43 个成功任务的单项中位用时依次为 Electricity 6.55 分钟、PM2.5 21.15 分钟和 Exchange 24.89 分钟。失败记录没有计时，整次队列的墙钟耗时和 GPU 利用率仍缺少相应记录。[任务耗时](../results/chronological/delelstm45/runs.csv)

## DeLELSTM 推理效率

正式训练的 `seconds` 覆盖数据准备、训练、验证、测试和解释导出。单独测速时，加载第三折预先选定种子的最佳验证权重，Electricity 和 Exchange 用 555，PM2.5 用 556。服务器在项目根目录执行以下命令，入口只选择物理 GPU 1。

```bash
git -c http.version=HTTP/1.1 pull --ff-only origin main &&
/usr/bin/python scripts/benchmark_inference.py
```

在 FP32 下，每个真实测试窗口单独前向一次，记录延迟中位数和 P95。批量模式用 64、32、128 个窗口分别处理 Electricity、PM2.5 和 Exchange，每个数据集完整跑十遍，包含尾批，报告每秒处理的窗口数。两项测量都先预热，模型权重和输入已在设备上，计时包括递归预测和解释系数计算。CUDA 在计时边界同步。结果同时输出到终端和 `outputs/h100_gpu1_delelstm45/inference/`。[测速入口](../scripts/benchmark_inference.py) · [PyTorch 同步计时说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)

本地 CPU 已用三个数据集的短测权重验证测速流程，按原批量重算的预测与保存值逐项一致。正式 H100 推理数据待服务器回传。

## 下一步

小组可以用 Word 表 3 和逐折汇总撰写汇报，把“按未来时段检验泛化、消除时间泄漏”作为已验证的贡献。如果继续研究两项 PM2.5 失败，先保留服务器 `outputs/h100_gpu1_delelstm45/jobs/` 中两个原始 `attempt_001` 的 `run.log`、`failure_location.json`、`failed_batch.npz` 和 `failed_model.pt`，再按原种子定位首次非有限运算。若要检验修复本身能否降低 RMSE，后续对照需固定同一未来测试段、目标掩码和种子。复跑入口与数据获取方式见[服务器运行说明](../doc/ablation_workflow.md#在服务器运行)。

## 报告中已有的结果

三个数据集各训练五种模型，每种模型成功完成五次，共 75 次。历史训练使用 macOS CPU，固定随机划分种子为 555。五次重复衡量同一划分下的训练波动。逐项核对 75 份测试指标和 75 个权重文件后，从明细重算的 RMSE、MAE、MAPE 均值与样本标准差与表 1、附录 A 的显示精度一致。[逐次记录](../results/five_run/verified_detail.csv) · [汇总表](../results/five_run/verified_summary.csv) · [训练环境](../results/five_run/environment.json)

下表两列 RMSE 均来自同一批 DeLELSTM 权重。左列是原代码丢弃测试尾批后的五次结果。右列是固定每次权重、补回尾批后的五次结果。

| 数据集 | 原测试 RMSE，均值 ± SD | 完整随机测试集 RMSE，均值 ± SD | 原保留 / 全部窗口 | 原测试中平均 RMSE 最低的模型 |
| --- | ---: | ---: | ---: | --- |
| Electricity | 1.7185 ± 0.0596 | 1.6825 ± 0.0357 | 64 / 92 | DeLELSTM |
| PM2.5 | 26.3124 ± 0.4471 | 25.6103 ± 0.4114 | 160 / 184 | RETAIN（24.9538） |
| Exchange | 0.002293 ± 0.000069 | 0.002273 ± 0.000060 | 640 / 757 | DeLELSTM |

- **图 3 的完整测试重评估已完成。** 加载 15 个 DeLELSTM 权重逐个复算，原保留窗口上的 RMSE 均在附录 B 的容差内。补回尾批后，Electricity 有 4/5 次误差下降、1/5 次上升，另两组各 5/5 次下降。模型仍沿用原验证集选出的权重。[15 组配对明细](../results/figures/audit_full_test_five_runs.csv) · [五次训练权重](../outputs/five_run/)
- **图 2 的 PM2.5 解释曲线已完成。** 从五份测试解释系数重算了 5 × 22 个位置值。目标变量首位置的平均瞬时占比为 0.603 ± 0.183，末位置为 0.023 ± 0.022，五次均为末值低于首值。横轴表示窗口内的输出位置，系数已在每次运行的保留测试窗口上取平均。[110 个位置明细](../results/figures/pm25_instantaneous_share_five_runs.csv) · [复算和绘图脚本](../scripts/reproduce_paper_figures.py)
- **原协议的局限已有定量记录。** Exchange 的 757 个随机测试窗口全部与训练窗口共享原始时点，其中 713 个与相差一天的训练窗口重叠。PM2.5 另有一次种子 555 的早期训练在第 85 轮出现 NaN，原因待查。五次成功结果使用种子 556–560。[窗口重叠记录](../results/audits/exchange_overlap.txt) · [失败日志](../results/audits/pm_seed555_nan.txt)

## 时间划分与补充实验

`chronological` 是 `main` 内的一种实验协议。当前三种协议的区别如下。[数据划分](../scripts/ablation_data.py) · [选模和评分](../scripts/ablation_engine.py)

| 协议 | 划分与归一化 | 验证选模和测试主指标 |
| --- | --- | --- |
| `released` | 随机窗口，全数据 min–max | 验证和测试均沿用原保留批次 |
| `full_eval` | 同一随机划分，全数据 min–max | 完整验证集选模，完整测试集评分 |
| `chronological` | 按时间扩展训练区间，每折用训练集拟合 min–max | 完整验证集选模，随后在更晚的测试区间评分 |

**已验证。** 三个数据集各三折，训练、验证、测试时间严格有序，任意两集合共享的原始时点均为零。Exchange 每处边界剔除后一集合的前 29 个窗口。扰动验证及测试输入后，九折的训练极值和缩放后的训练输入均保持一致。原随机划分下，PM2.5 训练窗口的污染物输入范围是 1–886，全数据缩放却用了 0–994。九折时间范围、有效目标数量和原协议缩放范围均已归档。[九折核查表](../results/audits/temporal_protocol.csv) · [复算脚本](../scripts/audit_temporal_protocol.py)

**验证与可选实验。** 此前在本地 CPU 通过 13 项协议、环境与报告测试，以及 32 项一轮端到端短测试。通用脚本支持五模型预测比较和求解器、变量充分性消融。本阶段通过 `--forecast-only --models Delelstm` 选择其中的 45 项。[测试代码](../tests/test_ablation.py) · [任务定义](../scripts/ablation_suite.py) · [运行说明](../doc/ablation_workflow.md)

本地已通过 15 项测试。单模型续跑另用 CPU 在 Electricity 和 PM2.5 的末折做了一轮小规模验证，每个分区取 16 个窗口。Electricity 种子 555 的预测文件复用前后逐字节相同，两个 PM2.5 任务成功续跑。Electricity 种子 556 在首个批次出现非有限梯度，单独从头运行也复现了该异常，失败按原种子记录。此处的四项均为流程验证，H100 正式成绩单独统计。[单模型续跑测试](../tests/test_delelstm_forecast.py)

## 后续任务

**小组协作资料**

- [x] 仓库首页与结果索引指向可共同编辑的 Word 稿。共享稿保留六位组员署名、报告正文和小组编写的附录程序。含学号的课程提交稿在本地 `doc/6013_Paper.docx`。报告图 1–3 的原图、逐次数据及复算脚本均已归档。[共享稿](../doc/6013_Paper_public.docx) · [图表与明细](../results/figures/) · [结果索引](../results/README.md)
- [x] 固定上游模型和输入数据已接入小组 Git 仓库。服务器准备时校验 6 个核心源码文件和 9 份 CSV，含 PM2.5 的原始小时观测。[固定版本](../vendor/upstream) · [校验清单](../patches/upstream.json)
- [x] 2026-11-18 从 GitHub 递归克隆新副本，取得 9 份 CSV 和 75 个历史训练权重。在该副本用 Windows CPU 和现有 Python 环境运行服务器入口，三个数据集、五种模型的 15 项一轮测试全部成功。重新生成的九折核查表与仓库版本逐字节一致，正式计划显示 225 次训练。明细表的 `github_recursive_clone` 行记录了这次试跑。[验证过的代码版本](https://github.com/CanyonChen/DSC6013-Group17-DeLELSTM-Recap/commit/c9d92f5aa1781e308ce7a31ea972b4d436665c6b) · [短测试明细](../results/audits/temporal_smoke.csv)
- [x] Word 参考文献第 2 条已按出版社记录修订为 ICML 2019，PMLR 97，第 2494–2504 页。[出版社记录](https://proceedings.mlr.press/v97/guo19b.html)

**未来时段预测与变量充分性**的完成记录如下。

1. [x] **PM2.5 输入与评分已修正，2026-11-18。** 按原始 PRSA 小时记录前向填补输入，剔除无历史可用的首日。训练损失、验证选模及测试 RMSE/MAE/MAPE 均使用原始观测目标。末折训练、验证、测试分别有 28,338、5,975、3,964 个有效目标，测试时剔除了 4,026 个候选位置中的 62 个。[数据读取](../scripts/ablation_data.py) · [训练与评分](../scripts/ablation_engine.py)
2. [x] **Electricity 变量充分性流程已切换到最后一个时间折，2026-11-18。** 四种解释模型在同一末折训练集导出排名。前 8 个变量、随机 8 个和全部 16 个变量的 LSTM 共用种子、划分及测试窗口。排名依赖、重训、配对汇总均通过短测试。[任务定义](../scripts/ablation_suite.py) · [配对汇总](../scripts/ablation_report.py)
3. [x] **数据与训练路径已验证，2026-11-18。** 改动未来原始 PM2.5 观测不影响此前填补输入。只改动缺失目标处的预测，训练损失、有效位置梯度和评分保持一致。9 个完整时间划分均已核对，32 项短测试全部完成。另用 PM2.5 末折的全部窗口训练 DeLELSTM 一轮，从保存的 183 × 22 个预测位置中取出 3,964 个真实目标，独立重算的 RMSE、MAE 与运行记录一致。[协议测试](../tests/test_ablation.py) · [各折明细](../results/audits/temporal_protocol.csv)
4. [x] **正式回测及报告回填，2026-11-19。** 全部 45 项均已有最终状态，43 项产生预测成绩。九组指标、两项训练失败及历史参照已写入共享报告第 6.4、7 节。[表 3 所用汇总](../results/chronological/delelstm45/summary.csv) · [全部 45 条记录](../results/chronological/delelstm45/runs.csv)
5. 同一未来测试段上的修复前后配对、其他模型的预测排名、Top-50% 和求解器消融均列为后续可选实验。[扩展实验配置](../doc/ablation_workflow.md)

后续更新时，在对应任务旁补上完成日期、实际完成数量和结果链接。
