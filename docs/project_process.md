# DSC6013 第 17 组｜DeLELSTM 复现项目进度

更新于 **2026-11-18**。原随机划分下的 75 次训练已完成。报告第 6.1、6.3 节及表 2 已写入未来信息泄漏的证据、每折归一化方法和有效目标数。修正后的九个时间划分已通过核查，**下一项是服务器上的 225 次正式时间回测**。当前报告的模型成绩来自原随机协议。[共享 Word 报告](../doc/6013_Paper_public.docx) · [九折数据核查](../results/audits/temporal_protocol.csv) · [逐项短测试记录](../results/audits/temporal_smoke.csv)

## 下一步：在服务器运行正式时间回测

首次使用，在 Linux 服务器执行以下两行。一次递归克隆会取回本组代码、既有权重，以及固定作者版本内三个数据集的原始 CSV 和处理后 CSV。已有小组仓库的成员在仓库根目录执行 `git pull --ff-only --recurse-submodules origin main` 即可。首次接收此更新时，准备脚本会自动初始化新增的数据子模块。

```bash
git clone --recurse-submodules https://github.com/CanyonChen/DSC6013-Group17-DeLELSTM-Recap.git
cd DSC6013-Group17-DeLELSTM-Recap
```

在仓库根目录运行以下命令。示例使用两张可见 GPU。只分到一张时，两行都改为 `--gpus 0`。

```bash
bash scripts/eval_and_ablation.sh --stage plan --forecast-only --gpus 0,1 --output-dir outputs/forecast_chronological
bash scripts/eval_and_ablation.sh --stage all --forecast-only --gpus 0,1 --output-dir outputs/forecast_chronological
```

第一行应显示 225 项训练。第二行准备环境、核验代码和数据，先跑 15 项一轮短测试，再从头运行 3 个数据集 × 5 种模型 × 5 个种子 × 3 折。Electricity 每次 350 轮，PM2.5 与 Exchange 各 300 轮。已装好依赖的服务器可在第二行追加 `--python /实际环境路径/bin/python`。正式训练中断后，用相同设置和输出目录续跑。

```bash
bash scripts/eval_and_ablation.sh --stage run --resume --forecast-only --gpus 0,1 --output-dir outputs/forecast_chronological
```

训练结束后，检查 `outputs/forecast_chronological/report/report.md` 的成功数和计划数，逐个核对 `runs.csv` 的种子、状态与有效测试目标数。用同目录下的 `summary.csv` 填写 45 个“数据集 × 模型 × 时间折”组合的 RMSE、MAE、MAPE，列出每组实际成功数、均值和样本标准差。保留失败任务及其日志。划分、归一化和 PM2.5 目标处理同时变化，原随机成绩与新成绩的差额包含这些变化及测试时段差异。[服务器说明](../doc/ablation_workflow.md)

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

**现有运行范围。** 2026-11-18 修订后在本地 CPU 通过 12/12 项协议与报告测试，以及 32/32 项一轮端到端短测试，覆盖三个数据集、五种模型、求解器及同折变量筛选。旧版另有 11 项小规模集成任务和 1 项全测试集 Exchange 权重审计。`--forecast-only` 默认只运行 225 项正式时间回测。完整计划仍为 554 项，含历史对照、60 项求解器消融和 25 项末折变量充分性训练。正式时间回测未在本轮运行。[测试代码](../tests/test_ablation.py) · [任务定义](../scripts/ablation_suite.py) · [运行说明](../doc/ablation_workflow.md)

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
4. [ ] **完成 225 次正式时间回测。** 归档逐次与汇总结果，更新报告第 6、7 节，保留各折日期或源行号、有效目标数和失败记录。
5. [ ] **补充分模型解释实验。** 在末折完成 25 项 Top-50% 变量充分性和 60 项求解器消融，连同对应的全变量基线、条件数、重建误差和解释稳定性归档。[正式实验配置](../doc/ablation_workflow.md)

后续更新时，在对应任务旁补上完成日期、实际完成数量和结果链接。
