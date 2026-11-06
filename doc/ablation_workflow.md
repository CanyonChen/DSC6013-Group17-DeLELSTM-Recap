# 补充实验：服务器全流程

入口是 [`scripts/eval_and_ablation.sh`](../scripts/eval_and_ablation.sh)。它完成独立 Conda 环境安装、上游代码与数据校验、GPU 检查、短测试、正式实验和中文 Markdown/HTML 报告生成。默认使用两张可见 GPU，每张卡运行一个独立实验；额外的 CPU 对照使用一个 CPU 进程。模型直接来自固定版本的上游代码，求解器消融通过运行时替换实现。

目标环境为 Linux x86_64、Bash、Git、NVIDIA GPU。按此前服务器的双 H100、SSH 使用方式设计；当前没有连接服务器，也未验证其驱动或运行速度。脚本不依赖桌面或浏览器。

## 在服务器运行

首次获取仓库：

```bash
git clone https://github.com/CanyonChen/DSC6013-Group17-DeLELSTM-Recap.git
cd DSC6013-Group17-DeLELSTM-Recap
```

已有仓库时，在该仓库中获取更新：

```bash
git pull --ff-only origin main
```

查看计划后启动全流程：

```bash
bash scripts/eval_and_ablation.sh --stage plan
bash scripts/eval_and_ablation.sh --gpus 0,1
```

正式训练持续时间取决于服务器实测。SSH 运行可使用现有的 tmux 会话；也可以用以下命令后台启动，日志仍保留在仓库中：

```bash
mkdir -p outputs/launch_logs
nohup bash scripts/eval_and_ablation.sh --gpus 0,1 > outputs/launch_logs/ablation.log 2>&1 &
tail -f outputs/launch_logs/ablation.log
```

两个启动方式选一种。默认 `all` 依次执行环境安装、32 项短测试、554 项正式任务和报告生成。短测试失败时停止，不启动正式实验。已有正式任务失败时继续执行其他独立任务，最后生成含失败清单的报告并以非零状态退出。

## 环境与设备

- 默认创建独立环境 `.conda/delelstm-ablation`，无需 `conda activate`，也不复用 SuperMimic 的环境。
- 现有 Conda 可用时直接用它创建项目环境；没有 Conda 时下载 [Miniforge 官方 Linux x86_64 安装程序](https://github.com/conda-forge/miniforge)，核验 SHA-256 后安装到 `.conda/miniforge3`。需要服务器能访问 GitHub、conda-forge、PyPI 和 PyTorch 下载源。
- Python 3.13；依赖继承根目录 `requirements.txt` 的 PyTorch 2.10.0 CUDA 12.6 配置，并通过 [`environment/requirements-ablation.txt`](../environment/requirements-ablation.txt) 增加 Matplotlib。安装依据见 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/#v2100)。驱动仍须支持该 CUDA 运行库。
- 启动时实际检查指定 GPU 上的张量运算和线性求解，记录 Python、依赖、设备信息和 `pip freeze`。
- `--gpus 0,1` 使用可见设备的逻辑编号。调度器已设置 `CUDA_VISIBLE_DEVICES` 时沿用该限制；例如只分配了一张可见卡，应使用 `--gpus 0`。
- 使用 FP32，关闭 TF32，不使用混合精度。两个 GPU 分别处理不同任务，不需要 `torchrun`。

可以分阶段执行：

```bash
bash scripts/eval_and_ablation.sh --stage setup --gpus 0,1
bash scripts/eval_and_ablation.sh --stage smoke --gpus 0,1
bash scripts/eval_and_ablation.sh --stage run --gpus 0,1
bash scripts/eval_and_ablation.sh --stage report
```

使用已经安装好依赖的 Python 时，传入 `--python /absolute/path/to/python`；这会跳过环境安装。`--skip-setup` 则复用默认项目环境。路径参数相对于仓库根目录解释。

## 正式实验计划

默认包含三数据集、五模型、五种子。Electricity/Exchange 使用 `555–559`，PM2.5 使用 `556–560`；划分随机种子固定为 `555`。每个模型在独立进程中重新设种子，适合新对照实验，但不等同于旧脚本一次运行五模型、连续使用随机数状态的逐位复现。Electricity 每次训练 350 轮，PM2.5/Exchange 各 300 轮。

| 任务 | 默认数量 | 要回答的问题 |
|---|---:|---|
| 历史检查点重评估 | 90 | 15 个单次、75 个五次实验权重，完整测试与原保留窗口的指标差多少？ |
| `released` 训练 | 75 | 原随机划分、全数据归一化、验证丢弃尾批的对照 |
| `full_eval` 训练 | 75 | 保持上述数据划分与归一化，完整验证参与选模后的结果 |
| `chronological` 训练 | 225 | 三个时间回测区间、训练集归一化、完整验证/测试下的结果 |
| 求解器消融 | 60 | 最后一个时间折上，直接伪逆与三档 ridge 相对原求解器的变化 |
| Top-50% 变量充分性 | 25 | 四种解释模型选一半变量、随机一半变量，分别重训 LSTM |
| CPU 训练对照 | 3 | 每数据集一个匹配种子的 DeLELSTM CPU/GPU 对照 |
| PM 种子 555 诊断 | 1 | 当前环境是否再次出现历史非有限值问题，保留失败位置与批次 |
| **合计** | **554** | **90 次历史重评估、464 次新训练；另有 32 项短测试** |

训练、验证和测试中的每个“样本”是一个多步窗口。`released` 主指标沿用原测试 `drop_last=True` 保留的窗口；所有任务同时保存完整测试预测，`full_eval` 与时间划分以完整测试指标为主。固定同一检查点的完整/保留测试差异与重新训练后的差异分别报告。

### 时间划分

默认三折按源时间顺序使用以下边界：

| 折 | 训练区间 | 验证区间 | 测试区间 |
|---|---|---|---|
| 0 | 0–55% | 55–70% | 70–80% |
| 1 | 0–65% | 65–80% | 80–90% |
| 2 | 0–75% | 75–90% | 90–100% |

边界按窗口数量取整。Exchange 每个窗口覆盖 30 个原始时点，因此每个边界从后一分区剔除共享原始时点的 29 个窗口。每折只在训练集拟合 min–max，不裁剪超出训练范围的值。发布的 CSV 已经过上游缺失值处理；本流程修复窗口划分和缩放的泄漏，不重新从原始数据构建全部预处理。

`--temporal-folds 1` 只保留最后的 75/15/10 时间划分；`2` 保留后两折。不同时间折测试集合不同，不与随机划分分数进行配对显著性检验。

### 解释、充分性与数值稳定性

- DeLELSTM 导出逐样本、逐位置的两组系数、窗口 ID、变量名、输入时间和预测时间。系数位置对应输入位置 `1..T−2`，预测对应 `2..T−1`。Electricity/PM2.5 用实际输入时钟时间聚合，Exchange 只有源行号。
- 对两组系数分别取绝对值并做变量方向 L1 归一化，得到瞬时占比和欧氏范数重要性；同时比较原始幅度、先按样本归一化再聚合、先平均系数再归一化的排名。多种子之间计算 Spearman 和 Top-50% Jaccard。
- 充分性仅在 Electricity **训练集**上提取排名，选 8/16 个变量，重新训练相同配置的 LSTM，并比较全变量 LSTM。DeLELSTM 使用上述重要性，IMV 使用变量注意力 beta，RETAIN 使用绝对贡献；按样本和时间求均值。原论文没有完全说明这些聚合细节，因此这是明确定义的补充协议。
- 原求解器保留上游实现；`pinv` 直接对设计矩阵 A 求伪逆；`ridge` 求解 `(AᵀA + λI)c = AᵀH`，默认 λ 为 `1e-6, 1e-4, 1e-2`。它们参与整个训练与前向过程。所有 λ 都报告，不根据测试集挑选后只展示最优项。
- 每个 DeLELSTM 检查点默认均匀抽样 64 个测试窗口，在 float64 下诊断 cond(A)、cond(A)²、重建相对误差和系数幅度；模型和求解器仍为 FP32。
- 训练损失或梯度非有限时保存模型、输入批次和失败位置；其他异常保留 traceback。CPU/GPU 同检查点前向比较与重新训练对照单独记录。当前 seed 555 未失败不能否定历史 NaN，数值相关性也不能直接证明原因。

### 缩小范围与短测试

以下是只做 Electricity、两个模型、一个种子、一个时间折的**部分实验**，仍使用正式训练轮数：

```bash
bash scripts/eval_and_ablation.sh --gpus 0 \
  --datasets Electricity --models Delelstm LSTM --seeds 555 \
  --temporal-folds 1 --cpu-control-runs 0 --output-dir outputs/ablation_electricity
```

`--cpu-control-runs 0` 同时关闭 CPU 训练对照和历史 PM 种子 555 探测。`--diagnostic-windows` 控制矩阵诊断抽样量。完整选项见 `--help`。

`--stage smoke` 每个选定数据集/模型跑一个种子、一轮训练，每个分区最多 16 个窗口，额外覆盖所选求解器与充分性实验。它不运行历史审计和时间回测；这些路径另外由协议测试及本地集成测试检查。传入 `--epochs` 或 `--max-windows` 的结果都会标记为短测试/自定义运行，不能冒充正式结果。

## 恢复、报告和交付

正式结果默认在 `outputs/ablation/`，短测试在 `outputs/ablation-smoke/`，环境单独检查在 `outputs/ablation-environment-check/`。成功的任务不覆盖；每次失败或中断的重试使用原种子，新建 `attempt_002` 等目录。

```bash
# 与原运行使用同一输出目录及实验参数
bash scripts/eval_and_ablation.sh --stage run --gpus 0,1 --resume

# 随时根据已有文件重新生成报告，无需 GPU
bash scripts/eval_and_ablation.sh --stage report
```

`--resume` 是任务级恢复：跳过已成功任务，未完成的训练从头开始。代码哈希、实验参数或设备配置改变时要求新输出目录，避免把不同方案的结果混在一起。短测试的续跑使用 `--stage smoke --resume`。

| 文件 | 内容 |
|---|---|
| `settings.json` / `plan.json` | 参数、源码哈希及完整任务清单 |
| `environment.json` / `requirements-installed.txt` | 本次实际环境 |
| `jobs/<任务>/attempt_*/run.log` | 训练进度、异常和 traceback |
| `split.json` / `job.json` | 窗口划分、归一化参数和模型配置标识 |
| `best.pt` / `training.csv` | 最佳验证权重及每轮轨迹；历史审计复用已有权重 |
| `predictions.npz` / `result.json` | 完整预测、时间、指标与成功状态 |
| `sample_explanations.npz` / `importance.csv` | 逐样本解释和变量重要性 |
| `report/report.md` / `report/report.html` | 中文报告，可离线查看 HTML |
| `report/*.csv` / `report/*.png` | 汇总、完整测试审计、配对比较、解释稳定性和图表 |

报告使用成功运行的均值与样本标准差，同时显示成功数/计划数和失败原因。只对相同测试集、模型和种子的 RMSE 差值做配对符号翻转检验，并进行 Holm 校正；5 对样本时双侧精确检验最小 p 值为 0.0625。不会自动把数值下降写成显著改善，也不会改写原课程报告。

训练权重、正式实验产物、短测试产物和运行日志都可以提交到小组仓库；`.gitignore` 只排除本地环境、缓存和 `outputs/verification/` 等开发验证目录。训练结束后检查 `git status`，把选定输出目录与相关配置一起提交；脚本本身不在服务器自动执行 Git 提交或推送。

## 本地验证范围

在 Windows CPU 隔离环境中通过了 9 项协议与报告测试、32 项短测试以及 11 项包含历史检查点和时间划分的集成任务，并验证已完成任务可跳过续跑。测试覆盖原预处理对齐、时间边界去重、训练集归一化、目标/时间对齐、求解器梯度、统计计算和失败记录；报告由这些真实短测试生成。

```bash
python -m unittest discover -s tests -v
```

上述结果证明本地流程可以运行，不代表 Linux 自动安装、H100 数值行为或正式 300/350 轮实验已经验证。服务器第一次运行仍会先执行环境检查与短测试。
