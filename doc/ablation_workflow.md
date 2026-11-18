# 补充实验：服务器全流程

入口是 [`scripts/eval_and_ablation.sh`](../scripts/eval_and_ablation.sh)。它完成独立 Conda 环境安装、上游代码与数据校验、GPU 检查、短测试、正式实验和中文 Markdown/HTML 报告生成。默认使用两张可见 GPU，每张卡运行一个独立实验；额外的 CPU 对照使用一个 CPU 进程。模型直接来自固定版本的上游代码，求解器消融通过运行时替换实现。

目标环境为 Linux x86_64、Bash、Git 和 NVIDIA GPU。当前 H100 任务只使用 `nvidia-smi` 编号为 1 的物理 GPU。服务器已确认 `/usr/bin/python` 可识别两张 GPU，PyTorch 为 `2.7.1+cu126`。

## 在服务器运行

首次获取仓库：

```bash
git clone --recurse-submodules https://github.com/CanyonChen/DSC6013-Group17-DeLELSTM-Recap.git
cd DSC6013-Group17-DeLELSTM-Recap
```

已有仓库时，在该仓库中获取更新：

```bash
git pull --ff-only --recurse-submodules origin main
```

本轮先运行未来时段预测。Git 子模块 `vendor/upstream` 含作者固定版本的代码、三个数据集的原始 CSV 及处理后 CSV。首次运行会从这份本地副本应用补丁、检验 6 个源码文件和 9 份数据。已有普通克隆会自动初始化子模块。递归克隆完成后，准备训练数据可在离线状态下进行。[版本和校验值](../patches/upstream.json)

```bash
/usr/bin/python scripts/run_delelstm_gpu1.py
```

本阶段仅运行 DeLELSTM 的 45 项正式时间回测，结果保存在 `outputs/h100_gpu1_delelstm45/report/`。已有五模型训练结果时，先停止旧队列，再在命令末尾追加 `--reuse-from outputs/h100_gpu1_torch271`。入口按 UUID 选择物理 GPU 1，进程内编号为 `cuda:0`。`environment.json` 保存实际设备和 PyTorch 版本。[切换与验收](../docs/project_process.md) · [CUDA 设备编号说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)

以下命令用于另行启动完整套件，包括历史检查点和求解器消融。

```bash
bash scripts/eval_and_ablation.sh --stage plan --gpus 0
gpu1_uuid=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader) &&
CUDA_VISIBLE_DEVICES="$gpu1_uuid" bash scripts/eval_and_ablation.sh --gpus 0
```

正式训练持续时间取决于服务器实测。SSH 运行可使用现有的 tmux 会话；也可以用以下命令后台启动，日志仍保留在仓库中：

```bash
mkdir -p outputs/launch_logs
CUDA_VISIBLE_DEVICES="$gpu1_uuid" nohup bash scripts/eval_and_ablation.sh --gpus 0 > outputs/launch_logs/ablation.log 2>&1 &
tail -f outputs/launch_logs/ablation.log
```

两个启动方式选一种。默认 `all` 依次执行环境安装、32 项短测试、554 项正式任务和报告生成。短测试失败时停止，不启动正式实验。已有正式任务失败时继续执行其他独立任务，最后生成含失败清单的报告并以非零状态退出。

### 下载中断后重试

主仓库克隆时出现 `early EOF` 或 `unexpected disconnect`，可以从原先执行克隆的父目录分两次取回本轮预测需要的文件。第一步选取训练代码与环境配置，第二步取回含三个数据集的固定上游版本。下面的命令使用新目录 `DSC6013-forecast`。

```bash
git -c http.version=HTTP/1.1 clone --depth 1 --filter=blob:none --sparse --branch main https://github.com/CanyonChen/DSC6013-Group17-DeLELSTM-Recap.git DSC6013-forecast &&
cd DSC6013-forecast &&
git -c http.version=HTTP/1.1 sparse-checkout set scripts environment patches tests vendor docs &&
git -c http.version=HTTP/1.1 submodule update --init --depth 1 -- vendor/upstream
```

最后一行成功时会显示子模块已检出 `b9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9`。如果这一行中断，在 `DSC6013-forecast` 中重新执行该行即可。接着执行本页的 `run_delelstm_gpu1.py` 命令。日后需要复评已有权重或编辑 Word 报告时，在同一目录执行 `git sparse-checkout disable` 即可补齐当前版本的其他文件。[Git 按需克隆说明](https://git-scm.com/docs/git-clone)

### cuDNN 下载超时

项目固定环境使用 PyTorch `2.10.0+cu126`。当前安装清单从 PyTorch 的 torch 文件列表选择这个构建，cuDNN 和其他运行库从常规 Python 包索引下载。NVIDIA 在 PyPI 发布的 Linux x86-64 cuDNN `9.10.2.21` 与原下载链接的 SHA-256 一致。若旧安装反复显示 `pypi.nvidia.com` 连接超时，在安装终端按 `Ctrl+C`，然后从仓库根目录执行下列命令。以下设置只对这次安装生效。[cuDNN 发布文件及校验值](https://pypi.org/project/nvidia-cudnn-cu12/9.10.2.21/#files)

```bash
git -c http.version=HTTP/1.1 pull --ff-only origin main &&
gpu1_uuid=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader) &&
env -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple PIP_RETRIES=2 PIP_TIMEOUT=30 CUDA_VISIBLE_DEVICES="$gpu1_uuid" bash scripts/eval_and_ablation.sh --stage all --forecast-only --models Delelstm --gpus 0 --output-dir outputs/h100_gpu1_torch210_delelstm45
```

此路径复用已创建的项目 Conda 环境，在验证依赖和 GPU 后开始短测试。使用 `/usr/bin/python` 的运行保存在独立的 `torch271` 目录。两种环境的实际版本分别记入对应的实验记录。

## 环境与设备

- 默认创建独立环境 `.conda/delelstm-ablation`，无需 `conda activate`，也不复用 SuperMimic 的环境。
- 现有 Conda 可用时直接用它创建项目环境；没有 Conda 时下载 [Miniforge 官方 Linux x86_64 安装程序](https://github.com/conda-forge/miniforge)，核验 SHA-256 后安装到 `.conda/miniforge3`。需要服务器能访问 GitHub、conda-forge、PyPI 和 PyTorch 下载源。
- Python 3.13；依赖继承根目录 `requirements.txt` 的 PyTorch 2.10.0 CUDA 12.6 配置，并通过 [`environment/requirements-ablation.txt`](../environment/requirements-ablation.txt) 增加 Matplotlib。安装依据见 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/#v2100)。驱动仍须支持该 CUDA 运行库。
- 启动时实际检查指定 GPU 上的张量运算和线性求解，记录 Python、依赖、设备信息和 `pip freeze`。
- `--gpus 0,1` 使用可见设备的逻辑编号。调度器已设置 `CUDA_VISIBLE_DEVICES` 时沿用该限制；例如只分配了一张可见卡，应使用 `--gpus 0`。
- 使用 FP32，关闭 TF32，不使用混合精度。两个 GPU 分别处理不同任务，不需要 `torchrun`。

可以分阶段执行：

```bash
CUDA_VISIBLE_DEVICES="$gpu1_uuid" bash scripts/eval_and_ablation.sh --stage setup --gpus 0
CUDA_VISIBLE_DEVICES="$gpu1_uuid" bash scripts/eval_and_ablation.sh --stage smoke --gpus 0
CUDA_VISIBLE_DEVICES="$gpu1_uuid" bash scripts/eval_and_ablation.sh --stage run --gpus 0
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

边界先按原始窗口数量取整。Exchange 每个窗口覆盖 30 个原始时点，每处边界从后一分区剔除前 29 个窗口。每折只在保留的训练窗口拟合逐变量 min–max，验证与测试复用相同参数，超出训练范围的值不裁剪。目标始终保留原单位。

PM2.5 的时间协议读取原始 PRSA 小时记录，按时间前向填补输入，剔除起始无历史可用的一天。训练损失、验证选模与测试评分只使用有原始观测的目标。各集合的时间范围、候选目标数与有效目标数保存在 `split.json`，测试预测同时保存 `target_observed` 掩码。原随机协议沿用发布数据以复核历史成绩。[九折核查结果](../results/audits/temporal_protocol.csv)

`--temporal-folds 1` 只保留最后的 75/15/10 时间划分；`2` 保留后两折。不同时间折测试集合不同，不与随机划分分数进行配对显著性检验。

### 解释、充分性与数值稳定性

- DeLELSTM 导出逐样本、逐位置的两组系数、窗口 ID、变量名、输入时间和预测时间。系数位置对应输入位置 `1..T−2`，预测对应 `2..T−1`。Electricity/PM2.5 用实际输入时钟时间聚合，Exchange 只有源行号。
- 对两组系数分别取绝对值并做变量方向 L1 归一化，得到瞬时占比和欧氏范数重要性；同时比较原始幅度、先按样本归一化再聚合、先平均系数再归一化的排名。多种子之间计算 Spearman 和 Top-50% Jaccard。
- 充分性在 Electricity 最后一个时间折的训练集上提取排名，选 8/16 个变量，重新训练同配置 LSTM。随机 8 个变量、全部 16 个变量的 LSTM 均使用同一时间折、种子和测试窗口。DeLELSTM 使用上述重要性，IMV 使用变量注意力 beta，RETAIN 使用绝对贡献，按样本和时间求均值。
- 原求解器保留上游实现；`pinv` 直接对设计矩阵 A 求伪逆；`ridge` 求解 `(AᵀA + λI)c = AᵀH`，默认 λ 为 `1e-6, 1e-4, 1e-2`。它们参与整个训练与前向过程。所有 λ 都报告，不根据测试集挑选后只展示最优项。
- 每个 DeLELSTM 检查点默认均匀抽样 64 个测试窗口，在 float64 下诊断 cond(A)、cond(A)²、重建相对误差和系数幅度；模型和求解器仍为 FP32。
- 训练损失或梯度非有限时保存模型、输入批次和失败位置；其他异常保留 traceback。CPU/GPU 同检查点前向比较与重新训练对照单独记录。当前 seed 555 未失败不能否定历史 NaN，数值相关性也不能直接证明原因。

### 缩小范围与短测试

以下是只做 Electricity、两个模型、一个种子、一个时间折的**部分实验**，仍使用正式训练轮数：

```bash
CUDA_VISIBLE_DEVICES="$gpu1_uuid" bash scripts/eval_and_ablation.sh --gpus 0 \
  --datasets Electricity --models Delelstm LSTM --seeds 555 \
  --temporal-folds 1 --cpu-control-runs 0 --output-dir outputs/ablation_electricity
```

`--cpu-control-runs 0` 同时关闭 CPU 训练对照和历史 PM 种子 555 探测。`--diagnostic-windows` 控制矩阵诊断抽样量。完整选项见 `--help`。

`--forecast-only` 单独运行时间回测的主预测实验。完整套件的 `--stage smoke` 在最后一个时间折对每个数据集、模型跑一个种子和一轮训练，每个分区最多 16 个窗口，同时覆盖求解器和变量充分性。使用 `--epochs` 或 `--max-windows` 的结果标记为短测试或自定义运行。

## 恢复、报告和交付

正式结果默认在 `outputs/ablation/`，短测试在 `outputs/ablation-smoke/`，环境单独检查在 `outputs/ablation-environment-check/`。成功的任务不覆盖；每次失败或中断的重试使用原种子，新建 `attempt_002` 等目录。

```bash
# 与原运行使用同一输出目录及实验参数
CUDA_VISIBLE_DEVICES="$gpu1_uuid" bash scripts/eval_and_ablation.sh --stage run --gpus 0 --resume

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

2026-11-18 的 Windows CPU 验证通过了 13 项协议、环境与报告测试和 32 项时间划分端到端短测试。另有早期版本完成的 11 项集成任务。当前测试覆盖九折时间边界、训练集缩放、PM2.5 前向填补、缺失位置的损失与梯度、验证和测试评分，以及末折训练集排名与重训依赖。

```bash
python -m unittest discover -s tests -v
```

## H100 服务器验证

组员在 Linux 上复用 `/usr/bin/python` 运行了 `--stage smoke --forecast-only`。提交 `6a0e3a6` 的服务器报告记录 15/15 项成功，环境为单张可见的 NVIDIA H100 80GB HBM3、PyTorch 2.7.1 和 CUDA 12.6。后续回传的正式运行中，已有两个 Electricity 第一折的 DeLELSTM 种子完成，45 项单模型汇总待回传。[服务器短测试摘要](../results/audits/h100_forecast_smoke.json) · [GPU 1 正式启动命令](../docs/project_process.md)
