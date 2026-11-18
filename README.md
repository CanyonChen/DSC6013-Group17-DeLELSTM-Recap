# DeLELSTM Replication and Audit

DS6013 课程项目：复现 IJCAI 2023 的 **DeLELSTM**，比较其在 Electricity、PM2.5 和 Exchange 数据集上的预测结果，并核查数据处理、测试评估与解释系数。

已完成 **3 个数据集 × 5 个模型 × 5 次训练，共 75 次训练**。本仓库提供实验调度脚本、相对于上游的修改补丁、课程报告，以及可重新汇总的原始指标。

- [小组协作版 Word 报告](doc/6013_Paper_public.docx)
- [项目进度与服务器下一步](docs/project_process.md)
- [实验结果与证据索引](results/README.md)
- [代码与实验核查](doc/code_audit.md)
- [服务器补充实验全流程](doc/ablation_workflow.md)
- [原论文](https://doi.org/10.24963/ijcai.2023/478) · [作者代码](https://github.com/wangcq01/DeLELSTM)

## 原随机划分下的主要结果

测试 RMSE，均值 ± 样本标准差，`n = 5`，越低越好：

| 模型 | Electricity | PM2.5 | Exchange |
|---|---:|---:|---:|
| DeLELSTM | **1.7185 ± 0.0596** | 26.3124 ± 0.4471 | **0.002293 ± 0.000069** |
| IMV-Full | 1.9995 ± 0.0751 | 28.6004 ± 0.6386 | 0.002626 ± 0.000011 |
| IMV-Tensor | 2.3636 ± 0.1947 | 29.3094 ± 0.9821 | 0.002638 ± 0.000004 |
| LSTM | 5.1152 ± 6.9259 | 26.5177 ± 0.1724 | 0.002619 ± 0.000005 |
| RETAIN | 2.6980 ± 0.2287 | **24.9538 ± 0.0536** | 0.002485 ± 0.000020 |

以上来自报告中的 **macOS CPU** 五次实验，详见 [运行环境原始记录](results/five_run/environment.json) 和 [75 次实验明细](results/five_run/verified_detail.csv)。所有重复使用同一划分种子 `555`，衡量固定划分上的训练随机性。下方 GPU 环境供后续运行，不能视为上述结果的原始环境。

## 目录结构

```text
.
├── README.md
├── environment.yml           # Conda 环境定义
├── requirements.txt          # CUDA PyTorch 与 Python 依赖
├── LICENSE                   # 本项目原创贡献的 MIT 协议
├── THIRD_PARTY_NOTICES.md     # 上游代码、数据和论文来源
├── doc/                      # 当前 Word 稿、历史报告和实验说明
├── docs/                     # 项目进度及下一轮实验
├── environment/              # 安装说明与历史环境差异
├── scripts/                  # 准备代码、训练、汇总、解释与审计
├── patches/                  # 固定上游版本、修改补丁、源码校验值
├── results/                  # 提交到 Git 的小型实验指标与审计证据
├── vendor/upstream/           # Git 子模块，固定版本的作者代码和三个数据集
├── vendor/DeLELSTM/           # 从子模块生成的本地训练副本
├── outputs/                  # 完整实验结果、训练权重和后续实验产出
└── local_archive/            # 历史日志、脚本与环境记录；第三方原件单独忽略
```

小组成员递归克隆后即可获得本组代码、训练权重、历史日志、作者模型和三个数据集。`results/` 保存报告图表与指标明细，`outputs/` 保存对应的训练产物。[实验产物说明](outputs/README.md) · [历史材料说明](local_archive/README.md)

## 安装 GPU 环境

需要 Git、Conda，以及可用的 NVIDIA GPU 和兼容驱动。以下命令在仓库根目录执行，适用于 Windows/Linux x86-64：

```bash
conda env create -f environment.yml
conda activate delelstm-gpu
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

预期输出包括 `2.10.0+cu126`、`12.6` 和 `True`。Conda 管理 Python 3.13 和环境隔离，环境内的 pip 通过 `requirements.txt` 安装 CUDA 12.6 版 PyTorch。[PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v2100)提供该构建。

其他依赖固定为 NumPy 2.4.6、pandas 3.0.1、scikit-learn 1.8.0、TorchMetrics 1.9.0。报告原 NumPy 2.4.0 已被 PyPI 撤回，故新环境使用补丁版本；详情见 [环境说明](environment/README.md)。

## 准备代码与数据

```bash
python scripts/setup_upstream.py
```

脚本从 Git 子模块的固定提交 `b9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9` 准备训练副本，应用小组补丁并校验 6 个核心源码文件与 9 份 CSV。普通克隆会在首次准备时自动补取子模块，递归克隆后可离线准备代码和数据。原协议读取处理后 CSV，时间回测用原始 PM2.5 小时记录构造输入和观测目标。[复现补丁](patches/reproduction.patch) · [数据与源码校验清单](patches/upstream.json)

`vendor/DeLELSTM/` 中的三个训练入口分别为 `Electricity.py`、`PM.py`、`Exchange.py`；模型定义在 `networks.py`，训练循环在 `model_training.py`，配置在 `config.py`。

## 服务器补充实验

预测未来时段时，使用以下入口。三个扩展时间折分别用训练数据拟合输入缩放参数，验证集选模，测试集在更晚的时段。Exchange 清除跨集合重叠窗口，PM2.5 只使用过去已知的输入并在原始观测上评分。

```bash
bash scripts/eval_and_ablation.sh --stage plan --forecast-only --gpus 0,1 --output-dir outputs/forecast_chronological
bash scripts/eval_and_ablation.sh --stage all --forecast-only --gpus 0,1 --output-dir outputs/forecast_chronological
```

预测专用入口先跑 15 项短测试，再运行 225 项正式时间回测。完成后查看 `outputs/forecast_chronological/report/report.md`。分到一张可见 GPU 时使用 `--gpus 0`。各折窗口数、测试区间和有效目标数见[时间协议核查表](results/audits/temporal_protocol.csv)。下方完整套件另含历史随机协议对照和消融。[进度与验收方法](docs/project_process.md)

Linux 双 GPU 服务器可从环境配置一直运行到补充实验报告：

```bash
bash scripts/eval_and_ablation.sh --stage plan
bash scripts/eval_and_ablation.sh --gpus 0,1
```

脚本创建独立 Conda 环境，先执行短测试，再完成完整测试审计、时间划分、Top-50% 变量充分性、求解器消融、解释稳定性及 CPU/GPU 对照。默认正式计划为 **554 项任务（90 次历史重评估、464 次新训练）**，每张 GPU 独立运行一个任务；另有 32 项短测试。

报告保存至 `outputs/ablation/report/report.html` 和 `report.md`，权重、预测与日志保存在同一实验目录。使用 `--stage run --resume --gpus 0,1` 跳过已成功任务；`--stage report` 可单独重建报告。参数、实验协议和后台运行方法见 [全流程说明](doc/ablation_workflow.md)。目前已完成本地 CPU 短测试，服务器正式结果须运行后生成。

## 复跑报告原实验

先查看计划，再运行小规模检查：

```bash
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --datasets Electricity --models Delelstm --num-runs 1 --epochs 2 --output-dir outputs/smoke
```

按报告的独立初始化协议执行全部 75 次训练：

```bash
python scripts/run_experiments.py --output-dir outputs/rerun
python scripts/summarize.py outputs/rerun --output-dir outputs/rerun_summary
```

- 默认串行使用 `cuda:0`；CUDA 不可用时明确退出。若需要 CPU，添加 `--device cpu`。
- 训练/验证/测试按窗口随机划分为 `75% / 15% / 约10%`，划分种子始终为 `555`。
- Electricity、Exchange 的五次训练种子为 `555–559`；PM2.5 为 `556–560`。
- Electricity 默认 350 轮，PM2.5、Exchange 默认 300 轮。`--epochs` 仅用于自定义试跑。
- 每个数据集的每次重复依次训练五种模型。失败时最多尝试六次，每次重试将训练种子增加 `100`；保留失败日志，并记录实际成功种子。
- 输出目录必须是新目录，以保留已有实验。日志在其 `logs/` 下，成功结果在 `{dataset}/exp_555_{run}/` 下。

可选模型参数为 `Delelstm IMV_full IMV_tensor Retain LSTM`；数据集参数为 `Electricity PM Exchange`。例如只跑 PM2.5 的 DeLELSTM：

```bash
python scripts/run_experiments.py --datasets PM --models Delelstm --output-dir outputs/pm_delelstm
```

直接调用上游入口的 `--num_exp 5` 使用连续随机数状态，与上述独立进程种子协议不同；报告复现请使用统一入口。原始训练文件的 `--resample_split` 选项可用于其他实验，但不属于本报告的固定划分结果。

## 查看和核验已有结果

以下两项只依赖 Python 标准库，无需 GPU 或下载上游：

```bash
python scripts/summarize.py
python scripts/interpret.py
```

汇总默认读取 `results/five_run/`，输出到 `outputs/summary/`；解释分析默认读取 `results/single_run/`，输出到 `outputs/interpret/`。历史结果保持原样。MAPE 汇总单位为百分比，标准差使用 `ddof=1`。

安装依赖并准备上游后，可重新核查 Exchange 窗口重叠：

```bash
python scripts/audit_exchange_overlap.py
```

完整测试集审计使用仓库中较早单次实验的原始检查点，默认路径为 `outputs/single_run/`：

```bash
python scripts/audit_full_test.py Electricity
python scripts/audit_full_test.py PM
python scripts/audit_full_test.py Exchange
```

可以对照 [已保存的审计结果](results/audits/full_test.txt) 核查运行输出。五次新训练的权重不应替代单次历史检查点来声称复核了同一组数值。

## 已知局限

原实现使用全数据 min–max 归一化和随机窗口划分；Exchange 滚动窗口之间有重叠，验证/测试还使用 `drop_last=True`。这些设置影响结果解读，不能将随机测试结果等同于未来时段预测表现。

当前报告图 2 使用五次 PM2.5 训练各自保存的平均解释系数，图 3 使用 15 个固定检查点的配对重评估。这些结果属于原随机划分。九折时间边界、训练集缩放和 PM2.5 观测目标已完成核查，未来时段预测与 Top-50% 的正式多种子比较仍待训练。[图表与数据](results/figures/) · [当前课程报告](doc/6013_Paper_public.docx)

## 获取和更新项目

远端仓库：[CanyonChen/DSC6013-Group17-DeLELSTM-Recap](https://github.com/CanyonChen/DSC6013-Group17-DeLELSTM-Recap)，默认分支为 `main`。

首次下载后，在仓库根目录执行前述环境安装与代码准备命令：

```bash
git clone --recurse-submodules https://github.com/CanyonChen/DSC6013-Group17-DeLELSTM-Recap.git
cd DSC6013-Group17-DeLELSTM-Recap
```

已有本地副本时，获取更新：

```bash
git pull --ff-only --recurse-submodules origin main
```

有写入权限的成员可提交并推送自己的改动：

```bash
git status --short
git add .
git commit -m "Describe your changes"
git push origin main
```

推送需要本机 Git 的 GitHub 身份认证。训练权重、完整实验产出和日志均纳入版本管理；`.gitignore` 排除缓存、虚拟环境、临时验证/上传文件，以及单独下载的第三方代码、数据和论文原件。

## 许可证与引用

本项目原创脚本、文档及修改采用 [MIT License](LICENSE)。上游版本未附显式许可证，因此以独立下载依赖和补丁的方式提供复现入口；MIT 不覆盖上游原代码、数据或论文。详见 [第三方材料说明](THIRD_PARTY_NOTICES.md)。

```bibtex
@inproceedings{wang2023delelstm,
  title     = {DeLELSTM: Decomposition-based Linear Explainable LSTM to Capture Instantaneous and Long-term Effects in Time Series},
  author    = {Wang, Chaoqun and Li, Yijun and Sun, Xiangqian and Wu, Qi and Wang, Dongdong and Huang, Zhixiang},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages     = {4299--4307},
  year      = {2023},
  doi       = {10.24963/ijcai.2023/478}
}
```
