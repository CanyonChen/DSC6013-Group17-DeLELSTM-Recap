# DeLELSTM Replication and Audit

DS6013 课程项目：复现 IJCAI 2023 的 **DeLELSTM**，比较其在 Electricity、PM2.5 和 Exchange 数据集上的预测结果，并核查数据处理、测试评估与解释系数。

已完成 **3 个数据集 × 5 个模型 × 5 次训练，共 75 次训练**。本仓库提供实验调度脚本、相对于上游的修改补丁、课程报告，以及可重新汇总的原始指标。

- [课程报告](doc/DeLELSTM_Five_Run_Course_Paper.docx)
- [实验结果与证据索引](results/README.md)
- [代码与实验核查](doc/code_audit.md)
- [原论文](https://doi.org/10.24963/ijcai.2023/478) · [作者代码](https://github.com/wangcq01/DeLELSTM)

## 主要结果

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
├── doc/                      # 最终课程报告与代码核查
├── environment/              # 安装说明与历史环境差异
├── scripts/                  # 准备代码、训练、汇总、解释与审计
├── patches/                  # 固定上游版本、修改补丁、源码校验值
├── results/                  # 提交到 Git 的小型实验指标与审计证据
├── vendor/DeLELSTM/           # 本地下载的上游代码和数据，Git 忽略
├── outputs/                  # 完整实验结果、训练权重和后续实验产出
└── local_archive/            # 历史日志、脚本与环境记录；第三方原件单独忽略
```

小组成员克隆后即可获得现有训练权重、完整实验产出和历史日志。`results/` 保存便于阅读的小型证据，`outputs/` 保存对应完整产物；各目录的用途见 [实验产物说明](outputs/README.md) 与 [历史材料说明](local_archive/README.md)。`setup_upstream.py` 获取运行所需的上游代码与处理后数据。

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

脚本下载上游提交 `b9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9`，应用 [复现补丁](patches/reproduction.patch)，并核验核心源码 SHA-256。再次运行会检查已有文件。训练直接使用随上游发布的处理后 CSV，无需重新执行原始预处理脚本。

`vendor/DeLELSTM/` 中的三个训练入口分别为 `Electricity.py`、`PM.py`、`Exchange.py`；模型定义在 `networks.py`，训练循环在 `model_training.py`，配置在 `config.py`。

## 运行实验

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

解释分析来自较早单次 GPU 实验的平均系数，没有逐样本系数或原时间戳。报告未完成 top-50% 变量充分性实验，也不声称解释具有因果性或结果具有统计显著性。详细证据见 [代码核查](doc/code_audit.md) 与 [课程报告](doc/DeLELSTM_Five_Run_Course_Paper.docx)。

## 获取和更新项目

远端仓库：[CanyonChen/DSC6013-Group17-DeLELSTM-Recap](https://github.com/CanyonChen/DSC6013-Group17-DeLELSTM-Recap)，默认分支为 `main`。

首次下载后，在仓库根目录执行前述环境安装与代码准备命令：

```bash
git clone https://github.com/CanyonChen/DSC6013-Group17-DeLELSTM-Recap.git
cd DSC6013-Group17-DeLELSTM-Recap
```

已有本地副本时，获取更新：

```bash
git pull --ff-only origin main
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
