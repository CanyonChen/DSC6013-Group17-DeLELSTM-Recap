# 运行环境

在仓库根目录执行：

```powershell
conda env create -f environment.yml
conda activate delelstm-gpu
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Conda 管理 Python 和环境隔离，环境内的 pip 安装根目录 `requirements.txt`。该文件固定 Python 3.13 对应的 PyTorch 2.10.0 CUDA 12.6 构建，以及 NumPy 2.4.6、pandas 3.0.1、scikit-learn 1.8.0、TorchMetrics 1.9.0。配置面向 Windows/Linux x86-64 和 NVIDIA GPU；CUDA 检查应输出 `2.10.0+cu126`、`12.6`、`True`。

PyTorch CUDA wheel 自带所需 CUDA 运行库，仍需要兼容的 NVIDIA 驱动。安装方案依据 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v2100)；项目不需要 torchvision 或 torchaudio。

这是供后续复现实验使用的 GPU 环境。报告中的五次正式实验在 macOS CPU 上运行，原环境记录在 [`results/five_run/environment.json`](../results/five_run/environment.json)。PyTorch、pandas、scikit-learn 和 TorchMetrics 主包版本与该记录对齐；原记录中的 [NumPy 2.4.0 已因兼容性问题被 PyPI 撤回](https://pypi.org/project/numpy/2.4.0/)，当前配置使用同系列补丁版 2.4.6。操作系统、计算设备和 NumPy 补丁版本不同，不能把新 GPU 运行当作原报告的精确环境，也不保证结果逐位一致。此前单次 GPU 实验的旧依赖快照保留在本地归档，未用作当前安装清单。

`requirements.txt` 固定直接依赖版本，传递依赖由 pip 解析；它不是跨平台完整锁文件。训练完成后，可在对应输出目录保存实际环境：

```powershell
python -m pip freeze > outputs/rerun/requirements-installed.txt
```
