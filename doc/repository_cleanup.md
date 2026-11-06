# 仓库整理记录

首次整理日期：2026-11-06。小组协作上传范围更新：2026-11-07。

## 当前布局

| 内容 | 当前位置 | Git 提交 |
|---|---|---|
| 最终课程报告 | `doc/DeLELSTM_Five_Run_Course_Paper.docx` | 是，原文件未修改 |
| 代码核查 | `doc/code_audit.md` | 是，已更新路径 |
| 环境与依赖 | `environment.yml`、`requirements.txt`、`environment/README.md` | 是 |
| 项目运行、分析和审计脚本 | `scripts/` | 是 |
| 上游版本和修改补丁 | `patches/` | 是 |
| 报告的小型指标、元数据、解释表和审计日志 | `results/` | 是 |
| 处理后的上游源码、数据和补充材料 | `vendor/DeLELSTM/` | 否，可用准备脚本恢复 |
| 原完整五次实验 | `outputs/five_run/` | 是 |
| 原单次 GPU 实验，包括审计所需检查点 | `outputs/single_run/` | 是 |
| 早期未完成实验 | `outputs/incomplete_run/` | 是，标明未完成 |
| 原日志 | `local_archive/logs/{five_run,single_run,incomplete_run}/` | 是 |
| 原代码 ZIP | `local_archive/original/DeLELSTM-main.zip` | 否，本地保留 |
| 原论文 PDF | `local_archive/references/DeLELSTM_IJCAI2023.pdf` | 否，本地保留 |
| 旧报告生成器、调度器、补丁和说明 | `local_archive/legacy_scripts/`、`local_archive/legacy_notes/` | 是，仅用于历史追溯 |
| 旧 GPU 依赖快照 | `local_archive/environment/requirements-gpu-original.txt` | 是，仅用于历史追溯 |

旧 `MANIFEST_SHA256.json` 保存为 `local_archive/MANIFEST_SHA256_original.json`。
其中路径对应原交付包，不是新目录的文件清单。过期的 70 次训练进度表仍随
完整原实验保存在 `outputs/five_run/`，公开结果只使用最终 75 次的 verified 表。

## 清理与代码调整

- 删除与保留 PDF 的 SHA-256 完全一致的根目录论文副本。
- 删除 `.DS_Store`、Python 字节码缓存和无文件的 staging 目录。
- 原源码和数据与准备脚本生成的内容逐文件核对后，删除旧目录中的重复副本；历史审计脚本另存本地归档。
- 将两阶段旧补丁整合为一个相对于固定上游提交的补丁，并保留最终源码校验值。
- 新训练入口采用串行执行，保留报告的固定划分、独立初始化与重试协议。
- 汇总脚本改为标准库实现，可处理公开指标中缺少训练轨迹的情况；分析结果写到 `outputs/`。
- 旧报告生成器依赖原机器绝对路径及缺失模板，移入历史归档；现有最终报告可直接阅读。
- 首次整理时初始化了本地 `main`；小组协作发布使用 `CanyonChen/DSC6013-Group17-DeLELSTM-Recap`。

## 已完成的验证

1. 首先用 Microsoft MarkItDown 读取现有 Word 报告，核对实验协议和主要结果。
2. 114 个历史证据文件复制前后 SHA-256 一致；75 份测试指标与原 verified 明细一致。
3. 新汇总脚本重算的 15 组 × 6 项均值/标准差与原 verified 汇总一致，比较容限为 `1e-12`。
4. 新解释脚本的 10 份 CSV 与原解释产物逐字节一致。
5. 从本地 ZIP、从 GitHub 指定提交下载两条准备路径均完成补丁应用和六个核心 Python 文件的哈希核验。
6. 六个项目 Python 脚本通过语法和帮助入口检查；训练入口 dry-run 正确列出数据集、模型和种子。
7. Conda YAML 与相对 requirements 路径完成实际解析；Conda Python/pip 环境完成 dry-run 求解，直接依赖的 CPython 3.13 Windows/Linux wheel 可用性已核查。
8. 首次整理时核对 Markdown 链接和小型证据的发布范围。2026-11-07 按小组协作要求扩展为提交全部训练权重、实验产物和日志；缓存、虚拟环境、临时验证文件及单独下载的第三方原件继续排除。

本次未安装完整 GPU 环境，未重新训练模型，也未重新执行依赖检查点的数值审计。
因此 GPU 训练兼容性仍以用户在目标设备上的环境检查和短训练为准；上述验证不等同于新的 75 次复现实验。
