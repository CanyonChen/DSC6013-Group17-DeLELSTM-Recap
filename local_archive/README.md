# 历史实验材料

下列材料随小组仓库保留，供追溯报告和复核实验：

- `logs/five_run/`：正式五次实验的训练日志和原 CPU 环境记录。
- `logs/single_run/`：较早单次 GPU 实验的日志。
- `logs/incomplete_run/`：早期失败/未完成实验日志，以及当时的审计输出。
- `environment/`：原单次 GPU 依赖快照，不作为当前安装清单。
- `legacy_scripts/`：原调度、汇总、报告生成器和补丁。部分脚本依赖旧机器路径或缺失模板，日常运行请使用根目录下的 `scripts/`。
- `legacy_notes/`：历史阶段说明，其中的完成状态和旧路径不代表当前项目状态。
- `MANIFEST_SHA256_original.json`：原交付包清单，其路径对应旧目录布局。

`original/` 中的上游源码压缩包和 `references/` 中的原论文 PDF
继续仅在本地保存；第三方源码和数据可用 `scripts/setup_upstream.py`
获取，论文入口见 [第三方材料说明](../THIRD_PARTY_NOTICES.md)。
