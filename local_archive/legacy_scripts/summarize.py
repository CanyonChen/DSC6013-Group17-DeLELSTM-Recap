# -*- coding: utf-8 -*-
"""汇总复现结果并与论文 Table 1（papers/0478.pdf）对照，输出 mean ± std。

用法:
    python summarize.py                         # 汇总 ./results（默认）
    python summarize.py results_A results_B     # 汇总多个结果根目录
    python summarize.py ..\\rerun_results         # 汇总任意目录

输出（写到每个结果根目录下）:
    summary.csv          每个 数据集×模型 一行，含 mean 与 std（[改动 10]）
    summary_detail.csv   每次运行的明细（[改动 10] 明细与聚合分离）
    summary.md           与论文 Table 1 并排的对照表
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if len(sys.argv) > 1:
    RES_DIRS = [Path(a) if Path(a).is_absolute() else ROOT / a for a in sys.argv[1:]]
else:
    RES_DIRS = [ROOT / "results"]

# Table 1 of IJCAI-23 paper 0478.pdf (mean over 5 runs, raw target scale)
PAPER = {
    ("Electricity", "LSTM"): (5.1049, 3.9574, 6.49),
    ("Electricity", "Retain"): (2.6685, 2.0000, 3.21),
    ("Electricity", "IMV_full"): (2.0284, 1.4085, 2.28),
    ("Electricity", "IMV_tensor"): (2.6069, 1.7526, 2.74),
    ("Electricity", "Delelstm"): (1.7247, 1.1025, 1.69),
    ("PM2.5", "LSTM"): (26.566, 13.466, 22.47),
    ("PM2.5", "Retain"): (24.913, 13.402, 21.23),
    ("PM2.5", "IMV_full"): (28.465, 14.087, 25.21),
    ("PM2.5", "IMV_tensor"): (28.866, 14.480, 26.67),
    ("PM2.5", "Delelstm"): (26.612, 13.553, 22.16),
    ("Exchange", "LSTM"): (0.0026, 0.0015, 0.23),
    ("Exchange", "Retain"): (0.0025, 0.0015, 0.23),
    ("Exchange", "IMV_full"): (0.0026, 0.0015, 0.22),
    ("Exchange", "IMV_tensor"): (0.0026, 0.0015, 0.23),
    ("Exchange", "Delelstm"): (0.0023, 0.0015, 0.23),
}

# 结果目录名 -> 数据集名。作者代码里 PM2.5 写成 newrevision_pmc，这里都兼容。
DATASET_DIRS = {
    "Electricity": ["Electricity"],
    "PM2.5": ["newrevision_pmc", "newrevision_pm"],
    "Exchange": ["Exchange"],
}

MODEL_ORDER = ["Delelstm", "IMV_full", "IMV_tensor", "LSTM", "Retain"]


def collect(root):
    rows = []
    for dataset, subdirs in DATASET_DIRS.items():
        for sub in subdirs:
            base = root / sub
            if not base.exists():
                continue
            for exp in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("exp_")):
                for model_dir in sorted(p for p in exp.iterdir() if p.is_dir()):
                    tests = list(model_dir.glob("*_Explain_test_results.csv"))
                    if not tests:
                        continue
                    t = pd.read_csv(tests[0])
                    train = list(model_dir.glob("*_Expalin_train_results.csv"))
                    tr = pd.read_csv(train[0]) if train else None
                    rows.append({
                        "root": root.name,
                        "dataset": dataset,
                        "exp": exp.name,
                        "model": model_dir.name,
                        "rmse": float(t.loc[0, "rmse_test"]),
                        "mae": float(t.loc[0, "mae_test"]),
                        "mape_percent": float(t.loc[0, "mape_test"]) * 100,
                        "epochs": int(len(tr)) if tr is not None else None,
                        "best_val_rmse": float(tr["rmse_val"].min()) if tr is not None else None,
                        "best_val_epoch": int(tr["rmse_val"].idxmin()) if tr is not None else None,
                    })
    return pd.DataFrame(rows)


def fmt(mean, std, nd):
    """n>1 时输出 mean±std；n=1 时只输出 mean 并显式标注。"""
    if std is None or pd.isna(std):
        return f"{mean:.{nd}f} (n=1)"
    return f"{mean:.{nd}f}±{std:.{nd}f}"


def main():
    if not any(d.exists() for d in RES_DIRS):
        raise SystemExit(f"结果目录不存在: {[str(d) for d in RES_DIRS]}")

    detail = pd.concat([collect(d) for d in RES_DIRS if d.exists()], ignore_index=True)
    if detail.empty:
        raise SystemExit("没有找到任何 *_Explain_test_results.csv，检查 --save_dirs 是否正确")

    detail.to_csv(ROOT / "summary_detail.csv", index=False)
    print(f"[ok] 明细 {len(detail)} 行 -> summary_detail.csv "
          f"({detail['exp'].nunique()} 个 exp, {len(detail['root'].unique())} 个结果目录)")

    agg = (detail.groupby(["dataset", "model"])
                 .agg(n=("rmse", "size"),
                      rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                      mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                      mape_mean=("mape_percent", "mean"), mape_std=("mape_percent", "std"),
                      epochs=("epochs", "max"),
                      best_val_rmse=("best_val_rmse", "mean"))
                 .reset_index())
    agg.to_csv(ROOT / "summary.csv", index=False)

    lines = [
        "| Dataset | Model | n | epochs | best val RMSE | test RMSE (ours) | paper RMSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset in ["Electricity", "PM2.5", "Exchange"]:
        sub = agg[agg["dataset"] == dataset]
        if sub.empty:
            continue
        sub = sub.set_index("model").reindex([m for m in MODEL_ORDER if m in set(sub["model"])])
        for model, r in sub.iterrows():
            if pd.isna(r["rmse_mean"]):
                continue
            paper = PAPER.get((dataset, model), (float("nan"),) * 3)[0]
            nd = 6 if dataset == "Exchange" else 4
            lines.append(
                f"| {dataset} | {model} | {int(r['n'])} | {int(r['epochs'])} | {r['best_val_rmse']:.4f} | "
                f"{fmt(r['rmse_mean'], r['rmse_std'], nd)} | {paper:.4g} |"
            )
    md = "\n".join(lines)
    (ROOT / "summary.md").write_text(md, encoding="utf-8")
    print(md)

    n_runs = detail.groupby(["dataset", "model"]).size()
    if (n_runs == 1).all():
        print("\n[!] 每个 数据集×模型 只有 1 次运行，std 无法计算（输出标为 n=1）。")
        print("    要得到 mean±std，请用 --num_exp 5 重跑（可加 --resample_split 让划分也随机）。")


if __name__ == "__main__":
    main()
