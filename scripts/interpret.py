#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpret.py — 从 DeLELSTM 已导出的解释权重反推论文的两个可解释性指标。

背景
----
作者代码在 model_training.py 中把 DeLELSTM 的回归系数存成
    {exp_id}_Explain_test_weight.csv
形状为  (depth-2) x (2 * input_dim)，其中
    第 0..D-1 列  = alpha，作用于张量 LSTM 的 h_{t-1}（长期项）
    第 D..2D-1 列 = beta ，作用于 delta h_t = h_t - h_{t-1}（瞬时项）
且已经对 batch 取了均值（因此只有"平均样本"的解释，无逐样本方差）。

本脚本据此计算论文定义的两个指标：
    In_t^d = beta~ / (alpha~ + beta~)            瞬时占比，越接近 1 越"瞬时"
    Gl_T^d = mean_t sqrt(alpha~^2 + beta~^2)     全局重要性，用于变量排名
其中 ~ 表示按时间步对 |.| 做归一化（论文未写明具体归一化形式，此处取
"该时间步内各变量绝对值之和"归一化，这是最自然的一种；脚本同时输出
不归一化的原始比值 raw_In 供对照）。

只依赖 Python 标准库，可在任何环境直接运行。

用法
----
    python scripts/interpret.py          # 默认读 results/single_run，写 outputs/interpret
    python scripts/interpret.py --res outputs/rerun --output-dir outputs/rerun/interpret
"""
import argparse
import csv
import math
import os
import statistics
from pathlib import Path

# 各数据集的变量名，顺序与 newX_train.csv 中被喂进模型的列完全一致
# PM2.5   : iloc[:, 6:]   -> pm2.5, DEWP, TEMP, PRES, Iws, Is, Ir, cbwd
# Exchange: iloc[:, 1:]   -> 8 国汇率
# Electric: iloc[:, 2:]   -> 15 个天气变量 + Consumption
DATASETS = {
    "Electricity": {
        "dir": "Electricity",
        "names": [
            "MaxtempC", "MintempC", "DewPointC", "FeelsLikeC", "HeatIndexC",
            "WindChillC", "WindGustKmph", "Cloudcover", "Humidity", "PrecipMM",
            "Pressure", "TempC", "Visibility", "WinddirDegree",
            "WindspeedKmph", "Consumption",
        ],
    },
    "PM2.5": {
        "dir": ["newrevision_pmc", "newrevision_pm"],
        "names": ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "cbwd"],
    },
    "Exchange": {
        "dir": "Exchange",
        "names": [
            "Australia", "British", "Canada", "Switzerland",
            "China", "Japan", "New Zealand", "Singapore",
        ],
    },
}


def read_weight_csv(path):
    """读 _Explain_test_weight.csv，返回 list[list[float]]（行=时间步，列=2D）。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    out = []
    for r in body:
        if not r:
            continue
        vals = [float(x) for x in r[1:]]  # 去掉索引列
        out.append(vals)
    return out


def find_weight_files(dataset_dir):
    """找出该数据集下所有 exp_*/Delelstm/*_Explain_test_weight.csv。"""
    base = Path(dataset_dir)
    if not base.exists():
        return []
    hits = sorted(base.glob("exp_*/Delelstm/*_Explain_test_weight.csv"))
    return hits


def compute(weight_rows, D):
    """weight_rows: (T, 2D)；返回 In / raw_In / Gl 以及诊断量。"""
    alpha = [row[:D] for row in weight_rows]
    beta = [row[D:] for row in weight_rows]

    # 按时间步做 |.| 归一化
    a_norm, b_norm = [], []
    for a_row, b_row in zip(alpha, beta):
        sa = sum(abs(v) for v in a_row)
        sb = sum(abs(v) for v in b_row)
        a_norm.append([abs(v) / sa if sa > 0 else 0.0 for v in a_row])
        b_norm.append([abs(v) / sb if sb > 0 else 0.0 for v in b_row])

    In, raw_In = [], []
    for (a, b), (ra, rb) in zip(zip(a_norm, b_norm), zip(alpha, beta)):
        row = []
        rrow = []
        for av, bv, rav, rbv in zip(a, b, ra, rb):
            denom = av + bv
            row.append(bv / denom if denom > 0 else 0.5)
            rdenom = abs(rav) + abs(rbv)
            rrow.append(abs(rbv) / rdenom if rdenom > 0 else 0.5)
        In.append(row)
        raw_In.append(rrow)

    T = len(a_norm)
    Gl = []
    for d in range(D):
        Gl.append(sum(math.sqrt(a_norm[t][d] ** 2 + b_norm[t][d] ** 2) for t in range(T)) / T)

    flat = [abs(v) for row in weight_rows for v in row]
    flat_sorted = sorted(flat)
    diag = {
        "T": T,
        "D": D,
        "abs_max": max(flat),
        "abs_median": statistics.median(flat),
        "ratio_max_over_median": (max(flat) / statistics.median(flat)) if statistics.median(flat) else float("inf"),
        "frac_gt10": sum(1 for v in flat if v > 10) / len(flat),
    }
    return In, raw_In, Gl, diag


def write_matrix(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t"] + header)
        for i, r in enumerate(rows):
            w.writerow([i] + [f"{v:.6f}" for v in r])


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--res", default=str(root / "results" / "single_run"), help="结果目录")
    ap.add_argument("--output-dir", "--out", dest="out", default=str(root / "outputs" / "interpret"), help="输出目录，默认 outputs/interpret")
    args = ap.parse_args()

    res = Path(args.res)
    out = Path(args.out) if args.out else res / "interpret"
    out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for ds, meta in DATASETS.items():
        names = meta["names"]
        D = len(names)
        dirs = meta["dir"] if isinstance(meta["dir"], list) else [meta["dir"]]
        files = []
        for d in dirs:
            files += find_weight_files(res / d)
        if not files:
            print(f"[{ds}] 未找到 Delelstm 的解释权重，跳过")
            continue

        # 多次实验时对所有 exp 求平均
        per_file = [compute(read_weight_csv(p), D) for p in files]
        T = per_file[0][3]["T"]
        In = [[sum(pf[0][t][d] for pf in per_file) / len(per_file) for d in range(D)] for t in range(T)]
        raw_In = [[sum(pf[1][t][d] for pf in per_file) / len(per_file) for d in range(D)] for t in range(T)]
        Gl = [sum(pf[2][d] for pf in per_file) / len(per_file) for d in range(D)]
        diag = per_file[0][3]

        write_matrix(out / f"{ds}_In_t.csv", names, In)
        write_matrix(out / f"{ds}_In_t_raw.csv", names, raw_In)
        with open(out / f"{ds}_Gl_T.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "variable", "Gl_T"])
            for i, (nm, v) in enumerate(sorted(zip(names, Gl), key=lambda x: -x[1]), start=1):
                w.writerow([i, nm, f"{v:.6f}"])

        print(f"\n===== {ds}  (exp 数={len(files)}, T={T}, D={D}) =====")
        print("全局重要性 Gl_T 排名（论文 Figure 的可比量）:")
        for i, (nm, v) in enumerate(sorted(zip(names, Gl), key=lambda x: -x[1]), start=1):
            print(f"  {i:2d}. {nm:<16s} {v:.4f}")
        print("瞬时占比 In_t 按时间步的均值（1=纯瞬时, 0=纯长期）:")
        avg_in = [statistics.mean(In[t][d] for t in range(T)) for d in range(D)]
        for nm, v in sorted(zip(names, avg_in), key=lambda x: -x[1]):
            print(f"      {nm:<16s} {v:.4f}")
        print("系数幅度诊断（首个实验的 |alpha|,|beta| 分布；不能据此证明矩阵病态）:")
        print(f"      max={diag['abs_max']:.2f}  中位={diag['abs_median']:.3f}  "
              f"max/中位={diag['ratio_max_over_median']:.1f}  |w|>10 占比={diag['frac_gt10']*100:.1f}%")

        summary_rows.append({"dataset": ds, "D": D, "T": T, **diag,
                             "top1": sorted(zip(names, Gl), key=lambda x: -x[1])[0][0]})

    if summary_rows:
        with open(out / "diagnostics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n输出已写入: {out}")


if __name__ == "__main__":
    main()
