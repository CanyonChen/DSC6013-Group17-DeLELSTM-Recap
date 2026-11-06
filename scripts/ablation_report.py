"""Generate evidence-linked Chinese Markdown/HTML reports from actual job results."""
import csv
import html
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics

KEYS = ("kind", "protocol", "dataset", "model", "fold", "solver", "ridge", "selector", "source")


def group_key(row):
    return tuple(str(row.get(key, "")) for key in KEYS)


def csv_write(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def ranks(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.] * len(values)
    for value, indices in itertools.groupby(ordered, key=lambda i: values[i]):
        indices = list(indices)
        rank = statistics.mean(ordered.index(i) + 1 for i in indices)
        for i in indices:
            result[i] = rank
    return result


def correlation(a, b):
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return statistics.correlation(a, b)


def sign_flip(differences):
    n = len(differences)
    if n < 2:
        return None
    observed = abs(statistics.mean(differences))
    if n <= 16:
        values = [abs(sum(d*s for d, s in zip(differences, signs))/n)
                  for signs in itertools.product([-1, 1], repeat=n)]
        return sum(v >= observed - 1e-15 for v in values) / len(values)
    rng = random.Random(555)
    extreme = sum(abs(sum(d*rng.choice([-1, 1]) for d in differences)/n) >= observed - 1e-15 for _ in range(10000))
    return (extreme + 1) / 10001


def table(headers, rows):
    def clean(value):
        return str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows])


def simple_html(markdown):
    import re
    output, in_table = [], False
    def inline(text):
        text = html.escape(text)
        text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        return text
    for line in markdown.splitlines():
        if line.startswith("| "):
            if "---" in line and set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue
            if not in_table:
                output.append('<div class="table"><table>')
                in_table = True
            output.append("<tr>" + "".join(f"<td>{inline(cell.strip())}</td>" for cell in line.strip("|").split("|")) + "</tr>")
            continue
        if in_table:
            output.append("</table></div>")
            in_table = False
        if line.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                output.append(f'<figure><img src="{html.escape(match[2])}" alt="{html.escape(match[1])}"></figure>')
        elif line.startswith("#"):
            level = min(len(line)-len(line.lstrip("#")), 6)
            output.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line:
            output.append(f"<p>{inline(line)}</p>")
    if in_table:
        output.append("</table></div>")
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DeLELSTM 补充实验报告</title><style>' + \
           'body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui,sans-serif;color:#17212b}h1,h2{line-height:1.3}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #dce1e6;padding:7px}tr:first-child{background:#edf2f6;font-weight:600}img{max-width:100%}code{background:#edf2f6;padding:2px 5px}a{color:#145da0}</style>' + "\n".join(output) + '</html>'


def build_report(root):
    root = Path(root)
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    report = root / "report"
    report.mkdir(exist_ok=True)
    runs = []
    for job in plan:
        attempts = sorted((root / "jobs" / job["id"]).glob("attempt_*/result.json"))
        result = dict(job, status="missing", reason="not completed")
        for path in reversed(attempts):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if result["status"] == "missing" or candidate["status"] == "success":
                result = candidate
                result["artifact_dir"] = path.parent.relative_to(root).as_posix()
            if candidate["status"] == "success":
                break
        runs.append(result)
    flat = []
    for row in runs:
        record = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for group in ("metrics_all", "metrics_retained"):
            record.update({f"{group}_{key}": value for key, value in row.get(group, {}).items()})
        flat.append(record)
    csv_write(report / "runs.csv", flat)
    summary, audits = [], []
    for key in sorted(set(group_key(job) for job in plan)):
        rows = [r for r in runs if group_key(r) == key]
        success = [r for r in rows if r["status"] == "success"]
        entry = dict(zip(KEYS, key), planned=len(rows), successful=len(success),
                     failed=sum(r["status"]=="failed" for r in rows),
                     blocked=sum(r["status"]=="blocked" for r in rows),
                     missing=sum(r["status"]=="missing" for r in rows))
        scope = "metrics_retained" if entry["protocol"] == "released" else "metrics_all"
        entry["metric_scope"] = scope
        for metric in ("rmse", "mae", "mape_percent"):
            values = [r[scope][metric] for r in success]
            entry[f"{metric}_mean"] = statistics.mean(values) if values else None
            entry[f"{metric}_std"] = statistics.stdev(values) if len(values)>1 else None
        summary.append(entry)
        if entry["kind"] == "audit" and success:
            audit = dict(dataset=entry["dataset"], model=entry["model"], source=entry["source"],
                         successful=len(success), planned=len(rows),
                         complete_windows=success[0]["test_windows"], retained_windows=success[0]["retained_windows"])
            for label, field in [("complete", "metrics_all"), ("retained", "metrics_retained")]:
                values = [r[field]["rmse"] for r in success]
                audit[label+"_rmse_mean"] = statistics.mean(values)
                audit[label+"_rmse_std"] = statistics.stdev(values) if len(values)>1 else None
            audit["mean_rmse_difference"] = statistics.mean(r["metrics_all"]["rmse"]-r["metrics_retained"]["rmse"] for r in success)
            audits.append(audit)
    csv_write(report / "summary.csv", summary)
    csv_write(report / "full_test_audit.csv", audits)

    # Pair only the same model, dataset and training seed; never pair different temporal test periods.
    comparisons = []
    success = [r for r in runs if r["status"] == "success"]
    controls = [r for r in success if r["kind"] in ("solver", "sufficiency", "cpu_control") or
                (r["kind"] == "train" and r["protocol"] == "full_eval")]
    for key in sorted(set(group_key(r) for r in controls)):
        selected = [r for r in controls if group_key(r) == key]
        pairs = []
        for current in selected:
            reference = [r for r in success if r["kind"]=="train" and r["dataset"]==current["dataset"] and
                         r["model"]==current["model"] and r["seed"]==current["seed"] and
                         r.get("fold", 0)==current.get("fold", 0) and r.get("solver")=="original" and
                         r["protocol"] == ("released" if current["kind"]=="train" else current["protocol"])]
            if reference:
                pairs.append(current["metrics_all"]["rmse"] - reference[0]["metrics_all"]["rmse"])
        if pairs:
            comparisons.append(dict(zip(KEYS, key), n_paired=len(pairs), mean_rmse_difference=statistics.mean(pairs),
                                    p_sign_flip=sign_flip(pairs)))
    eligible = sorted([r for r in comparisons if r["p_sign_flip"] is not None], key=lambda r:r["p_sign_flip"])
    previous = 0.
    for i, row in enumerate(eligible):
        previous = max(previous, min(1., row["p_sign_flip"]*(len(eligible)-i)))
        row["p_holm"] = previous
    csv_write(report / "paired_comparisons.csv", comparisons)

    rankings, diagnostics, devices = [], [], []
    for row in success:
        folder = root / row["artifact_dir"]
        if (folder / "importance.csv").exists():
            ranking = read_csv(folder / "importance.csv")
            row["_ranking"] = [float(item["score"]) for item in ranking]
            raw = [float(item["raw_score"]) for item in ranking]
            mean_first = [float(item["mean_before_normalization"]) for item in ranking]
            rankings.append(dict(task=row["id"], comparison="normalization_vs_raw", rho=correlation(ranks(row["_ranking"]), ranks(raw))))
            rankings.append(dict(task=row["id"], comparison="sample_first_vs_mean_first", rho=correlation(ranks(row["_ranking"]), ranks(mean_first))))
        if (folder / "solver_diagnostics.csv").exists():
            values = read_csv(folder / "solver_diagnostics.csv")
            record = dict(task=row["id"], matrices=len(values))
            for field in ("condition_A", "condition_AtA_estimate", "relative_residual", "coefficient_abs_max"):
                finite = [float(v[field]) for v in values if v[field] and math.isfinite(float(v[field]))]
                record[field+"_median"] = statistics.median(finite) if finite else None
                record[field+"_max"] = max(finite) if finite else None
                record[field+"_nonfinite"] = len(values)-len(finite)
            diagnostics.append(record)
        if (folder / "device_comparison.json").exists():
            d = json.loads((folder / "device_comparison.json").read_text(encoding="utf-8"))
            devices.append(dict(task=row["id"], windows=d["windows"], prediction_max_abs_difference=d["prediction_max_abs_difference"]))
    explained = [r for r in success if "_ranking" in r]
    for key in sorted(set(group_key(r) for r in explained)):
        for a, b in itertools.combinations([r for r in explained if group_key(r)==key], 2):
            ra, rb = a["_ranking"], b["_ranking"]
            count = len(ra)//2
            sa = set(sorted(range(len(ra)), key=lambda i:-ra[i])[:count])
            sb = set(sorted(range(len(rb)), key=lambda i:-rb[i])[:count])
            rankings.append(dict(task=a["id"], comparison=b["id"], rho=correlation(ranks(ra), ranks(rb)),
                                 top50_jaccard=len(sa & sb)/len(sa | sb)))
    csv_write(report / "ranking_stability.csv", rankings)
    csv_write(report / "numerical_diagnostics.csv", diagnostics)
    csv_write(report / "device_comparisons.csv", devices)

    pictures = []
    if success:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".conda" / "matplotlib-cache"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Separate dataset axes: raw RMSE scales are not comparable across datasets.
        for dataset in settings["datasets"]:
            rows = [r for r in summary if r["dataset"]==dataset and r["kind"] in ("train", "solver") and r["rmse_mean"] is not None]
            if rows:
                labels = [f"{r['protocol']}/f{r['fold']}/{r['model']}/{r['solver']}:{r['ridge']}" for r in rows]
                fig, ax = plt.subplots(figsize=(11, max(3, .25*len(rows))))
                ax.barh(labels, [r["rmse_mean"] for r in rows], xerr=[r["rmse_std"] or 0 for r in rows])
                ax.invert_yaxis(); ax.set_xlabel("Test RMSE (mean and sample SD)")
                ax.set_title(dataset + (" - smoke/custom run" if settings["epochs"] or settings["max_windows"] else ""))
                fig.tight_layout(); path = report / f"rmse_{dataset}.png"; fig.savefig(path, dpi=160); plt.close(fig)
                pictures.append(path.name)
        for dataset in settings["datasets"]:
            rows = [r for r in explained if r["dataset"]==dataset and r["kind"]=="train" and r["protocol"]=="full_eval"]
            if rows:
                first = sorted(rows, key=lambda r:r["seed"])[0]
                folder = root / first["artifact_dir"]
                profile = folder / "clock_profiles.csv"
                if profile.exists():
                    data = read_csv(profile)
                    variables = list(dict.fromkeys(item["variable"] for item in data))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for color_index, variable in enumerate(variables):
                        values = [v for v in data if v["variable"]==variable]
                        ax.plot([int(v["hour"]) for v in values], [float(v["instantaneous"]) for v in values],
                                label=variable, color=plt.get_cmap("tab20")(color_index))
                    scope = "sampled test" if settings["max_windows"] else "full test"
                    label = " - smoke/custom run" if settings["epochs"] or settings["max_windows"] else ""
                    ax.set(xlabel="Input clock hour (not prediction hour)", ylabel="Instantaneous share", ylim=(0,1),
                           title=f"{dataset}, seed {first['seed']}, {scope}{label}")
                    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), fontsize=7, ncol=4); fig.tight_layout()
                    path = report / f"clock_{dataset}.png"; fig.savefig(path, dpi=160); plt.close(fig); pictures.append(path.name)

    failed = [r for r in runs if r["status"] != "success"]
    lines = ["# DeLELSTM 补充实验报告", "",
             "运行状态：" + ("短测试／自定义训练轮数，不能作为正式论文实验。" if settings["max_windows"] or settings["epochs"] else "正式设置，统计以实际完成记录为准。"), "",
             f"计划 {len(plan)} 项任务；成功 {len(success)} 项；失败、阻塞或未完成 {len(failed)} 项。", "",
             "本报告由实际文件生成。成功均值只描述完成的运行；失败不会更换种子后从统计中消失。原课程报告的历史结果保持不变。", "",
             "## 实验协议", "",
             "released：原固定随机划分、全数据归一化、验证丢弃尾批。full_eval：相同数据划分与初始化，验证使用全部窗口。两者均额外保存完整测试预测，便于在同一测试集合上比较选模差异。", "",
             "chronological：扩展时间回测、仅在训练集拟合归一化、完整验证/测试。Exchange 在边界剔除共享原始日期的窗口。不同时间折的测试集合不同，不与随机划分分数做配对显著性检验。处理后数据沿用上游的缺失值填充等预处理。", "",
             "Top-50%：仅在 Electricity 训练集上计算各解释模型的排名，选择一半变量，训练同架构 LSTM；提供随机一半和全变量 LSTM 对照。论文没有完全指定特征聚合及归一化细节，本实验采用明确记录的规则，不能声称与原表完全同协议。", "",
             "求解器：original 为原实现；pinv 直接对设计矩阵 A 求伪逆；ridge 求解 (AᵀA+λI)c=AᵀH，λ为绝对值。保留架构、损失和默认优化参数，禁用 TF32/混合精度。新训练按模型独立设种子，不等同于旧脚本多模型连续使用 RNG 的逐位复现。", "",
             "## 预测结果", "",
             "released 主指标使用原保留测试窗口；其余协议使用完整测试。历史审计按 single_run/five_run 分开。各运行两种测试口径均保存在 runs.csv。", "",
             table(["任务", "协议/折", "数据集", "模型/选择器", "求解器/λ", "成功/计划", "RMSE 均值±SD"],
                   [[r["kind"]+("/"+r["source"] if r["source"] else ""), f"{r['protocol']}/{r['fold']}", r["dataset"], r["model"]+("/"+r["selector"] if r["selector"] else ""),
                     f"{r['solver']}/{r['ridge']}", f"{r['successful']}/{r['planned']}",
                     "未获得" if r["rmse_mean"] is None else f"{r['rmse_mean']:.6g} ± {r['rmse_std']:.4g}" if r["rmse_std"] is not None else f"{r['rmse_mean']:.6g} (n=1)"] for r in summary]), "",
             "全部 RMSE、MAE、MAPE、成功数和失败数见 [summary.csv](summary.csv)，逐运行记录见 [runs.csv](runs.csv)。", "",
             "## 历史权重的完整测试审计", "",
             "固定同一检查点，比较完整测试与原丢弃尾批后的测试。差值为完整减保留；这是测试覆盖变化，不是重新训练收益。", "",
             table(["来源", "数据集", "模型", "成功/计划", "完整/保留窗口", "完整 RMSE 均值", "保留 RMSE 均值", "差值"],
                   [[r["source"], r["dataset"], r["model"], f"{r['successful']}/{r['planned']}",
                     f"{r['complete_windows']}/{r['retained_windows']}", f"{r['complete_rmse_mean']:.6g}",
                     f"{r['retained_rmse_mean']:.6g}", f"{r['mean_rmse_difference']:+.6g}"] for r in audits]) if audits else "当前没有已完成的历史检查点审计。", "",
             "[审计均值及样本标准差](full_test_audit.csv)。", "",
             "## 统计比较", "",
             "仅对相同数据集、模型、种子及同一测试集合的已完成配对进行双侧符号翻转检验；n≤16穷举，大于16使用固定种子的10000次抽样。表中差值为实验减对照，负值表示实验RMSE较低；Holm校正覆盖本报告所有可检验对照。训练种子不是独立时间样本，统计不能推导未来部署表现。", "",
             "[配对比较及校正 p 值](paired_comparisons.csv)。五个配对的双侧穷举检验最小 p 值为 0.0625，不自动宣称显著改善。", "",
             "## 解释与数值诊断", "",
             "逐样本系数、输入/预测时间及变量名保存于每个 DeLELSTM 任务的 sample_explanations.npz。输入位置1..T−2对应预测位置2..T−1；按输入时钟时间聚合，Exchange仅有源行号，不伪造日期。示例曲线取最小训练种子的 full_eval 运行，多种子统计见下表文件。", "",
             "[排名相关与 Top-50% Jaccard](ranking_stability.csv)；[条件数、重建相对误差及系数幅度](numerical_diagnostics.csv)。条件数只检查配置中指定数量的均匀抽样测试窗口，AtA条件数由cond(A)²估计。诊断与误差变化是关联证据，不单独证明NaN原因或解释因果性。", "",
             "## CPU/GPU 与失败分析", "",
             "[同检查点前向对照](device_comparisons.csv)使用同输入和FP32设置，仅比较设备数值差异。cpu_control 为另外执行的匹配种子训练，默认每数据集一个种子，不能据此归因全部硬件差异。nan_probe 尝试原 PM 种子555；如当前环境未失败，不表示历史失败不存在。", "",
             "非有限训练损失或梯度保留模型、输入批次与epoch；其他异常保留traceback。无论成功与否都保留原种子。", "",
             table(["任务", "状态", "原因/日志"], [[r["id"], r["status"], r.get("reason", "") +
                    (f" [目录](../{r['artifact_dir']}/)" if r.get("artifact_dir") else "")] for r in failed]) if failed else "全部计划任务已成功完成。", "",
             "## 图表", ""]
    lines.extend(f"![{path}]({path})\n" for path in pictures)
    lines += ["## 可追溯文件", "", "[完整配置](../settings.json) · [任务计划](../plan.json) · [实际环境](../environment.json) · [安装依赖快照](../requirements-installed.txt)", "",
              "每项任务保存 job.json、split.json、训练轨迹、best.pt（重新训练的任务）、完整测试 predictions.npz、日志及 result.json。任务级断点续跑跳过成功项，中断/失败项使用同种子新建 attempt 后从头重跑，不做训练轮次级恢复。", ""]
    text = "\n".join(lines)
    (report / "report.md").write_text(text, encoding="utf-8")
    (report / "report.html").write_text(simple_html(text), encoding="utf-8")
    print(f"Report: {report / 'report.html'}", flush=True)
