#!/usr/bin/env python3
"""Summarize saved test metrics as means and sample standard deviations."""
import argparse
import csv
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"Electricity": "Electricity", "newrevision_pmc": "PM2.5", "newrevision_pm": "PM2.5", "Exchange": "Exchange"}
MODELS = ["Delelstm", "IMV_full", "IMV_tensor", "LSTM", "Retain"]


def collect(root):
    rows = []
    for folder, dataset in DATASETS.items():
        for path in sorted((root / folder).glob("exp_*/*/*_Explain_test_results.csv")):
            with path.open(encoding="utf-8", newline="") as stream:
                result = next(csv.DictReader(stream))
            train_files = list(path.parent.glob("*_Expalin_train_results.csv"))
            train = []
            if train_files:
                with train_files[0].open(encoding="utf-8", newline="") as stream:
                    train = list(csv.DictReader(stream))
            rows.append({"root": str(root), "dataset": dataset, "exp": path.parent.parent.name, "model": path.parent.name,
                         "rmse": float(result["rmse_test"]), "mae": float(result["mae_test"]), "mape_percent": float(result["mape_test"]) * 100,
                         "epochs": len(train) if train else "", "best_val_rmse": min(float(row["rmse_val"]) for row in train) if train else ""})
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path, help="Result roots; relative paths use the current working directory.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "summary")
    args = parser.parse_args()
    roots = [path.resolve() for path in args.results] or [ROOT / "results" / "five_run"]
    if any(not path.is_dir() for path in roots):
        parser.error("Every result root must be an existing directory")
    detail = [row for root in roots for row in collect(root)]
    if not detail:
        parser.error("No *_Explain_test_results.csv files found")
    summary = []
    for dataset in ["Electricity", "PM2.5", "Exchange"]:
        for model in MODELS:
            group = [row for row in detail if row["dataset"] == dataset and row["model"] == model]
            if not group:
                continue
            record = {"dataset": dataset, "model": model, "n": len(group)}
            for metric in ["rmse", "mae", "mape_percent"]:
                values = [row[metric] for row in group]
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else ""
            epochs = [row["epochs"] for row in group if row["epochs"] != ""]
            val = [row["best_val_rmse"] for row in group if row["best_val_rmse"] != ""]
            record["epochs_max"] = max(epochs) if epochs else ""
            record["best_val_rmse_mean"] = statistics.mean(val) if val else ""
            summary.append(record)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "summary_detail.csv", detail)
    write_csv(output / "summary.csv", summary)
    lines = ["# Saved experiment summary", "", "Means and sample standard deviations; MAPE is a percentage. Missing training logs are shown as —.", "", "| Dataset | Model | n | RMSE | MAE | MAPE (%) | Epochs |", "| --- | --- | ---: | --- | --- | --- | ---: |"]
    for row in summary:
        metrics = []
        for metric in ["rmse", "mae", "mape_percent"]:
            digits = 6 if row["dataset"] == "Exchange" and metric != "mape_percent" else 4
            mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
            metrics.append(f"{mean:.{digits}f} ± {std:.{digits}f}" if std != "" else f"{mean:.{digits}f} (n=1)")
        lines.append(f"| {row['dataset']} | {row['model']} | {row['n']} | {' | '.join(metrics)} | {row['epochs_max'] or '—'} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summarized {len(detail)} test records into {len(summary)} groups: {output}")


if __name__ == "__main__":
    main()
