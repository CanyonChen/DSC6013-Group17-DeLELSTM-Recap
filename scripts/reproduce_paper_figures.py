"""Rebuild the three fixed-split figures and their run-level tables from the archived results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ablation_data import DATASETS, ROOT, load_windows, prepare_data
from ablation_engine import evaluate, make_model, metrics
from interpret import compute, read_weight_csv


def checkpoint_tables():
    curves, pairs = [], []
    device = torch.device("cpu")
    for dataset, spec in DATASETS.items():
        label = spec["folder"]
        folder = "newrevision_pmc" if dataset == "PM" else dataset
        windows = load_windows(dataset, protocol="released")
        inputs, (_, _, test), _ = prepare_data(windows, "released")
        batch = spec["batch"]
        retained = len(test) // batch * batch
        model = make_model(dataset, "Delelstm", len(windows.names), device)
        for run in range(5):
            archive = ROOT / "outputs" / "five_run" / folder / f"exp_555_{run}" / "Delelstm"
            checkpoint = archive / f"0_{spec['suffix']}_predict.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            pred, true, _ = evaluate(model, inputs, windows.y, test, batch, device)
            retained_rmse = metrics(pred[:retained], true[:retained])["rmse"]
            full_rmse = metrics(pred, true)["rmse"]
            stored = pd.read_csv(archive / "0_Explain_test_results.csv").iloc[0]["rmse_test"]
            if not np.isclose(retained_rmse, stored, rtol=3e-4, atol=2e-7):
                raise ValueError(f"Retained RMSE differs from saved run {run} in {label}.")
            pairs.append(dict(dataset=label, run=run, n_retained=retained, n_full=len(test),
                              retained_rmse=retained_rmse, full_rmse=full_rmse,
                              delta_rmse=full_rmse-retained_rmse, stored_rmse=stored))
            if dataset == "PM":
                weights = read_weight_csv(archive / "0_Explain_test_weight.csv")
                shares, _, _, _ = compute(weights, len(windows.names))
                target_column = windows.names.index("pm2.5")
                for position, row in enumerate(shares, start=1):
                    curves.append(dict(run=run, position=position, instantaneous_share=row[target_column]))
    return pd.DataFrame(curves), pd.DataFrame(pairs)


def plot_performance(records, output):
    names = {"Delelstm": "DeLELSTM", "IMV_full": "IMV-Full", "IMV_tensor": "IMV-Tensor",
             "Retain": "RETAIN", "LSTM": "LSTM"}
    colors = ["#156082", "#657f96", "#8a9aab", "#667d6a", "#bb8651"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.55), layout="constrained")
    for ax, dataset in zip(axes, ("Electricity", "PM2.5", "Exchange")):
        for i, (model, color) in enumerate(zip(names, colors)):
            rows = records[(records.dataset == dataset) & (records.model == model)].sort_values("exp")
            if rows.exp.tolist() != list(range(5)):
                raise ValueError(f"Expected five saved runs for {dataset}/{model}")
            ax.scatter(i + np.linspace(-.12, .12, 5), rows.rmse, color=color, s=24)
            ax.plot([i-.21, i+.21], [rows.rmse.mean()] * 2, color="#15293b")
        ax.set(title=dataset, ylabel="Test RMSE", ylim=(0, None))
        ax.set_xticks(range(5), names.values(), rotation=38, ha="right")
    fig.suptitle("Five-run predictive performance")
    fig.savefig(output / "figure_1_rmse_five_runs.png", dpi=300)
    plt.close(fig)


def plot_shares(curves, output):
    table = curves.pivot(index="run", columns="position", values="instantaneous_share")
    if table.shape != (5, 22) or table.isna().any().any():
        raise ValueError("Expected five complete 22-position PM2.5 curves")
    x, mean, sd = table.columns, table.mean(), table.std(ddof=1)
    fig, ax = plt.subplots(figsize=(8.2, 3.4), layout="constrained")
    ax.plot(x, table.to_numpy().T, color="#b0bec7", linewidth=1)
    ax.fill_between(x, (mean-sd).clip(0, 1), (mean+sd).clip(0, 1), color="#a8cddd",
                    alpha=.45, label="Mean ± 1 SD (clipped to [0, 1])")
    ax.plot(x, mean, color="#176486", linewidth=2.5, label="Mean of 5 runs")
    ax.set(xlim=(1, 22), ylim=(0, 1), xticks=[1, 4, 7, 10, 13, 16, 19, 22],
           xlabel="Output position within the window (1–22)", ylabel="Instantaneous share",
           title="PM2.5: instantaneous share of the target variable")
    ax.legend(frameon=False)
    fig.savefig(output / "figure_2_pm25_five_run.png", dpi=300)
    plt.close(fig)


def plot_windows(pairs, output):
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4), layout="constrained")
    for ax, (dataset, rows) in zip(axes, pairs.groupby("dataset", sort=False)):
        values = rows[["retained_rmse", "full_rmse"]]
        ax.plot([0, 1], values.to_numpy().T, color="#aab9c3", marker="o", linewidth=1)
        ax.errorbar([0, 1], values.mean(), yerr=values.std(ddof=1), fmt="o", color="#176486", capsize=4)
        ax.set_xticks([0, 1], [f"Retained\n(n={rows.n_retained.iloc[0]})", f"All test\n(n={rows.n_full.iloc[0]})"])
        ax.set(title=dataset, ylabel="Test RMSE")
        ax.ticklabel_format(axis="y", useOffset=False)
    fig.suptitle("Paired test-window evaluation across five runs")
    fig.savefig(output / "figure_3_test_windows_five_run.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "paper_figures")
    args = parser.parse_args()
    torch.set_num_threads(2)
    curves, pairs = checkpoint_tables()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output / "pm25_instantaneous_share_five_runs.csv", index=False)
    pairs.to_csv(output / "audit_full_test_five_runs.csv", index=False)
    plot_performance(pd.read_csv(ROOT / "results" / "five_run" / "verified_detail.csv"), output)
    plot_shares(curves, output)
    plot_windows(pairs, output)
    print(f"Wrote three figures, {len(curves)} PM2.5 positions and {len(pairs)} checkpoint pairs to {output}")


if __name__ == "__main__":
    main()
