"""Rebuild the three fixed-split figures and their run-level tables from the archived results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_ORDER = ("Electricity", "PM2.5", "Exchange")
MODEL_LABELS = {"Delelstm": "DeLELSTM", "IMV_full": "IMV-Full", "IMV_tensor": "IMV-Tensor",
                "LSTM": "LSTM", "Retain": "RETAIN"}
WIDTH, DPI = 6.5, 400  # inches at the intended Word placement
INK, RAW, MEAN, BAND = "#233541", "#586B75", "#005B6B", "#D4E5E9"
FIGURE_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK, "axes.labelsize": 9, "axes.edgecolor": "#72828A",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
    "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 10,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.major.size": 3, "ytick.major.size": 3, "grid.color": "#DCE2E4",
    "grid.linewidth": .55, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
}


def save_figure(fig, output, stem):
    fig.savefig(output / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    png = output / f"{stem}.png"
    fig.savefig(png, dpi=DPI, transparent=False)
    # Word embeds the opaque raster. The PDF retains editable text and vector marks.
    with Image.open(png) as raster:
        rgb = raster.convert("RGB")
    rgb.save(png, dpi=(DPI, DPI))
    plt.close(fig)


def checkpoint_tables():
    import torch
    from ablation_data import DATASETS, load_windows, prepare_data
    from ablation_engine import evaluate, make_model, metrics
    from interpret import compute, read_weight_csv

    torch.set_num_threads(2)
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
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.8), sharey=True)
    fig.subplots_adjust(left=.142, right=.982, top=.78, bottom=.2, wspace=.21)
    offsets = np.linspace(-.27, .06, 5)  # display offsets only, in saved-run order
    for ax, panel, dataset in zip(axes, "ABC", DATASET_ORDER):
        factor = 1000 if dataset == "Exchange" else 1
        if dataset == "Electricity":
            ax.set_xscale("log", base=10)
        for i, model in enumerate(MODEL_LABELS):
            rows = records[(records.dataset == dataset) & (records.model == model)].sort_values("exp")
            values = rows.rmse.to_numpy() * factor
            if rows.exp.tolist() != list(range(5)) or not np.isfinite(values).all():
                raise ValueError(f"Expected five saved runs for {dataset}/{model}")
            if dataset == "Electricity" and (values <= 0).any():
                raise ValueError("The Electricity log axis requires positive RMSE.")
            ax.plot(values, i + offsets, "o", ms=3.3, mfc="white", mec=RAW, mew=.8)
            ax.plot(values.mean(), i + .28, "D", ms=4.2, color=MEAN)
            if dataset == "Electricity" and model == "LSTM":
                worst = values.argmax()
                ax.annotate(f"{values[worst]:.2f}", (values[worst], i + offsets[worst]),
                            xytext=(-1, 6), textcoords="offset points", ha="right", fontsize=8)
        ax.set_title(f"{panel}   {dataset}", loc="left", pad=6)
        ax.set_yticks(range(5), MODEL_LABELS.values())
        ax.set_ylim(4.65, -.65)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        ax.margins(x=.10)
    axes[0].set_xticks([2, 5, 10, 20], ["2", "5", "10", "20"])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set_xlabel("RMSE (log₁₀ scale)")
    axes[1].set_xticks([25, 27, 29, 31])
    axes[1].set_xlabel("RMSE")
    axes[2].set_xticks([2.2, 2.4, 2.6])
    axes[2].set_xlabel("RMSE (×10⁻³)")
    fig.legend([Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=RAW),
                Line2D([], [], ls="", marker="D", ms=4.5, color=MEAN)],
               ["Individual run", "Arithmetic mean (5 runs)"], ncols=2, loc="upper center",
               bbox_to_anchor=(.53, 1.01), handlelength=1, columnspacing=2)
    save_figure(fig, output, "figure_1_rmse_five_runs")


def plot_shares(curves, output):
    table = curves.pivot(index="run", columns="position", values="instantaneous_share")
    if (table.index.tolist() != list(range(5)) or table.columns.tolist() != list(range(1, 23))
            or not np.isfinite(table).all().all() or ((table < 0) | (table > 1)).any().any()):
        raise ValueError("Expected five complete 22-position PM2.5 curves")
    x, mean, sd = table.columns, table.mean(), table.std(ddof=1)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.62))
    fig.subplots_adjust(left=.115, right=.976, top=.84, bottom=.21)
    ax.fill_between(x, (mean-sd).clip(0, 1), (mean+sd).clip(0, 1), color=BAND)
    runs = ax.plot(x, table.to_numpy().T, color=RAW, ls=(0, (3, 1.6)), lw=.85)
    average, = ax.plot(x, mean, color=MEAN, lw=1.8, marker="o", ms=2.8)
    ax.set(xlim=(.8, 22.2), ylim=(0, 1), xticks=[1, 4, 7, 10, 13, 16, 19, 22],
           yticks=[0, .25, .5, .75, 1], xlabel="Output position within the window",
           ylabel="Instantaneous share")
    ax.grid(axis="y")
    fig.legend([runs[0], average, Patch(facecolor=BAND, edgecolor=RAW, linewidth=.6)],
               ["Individual run", "Mean (5 runs)", "Mean ± 1 sample SD"], loc="upper center",
               bbox_to_anchor=(.53, 1.015), ncols=3, columnspacing=1.7)
    save_figure(fig, output, "figure_2_pm25_five_run")


def plot_windows(pairs, output):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.78))
    fig.subplots_adjust(left=.093, right=.987, top=.76, bottom=.2, wspace=.46)
    for ax, panel, dataset in zip(axes, "ABC", DATASET_ORDER):
        rows = pairs[pairs.dataset == dataset].sort_values("run")
        values = rows[["retained_rmse", "full_rmse"]]
        if rows.run.tolist() != list(range(5)) or not np.isfinite(values).all().all():
            raise ValueError(f"Expected five complete checkpoint pairs for {dataset}")
        if dataset == "Exchange":
            values = values * 1000
            ax.text(1, 1.03, "×10⁻³", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        for retained, full in values.itertuples(index=False, name=None):
            style = "--" if full > retained else "-"
            ax.plot([0, 1], [retained, full], linestyle=style, marker="o", ms=3.3,
                    mfc="white", mew=.8, color=RAW, lw=1)
        summary = ax.errorbar([-.14, 1.14], values.mean(), yerr=values.std(ddof=1),
                              fmt="D", ms=4.3, color=MEAN, elinewidth=1.3, capsize=2.3)
        ax.set_xticks([0, 1], [f"Retained\n{rows.n_retained.iloc[0]}", f"All\n{rows.n_full.iloc[0]}"])
        ax.set_xlim(-.34, 1.34)
        ax.margins(y=.09)
        ax.locator_params(axis="y", nbins=4)
        ax.grid(axis="y")
        ax.tick_params(axis="x", length=0, pad=4)
        ax.set_title(f"{panel}   {dataset}", loc="left", pad=6)
    axes[0].set_ylabel("Test RMSE")
    fig.legend([Line2D([], [], color=RAW, lw=1, marker="o", ms=3.3, mfc="white"),
                Line2D([], [], color=RAW, lw=1, ls="--", marker="o", ms=3.3, mfc="white"),
                summary], ["RMSE decreased", "RMSE increased", "Mean ± 1 sample SD"],
               loc="upper center", bbox_to_anchor=(.52, 1.01), ncols=3, columnspacing=1.5)
    save_figure(fig, output, "figure_3_test_windows_five_run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "paper_figures")
    parser.add_argument("--from-archived", action="store_true",
                        help="Draw from the committed run-level CSVs without loading checkpoints.")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    figures = ROOT / "results" / "figures"
    curves_name, pairs_name = "pm25_instantaneous_share_five_runs.csv", "audit_full_test_five_runs.csv"
    if args.from_archived:
        curves = pd.read_csv(figures / curves_name, float_precision="round_trip")
        pairs = pd.read_csv(figures / pairs_name, float_precision="round_trip")
    else:
        curves, pairs = checkpoint_tables()
        curves.to_csv(output / curves_name, index=False)
        pairs.to_csv(output / pairs_name, index=False)
    with plt.rc_context(FIGURE_STYLE):
        plot_performance(pd.read_csv(ROOT / "results" / "five_run" / "verified_detail.csv"), output)
        plot_shares(curves, output)
        plot_windows(pairs, output)
    print(f"Wrote three figures, {len(curves)} PM2.5 positions and {len(pairs)} checkpoint pairs to {output}")


if __name__ == "__main__":
    main()
