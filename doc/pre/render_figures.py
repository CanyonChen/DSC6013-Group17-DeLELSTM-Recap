"""Redraw the three presentation figures from the group's archived scores."""

from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce_paper_figures as paper


def main():
    css = (HERE / "deck.html").read_text(encoding="utf-8")
    accent = re.search(r"--accent:\s*(#[0-9a-fA-F]{6})\s*;", css).group(1)
    paper.MEAN = accent
    paper.BAND = colors.to_hex([0.82 + 0.18 * channel for channel in colors.to_rgb(accent)])
    paper.RAW = "#595959"
    style = {
        **paper.FIGURE_STYLE,
        "text.color": "#282828",
        "axes.labelcolor": "#282828",
        "xtick.color": "#282828",
        "ytick.color": "#282828",
        "axes.edgecolor": "#777777",
        "grid.color": "#dddddd",
    }

    results = ROOT / "results"
    figures = results / "figures"
    with TemporaryDirectory() as temp, plt.rc_context(style):
        out = Path(temp)
        paper.plot_performance(pd.read_csv(results / "five_run" / "verified_detail.csv"), out)
        paper.plot_shares(pd.read_csv(figures / "pm25_instantaneous_share_five_runs.csv",
                                      float_precision="round_trip"), out)
        paper.plot_windows(pd.read_csv(figures / "audit_full_test_five_runs.csv",
                                       float_precision="round_trip"), out)
        for image in out.glob("*.png"):
            shutil.copyfile(image, HERE / "media" / image.name)


if __name__ == "__main__":
    main()
