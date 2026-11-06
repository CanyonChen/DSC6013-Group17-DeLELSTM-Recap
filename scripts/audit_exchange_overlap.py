#!/usr/bin/env python3
"""Count test windows sharing observations with a training window in the seed-555 split."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, default=ROOT / "vendor" / "DeLELSTM")
    args = parser.parse_args()
    path = args.code_dir.resolve() / "DATA" / "Exchange" / "newX_train.csv"
    if not path.is_file():
        parser.error(f"Prepared Exchange data is missing: {path}")
    import numpy as np
    import pandas as pd

    ids = np.array(list(set(pd.read_csv(path, usecols=["idx"]).idx)))
    np.random.seed(555)
    train = np.random.choice(ids, int(.75 * len(ids)), replace=False)
    remaining = ids[~np.isin(ids, train)]
    val = np.random.choice(remaining, int(.15 * len(ids)), replace=False)
    test = np.sort(remaining[~np.isin(remaining, val)])
    train.sort()
    position = np.searchsorted(train, test)
    distance = np.minimum(abs(test - train[np.maximum(position - 1, 0)]), abs(test - train[np.minimum(position, len(train) - 1)]))
    for label, subset in [("all_test", distance), ("retained_by_drop_last", distance[:len(distance) // 128 * 128])]:
        print(label, "total", len(subset), "overlap_under_30_days", int((subset < 30).sum()), "adjacent_day", int((subset == 1).sum()))


if __name__ == "__main__":
    main()
