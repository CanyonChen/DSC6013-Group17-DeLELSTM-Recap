"""Reproduce the chronological split, scaling and target counts reported in section 6."""
import argparse
import csv
from contextlib import nullcontext
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

from ablation_data import DATASETS, load_windows, prepare_data, split_indices


def source_points(windows):
    if windows.name == "Exchange":
        return windows.ids[:, None] + np.arange(windows.x.shape[1])
    return windows.times.astype("datetime64[s]").astype(np.int64)


def audit_dataset(name):
    released = load_windows(name)
    train, _, test = split_indices(released, "released")
    points = source_points(released)
    overlap = np.isin(points[test], np.unique(points[train])).any(axis=1).sum()
    old_train = released.x[train]
    target_name = {"Electricity": "Consumption", "PM": "pm2.5", "Exchange": "Singapore"}[name]
    target_column = released.names.index(target_name)
    train_input = old_train[:, :, target_column]
    full_input = released.x[:, :, target_column]
    heldout_extrema = ((released.x.min(axis=(0, 1)) != old_train.min(axis=(0, 1))) |
                       (released.x.max(axis=(0, 1)) != old_train.max(axis=(0, 1))))
    windows = load_windows(name, protocol="chronological")
    points = source_points(windows)
    for fold in range(3):
        x, parts, split = prepare_data(windows, "chronological", fold, 3)
        grids = [np.unique(points[part]) for part in parts]
        common = [np.intersect1d(a, b, assume_unique=True).size for a, b in combinations(grids, 2)]
        if any(common) or not (grids[0][-1] < grids[1][0] and grids[1][-1] < grids[2][0]):
            raise ValueError(f"Temporal boundary violation in {name}, fold {fold}")
        train, val, test = parts
        changed = windows.x.copy()
        changed[val] -= 10000
        changed[test] += 20000
        altered, _, alt_split = prepare_data(replace(windows, x=changed), "chronological", fold, 3)
        np.testing.assert_array_equal(split["minimum"], windows.x[train].min(axis=(0, 1)))
        np.testing.assert_array_equal(split["maximum"], windows.x[train].max(axis=(0, 1)))
        np.testing.assert_array_equal(split["minimum"], alt_split["minimum"])
        np.testing.assert_array_equal(split["maximum"], alt_split["maximum"])
        np.testing.assert_array_equal(x[train], altered[train])
        row = dict(dataset=name, fold=fold, released_overlapping_test_windows=int(overlap),
                   released_features_with_heldout_extrema="|".join(np.asarray(windows.names)[heldout_extrema]),
                   target_input=target_name, released_train_input_min=train_input.min(),
                   released_train_input_max=train_input.max(), released_full_input_min=full_input.min(),
                   released_full_input_max=full_input.max(),
                   train_val_shared=common[0], train_test_shared=common[1], val_test_shared=common[2],
                   train_scaling_invariant=True)
        for key, part in zip(("train", "val", "test"), parts):
            row[key + "_windows"] = len(part)
            row.update({f"{key}_{k}": v for k, v in split["partitions"][key].items()})
        yield row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [row for dataset in DATASETS for row in audit_dataset(dataset)]
    with (args.output.open("w", encoding="utf-8", newline="") if args.output else nullcontext(sys.stdout)) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
