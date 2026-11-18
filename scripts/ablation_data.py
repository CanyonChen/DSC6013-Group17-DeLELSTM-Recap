"""Data and split definitions for the follow-up experiments."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "Electricity": dict(folder="Electricity", depth=24, offset=2, units=64,
                        batch=64, epochs=350, lr=.01, decay=.001, config="electricity", suffix="elec"),
    "PM": dict(folder="PM2.5", depth=24, offset=6, units=32,
               batch=32, epochs=300, lr=.05, decay=.03, config="PM", suffix="pm"),
    "Exchange": dict(folder="Exchange", depth=30, offset=1, units=32,
                     batch=128, epochs=300, lr=.01, decay=0., config="exchange", suffix="exchange"),
}
MODELS = ["Delelstm", "IMV_full", "IMV_tensor", "Retain", "LSTM"]


@dataclass
class Windows:
    name: str
    x: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    names: list
    times: np.ndarray
    # Chronological PM retains the original day IDs even when an early day has no history.
    usable_inputs: np.ndarray | None = None


def raw_pm_windows(folder):
    raw = pd.read_csv(folder / "PRSA_data_2010.1.1-2014.12.31.csv")
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw[["year", "month", "day", "hour"]]))
    raw = raw.sort_index()
    # Align to calendar days before partitioning. A missing source row has an unobserved target.
    hours = pd.date_range(raw.index[0].normalize(), raw.index[-1].normalize() + pd.Timedelta(hours=23), freq="h")
    raw = raw.reindex(hours)
    names = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "cbwd"]
    features = raw[names].copy()
    features["cbwd"] = features["cbwd"].map({"cv": 1, "NE": 2, "NW": 3, "SE": 4})
    depth = DATASETS["PM"]["depth"]
    x = features.ffill().to_numpy(dtype=np.float64).reshape(-1, depth, len(names))
    y = raw["pm2.5"].to_numpy(dtype=np.float32).reshape(-1, depth)[:, 2:]
    times = hours.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype=str).reshape(-1, depth)
    return Windows("PM", x, y, np.arange(len(x)), names, times, np.isfinite(x).all(axis=(1, 2)))


def load_windows(name, code_dir=ROOT / "vendor" / "DeLELSTM", *, protocol="released"):
    spec = DATASETS[name]
    folder = Path(code_dir) / "DATA" / spec["folder"]
    if name == "PM" and protocol == "chronological":
        return raw_pm_windows(folder)
    frame = pd.read_csv(folder / "newX_train.csv")
    target = pd.read_csv(folder / "second_y.csv")
    groups = frame.groupby("idx", sort=True)
    ids = np.array(sorted(groups.groups))
    depth = spec["depth"]
    if not (groups.size().to_numpy() == depth).all():
        raise ValueError(f"{name}: unexpected window sizes")
    frame = frame.sort_values("idx", kind="stable").reset_index(drop=True)
    target = target.sort_values("idx", kind="stable").reset_index(drop=True)
    if not np.array_equal(target.groupby("idx", sort=True).size().index, ids):
        raise ValueError("Input/target window IDs differ")
    if not (target.groupby("idx", sort=True).size().to_numpy() == depth - 2).all():
        raise ValueError("Unexpected target window size")
    names = list(frame.columns[spec["offset"]:])
    x = frame[names].to_numpy(dtype=np.float64).reshape(len(ids), depth, -1)
    y = target.target.to_numpy(dtype=np.float32).reshape(len(ids), depth - 2)
    if name == "Electricity":
        times = pd.to_datetime(frame.Date, format="%m/%d/%Y %H:%M").astype(str).to_numpy()
    elif name == "PM":
        times = pd.to_datetime(frame[["year", "month", "day", "hour"]]).astype(str).to_numpy()
    else:
        # The released Exchange data has no calendar timestamps. Preserve source positions.
        times = np.array([f"source_row_{int(i) + t}" for i in ids for t in range(depth)])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite processed data: {name}")
    return Windows(name, x, y, ids, names, np.asarray(times, dtype=str).reshape(len(ids), depth))


def split_indices(windows, protocol, fold=0, temporal_folds=1):
    n = len(windows.ids)
    if protocol in ("released", "full_eval"):
        # Match the released split membership and original CSV row order.
        rng = np.random.RandomState(555)
        train_ids = rng.choice(windows.ids, int(.75 * n), replace=False)
        left = windows.ids[~np.isin(windows.ids, train_ids)]
        val_ids = rng.choice(left, int(.15 * n), replace=False)
        test_ids = left[~np.isin(left, val_ids)]
        return [np.flatnonzero(np.isin(windows.ids, group)) for group in (train_ids, val_ids, test_ids)]
    if protocol != "chronological":
        raise ValueError(protocol)
    # Final fold is 75/15/10; earlier folds are expanding-window backtests.
    train_percent = 75 - 10 * (temporal_folds - 1 - fold)
    a, b, c = [n * percent // 100 for percent in (train_percent, train_percent + 15, train_percent + 25)]
    train, val, test = np.arange(a), np.arange(a, b), np.arange(b, c)
    if windows.name == "Exchange":
        # Each idx starts a 30-day window. Purge boundary windows sharing any raw day.
        gap = DATASETS[windows.name]["depth"] - 1
        val = val[windows.ids[val] > windows.ids[train[-1]] + gap]
        if len(val):
            test = test[windows.ids[test] > windows.ids[val[-1]] + gap]
    if windows.usable_inputs is not None:
        train, val, test = [part[windows.usable_inputs[part]] for part in (train, val, test)]
    if any(len(part) == 0 for part in (train, val, test)):
        raise ValueError("Empty chronological partition")
    return [train, val, test]


def prepare_data(windows, protocol, fold=0, temporal_folds=1, max_windows=0):
    if windows.name == "PM" and (protocol == "chronological") != (windows.usable_inputs is not None):
        raise ValueError("Load PM windows with load_windows('PM', protocol=protocol) for this experiment")
    splits = split_indices(windows, protocol, fold, temporal_folds)
    if max_windows:
        splits = [part[np.linspace(0, len(part)-1, min(max_windows, len(part)), dtype=int)]
                  for part in splits]
    fit = windows.x[splits[0]] if protocol == "chronological" else windows.x
    minimum, maximum = fit.min(axis=(0, 1)), fit.max(axis=(0, 1))
    span = maximum - minimum
    span[span == 0] = 1.
    x = ((windows.x - minimum) / span).astype(np.float32)
    metadata = {
        "protocol": protocol, "fold": fold, "temporal_folds": temporal_folds,
        "normalization_fit": "train_only" if protocol == "chronological" else "all_windows",
        "minimum": minimum.tolist(), "maximum": maximum.tolist(),
        "exchange_boundary_purge": 29 if protocol == "chronological" and windows.name == "Exchange" else 0,
        "window_ids": {key: windows.ids[part].tolist() for key, part in zip(("train", "val", "test"), splits)},
        "smoke_max_windows": max_windows,
        "target_policy": "observed_only" if windows.usable_inputs is not None else "published_targets",
        "excluded_no_history_ids": (windows.ids[~windows.usable_inputs].tolist()
                                    if windows.usable_inputs is not None else []),
        "partitions": {
            key: {"first": str(windows.times[part[0], 0]), "last": str(windows.times[part[-1], -1]),
                  "targets": int(windows.y[part].size), "valid_targets": int(np.isfinite(windows.y[part]).sum())}
            for key, part in zip(("train", "val", "test"), splits)
        },
    }
    return x, splits, metadata
