#!/usr/bin/env python3
"""Compare retained and full test windows for a saved single-run DeLELSTM checkpoint."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["Electricity", "PM", "Exchange"])
    parser.add_argument("--results", type=Path, default=ROOT / "outputs" / "single_run", help="Root containing original exp_0 checkpoints.")
    parser.add_argument("--code-dir", type=Path, default=ROOT / "vendor" / "DeLELSTM")
    args = parser.parse_args()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    code = args.code_dir.resolve()
    if not (code / f"{args.dataset}.py").is_file():
        parser.error("Upstream code is missing; run scripts/setup_upstream.py first")
    folder = {"Electricity": "Electricity", "PM": "newrevision_pmc", "Exchange": "Exchange"}[args.dataset]
    suffix = {"Electricity": "elec", "PM": "pm", "Exchange": "exchange"}[args.dataset]
    checkpoint = args.results.resolve() / folder / "exp_0" / "Delelstm" / f"0_{suffix}_predict.pt"
    if not checkpoint.is_file():
        parser.error(f"Original single-run checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(code))
    original_argv = sys.argv
    sys.argv = [args.dataset + ".py"]
    try:
        module = importlib.import_module(args.dataset)
    finally:
        sys.argv = original_argv
    from networks import Delelstm
    import config

    _, _, loader = module.prepare(555)
    model = Delelstm(config.config("Delelstm", module.args), 0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in DataLoader(loader.dataset, batch_size=module.args.batch_size, shuffle=False, drop_last=False):
            pred, _, _ = model(x.float(), torch.device("cpu"))
            predictions.append(pred.squeeze(-1).numpy())
            targets.append(y.numpy())
    pred, true = np.concatenate(predictions), np.concatenate(targets)
    retained = len(loader) * module.args.batch_size
    for title, subset in [("retained", slice(None, retained)), ("omitted", slice(retained, None)), ("all", slice(None))]:
        error = pred[subset] - true[subset]
        print(json.dumps({"dataset": args.dataset, "subset": title, "windows": len(error), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}))


if __name__ == "__main__":
    main()
