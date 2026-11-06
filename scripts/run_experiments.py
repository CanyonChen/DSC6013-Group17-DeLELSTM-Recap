#!/usr/bin/env python3
"""Run the report's independent initializations sequentially on a fixed split."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "vendor" / "DeLELSTM"
FOLDERS = {"Electricity": "Electricity", "PM": "newrevision_pmc", "Exchange": "Exchange"}
MODELS = ["Delelstm", "IMV_full", "IMV_tensor", "Retain", "LSTM"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--datasets", nargs="+", choices=list(FOLDERS), default=list(FOLDERS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--epochs", type=int, help="Override epochs for a smoke run; omit for the report defaults.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "rerun")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without importing torch or writing files.")
    args = parser.parse_args()
    if args.num_runs < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("--num-runs and --epochs must be positive")
    args.datasets = list(dict.fromkeys(args.datasets))
    args.models = list(dict.fromkeys(args.models))
    args.output_dir = args.output_dir.resolve()
    return args


def metrics_complete(experiment, models):
    for model in models:
        files = list((experiment / model).glob("*_Explain_test_results.csv"))
        if len(files) != 1:
            return False
        with files[0].open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not all(math.isfinite(float(rows[0][key])) for key in ("rmse_test", "mae_test", "mape_test")):
            return False
    return True


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists; choose a new --output-dir: {args.output_dir}")
    print(f"Code: {CODE}\nOutput: {args.output_dir}\nDevice: {args.device}")
    print("Split seed: 555; sequential execution; at most 6 attempts per dataset/run.")
    for dataset in args.datasets:
        seeds = [555 + i + (dataset == "PM") for i in range(args.num_runs)]
        print(f"{dataset}: initialization seeds {seeds}; models {', '.join(args.models)}")
    if args.dry_run:
        return
    for dataset in args.datasets:
        if not (CODE / f"{dataset}.py").is_file():
            raise SystemExit("Upstream code is missing. Run python scripts/setup_upstream.py first.")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable. Activate the GPU environment or pass --device cpu.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    args.output_dir.mkdir(parents=True)
    logs = args.output_dir / "logs"
    logs.mkdir()
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", DELE_MODELS=",".join(args.models))
    environment.pop("DELE_EPOCHS", None)
    if args.epochs is not None:
        environment["DELE_EPOCHS"] = str(args.epochs)
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    for dataset in args.datasets:
        for run in range(args.num_runs):
            for attempt in range(6):
                seed = 555 + run + (dataset == "PM") + 100 * attempt
                staging = args.output_dir / "attempts" / f"{dataset}_{run}_{attempt}"
                staging.mkdir(parents=True)
                environment["DELE_TORCH_SEED"] = str(seed)
                command = [sys.executable, "-u", str(CODE / f"{dataset}.py"), "--seed", "555", "--num_exp", "1", "--save_dirs", str(staging)]
                log = logs / f"{dataset}_run{run}_attempt{attempt}_seed{seed}.log"
                metadata = {"dataset": dataset, "run": run, "attempt": attempt, "torch_seed": seed, "split_seed": 555, "device": args.device, "models": args.models, "epochs_override": args.epochs, "command": command}
                print(f"Running {dataset} run={run} seed={seed}; log={log}", flush=True)
                with log.open("w", encoding="utf-8") as stream:
                    stream.write(json.dumps(metadata) + "\n")
                    stream.flush()
                    process = subprocess.run(command, cwd=CODE, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                source = staging / FOLDERS[dataset] / "exp_555_0"
                if process.returncode == 0 and metrics_complete(source, args.models):
                    target = args.output_dir / FOLDERS[dataset] / f"exp_555_{run}"
                    target.parent.mkdir(exist_ok=True)
                    shutil.move(str(source), str(target))
                    (target / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                    break
                print(f"Attempt failed; preserved files in {staging}", flush=True)
            else:
                raise SystemExit(f"{dataset} run={run} failed after 6 attempts; see {logs}")
    print(f"Completed {len(args.datasets) * args.num_runs} dataset runs. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
