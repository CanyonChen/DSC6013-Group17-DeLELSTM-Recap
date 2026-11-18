"""Run the 45 chronological DeLELSTM jobs on physical GPU 1."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from ablation_suite import ROOT, build_plan, configuration, latest_success, parser as suite_parser, save


FORECAST_ARGS = ["--forecast-only", "--models", "Delelstm", "--gpus", "0"]
DEFAULT_OUTPUT = ROOT / "outputs" / "h100_gpu1_delelstm45"


def import_completed(source, destination, settings):
    """Copy completed matching attempts into a separate campaign, keeping the originals."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Choose a different output directory for the DeLELSTM-only plan.")
    previous = json.loads((source / "settings.json").read_text(encoding="utf-8"))
    if "Delelstm" not in previous["models"] or dict(previous, models=["Delelstm"]) != settings:
        raise ValueError("Source settings or code differ from this formal chronological run.")
    source_env = json.loads((source / "environment.json").read_text(encoding="utf-8"))
    source_torch = source_env["packages"]["torch"]
    if source_torch != version("torch"):
        raise ValueError(f"Source used PyTorch {source_torch}. Run this entry with the matching interpreter.")

    plan = build_plan(settings)
    config = destination / "settings.json"
    if destination.exists() and any(destination.iterdir()):
        if not config.is_file() or json.loads(config.read_text(encoding="utf-8")) != settings:
            raise ValueError("Output directory already contains a different campaign.")
    else:
        save(config, settings)
        save(destination / "plan.json", plan)

    imported = 0
    for job in plan:
        task = job["id"]
        if latest_success(destination, task):
            continue
        try:
            attempt = latest_success(source, task)
        except json.JSONDecodeError:
            print(f"Skipping an interrupted source result for {task}. It will be trained.", flush=True)
            continue
        if attempt is None:
            continue
        if not all((attempt / name).is_file() for name in ("best.pt", "split.json", "predictions.npz")):
            print(f"Skipping incomplete source artifacts for {task}. It will be trained.", flush=True)
            continue
        target = destination / "jobs" / task
        target.mkdir(parents=True, exist_ok=True)
        next_attempt = len(list(target.glob("attempt_*"))) + 1
        with tempfile.TemporaryDirectory(dir=target) as tmp:
            staged = Path(tmp) / "attempt"
            shutil.copytree(attempt, staged)
            save(staged / "reused_from.json", {"source_attempt": str(attempt), "environment": source_env})
            staged.rename(target / f"attempt_{next_attempt:03d}")
        imported += 1
    ready = sum(latest_success(destination, job["id"]) is not None for job in plan)
    print(f"Imported {imported}. Completed {ready}/{len(plan)}. To train: {len(plan) - ready}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-from", type=Path, help="Previous formal forecasting output directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if not uuid.startswith("GPU-") or len(uuid.splitlines()) != 1:
        raise RuntimeError("Could not identify physical GPU 1.")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    if args.reuse_from:
        settings = configuration(suite_parser().parse_args(["--stage", "run", *FORECAST_ARGS]))
        import_completed(args.reuse_from, args.output_dir, settings)
    launcher = ROOT / "scripts" / "eval_and_ablation.sh"
    os.execlp("bash", "bash", str(launcher), "--stage", "run", "--python", sys.executable,
              *FORECAST_ARGS, "--output-dir", str(args.output_dir.resolve()), "--resume")


if __name__ == "__main__":
    main()
