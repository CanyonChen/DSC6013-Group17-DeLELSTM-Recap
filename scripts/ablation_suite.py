#!/usr/bin/env python3
"""Plan and execute isolated follow-up experiments, one process per device."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Electricity", "PM", "Exchange"]
MODELS = ["Delelstm", "IMV_full", "IMV_tensor", "Retain", "LSTM"]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def task_id(job):
    keys = ("kind", "protocol", "dataset", "model", "seed", "fold", "solver", "ridge", "selector", "source")
    return "__".join(str(job[k]) for k in keys if k in job)


def build_plan(settings, smoke=False):
    jobs = []
    seeds = settings["seeds"]
    if not smoke:
        for dataset in settings["datasets"]:
            folder = "newrevision_pmc" if dataset == "PM" else dataset
            suffix = {"Electricity": "elec", "PM": "pm", "Exchange": "exchange"}[dataset]
            for source, runs in [("single_run", [0]), ("five_run", list(range(5)))]:
                for run in runs:
                    for model in settings["models"]:
                        exp = "exp_0" if source == "single_run" else f"exp_555_{run}"
                        jobs.append(dict(kind="audit", protocol="released", dataset=dataset, model=model,
                                         seed=555+run+(source == "five_run" and dataset == "PM"), source=source,
                                         seed_role="historical_training_seed" if source == "five_run" else "evaluation_only",
                                         checkpoint=f"outputs/{source}/{folder}/{exp}/{model}/0_{suffix}_predict.pt",
                                         device_compare=(model == "Delelstm" and run == 0)))
    protocols = ["full_eval"] if smoke else ["released", "full_eval", "chronological"]
    for protocol in protocols:
        folds = range(settings["temporal_folds"]) if protocol == "chronological" else [0]
        for fold in folds:
            for dataset in settings["datasets"]:
                for seed in seeds:
                    for model in settings["models"]:
                        jobs.append(dict(kind="train", protocol=protocol, dataset=dataset, model=model,
                                         seed=seed+(dataset == "PM"), fold=fold, solver="original", ridge=0.))
    if "Delelstm" in settings["models"]:
        for dataset in settings["datasets"]:
            for seed in seeds:
                for solver, ridge in [("pinv", 0.)] + [("ridge", v) for v in settings["ridge_values"]]:
                    jobs.append(dict(kind="solver", protocol="full_eval" if smoke else "chronological",
                                     dataset=dataset, model="Delelstm", seed=seed+(dataset == "PM"),
                                     fold=0 if smoke else settings["temporal_folds"]-1, solver=solver, ridge=ridge))
    if "Electricity" in settings["datasets"] and "LSTM" in settings["models"]:
        selectors = [m for m in settings["models"] if m != "LSTM"] + ["random"]
        for seed in seeds:
            for selector in selectors:
                job = dict(kind="sufficiency", protocol="full_eval", dataset="Electricity", model="LSTM",
                           seed=seed, selector=selector, fold=0, solver="original", ridge=0.)
                if selector != "random":
                    source = dict(kind="train", protocol="full_eval", dataset="Electricity", model=selector,
                                  seed=seed, fold=0, solver="original", ridge=0.)
                    job["dependency"] = task_id(source)
                jobs.append(job)
    if not smoke and settings["devices"] != ["cpu"] and "Delelstm" in settings["models"]:
        for dataset in settings["datasets"]:
            for seed in seeds[:settings["cpu_control_runs"]]:
                jobs.append(dict(kind="cpu_control", protocol="full_eval", dataset=dataset, model="Delelstm",
                                 seed=seed+(dataset == "PM"), fold=0, solver="original", ridge=0., fixed_device="cpu"))
        if "PM" in settings["datasets"] and settings["cpu_control_runs"]:
            jobs.append(dict(kind="nan_probe", protocol="released", dataset="PM", model="Delelstm",
                             seed=555, fold=0, solver="original", ridge=0., fixed_device="cpu"))
    for job in jobs:
        job["id"] = task_id(job)
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate task IDs")
    return jobs


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["plan", "preflight", "smoke", "run", "report", "worker"], default="plan")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ablation")
    ap.add_argument("--gpus", default="0,1", help="Logical visible GPU indices; use cpu for local validation")
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[555, 556, 557, 558, 559])
    ap.add_argument("--temporal-folds", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--ridge-values", nargs="+", type=float, default=[1e-6, 1e-4, 1e-2])
    ap.add_argument("--cpu-control-runs", type=int, default=1)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--diagnostic-windows", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=0, help="0 = released defaults; overrides are labelled custom/smoke")
    ap.add_argument("--max-windows", type=int, default=0, help="0 = full data; positive values are smoke-only")
    ap.add_argument("--resume", action="store_true", help="Skip completed jobs; rerun interrupted/failed jobs in a new attempt")
    ap.add_argument("--task", help=argparse.SUPPRESS)
    ap.add_argument("--attempt", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--device", help=argparse.SUPPRESS)
    return ap


def configuration(args):
    devices = ["cpu"] if args.gpus == "cpu" else [f"cuda:{int(v)}" for v in args.gpus.split(",")]
    if len(set(devices)) != len(devices) or any(v.startswith("cuda:-") for v in devices):
        raise ValueError("GPU indices must be distinct and non-negative")
    for name in ("datasets", "models", "seeds", "ridge_values"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            raise ValueError(f"Duplicate {name}")
    if args.epochs < 0 or args.max_windows < 0 or args.cpu_threads < 1 or args.diagnostic_windows < 1:
        raise ValueError("Invalid count")
    if any(value <= 0 for value in args.ridge_values) or not 0 <= args.cpu_control_runs <= len(args.seeds):
        raise ValueError("Invalid ridge values or CPU control count")
    settings = {key: getattr(args, key) for key in ("datasets", "models", "seeds", "temporal_folds", "ridge_values",
                "cpu_control_runs", "cpu_threads", "diagnostic_windows", "epochs", "max_windows")}
    settings["devices"] = devices
    if args.stage == "smoke":
        settings.update(seeds=[args.seeds[0]], temporal_folds=1, cpu_control_runs=0,
                        epochs=1, max_windows=args.max_windows or 16, diagnostic_windows=4)
    paths = list((ROOT / "scripts").glob("ablation_*.py")) + [ROOT / "scripts" / "eval_and_ablation.sh",
             ROOT / "patches" / "upstream.json", ROOT / "requirements.txt", ROOT / "environment" / "requirements-ablation.txt"]
    settings["code_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return settings


def preflight(out, settings):
    import importlib.metadata
    import platform
    import torch
    from ablation_engine import seed_everything
    seed_everything(555)
    devices = []
    for name in settings["devices"]:
        device = torch.device(name)
        if device.type == "cuda" and (not torch.cuda.is_available() or device.index >= torch.cuda.device_count()):
            raise RuntimeError(f"Requested {name} is unavailable; visible GPUs={torch.cuda.device_count()}")
        a = torch.eye(8, device=device)
        torch.linalg.solve(a, torch.ones(8, 1, device=device))
        devices.append({"device": name, "name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    env = dict(python=sys.version, platform=platform.platform(), cuda=torch.version.cuda, devices=devices,
               git_commit=revision.stdout.strip() if revision.returncode == 0 else None,
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               packages={p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "torchmetrics", "matplotlib"]})
    save(out / "environment.json", env)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (out / "requirements-installed.txt").write_text(freeze, encoding="utf-8")
    print(json.dumps(env, ensure_ascii=False, indent=2))


def latest_success(root, task):
    attempts = sorted((root / "jobs" / task).glob("attempt_*/result.json"))
    for result in reversed(attempts):
        if json.loads(result.read_text(encoding="utf-8"))["status"] == "success":
            return result.parent
    return None


def run_campaign(out, settings, jobs, resume):
    config_path = out / "settings.json"
    if config_path.exists():
        if not resume:
            raise ValueError("Campaign exists. Use --resume or a new --output-dir")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != settings:
            raise ValueError("Settings or code changed; use a new output directory")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not an existing campaign or empty")
    out.mkdir(parents=True, exist_ok=True)
    save(config_path, settings)
    save(out / "plan.json", jobs)
    preflight(out, settings)
    for job in jobs:
        save(out / "jobs" / job["id"] / "job.json", job)
    errors = []

    def execute(job, device):
        task = job["id"]
        if latest_success(out, task):
            print(f"SKIP completed {task}", flush=True)
            return
        folder = out / "jobs" / task
        index = len(list(folder.glob("attempt_*"))) + 1
        attempt = folder / f"attempt_{index:03d}"
        attempt.mkdir()
        if job.get("dependency") and not latest_success(out, job["dependency"]):
            save(attempt / "result.json", dict(job, status="blocked", reason="selector training did not succeed"))
            errors.append(task)
            return
        print(f"START [{device}] {task}", flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", OMP_NUM_THREADS=str(settings["cpu_threads"]),
                   OPENBLAS_NUM_THREADS=str(settings["cpu_threads"]), MPLBACKEND="Agg")
        env.pop("PYTHONPATH", None)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "worker", "--output-dir", str(out),
               "--task", task, "--attempt", str(attempt), "--device", device]
        with (attempt / "run.log").open("w", encoding="utf-8") as stream:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if proc.returncode:
            errors.append(task)
            if not (attempt / "result.json").exists():
                save(attempt / "result.json", dict(job, status="failed", reason=f"worker exit {proc.returncode}"))
        print(f"{'DONE' if proc.returncode == 0 else 'FAILED'} {task}", flush=True)

    def wave(selected):
        queues = {"normal": queue.Queue(), "cpu": queue.Queue()}
        for job in selected:
            queues["cpu" if job.get("fixed_device") == "cpu" and settings["devices"] != ["cpu"] else "normal"].put(job)
        lanes = [(device, queues["normal"]) for device in settings["devices"]]
        if not queues["cpu"].empty():
            lanes.append(("cpu", queues["cpu"]))
        def lane(device, work):
            while True:
                try:
                    job = work.get_nowait()
                except queue.Empty:
                    return
                execute(job, device)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            futures = [pool.submit(lane, *item) for item in lanes]
            for future in futures:
                future.result()
    try:
        wave([j for j in jobs if j["kind"] != "sufficiency"])
        wave([j for j in jobs if j["kind"] == "sufficiency"])
    finally:
        from ablation_report import build_report
        build_report(out)
    return 1 if errors else 0


def main():
    args = parser().parse_args()
    out = args.output_dir.resolve()
    if args.stage == "report":
        from ablation_report import build_report
        build_report(out)
        return 0
    if args.stage == "worker":
        settings = json.loads((out / "settings.json").read_text(encoding="utf-8"))
        job = json.loads((out / "jobs" / args.task / "job.json").read_text(encoding="utf-8"))
        job["device"] = args.device
        dependency = latest_success(out, job["dependency"]) if job.get("dependency") else None
        try:
            from ablation_engine import worker
            worker(job, settings, args.attempt, str(dependency) if dependency else None)
            return 0
        except Exception as exc:
            traceback.print_exc()
            save(args.attempt / "result.json", dict(job, status="failed", reason=f"{type(exc).__name__}: {exc}"))
            return 1
    settings = configuration(args)
    jobs = build_plan(settings, smoke=args.stage == "smoke")
    if args.stage == "plan":
        from collections import Counter
        print(json.dumps({"jobs": len(jobs), "by_kind": dict(Counter(j["kind"] for j in jobs)),
                          "devices": settings["devices"], "output": str(out), "settings": settings}, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "preflight":
        out.mkdir(parents=True, exist_ok=True)
        preflight(out, settings)
        return 0
    return run_campaign(out, settings, jobs, args.resume)


if __name__ == "__main__":
    sys.exit(main())
