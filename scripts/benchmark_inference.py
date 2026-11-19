"""Measure FP32 DeLELSTM forward latency and throughput using saved checkpoints."""
import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ablation_suite import ROOT, latest_success, save
from run_delelstm_gpu1 import DEFAULT_OUTPUT, select_gpu1


WARMUP_WINDOWS = 10
TIMED_PASSES = 10


def measure(model, inputs, batch_size, reference):
    """Time resident-input forwards, with one completion barrier per request or full pass."""
    import torch

    device = inputs.device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    model.eval()
    singles = inputs.split(1)
    batches = inputs.split(batch_size)
    with torch.inference_mode():
        for i in range(WARMUP_WINDOWS):
            model(singles[i % len(singles)], device)
        synchronize()
        single_ms, single_predictions = [], []
        for window in singles:
            start = time.perf_counter()
            pred = model(window, device)[0]
            synchronize()
            single_ms.append((time.perf_counter() - start) * 1000)
            single_predictions.append(pred)
        if not torch.cat(single_predictions).isfinite().all().item():
            raise ValueError("Non-finite prediction in single-window inference")
        del single_predictions, pred

        # Warm every real batch shape, including the tail. Check predictions outside the timer.
        prediction = torch.cat([model(batch, device)[0] for batch in batches]).squeeze(-1).cpu().numpy()
        if (prediction.shape != reference.shape or not np.isfinite(prediction).all()
                or not np.isfinite(reference).all()):
            raise ValueError("Invalid batched predictions or saved reference")
        max_difference = float(np.max(np.abs(prediction - reference)))
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        pass_seconds = []
        for _ in range(TIMED_PASSES):
            start = time.perf_counter()
            for batch in batches:
                model(batch, device)
            synchronize()
            pass_seconds.append(time.perf_counter() - start)
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    return {
        "single_p50_ms": float(np.median(single_ms)),
        "single_p95_ms": float(np.percentile(single_ms, 95)),
        "batch_windows_per_s": len(inputs) * TIMED_PASSES / sum(pass_seconds),
        "full_test_pass_median_ms": float(np.median(pass_seconds) * 1000),
        "batch_peak_allocated_mib": peak / (1024 ** 2) if peak is not None else None,
        "batch_prediction_max_abs_diff": max_difference,
    }


def benchmark_job(campaign, job, device):
    import torch
    from ablation_data import DATASETS, load_windows, prepare_data
    from ablation_engine import make_model, seed_everything

    attempt = latest_success(campaign, job["id"])
    if attempt is None:
        raise ValueError(f"No successful checkpoint for {job['id']}")
    prior = json.loads((attempt / "split.json").read_text(encoding="utf-8"))
    dataset = job["dataset"]
    windows = load_windows(dataset, protocol="chronological")
    x, (_, _, test), split = prepare_data(windows, "chronological", job["fold"],
                                          prior["temporal_folds"], prior["smoke_max_windows"])
    if any(split[k] != prior[k] for k in ("window_ids", "minimum", "maximum", "target_policy")):
        raise ValueError(f"{dataset}: test split or training normalization differs from the checkpoint")
    with np.load(attempt / "predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["window_ids"], windows.ids[test]):
            raise ValueError(f"{dataset}: saved predictions use different test windows")
        reference = saved["prediction"]
    seed_everything(job["seed"])
    model = make_model(dataset, "Delelstm", len(windows.names), device)
    model.load_state_dict(torch.load(attempt / "best.pt", map_location="cpu", weights_only=True))
    inputs = torch.from_numpy(x[test]).to(device)
    batch_size = min(DATASETS[dataset]["batch"], len(test))
    return {
        "dataset": dataset, "fold": job["fold"], "seed": job["seed"],
        "checkpoint": (attempt / "best.pt").relative_to(campaign).as_posix(),
        "test_windows": len(test), "input_shape": list(inputs.shape[1:]),
        "outputs_per_window": reference.shape[1], "batch_size": batch_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        **measure(model, inputs, batch_size, reference),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_OUTPUT, help="Existing training output directory")
    parser.add_argument("--cpu", action="store_true", help="Run a CPU measurement instead of using physical GPU 1")
    args = parser.parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        select_gpu1()
    # Set visibility before importing PyTorch or initializing its CUDA runtime.
    import torch
    if not args.cpu and torch.cuda.device_count() != 1:
        parser.error("Expected exactly one visible CUDA device after selecting physical GPU 1")
    device = torch.device("cpu" if args.cpu else "cuda:0")
    campaign = args.campaign.resolve()
    settings = json.loads((campaign / "settings.json").read_text(encoding="utf-8"))
    plan = json.loads((campaign / "plan.json").read_text(encoding="utf-8"))
    environment = json.loads((campaign / "environment.json").read_text(encoding="utf-8"))
    torch.set_num_threads(settings["cpu_threads"])
    fold = settings["temporal_folds"] - 1
    jobs = [j for j in plan if j["kind"] == "train" and j["model"] == "Delelstm"
            and j["protocol"] == "chronological" and j["fold"] == fold and j["solver"] == "original"]
    measured = []
    for dataset in settings["datasets"]:
        # Select the lowest planned seed before consulting success or speed.
        candidates = [j for j in jobs if j["dataset"] == dataset]
        if not candidates:
            raise ValueError(f"{dataset}: the plan has no chronological DeLELSTM training job in fold {fold}")
        selected = min(candidates, key=lambda j: j["seed"])
        measured.append(benchmark_job(campaign, selected, device))

    stamp = datetime.now(timezone.utc)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    result = {
        "measured_at_utc": stamp.isoformat(),
        "benchmark_commit": revision.stdout.strip() or None,
        "training_commit": environment["git_commit"],
        "training_settings_custom": bool(settings["max_windows"] or settings["epochs"]),
        "device": platform.processor() if args.cpu else torch.cuda.get_device_name(device),
        "physical_gpu": None if args.cpu else 1, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(), "precision": "FP32 (TF32 off)",
        "scope": "resident-input model forward including decomposition; synchronized wall time",
        "single_warmups": WARMUP_WINDOWS, "single_timed": "one call per test window",
        "batch_warmup_passes": 1, "batch_timed_passes": TIMED_PASSES, "results": measured,
    }
    filename = stamp.strftime("%Y%m%dT%H%M%S%fZ") + ("-cpu.json" if args.cpu else "-gpu1.json")
    output = campaign / "inference" / filename
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    save(output, result)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
