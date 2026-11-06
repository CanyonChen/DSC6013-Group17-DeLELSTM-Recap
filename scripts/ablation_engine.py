"""Training, checkpoint audits and explanation export using the pinned upstream models."""
import csv
import importlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from ablation_data import ROOT, DATASETS, load_windows, prepare_data


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def modules():
    code = ROOT / "vendor" / "DeLELSTM"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    return importlib.import_module("networks"), importlib.import_module("config")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class Solver:
    def __init__(self, mode="original", ridge=0.):
        self.mode, self.ridge = mode, ridge
        self.diagnostics = []
        self.capture = False
        self.original = modules()[0]._estimate_alpha1

    def __call__(self, feature_reps, targets, device):
        a = feature_reps.transpose(1, 2)
        if self.mode == "original":
            coeff = self.original(feature_reps, targets, device)
        elif self.mode == "pinv":
            coeff = torch.linalg.pinv(a) @ targets.unsqueeze(-1)
        elif self.mode == "ridge":
            gram = feature_reps @ a
            identity = torch.eye(gram.shape[-1], device=device, dtype=gram.dtype)
            coeff = torch.linalg.solve(gram + self.ridge * identity,
                                       feature_reps @ targets.unsqueeze(-1))
        else:
            raise ValueError(self.mode)
        if self.capture:
            with torch.no_grad():
                # Diagnostics in float64, solver itself retains the model's FP32 arithmetic.
                design = a.detach().double()
                singular = torch.linalg.svdvals(design)
                cond = singular[:, 0] / singular[:, -1].clamp_min(torch.finfo(torch.float64).tiny)
                residual = (design @ coeff.detach().double()).squeeze(-1) - targets.detach().double()
                relative = residual.norm(dim=-1) / targets.detach().double().norm(dim=-1).clamp_min(1e-15)
                for c, r, maximum in zip(cond.cpu().tolist(), relative.cpu().tolist(),
                                          coeff.detach().abs().amax(dim=(1, 2)).cpu().tolist()):
                    self.diagnostics.append({"condition_A": c if np.isfinite(c) else None,
                                             "condition_AtA_estimate": c*c if np.isfinite(c*c) else None,
                                             "relative_residual": r if np.isfinite(r) else None,
                                             "coefficient_abs_max": maximum if np.isfinite(maximum) else None})
        return coeff

    def __enter__(self):
        modules()[0]._estimate_alpha1 = self
        return self

    def __exit__(self, *exc):
        modules()[0]._estimate_alpha1 = self.original


def make_model(dataset, model, dimension, device):
    networks, config = modules()
    spec = DATASETS[dataset]
    args = SimpleNamespace(input_dim=dimension, n_units=spec["units"], N_units=spec["units"],
                           depth=spec["depth"], output_dim=1, dataset=spec["config"], batch_size=spec["batch"])
    classes = {"Delelstm": "Delelstm", "IMV_full": "IMVFullLSTM_pertime",
               "IMV_tensor": "IMVTensorLSTM_pertime", "Retain": "Retain_pertime", "LSTM": "normalLSTMpertime"}
    return getattr(networks, classes[model])(config.config(model, args), 0).to(device)


def metrics(prediction, target):
    error = prediction.astype(np.float64) - target.astype(np.float64)
    if not np.isfinite(error).all():
        raise FloatingPointError("Non-finite predictions")
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                mape_percent=float(np.mean(np.abs(error) / np.maximum(np.abs(target), 1.17e-6)) * 100))


def loader(x, y, indices, batch, shuffle=False, drop_last=False):
    return DataLoader(TensorDataset(torch.from_numpy(x[indices]), torch.from_numpy(y[indices])),
                      batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_workers=0)


def evaluate(model, x, y, indices, batch, device, drop_last=False, explanation=None):
    predictions, targets, coefficients = [], [], []
    model.eval()
    with torch.no_grad():
        for bx, by in loader(x, y, indices, batch, drop_last=drop_last):
            pred, weight, beta = model(bx.to(device), device)
            predictions.append(pred.squeeze(-1).cpu().numpy())
            targets.append(by.numpy())
            if explanation == "Delelstm":
                coefficients.append(weight.squeeze(-1).permute(1, 0, 2).cpu().numpy())
            elif explanation in ("IMV_full", "IMV_tensor"):
                coefficients.append(beta.cpu().numpy())
            elif explanation == "Retain":
                coefficients.append(weight.cpu().numpy())
    if not predictions:
        raise ValueError("No evaluation batches; increase smoke window count")
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(coefficients) if coefficients else None


def normalized_components(coeff):
    a, b = np.split(np.abs(coeff.astype(np.float64)), 2, axis=-1)
    an = a / np.maximum(a.sum(axis=-1, keepdims=True), 1e-30)
    bn = b / np.maximum(b.sum(axis=-1, keepdims=True), 1e-30)
    instantaneous = np.divide(bn, an+bn, out=np.full_like(bn, .5), where=(an+bn)>0)
    return instantaneous, np.sqrt(an**2 + bn**2)


def rank_features(model_name, weights):
    if not np.isfinite(weights).all():
        raise FloatingPointError("Non-finite explanation coefficients")
    if model_name == "Delelstm":
        return normalized_components(weights)[1].mean(axis=(0, 1))
    return np.abs(weights).mean(axis=(0, 1))


def export_explanations(out, windows, indices, coeff, solver, model, x, batch, device, diagnostic_windows):
    if not np.isfinite(coeff).all():
        raise FloatingPointError("Non-finite explanation coefficients")
    instantaneous, importance = normalized_components(coeff)
    global_score = importance.mean(axis=(0, 1))
    raw_a, raw_b = np.split(np.abs(coeff), 2, axis=-1)
    raw_score = np.sqrt(raw_a.astype(float)**2 + raw_b.astype(float)**2).mean(axis=(0, 1))
    mean_first = normalized_components(coeff.mean(axis=0, keepdims=True))[1].mean(axis=(0, 1))
    write_csv(out / "importance.csv", [dict(variable=name, score=float(score),
              raw_score=float(raw), mean_before_normalization=float(old))
              for name, score, raw, old in zip(windows.names, global_score, raw_score, mean_first)])
    np.savez_compressed(out / "sample_explanations.npz", coefficients=coeff,
                        window_ids=windows.ids[indices], input_time=windows.times[indices, 1:-1],
                        prediction_time=windows.times[indices, 2:], variables=np.array(windows.names))
    rows = []
    for t in range(coeff.shape[1]):
        for d, name in enumerate(windows.names):
            rows.append(dict(position=t+1, variable=name, instantaneous=float(instantaneous[:, t, d].mean()),
                             importance=float(importance[:, t, d].mean())))
    write_csv(out / "position_profiles.csv", rows)
    if windows.name != "Exchange":
        hours = np.array([[int(value[11:13]) for value in row] for row in windows.times[indices, 1:-1]])
        rows = []
        for hour in sorted(np.unique(hours)):
            mask = hours == hour
            for d, name in enumerate(windows.names):
                rows.append(dict(hour=int(hour), variable=name, samples=int(mask.sum()),
                                 instantaneous=float(instantaneous[..., d][mask].mean()),
                                 importance=float(importance[..., d][mask].mean())))
        write_csv(out / "clock_profiles.csv", rows)
    selected = indices[np.linspace(0, len(indices)-1, min(diagnostic_windows, len(indices)), dtype=int)]
    solver.capture = True
    try:
        evaluate(model, x, windows.y, selected, min(batch, len(selected)), device)
    finally:
        solver.capture = False
    write_csv(out / "solver_diagnostics.csv", solver.diagnostics)


def worker(job, settings, out, dependency=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "job.json", job)
    start = time.monotonic()
    seed_everything(job["seed"])
    torch.set_num_threads(settings["cpu_threads"])
    device = torch.device(job["device"])
    windows = load_windows(job["dataset"])
    x, parts, split = prepare_data(windows, job["protocol"], job.get("fold", 0),
                                  settings["temporal_folds"], settings["max_windows"])
    train, val, test = parts
    save_json(out / "split.json", split)
    columns = np.arange(len(windows.names))
    if job["kind"] == "sufficiency":
        if job["selector"] == "random":
            columns = np.sort(np.random.default_rng(job["seed"]).choice(len(columns), len(columns)//2, replace=False))
        else:
            rank = json.loads((Path(dependency) / "train_ranking.json").read_text(encoding="utf-8"))
            columns = np.sort(np.argsort(-np.array(rank["scores"]), kind="stable")[:len(columns)//2])
        x = x[..., columns].copy()
        save_json(out / "selected_features.json", {"selector": job["selector"], "ranking_split": "train",
                  "columns": columns.tolist(), "variables": [windows.names[i] for i in columns], "source": dependency})
    model = make_model(job["dataset"], job["model"], len(columns), device)
    spec = DATASETS[job["dataset"]]
    batch = min(spec["batch"], settings["max_windows"]) if settings["max_windows"] else spec["batch"]
    drop = job["protocol"] == "released"
    model_file = out / "best.pt"
    with Solver(job.get("solver", "original"), job.get("ridge", 0.)) as solver:
        if job["kind"] == "audit":
            model.load_state_dict(torch.load(ROOT / job["checkpoint"], map_location=device, weights_only=True))
            best_epoch, best_value = None, None
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=spec["lr"], weight_decay=spec["decay"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=.5, patience=20, min_lr=.0001)
            best_value, best_epoch = float("inf"), None
            history = []
            epochs = settings["epochs"] or spec["epochs"]
            for epoch in range(epochs):
                model.train()
                total_loss, count = 0., 0
                for batch_id, (bx, by) in enumerate(loader(x, windows.y, train, batch, shuffle=True, drop_last=True)):
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(bx.to(device), device)[0].squeeze(-1)
                    loss = torch.nn.functional.mse_loss(pred, by.to(device))
                    if not torch.isfinite(loss):
                        torch.save(model.state_dict(), out / "failed_model.pt")
                        np.savez_compressed(out / "failed_batch.npz", x=bx.numpy(), y=by.numpy())
                        save_json(out / "failure_location.json", dict(epoch=epoch, batch=batch_id, phase="loss"))
                        raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, batch={batch_id}")
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        torch.save(model.state_dict(), out / "failed_model.pt")
                        np.savez_compressed(out / "failed_batch.npz", x=bx.numpy(), y=by.numpy())
                        save_json(out / "failure_location.json", dict(epoch=epoch, batch=batch_id, phase="gradient"))
                        raise FloatingPointError(f"Non-finite gradient at epoch={epoch}, batch={batch_id}")
                    optimizer.step()
                    total_loss += loss.item() * len(bx)
                    count += len(bx)
                if not count:
                    raise ValueError("No training batches")
                pv, yv, _ = evaluate(model, x, windows.y, val, batch, device, drop_last=drop)
                val_metrics = metrics(pv, yv)
                mse = val_metrics["rmse"]**2
                if mse < best_value:
                    best_value, best_epoch = mse, epoch
                    torch.save(model.state_dict(), model_file)
                scheduler.step(mse)
                history.append(dict(epoch=epoch, train_mse=total_loss/count, val_rmse=val_metrics["rmse"],
                                    lr=optimizer.param_groups[0]["lr"], val_windows=len(pv)))
                write_csv(out / "training.csv", history)
                print(f"epoch={epoch+1}/{epochs} train_mse={total_loss/count:.6g} val_rmse={val_metrics['rmse']:.6g}", flush=True)
            model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        # Always export complete test predictions; retained scores use exactly the old prefix.
        pred, true, coeff = evaluate(model, x, windows.y, test, batch, device,
                                     explanation="Delelstm" if job["model"] == "Delelstm" else None)
        keep = len(test)//batch*batch
        result = dict(job, status="success", seconds=time.monotonic()-start,
                      best_epoch=best_epoch, best_val_mse=best_value, train_windows=len(train),
                      val_windows=len(val), test_windows=len(test), retained_windows=keep,
                      metrics_all=metrics(pred, true), metrics_retained=metrics(pred[:keep], true[:keep]),
                      smoke=bool(settings["max_windows"] or settings["epochs"]))
        np.savez_compressed(out / "predictions.npz", prediction=pred, target=true,
                            window_ids=windows.ids[test], prediction_time=windows.times[test, 2:])
        if job["model"] == "Delelstm":
            export_explanations(out, windows, test, coeff, solver, model, x, batch, device, settings["diagnostic_windows"])
        if job["kind"] == "train" and job["protocol"] == "full_eval" and job["dataset"] == "Electricity" and job["model"] != "LSTM":
            _, _, weights = evaluate(model, x, windows.y, train, batch, device, explanation=job["model"])
            save_json(out / "train_ranking.json", {"variables": windows.names,
                      "scores": rank_features(job["model"], weights).tolist(), "split": "train"})
        if job.get("device_compare") and device.type == "cuda":
            # Identical checkpoint and inputs, strict FP32 on both devices; no training claim.
            sample = test[:min(8, len(test))]
            pg, tg, _ = evaluate(model, x, windows.y, sample, len(sample), device)
            model.to("cpu")
            pc, _, _ = evaluate(model, x, windows.y, sample, len(sample), torch.device("cpu"))
            save_json(out / "device_comparison.json", {"scope": "same_checkpoint_forward_only", "windows": len(sample),
                      "prediction_max_abs_difference": float(np.max(np.abs(pc-pg))),
                      "cpu_metrics": metrics(pc, tg), "gpu_metrics": metrics(pg, tg)})
    result["seconds"] = time.monotonic() - start
    save_json(out / "result.json", result)
    return result
