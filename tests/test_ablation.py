"""Scientific protocol checks; run with python -m unittest discover -s tests -v."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ablation_data import load_windows, prepare_data, raw_pm_windows, split_indices
from ablation_engine import Solver, make_model, metrics, modules, normalized_components, observed_mse, seed_everything
from ablation_report import build_report, read_csv, sign_flip
from ablation_suite import build_plan, configuration, parser
from audit_temporal_protocol import audit_dataset


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {name: load_windows(name) for name in ("Electricity", "PM", "Exchange")}
        torch.set_num_threads(1)

    def test_released_data_matches_upstream_prepare(self):
        modules()
        for name, data in self.data.items():
            with self.subTest(dataset=name):
                old = sys.argv
                sys.argv = [name + ".py"]
                try:
                    upstream = importlib.import_module(name)
                finally:
                    sys.argv = old
                loaders = upstream.prepare(555)
                x, splits, _ = prepare_data(data, "released")
                for loader, indices in zip(loaders, splits):
                    original_x, original_y = loader.dataset.tensors
                    np.testing.assert_allclose(x[indices], original_x.numpy(), rtol=1e-6, atol=1e-7)
                    np.testing.assert_array_equal(data.y[indices], original_y.numpy())

    def test_chronological_exchange_has_no_shared_raw_rows(self):
        data = self.data["Exchange"]
        for fold in range(3):
            train, val, test = split_indices(data, "chronological", fold, 3)
            self.assertLess(data.ids[train[-1]] + 29, data.ids[val[0]])
            self.assertLess(data.ids[val[-1]] + 29, data.ids[test[0]])

    def test_train_only_normalization_ignores_future_values(self):
        from dataclasses import replace
        data = self.data["Electricity"]
        x, splits, metadata = prepare_data(data, "chronological")
        changed = data.x.copy()
        changed[splits[2]] += 10000
        _, _, altered = prepare_data(replace(data, x=changed), "chronological")
        self.assertEqual(metadata["minimum"], altered["minimum"])
        self.assertEqual(metadata["maximum"], altered["maximum"])
        self.assertAlmostEqual(float(x[splits[0]].min()), 0.)
        self.assertAlmostEqual(float(x[splits[0]].max()), 1.)

    def test_timestamp_and_target_alignment(self):
        for name, data in self.data.items():
            target_column = {"Electricity": "Consumption", "PM": "pm2.5", "Exchange": "Singapore"}[name]
            expected = data.x[:, 2:, data.names.index(target_column)].astype(np.float32)
            np.testing.assert_array_equal(data.y, expected)
            self.assertEqual(data.times[:, 1:-1].shape, data.y.shape)
            self.assertEqual(data.times.dtype.kind, "U")  # NPZ timestamps load without pickle.
        self.assertEqual(self.data["PM"].times[0, 2][11:13], "02")

    def test_all_nine_temporal_splits_and_observed_target_counts(self):
        for dataset in self.data:
            with self.subTest(dataset=dataset):
                folds = list(audit_dataset(dataset))
                self.assertEqual(len(folds), 3)
                self.assertTrue(all(row["train_scaling_invariant"] for row in folds))
                if dataset == "PM":
                    last = folds[-1]
                    self.assertEqual((last["train_windows"], last["val_windows"], last["test_windows"]),
                                     (1368, 274, 183))
                    self.assertEqual((last["train_valid_targets"], last["val_valid_targets"],
                                      last["test_valid_targets"], last["test_targets"]), (28338, 5975, 3964, 4026))
                if dataset == "Exchange":
                    self.assertEqual(folds[-1]["released_overlapping_test_windows"], 757)

    def test_pm_raw_input_uses_only_prior_observations(self):
        name = "PRSA_data_2010.1.1-2014.12.31.csv"
        raw = pd.read_csv(ROOT / "vendor" / "DeLELSTM" / "DATA" / "PM2.5" / name, nrows=120)
        raw.loc[29, "pm2.5"] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            raw.to_csv(path, index=False)
            before = raw_pm_windows(path.parent)
            raw.loc[48:, "pm2.5"] = 5000
            raw.to_csv(path, index=False)
            after = raw_pm_windows(path.parent)
        self.assertFalse(before.usable_inputs[0])
        self.assertTrue(np.isnan(before.x[0, :, 0]).all())
        self.assertEqual(before.x[1, 0, 0], 129)
        self.assertEqual(before.x[1, 5, 0], raw.loc[28, "pm2.5"])
        self.assertTrue(np.isnan(before.y[1, 3]))
        np.testing.assert_allclose(before.x[:2], after.x[:2], equal_nan=True)
        self.assertNotEqual(before.x[2, 0, 0], after.x[2, 0, 0])
        with self.assertRaisesRegex(ValueError, "Load PM windows"):
            prepare_data(self.data["PM"], "chronological")

    def test_missing_targets_do_not_change_loss_gradients_or_scores(self):
        target = torch.tensor([3., float("nan"), 5.])
        for missing_prediction in (0., 1.e8):
            prediction = torch.tensor([1., missing_prediction, 6.], requires_grad=True)
            loss, count = observed_mse(prediction, target)
            self.assertEqual((loss.item(), count), (2.5, 2))
            loss.backward()
            torch.testing.assert_close(prediction.grad, torch.tensor([-2., 0., 1.]))
            pred = prediction.detach().numpy()
            self.assertEqual(metrics(pred, target.numpy()), metrics(pred[[0, 2]], target.numpy()[[0, 2]]))
        missing = torch.full_like(target, float("nan"))
        self.assertEqual(observed_mse(prediction, missing), (None, 0))
        with self.assertRaisesRegex(ValueError, "No observed targets"):
            metrics(pred, missing.numpy())

    def test_original_forward_preserved_and_ridge_differentiable(self):
        seed_everything(555)
        model = make_model("PM", "Delelstm", 8, torch.device("cpu"))
        x = torch.rand(2, 24, 8)
        original = model(x, torch.device("cpu"))[0].detach()
        with Solver("original") as solver:
            solver.capture = True
            actual = model(x, torch.device("cpu"))[0]
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
            self.assertEqual(len(solver.diagnostics), 2*22)
        for mode, ridge in [("ridge", .01), ("pinv", 0.)]:
            model.zero_grad(set_to_none=True)
            with Solver(mode, ridge):
                result = model(x, torch.device("cpu"))[0]
                self.assertTrue(torch.isfinite(result).all())
                result.square().mean().backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_mape_matches_existing_metric_convention(self):
        from torchmetrics.functional.regression import mean_absolute_percentage_error
        y = np.array([[0., 2., -1.]], dtype=np.float32)
        p = np.array([[.1, 1., 2.]], dtype=np.float32)
        expected = mean_absolute_percentage_error(torch.tensor(p), torch.tensor(y)).item()*100
        self.assertAlmostEqual(metrics(p, y)["mape_percent"]/expected, 1., places=6)

    def test_zero_coefficients_and_small_sample_statistics(self):
        share, importance = normalized_components(np.zeros((2, 3, 4)))
        np.testing.assert_array_equal(share, .5)
        np.testing.assert_array_equal(importance, 0.)
        self.assertEqual(sign_flip([1., 1., 1., 1., 1.]), .0625)

    def test_forecast_preflight_without_legacy_packages(self):
        from importlib.metadata import PackageNotFoundError, version
        from unittest.mock import patch
        from ablation_suite import preflight

        optional = {"scipy", "scikit-learn", "torchmetrics"}

        def installed_version(name):
            if name in optional:
                raise PackageNotFoundError(name)
            return version(name)

        with tempfile.TemporaryDirectory() as folder, patch("importlib.metadata.version", side_effect=installed_version):
            out = Path(folder)
            preflight(out, {"devices": ["cpu"], "forecast_only": True})
            packages = json.loads((out / "environment.json").read_text(encoding="utf-8"))["packages"]
            self.assertTrue(all(packages[name] is None for name in optional))
            self.assertEqual(packages["torch"], version("torch"))
            with self.assertRaises(PackageNotFoundError):
                preflight(out, {"devices": ["cpu"], "forecast_only": False})

    def test_plan_has_all_protocols_and_sufficiency_dependencies(self):
        settings = configuration(parser().parse_args([]))
        plan = build_plan(settings)
        self.assertEqual(len(plan), 554)
        ids = {job["id"] for job in plan}
        self.assertTrue(all(job["dependency"] in ids for job in plan if "dependency" in job))
        self.assertEqual(sum(job["kind"] == "audit" for job in plan), 90)
        self.assertEqual(sum(job["kind"] == "sufficiency" for job in plan), 25)
        for job in plan:
            if job["kind"] == "sufficiency":
                self.assertEqual((job["protocol"], job["fold"]), ("chronological", 2))
                if job["selector"] != "random":
                    source = next(p for p in plan if p["id"] == job["dependency"])
                    self.assertEqual((source["model"], source["protocol"], source["fold"], source["seed"]),
                                     (job["selector"], "chronological", 2, job["seed"]))
            if job["kind"] == "audit" and job["source"] == "five_run":
                metadata_path = (ROOT / job["checkpoint"]).parent.parent / "run_metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(job["seed"], metadata["torch_seed"])
        forecast = build_plan(configuration(parser().parse_args(["--forecast-only"])))
        self.assertEqual(len(forecast), 225)
        self.assertTrue(all(j["protocol"] == "chronological" and j["kind"] == "train" for j in forecast))
        smoke = build_plan(configuration(parser().parse_args(["--stage", "smoke"])), smoke=True)
        self.assertEqual(len(smoke), 32)
        self.assertTrue(all(j["protocol"] == "chronological" for j in smoke))

    def test_failed_and_unstarted_tasks_remain_in_report(self):
        settings = configuration(parser().parse_args(["--datasets", "Exchange", "--models", "Delelstm"]))
        jobs = build_plan(settings)[:3]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
            (root / "plan.json").write_text(json.dumps(jobs), encoding="utf-8")
            for job, status in zip(jobs, ["failed", "blocked"]):
                folder = root / "jobs" / job["id"] / "attempt_001"
                folder.mkdir(parents=True)
                (folder / "result.json").write_text(json.dumps(dict(job, status=status, reason="test failure")), encoding="utf-8")
            build_report(root)
            runs = read_csv(root / "report" / "runs.csv")
            self.assertEqual([r["status"] for r in runs], ["failed", "blocked", "missing"])
            report = (root / "report" / "report.md").read_text(encoding="utf-8")
            self.assertIn("成功 0 项", report)
            self.assertIn("失败、阻塞或未完成 3 项", report)
            self.assertNotIn("全部计划任务已成功完成", report)


if __name__ == "__main__":
    unittest.main()
