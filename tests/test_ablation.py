"""Scientific protocol checks; run with python -m unittest discover -s tests -v."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ablation_data import load_windows, prepare_data, split_indices
from ablation_engine import Solver, make_model, metrics, modules, normalized_components, seed_everything
from ablation_report import build_report, read_csv, sign_flip
from ablation_suite import build_plan, configuration, parser


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

    def test_plan_has_all_protocols_and_sufficiency_dependencies(self):
        settings = configuration(parser().parse_args([]))
        plan = build_plan(settings)
        self.assertEqual(len(plan), 554)
        ids = {job["id"] for job in plan}
        self.assertTrue(all(job["dependency"] in ids for job in plan if "dependency" in job))
        self.assertEqual(sum(job["kind"] == "audit" for job in plan), 90)
        self.assertEqual(sum(job["kind"] == "sufficiency" for job in plan), 25)
        for job in plan:
            if job["kind"] == "audit" and job["source"] == "five_run":
                metadata_path = (ROOT / job["checkpoint"]).parent.parent / "run_metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(job["seed"], metadata["torch_seed"])

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
