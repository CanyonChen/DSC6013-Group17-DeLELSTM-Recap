import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_delelstm_gpu1 as forecast
from ablation_suite import MODELS, build_plan, configuration, latest_success, parser, save


class DelelstmForecastTests(unittest.TestCase):
    def test_transfer_completed_jobs_and_resume(self):
        settings = configuration(parser().parse_args(["--stage", "run", *forecast.FORECAST_ARGS]))
        original = dict(settings, models=MODELS)
        jobs = build_plan(settings)
        self.assertEqual(len(jobs), 45)
        self.assertEqual({(j["protocol"], j["model"]) for j in jobs}, {("chronological", "Delelstm")})
        other = next(j for j in build_plan(original) if j["model"] == "LSTM")
        with tempfile.TemporaryDirectory() as temp:
            source, dest = Path(temp) / "old", Path(temp) / "new"
            save(source / "settings.json", original)
            save(source / "environment.json", {"git_commit": "source-revision",
                                               "packages": {"torch": forecast.version("torch")}})
            for job in (jobs[0], jobs[1], other):
                folder = source / "jobs" / job["id"] / "attempt_001"
                save(folder / "result.json", dict(job, status="success"))
                for name in ("best.pt", "split.json", "predictions.npz"):
                    (folder / name).write_bytes(b"fixture")
            (source / "jobs" / jobs[1]["id"] / "attempt_001" / "predictions.npz").unlink()
            forecast.import_completed(source, dest, settings)
            forecast.import_completed(source, dest, settings)
            copied = latest_success(dest, jobs[0]["id"])
            self.assertEqual((copied / "best.pt").read_bytes(), b"fixture")
            self.assertEqual([p.name for p in copied.parent.iterdir()], ["attempt_001"])
            self.assertEqual(json.loads((copied / "reused_from.json").read_text())["environment"]["git_commit"],
                             "source-revision")
            self.assertIsNone(latest_success(dest, jobs[1]["id"]))
            self.assertFalse((dest / "jobs" / other["id"]).exists())
            self.assertTrue((source / "jobs" / other["id"] / "attempt_001" / "best.pt").is_file())
            self.assertEqual(json.loads((dest / "plan.json").read_text()), jobs)
            save(source / "settings.json", dict(original, epochs=1, max_windows=16))
            with self.assertRaisesRegex(ValueError, "settings or code differ"):
                forecast.import_completed(source, dest, settings)

    def test_launcher_pins_physical_gpu_one(self):
        with (patch.object(sys, "argv", ["run_delelstm_gpu1.py"]),
              patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}),
              patch.object(forecast.subprocess, "check_output", return_value="GPU-physical-one\n") as query,
              patch.object(os, "execlp") as launch):
            forecast.main()
            query.assert_called_once_with(
                ["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-physical-one")
            args = launch.call_args.args
            self.assertEqual(args[args.index("--models") + 1], "Delelstm")
            self.assertEqual(args[args.index("--gpus") + 1], "0")
            self.assertIn("--resume", args)


if __name__ == "__main__":
    unittest.main()
