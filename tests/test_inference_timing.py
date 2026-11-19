import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_inference as benchmark


class InferenceTimingTest(unittest.TestCase):
    def test_single_latency_and_full_pass_throughput_include_tail(self):
        calls = []

        class Model(torch.nn.Module):
            def forward(self, batch, device):
                calls.append((len(batch), self.training, torch.is_inference_mode_enabled()))
                return batch[:, :, :1], None, None

        inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2, 1)
        timestamps = []
        elapsed = 0.0
        for seconds in [0.001, 0.002, 0.003, 0.004, 0.005] + [0.25] * benchmark.TIMED_PASSES:
            timestamps.extend([elapsed, elapsed + seconds])
            elapsed += seconds
        with patch.object(benchmark.time, "perf_counter", side_effect=timestamps):
            result = benchmark.measure(Model(), inputs, 2, inputs.numpy().squeeze(-1))

        sizes = [1] * (benchmark.WARMUP_WINDOWS + 5) + [2, 2, 1] * (1 + benchmark.TIMED_PASSES)
        self.assertEqual(calls, [(n, False, True) for n in sizes])
        self.assertAlmostEqual(result["single_p50_ms"], 3.0)
        self.assertAlmostEqual(result["single_p95_ms"], 4.8)
        self.assertAlmostEqual(result["batch_windows_per_s"], 20.0)
        self.assertAlmostEqual(result["full_test_pass_median_ms"], 250.0)
        self.assertEqual(result["batch_prediction_max_abs_diff"], 0.0)
        self.assertIsNone(result["batch_peak_allocated_mib"])


if __name__ == "__main__":
    unittest.main()
