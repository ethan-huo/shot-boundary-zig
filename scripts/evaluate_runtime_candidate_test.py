#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluate_runtime_candidate.py")
SPEC = importlib.util.spec_from_file_location("evaluate_runtime_candidate", MODULE_PATH)
assert SPEC is not None
evaluate_runtime_candidate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_runtime_candidate)


def output(
    fps: float,
    single_frame: list[float] | None = None,
    many_hot: list[float] | None = None,
) -> dict:
    single_frame = single_frame or [0.1, 0.8, 0.2]
    many_hot = many_hot or [0.2, 0.7, 0.3]
    return {
        "implementation": "test",
        "environment": {"test": True},
        "runs": [
            {
                "frame_count": len(single_frame),
                "checksum_fnv1a64": "abc",
                "predictions": {
                    "single_frame": single_frame,
                    "many_hot": many_hot,
                },
                "scenes": [{"start": 0, "end": len(single_frame) - 1}],
                "frames_per_second": fps,
            }
        ],
        "summary": {
            "mean_frames_per_second": fps,
            "mean_total_ms": 1000.0 / fps,
            "mean_inference_ms": 900.0 / fps,
        },
    }


class RuntimeCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir_handle = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir_handle.cleanup)
        self.temp_dir = Path(self.temp_dir_handle.name)

    def test_passes_when_correct_and_faster_than_python(self) -> None:
        args = argparse.Namespace(
            prediction_threshold=5e-4,
            min_speedup_over_baseline=1.0,
            min_speedup_over_python=1.0,
            require_python_fps=True,
            candidate_name="fast",
            **self.write_outputs(output(100.0), output(40.0), output(130.0)),
        )

        report = evaluate_runtime_candidate.evaluate(args)

        self.assertTrue(report["passed"])
        self.assertGreater(report["performance"]["candidate_vs_python_speedup"], 1.0)

    def test_fails_when_probability_diff_exceeds_threshold(self) -> None:
        args = argparse.Namespace(
            prediction_threshold=5e-4,
            min_speedup_over_baseline=1.0,
            min_speedup_over_python=1.0,
            require_python_fps=False,
            candidate_name="wrong",
            **self.write_outputs(output(100.0), output(40.0), output(60.0, single_frame=[0.1, 0.7, 0.2])),
        )

        report = evaluate_runtime_candidate.evaluate(args)

        self.assertFalse(report["passed"])
        self.assertFalse(report["correctness"]["passed"])

    def test_can_require_python_fps(self) -> None:
        args = argparse.Namespace(
            prediction_threshold=5e-4,
            min_speedup_over_baseline=1.0,
            min_speedup_over_python=1.0,
            require_python_fps=True,
            candidate_name="mid",
            **self.write_outputs(output(100.0), output(40.0), output(60.0)),
        )

        report = evaluate_runtime_candidate.evaluate(args)

        self.assertFalse(report["passed"])
        self.assertIn("Python gate", report["performance"]["reasons"][0])

    def write_outputs(
        self,
        python_output: dict,
        baseline_rust_output: dict,
        candidate_output: dict,
    ) -> dict:
        paths = {
            "python_output": self.temp_dir / "python.json",
            "baseline_rust": self.temp_dir / "baseline-rust.json",
            "candidate": self.temp_dir / "candidate.json",
        }
        paths["python_output"].write_text(json.dumps(python_output))
        paths["baseline_rust"].write_text(json.dumps(baseline_rust_output))
        paths["candidate"].write_text(json.dumps(candidate_output))
        return paths


if __name__ == "__main__":
    unittest.main()
