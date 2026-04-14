#!/usr/bin/env python3
"""Evaluate a platform runtime candidate against local baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", dest="python_output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--prediction-threshold", type=float, default=5e-4)
    parser.add_argument("--min-speedup-over-baseline", type=float, default=1.0)
    parser.add_argument("--min-speedup-over-python", type=float, default=1.0)
    parser.add_argument("--require-python-fps", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def first_run(output: dict[str, Any]) -> dict[str, Any]:
    runs = output.get("runs") or []
    if not runs:
        raise SystemExit(f"{output.get('implementation', '<unknown>')} output has no runs")
    return runs[0]


def mean_fps(output: dict[str, Any]) -> float:
    summary = output.get("summary") or {}
    if "mean_frames_per_second" in summary:
        return float(summary["mean_frames_per_second"])

    runs = output.get("runs") or []
    if not runs:
        raise SystemExit(f"{output.get('implementation', '<unknown>')} output has no runs")
    return sum(float(run["frames_per_second"]) for run in runs) / len(runs)


def mean_timing(output: dict[str, Any], name: str) -> float | None:
    summary = output.get("summary") or {}
    key = f"mean_{name}_ms"
    if key in summary:
        return float(summary[key])
    return None


def prediction_stats(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise SystemExit(f"prediction lengths differ: {len(left)} != {len(right)}")
    if not left:
        raise SystemExit("prediction arrays are empty")

    diffs = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    max_index, max_abs_diff = max(enumerate(diffs), key=lambda item: item[1])
    sorted_diffs = sorted(diffs)
    p99_index = min(len(sorted_diffs) - 1, int(len(sorted_diffs) * 0.99))
    return {
        "count": len(diffs),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": sum(diffs) / len(diffs),
        "p99_abs_diff": sorted_diffs[p99_index],
        "max_abs_diff_frame": max_index,
    }


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def correctness_report(
    python_output: dict[str, Any],
    candidate_output: dict[str, Any],
    prediction_threshold: float,
) -> dict[str, Any]:
    python_run = first_run(python_output)
    candidate_run = first_run(candidate_output)
    single_stats = prediction_stats(
        python_run["predictions"]["single_frame"],
        candidate_run["predictions"]["single_frame"],
    )
    many_hot_stats = prediction_stats(
        python_run["predictions"]["many_hot"],
        candidate_run["predictions"]["many_hot"],
    )
    frame_counts_equal = python_run.get("frame_count") == candidate_run.get("frame_count")
    checksums_equal = python_run.get("checksum_fnv1a64") == candidate_run.get("checksum_fnv1a64")
    scenes_equal = python_run.get("scenes") == candidate_run.get("scenes")
    passed = (
        frame_counts_equal
        and checksums_equal
        and scenes_equal
        and single_stats["max_abs_diff"] <= prediction_threshold
        and many_hot_stats["max_abs_diff"] <= prediction_threshold
    )
    return {
        "passed": passed,
        "frame_counts_equal": frame_counts_equal,
        "checksums_equal": checksums_equal,
        "scenes_equal": scenes_equal,
        "single_frame": single_stats,
        "many_hot": many_hot_stats,
    }


def performance_report(
    python_output: dict[str, Any],
    baseline_output: dict[str, Any],
    candidate_output: dict[str, Any],
    min_speedup_over_baseline: float,
    min_speedup_over_python: float,
    require_python_fps: bool,
) -> dict[str, Any]:
    python_fps = mean_fps(python_output)
    baseline_fps = mean_fps(baseline_output)
    candidate_fps = mean_fps(candidate_output)
    baseline_gate_fps = baseline_fps * min_speedup_over_baseline
    python_gate_fps = python_fps * min_speedup_over_python

    reasons: list[str] = []
    if candidate_fps < baseline_gate_fps:
        reasons.append(
            f"candidate FPS {candidate_fps:.6g} is below baseline gate {baseline_gate_fps:.6g}"
        )
    if require_python_fps and candidate_fps < python_gate_fps:
        reasons.append(f"candidate FPS {candidate_fps:.6g} is below Python gate {python_gate_fps:.6g}")

    return {
        "passed": not reasons,
        "require_python_fps": require_python_fps,
        "python_mean_fps": python_fps,
        "baseline_mean_fps": baseline_fps,
        "candidate_mean_fps": candidate_fps,
        "candidate_vs_baseline_speedup": ratio(candidate_fps, baseline_fps),
        "candidate_vs_python_speedup": ratio(candidate_fps, python_fps),
        "baseline_gate_fps": baseline_gate_fps,
        "python_gate_fps": python_gate_fps if require_python_fps else None,
        "candidate_mean_total_ms": mean_timing(candidate_output, "total"),
        "candidate_mean_inference_ms": mean_timing(candidate_output, "inference"),
        "reasons": reasons,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    python_output = load(args.python_output)
    baseline_output = load(args.baseline)
    candidate_output = load(args.candidate)
    correctness = correctness_report(python_output, candidate_output, args.prediction_threshold)
    performance = performance_report(
        python_output,
        baseline_output,
        candidate_output,
        args.min_speedup_over_baseline,
        args.min_speedup_over_python,
        args.require_python_fps,
    )
    passed = correctness["passed"] and performance["passed"]
    recommendation = (
        "candidate is eligible for runtime replacement"
        if passed
        else "keep the current default path and continue runtime/kernel work"
    )
    return {
        "passed": passed,
        "candidate_name": args.candidate_name,
        "thresholds": {
            "prediction_max_abs_diff": args.prediction_threshold,
            "min_speedup_over_baseline": args.min_speedup_over_baseline,
            "min_speedup_over_python": args.min_speedup_over_python,
            "require_python_fps": args.require_python_fps,
        },
        "inputs": {
            "python": str(args.python_output),
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
        },
        "correctness": correctness,
        "performance": performance,
        "recommendation": recommendation,
        "environments": {
            "python": python_output.get("environment"),
            "baseline": baseline_output.get("environment"),
            "candidate": candidate_output.get("environment"),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    correctness = report["correctness"]
    performance = report["performance"]
    lines = [
        "# Runtime Candidate Report",
        "",
        f"Status: **{status}**",
        "",
        f"Candidate: `{report['candidate_name']}`",
        "",
        "## Correctness",
        "",
        f"- Passed: `{correctness['passed']}`",
        f"- Frame counts equal: `{correctness['frame_counts_equal']}`",
        f"- Decode checksums equal: `{correctness['checksums_equal']}`",
        f"- Scenes equal: `{correctness['scenes_equal']}`",
        f"- Single-frame max abs diff: `{correctness['single_frame']['max_abs_diff']}`",
        f"- Many-hot max abs diff: `{correctness['many_hot']['max_abs_diff']}`",
        "",
        "## Performance",
        "",
        f"- Passed: `{performance['passed']}`",
        f"- Python mean FPS: `{performance['python_mean_fps']}`",
        f"- Baseline mean FPS: `{performance['baseline_mean_fps']}`",
        f"- Candidate mean FPS: `{performance['candidate_mean_fps']}`",
        f"- Candidate vs baseline speedup: `{performance['candidate_vs_baseline_speedup']}`",
        f"- Candidate vs Python speedup: `{performance['candidate_vs_python_speedup']}`",
        f"- Require Python FPS: `{performance['require_python_fps']}`",
    ]
    if performance["reasons"]:
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend(f"- {reason}" for reason in performance["reasons"])
    lines.extend(["", f"Recommendation: {report['recommendation']}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.output_md.write_text(render_markdown(report) + "\n")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
