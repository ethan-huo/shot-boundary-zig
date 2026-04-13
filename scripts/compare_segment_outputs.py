#!/usr/bin/env python3
"""Compare Python TensorFlow and Rust Candle TransNetV2 segment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", dest="python_output", type=Path, required=True)
    parser.add_argument("--rust", dest="rust_output", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--prediction-threshold", type=float, default=5e-4)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def first_run(output: dict[str, Any]) -> dict[str, Any]:
    runs = output.get("runs") or []
    if not runs:
        raise SystemExit(f"{output.get('implementation', '<unknown>')} output has no runs")
    return runs[0]


def prediction_stats(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise SystemExit(f"prediction lengths differ: {len(left)} != {len(right)}")

    diffs = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    max_index, max_abs_diff = max(enumerate(diffs), key=lambda item: item[1])
    sorted_diffs = sorted(diffs)
    p99_index = min(len(sorted_diffs) - 1, int(len(sorted_diffs) * 0.99))

    return {
        "count": len(diffs),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "p99_abs_diff": sorted_diffs[p99_index] if diffs else 0.0,
        "max_abs_diff_frame": max_index,
    }


def compare(args: argparse.Namespace) -> tuple[dict[str, Any], str, bool]:
    python_output = load(args.python_output)
    rust_output = load(args.rust_output)
    python_run = first_run(python_output)
    rust_run = first_run(rust_output)

    single_stats = prediction_stats(
        python_run["predictions"]["single_frame"],
        rust_run["predictions"]["single_frame"],
    )
    many_hot_stats = prediction_stats(
        python_run["predictions"]["many_hot"],
        rust_run["predictions"]["many_hot"],
    )
    scenes_equal = python_run["scenes"] == rust_run["scenes"]
    checksums_equal = python_run.get("checksum_fnv1a64") == rust_run.get("checksum_fnv1a64")
    frame_counts_equal = python_run.get("frame_count") == rust_run.get("frame_count")

    passed = (
        scenes_equal
        and checksums_equal
        and frame_counts_equal
        and single_stats["max_abs_diff"] <= args.prediction_threshold
        and many_hot_stats["max_abs_diff"] <= args.prediction_threshold
    )

    report = {
        "passed": passed,
        "thresholds": {"prediction_max_abs_diff": args.prediction_threshold},
        "inputs": {
            "python": str(args.python_output),
            "rust": str(args.rust_output),
        },
        "frame_counts_equal": frame_counts_equal,
        "checksums_equal": checksums_equal,
        "scenes_equal": scenes_equal,
        "single_frame": single_stats,
        "many_hot": many_hot_stats,
        "python_summary": python_output.get("summary"),
        "rust_summary": rust_output.get("summary"),
        "python_environment": python_output.get("environment"),
        "rust_environment": rust_output.get("environment"),
    }

    markdown = render_markdown(report)
    return report, markdown, passed


def render_markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# TransNetV2 Local Alignment Report",
        "",
        f"Status: **{status}**",
        "",
        "## Correctness",
        "",
        f"- Frame counts equal: `{report['frame_counts_equal']}`",
        f"- Decode checksums equal: `{report['checksums_equal']}`",
        f"- Scenes equal: `{report['scenes_equal']}`",
        f"- Single-frame max abs diff: `{report['single_frame']['max_abs_diff']}`",
        f"- Single-frame mean abs diff: `{report['single_frame']['mean_abs_diff']}`",
        f"- Single-frame p99 abs diff: `{report['single_frame']['p99_abs_diff']}`",
        f"- Many-hot max abs diff: `{report['many_hot']['max_abs_diff']}`",
        f"- Many-hot mean abs diff: `{report['many_hot']['mean_abs_diff']}`",
        f"- Many-hot p99 abs diff: `{report['many_hot']['p99_abs_diff']}`",
        "",
        "## Performance",
        "",
        "| Implementation | Mean FPS | Mean total ms | Mean load ms | Mean decode ms | Mean inference ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for label, key in [("Python TensorFlow", "python_summary"), ("Rust Candle", "rust_summary")]:
        summary = report.get(key) or {}
        lines.append(
            "| {label} | {fps} | {total} | {load} | {decode} | {inference} |".format(
                label=label,
                fps=summary.get("mean_frames_per_second"),
                total=summary.get("mean_total_ms"),
                load=summary.get("mean_load_model_ms"),
                decode=summary.get("mean_decode_ms"),
                inference=summary.get("mean_inference_ms"),
            )
        )

    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Python output: `{report['inputs']['python']}`",
            f"- Rust output: `{report['inputs']['rust']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report, markdown, passed = compare(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.output_md.write_text(markdown + "\n")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
