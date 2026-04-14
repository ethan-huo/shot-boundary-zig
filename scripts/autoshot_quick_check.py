#!/usr/bin/env python3
"""Fast AutoShot feasibility checks.

This script is intentionally exploratory: it keeps the existing Zig ONNX ABI
and answers the two quick questions we care about first:

  1. Do the official predictions show better SHOT accuracy than the baseline?
  2. Can the real AutoShot checkpoint export to a dynamic-batch ONNX model and
     run faster/slower than the current TransNetV2 ONNX on the same windows?
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_WIDTH = 48
MODEL_HEIGHT = 27
MODEL_CHANNELS = 3
MODEL_CONTEXT_FRAMES = 25
MODEL_WINDOW_FRAMES = 100
MODEL_OUTPUT_FRAMES_PER_WINDOW = 50
FRAME_BYTES = MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS

SHOT_THRESHOLDS = np.array(
    [
        0.02,
        0.06,
        0.1,
        0.15,
        0.2,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.255,
        0.26,
        0.265,
        0.27,
        0.275,
        0.28,
        0.2833,
        0.2867,
        0.29,
        0.292,
        0.294,
        0.296,
        0.298,
        0.3,
        0.302,
        0.304,
        0.306,
        0.308,
        0.31,
        0.3133,
        0.3167,
        0.32,
        0.325,
        0.33,
        0.335,
        0.34,
        0.345,
        0.35,
        0.36,
        0.37,
        0.38,
        0.39,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("/tmp/autoshot-codex"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/ckpt_0_200_0.pth"))
    parser.add_argument("--output-onnx", type=Path, default=Path("models/autoshot.onnx"))
    parser.add_argument("--transnetv2-onnx", type=Path, default=Path("models/transnetv2.onnx"))
    parser.add_argument("--video", type=Path, default=Path("assets/333.mp4"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--window-batch-size", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=8, help="Limit speed benchmark windows; 0 means all windows.")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument(
        "--gt-scenes",
        type=Path,
        default=Path("target/autoshot/autoshot_gt_scenes_dict_baseline_v2.pickle"),
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=Path("target/autoshot/baseline_one_hot_pred_dict_baseline.pickle"),
    )
    parser.add_argument(
        "--autoshot-predictions",
        type=Path,
        default=Path("target/autoshot/autoshot_supernet_best_f1_predictions.pickle"),
    )
    parser.add_argument("--output-json", type=Path, default=Path("target/autoshot/quick-check.json"))
    return parser.parse_args()


def import_autoshot(upstream: Path):
    sys.path.insert(0, str(upstream))
    import torch
    import torch.nn.functional as F
    import supernet_flattransf_3_8_8_8_13_12_0_16_60 as autoshot

    def lookup_similarities(similarities, lookup_window):
        similarities_padded = F.pad(
            similarities,
            pad=[(lookup_window - 1) // 2, (lookup_window - 1) // 2, 0, 0, 0, 0],
        )
        batch_size = similarities.shape[0]
        time_window = similarities.shape[1]
        time_indices = torch.arange(time_window, device=similarities.device).reshape(1, time_window, 1)
        lookup_indices = (
            torch.arange(lookup_window, device=similarities.device).reshape(1, 1, lookup_window) + time_indices
        )
        lookup_indices = lookup_indices.expand(batch_size, time_window, lookup_window)
        return torch.gather(similarities_padded, 2, lookup_indices)

    def frame_similarity_forward(self, inputs):
        x = torch.cat([torch.mean(item, dim=[3, 4]) for item in inputs], dim=1)
        if self.stop_gradient:
            x = x.detach()
        x = x.permute(dims=[0, 2, 1])
        batch_size, time_window, old_channels = x.shape
        x = x.reshape(shape=[batch_size * time_window, old_channels])
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        _, new_channels = x.shape
        x = x.reshape(shape=[batch_size, time_window, new_channels])
        similarities = torch.matmul(x, x.permute(dims=[0, 2, 1]))
        return self.fc(lookup_similarities(similarities, self.lookup_window))

    def color_histograms_forward(self, inputs):
        x = self.compute_color_histograms(inputs)
        similarities = torch.matmul(x, x.permute(dims=[0, 2, 1]))
        similarities = lookup_similarities(similarities, self.lookup_window)
        if self.fc is None:
            return similarities
        return self.fc(similarities)

    autoshot.FrameSimilarity.forward = frame_similarity_forward
    autoshot.ColorHistograms.forward = color_histograms_forward
    return autoshot


def load_model(upstream: Path, checkpoint: Path):
    import torch

    autoshot = import_autoshot(upstream)
    loaded = torch.load(checkpoint, map_location="cpu")
    state_dict = loaded["net"] if isinstance(loaded, dict) and "net" in loaded else loaded
    model = autoshot.TransNetV2Supernet().eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model


def make_wrapper(model):
    import torch

    class AutoShotOnnxWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, frames):
            x = frames.permute(0, 4, 1, 2, 3).to(torch.float32)
            single_frame_logits, many_hot_logits = self.wrapped(x)
            start = MODEL_CONTEXT_FRAMES
            stop = MODEL_CONTEXT_FRAMES + MODEL_OUTPUT_FRAMES_PER_WINDOW
            return (
                torch.sigmoid(single_frame_logits[:, start:stop, 0]),
                torch.sigmoid(many_hot_logits[:, start:stop, 0]),
            )

    return AutoShotOnnxWrapper(model).eval()


def export_onnx(wrapper, output: Path) -> dict[str, Any]:
    import onnx
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, MODEL_WINDOW_FRAMES, MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNELS), dtype=torch.uint8)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["frames"],
        output_names=["single_frame", "many_hot"],
        dynamic_axes={
            "frames": {0: "batch"},
            "single_frame": {0: "batch"},
            "many_hot": {0: "batch"},
        },
        dynamo=False,
    )
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    return {"path": str(output), "bytes": output.stat().st_size, "nodes": len(model.graph.node)}


def smoke_onnx(wrapper, onnx_path: Path) -> list[dict[str, Any]]:
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    results = []
    for batch_size in (1, 2, 3):
        frames = torch.randint(
            0,
            256,
            (batch_size, MODEL_WINDOW_FRAMES, MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNELS),
            dtype=torch.uint8,
        )
        with torch.no_grad():
            expected = wrapper(frames)
        actual = session.run(["single_frame", "many_hot"], {"frames": frames.numpy()})
        diffs = [
            float(np.max(np.abs(actual_item - expected_item.detach().numpy())))
            for actual_item, expected_item in zip(actual, expected)
        ]
        results.append({"batch_size": batch_size, "max_abs_diff": max(diffs), "outputs": [list(item.shape) for item in actual]})
    return results


def decode_video(video: Path) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{MODEL_WIDTH}x{MODEL_HEIGHT}",
        "-",
    ]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    raw = completed.stdout
    if len(raw) == 0 or len(raw) % FRAME_BYTES != 0:
        raise SystemExit(f"invalid rawvideo byte length: {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape((-1, MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNELS))


def window_source_indices(frame_count: int) -> list[list[int]]:
    remainder = frame_count % MODEL_OUTPUT_FRAMES_PER_WINDOW
    padded_start = MODEL_CONTEXT_FRAMES
    padded_end = MODEL_CONTEXT_FRAMES + MODEL_OUTPUT_FRAMES_PER_WINDOW - (
        MODEL_OUTPUT_FRAMES_PER_WINDOW if remainder == 0 else remainder
    )
    padded_count = padded_start + frame_count + padded_end
    windows = []
    ptr = 0
    while ptr + MODEL_WINDOW_FRAMES <= padded_count:
        indices = []
        for padded_index in range(ptr, ptr + MODEL_WINDOW_FRAMES):
            if padded_index < padded_start:
                indices.append(0)
            elif padded_index < padded_start + frame_count:
                indices.append(padded_index - padded_start)
            else:
                indices.append(frame_count - 1)
        windows.append(indices)
        ptr += MODEL_OUTPUT_FRAMES_PER_WINDOW
    return windows


def build_windows(frames: np.ndarray) -> np.ndarray:
    windows = window_source_indices(len(frames))
    out = np.empty((len(windows), MODEL_WINDOW_FRAMES, MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNELS), dtype=np.uint8)
    for index, source_indices in enumerate(windows):
        out[index] = frames[source_indices]
    return out


def benchmark_onnx(path: Path, windows: np.ndarray, batch_size: int, runs: int, frame_count: int) -> dict[str, float | str]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batches = [windows[start : start + batch_size] for start in range(0, len(windows), batch_size)]
    for batch in batches:
        session.run(["single_frame", "many_hot"], {"frames": batch})

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        for batch in batches:
            session.run(["single_frame", "many_hot"], {"frames": batch})
        timings.append((time.perf_counter() - started) * 1_000.0)
    mean_ms = float(sum(timings) / len(timings))
    return {
        "path": str(path),
        "runs": runs,
        "window_count": len(windows),
        "batch_size": batch_size,
        "mean_inference_ms": mean_ms,
        "effective_frames_per_second": frame_count / (mean_ms / 1_000.0) if mean_ms > 0 else 0.0,
    }


def predictions_to_scenes(predictions: np.ndarray) -> np.ndarray:
    scenes = []
    t = -1
    t_prev = 0
    start = 0
    for index, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = index
        if t_prev == 0 and t == 1 and index != 0:
            scenes.append([start, index])
        t_prev = t
    if t == 0:
        scenes.append([start, index])
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def transition_counts(gt_scenes: np.ndarray, pred_scenes: np.ndarray) -> tuple[int, int, int]:
    shift = 1.0
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    gt_index = 0
    pred_index = 0
    tp = 0
    fp = 0
    fn = 0
    while gt_index < len(gt_trans) or pred_index < len(pred_trans):
        if pred_index == len(pred_trans):
            fn += 1
            gt_index += 1
        elif gt_index == len(gt_trans):
            fp += 1
            pred_index += 1
        elif pred_trans[pred_index, 1] < gt_trans[gt_index, 0]:
            fp += 1
            pred_index += 1
        elif pred_trans[pred_index, 0] > gt_trans[gt_index, 1]:
            fn += 1
            gt_index += 1
        else:
            gt_index += 1
            pred_index += 1
            tp += 1
    return tp, fp, fn


def evaluate_predictions(predictions: dict[str, np.ndarray], gt_scenes: dict[str, np.ndarray]) -> dict[str, float | int]:
    best = None
    for threshold in SHOT_THRESHOLDS:
        tp = 0
        fp = 0
        fn = 0
        for file_name, prediction in predictions.items():
            pred_scenes = predictions_to_scenes((prediction > np.array([threshold])).astype(np.uint8))
            next_tp, next_fp, next_fn = transition_counts(gt_scenes[file_name], pred_scenes)
            tp += next_tp
            fp += next_fp
            fn += next_fn
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else (precision * recall * 2.0) / (precision + recall)
        result = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        if best is None or f1 > best["f1"]:
            best = result
    assert best is not None
    return best


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def accuracy_report(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.skip_accuracy:
        return None
    required = [args.gt_scenes, args.baseline_predictions, args.autoshot_predictions]
    if not all(path.exists() for path in required):
        return {"skipped": True, "reason": "missing official prediction pickles", "paths": [str(path) for path in required]}
    gt_scenes = load_pickle(args.gt_scenes)
    baseline_predictions = load_pickle(args.baseline_predictions)
    autoshot_predictions = load_pickle(args.autoshot_predictions)
    baseline = evaluate_predictions(baseline_predictions, gt_scenes)
    autoshot = evaluate_predictions(autoshot_predictions, gt_scenes)
    return {
        "video_count": len(gt_scenes),
        "gt_scene_count": sum(len(value) for value in gt_scenes.values()),
        "baseline": baseline,
        "autoshot": autoshot,
        "autoshot_f1_delta": autoshot["f1"] - baseline["f1"],
    }


def speed_report(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.skip_speed:
        return None
    if not args.video.exists():
        return {"skipped": True, "reason": f"missing video: {args.video}"}
    frames = decode_video(args.video)
    windows = build_windows(frames)
    if args.max_windows > 0:
        windows = windows[: args.max_windows]
    effective_frame_count = min(len(frames), len(windows) * MODEL_OUTPUT_FRAMES_PER_WINDOW)
    report: dict[str, Any] = {
        "video": str(args.video),
        "frame_count": int(len(frames)),
        "effective_benchmark_frame_count": int(effective_frame_count),
        "window_count": int(len(windows)),
        "models": {},
    }
    if args.output_onnx.exists():
        try:
            report["models"]["autoshot"] = benchmark_onnx(
                args.output_onnx,
                windows,
                args.window_batch_size,
                args.runs,
                effective_frame_count,
            )
        except Exception as exc:
            report["models"]["autoshot"] = {"path": str(args.output_onnx), "error": f"{type(exc).__name__}: {exc}"}
    if args.transnetv2_onnx.exists():
        try:
            report["models"]["transnetv2"] = benchmark_onnx(
                args.transnetv2_onnx,
                windows,
                args.window_batch_size,
                args.runs,
                effective_frame_count,
            )
        except Exception as exc:
            report["models"]["transnetv2"] = {"path": str(args.transnetv2_onnx), "error": f"{type(exc).__name__}: {exc}"}
    models = report["models"]
    if (
        "autoshot" in models
        and "transnetv2" in models
        and "mean_inference_ms" in models["autoshot"]
        and "mean_inference_ms" in models["transnetv2"]
    ):
        autoshot_ms = models["autoshot"]["mean_inference_ms"]
        transnet_ms = models["transnetv2"]["mean_inference_ms"]
        report["autoshot_vs_transnetv2_speedup"] = transnet_ms / autoshot_ms if autoshot_ms > 0 else None
    return report


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be greater than zero")
    if args.window_batch_size <= 0:
        raise SystemExit("--window-batch-size must be greater than zero")

    report: dict[str, Any] = {"accuracy": accuracy_report(args)}

    wrapper = None
    if not args.skip_export:
        model = load_model(args.upstream, args.checkpoint)
        wrapper = make_wrapper(model)
        report["export"] = export_onnx(wrapper, args.output_onnx)
    if args.output_onnx.exists():
        if wrapper is None:
            model = load_model(args.upstream, args.checkpoint)
            wrapper = make_wrapper(model)
        report["dynamic_batch_smoke"] = smoke_onnx(wrapper, args.output_onnx)

    report["speed"] = speed_report(args)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
