#!/usr/bin/env python3
"""Run the AutoShot@F1 PyTorch reference and emit normalized JSON.

Recommended invocation:

  uv run --with torch --with einops --with numpy --with safetensors --with packaging \
    scripts/run_autoshot_reference.py \
    --upstream .scratch/autoshot \
    --weights models/autoshot.safetensors \
    --video assets/333.mp4 \
    --output target/autoshot-python-reference.json \
    --runs 5
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODEL_WIDTH = 48
MODEL_HEIGHT = 27
MODEL_CHANNELS = 3
MODEL_CONTEXT_FRAMES = 25
MODEL_WINDOW_FRAMES = 100
MODEL_OUTPUT_FRAMES_PER_WINDOW = 50
FRAME_BYTES = MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS
FNV1A64_OFFSET = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.296)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def fnv1a64(data: bytes) -> str:
    checksum = FNV1A64_OFFSET
    for byte in data:
        checksum ^= byte
        checksum = (checksum * FNV1A64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{checksum:016x}"


def decode_frames(video: Path, max_frames: int | None) -> tuple[Any, str]:
    import numpy as np

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{MODEL_WIDTH}x{MODEL_HEIGHT}", "-"])

    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    raw = completed.stdout
    if len(raw) == 0 or len(raw) % FRAME_BYTES != 0:
        raise SystemExit(f"decoded rawvideo byte length is invalid: {len(raw)}")

    frames = np.frombuffer(raw, dtype=np.uint8).reshape((-1, MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNELS))
    return frames, fnv1a64(raw)


def load_model(upstream: Path, weights: Path) -> Any:
    import torch
    from safetensors.torch import load_file

    sys.path.insert(0, str(upstream))
    import supernet_flattransf_3_8_8_8_13_12_0_16_60 as autoshot

    model = autoshot.TransNetV2Supernet().eval()
    state_dict = load_file(str(weights))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(f"weights mismatch: missing={missing}, unexpected={unexpected}")
    torch.set_grad_enabled(False)
    return model


def predict_frames_instrumented(model: Any, frames: Any) -> tuple[Any, Any, dict[str, float]]:
    import numpy as np
    import torch

    if len(frames) == 0:
        raise SystemExit("video did not decode any frames")

    timings = {"windowing_ms": 0.0, "inference_ms": 0.0}
    remainder = len(frames) % MODEL_OUTPUT_FRAMES_PER_WINDOW
    padded_frames_start = MODEL_CONTEXT_FRAMES
    padded_frames_end = MODEL_CONTEXT_FRAMES + MODEL_OUTPUT_FRAMES_PER_WINDOW - (
        MODEL_OUTPUT_FRAMES_PER_WINDOW if remainder == 0 else remainder
    )

    windowing_started = time.perf_counter()
    start_frame = np.expand_dims(frames[0], 0)
    end_frame = np.expand_dims(frames[-1], 0)
    padded_inputs = np.concatenate(
        [start_frame] * padded_frames_start + [frames] + [end_frame] * padded_frames_end,
        axis=0,
    )
    timings["windowing_ms"] += (time.perf_counter() - windowing_started) * 1_000.0

    predictions: list[tuple[Any, Any]] = []
    ptr = 0
    while ptr + MODEL_WINDOW_FRAMES <= len(padded_inputs):
        windowing_started = time.perf_counter()
        model_input = padded_inputs[ptr : ptr + MODEL_WINDOW_FRAMES][np.newaxis]
        ptr += MODEL_OUTPUT_FRAMES_PER_WINDOW
        timings["windowing_ms"] += (time.perf_counter() - windowing_started) * 1_000.0

        inference_started = time.perf_counter()
        inputs = torch.from_numpy(model_input).permute(0, 4, 1, 2, 3)
        single_logits, many_logits = model(inputs)
        predictions.append(
            (
                torch.sigmoid(
                    single_logits[0, MODEL_CONTEXT_FRAMES : MODEL_CONTEXT_FRAMES + MODEL_OUTPUT_FRAMES_PER_WINDOW, 0]
                ).numpy(),
                torch.sigmoid(
                    many_logits[0, MODEL_CONTEXT_FRAMES : MODEL_CONTEXT_FRAMES + MODEL_OUTPUT_FRAMES_PER_WINDOW, 0]
                ).numpy(),
            )
        )
        timings["inference_ms"] += (time.perf_counter() - inference_started) * 1_000.0

    single_frame = np.concatenate([single for single, _many in predictions])[: len(frames)]
    many_hot = np.concatenate([many for _single, many in predictions])[: len(frames)]
    return single_frame, many_hot, timings


def predictions_to_scenes(predictions: Any, threshold: float) -> Any:
    import numpy as np

    scenes = []
    previous_is_transition = False
    current_start = 0
    for index, prediction in enumerate(predictions):
        is_transition = prediction > threshold
        if previous_is_transition and not is_transition:
            current_start = index
        if not previous_is_transition and is_transition and index != 0:
            scenes.append([current_start, index])
        previous_is_transition = is_transition

    if not previous_is_transition:
        scenes.append([current_start, len(predictions) - 1])
    if not scenes and len(predictions) > 0:
        scenes.append([0, len(predictions) - 1])
    return np.asarray(scenes, dtype=np.int32)


def scene_dicts(scenes: Any) -> list[dict[str, int]]:
    return [{"start": int(start), "end": int(end)} for start, end in scenes.tolist()]


def summarize(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    total_ms = [run["timings"]["total_ms"] for run in runs]
    fps = [run["frames_per_second"] for run in runs]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "run_count": len(runs),
        "min_frames_per_second": min(fps),
        "max_frames_per_second": max(fps),
        "mean_frames_per_second": mean(fps),
        "min_total_ms": min(total_ms),
        "max_total_ms": max(total_ms),
        "mean_total_ms": mean(total_ms),
        "mean_load_model_ms": mean([run["timings"]["load_model_ms"] for run in runs]),
        "mean_decode_ms": mean([run["timings"]["decode_ms"] for run in runs]),
        "mean_windowing_ms": mean([run["timings"]["windowing_ms"] for run in runs]),
        "mean_inference_ms": mean([run["timings"]["inference_ms"] for run in runs]),
        "mean_postprocess_ms": mean([run["timings"]["postprocess_ms"] for run in runs]),
    }


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be greater than zero")

    import numpy as np
    import torch

    runs: list[dict[str, Any]] = []

    for run_index in range(args.runs):
        load_started = time.perf_counter()
        model = load_model(args.upstream, args.weights)
        load_model_ms = (time.perf_counter() - load_started) * 1_000.0

        decode_started = time.perf_counter()
        frames, checksum = decode_frames(args.video, args.max_frames)
        decode_ms = (time.perf_counter() - decode_started) * 1_000.0

        single_frame, many_hot, model_timings = predict_frames_instrumented(model, frames)

        postprocess_started = time.perf_counter()
        scenes = predictions_to_scenes(single_frame, args.threshold)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1_000.0

        total_ms = (
            load_model_ms
            + decode_ms
            + model_timings["windowing_ms"]
            + model_timings["inference_ms"]
            + postprocess_ms
        )
        frame_count = int(len(frames))

        runs.append(
            {
                "run_index": run_index,
                "frame_count": frame_count,
                "target_width": MODEL_WIDTH,
                "target_height": MODEL_HEIGHT,
                "checksum_fnv1a64": checksum,
                "limited_by_max_frames": args.max_frames is not None and frame_count >= args.max_frames,
                "predictions": {
                    "single_frame": [float(value) for value in single_frame.astype(np.float32)],
                    "many_hot": [float(value) for value in many_hot.astype(np.float32)],
                },
                "scenes": scene_dicts(scenes),
                "timings": {
                    "load_model_ms": load_model_ms,
                    "decode_ms": decode_ms,
                    "windowing_ms": model_timings["windowing_ms"],
                    "inference_ms": model_timings["inference_ms"],
                    "postprocess_ms": postprocess_ms,
                    "total_ms": total_ms,
                },
                "frames_per_second": frame_count / (total_ms / 1_000.0) if total_ms > 0 else 0.0,
            }
        )

    output = {
        "implementation": "python-autoshot-pytorch",
        "video": str(args.video),
        "weights": str(args.weights),
        "threshold": args.threshold,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "os": platform.system(),
            "machine": platform.machine(),
        },
        "runs": runs,
        "summary": summarize(runs),
    }
    content = json.dumps(output, indent=2, ensure_ascii=False) + "\n"

    if args.output is None:
        print(content, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
