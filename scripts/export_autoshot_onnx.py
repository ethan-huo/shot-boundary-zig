#!/usr/bin/env python3
"""Export AutoShot@F1 to the ONNX model used by the Zig Linux runtime.

Recommended invocation:

  CUDA_VISIBLE_DEVICES=-1 uv run --with torch --with einops --with numpy --with onnx --with onnxruntime \
    scripts/export_autoshot_onnx.py \
    --upstream /tmp/autoshot-codex \
    --checkpoint models/ckpt_0_200_0.pth \
    --output models/autoshot.onnx

The exported model keeps the existing runtime ABI:

  frames:       uint8   [batch, 100, 27, 48, 3]
  single_frame: float32 [batch, 50]
  many_hot:     float32 [batch, 50]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

MODEL_WIDTH = 48
MODEL_HEIGHT = 27
MODEL_CHANNELS = 3
MODEL_CONTEXT_FRAMES = 25
MODEL_WINDOW_FRAMES = 100
MODEL_OUTPUT_FRAMES_PER_WINDOW = 50
AUTOSHOT_RECOMMENDED_THRESHOLD = 0.296


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True, help="Path to a shallow clone of wentaozhu/AutoShot.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to ckpt_0_200_0.pth.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.manifest.json")


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
    return model, len(state_dict), sum(parameter.numel() for parameter in model.parameters())


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


def export_onnx(wrapper, output: Path, opset: int) -> int:
    import onnx
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, MODEL_WINDOW_FRAMES, MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNELS), dtype=torch.uint8)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        export_params=True,
        opset_version=opset,
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
    return len(model.graph.node)


def smoke_onnx(wrapper, output: Path) -> list[dict[str, Any]]:
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    smoke = []
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
        smoke.append(
            {
                "batch_size": batch_size,
                "max_abs_diff": max(diffs),
                "outputs": [list(item.shape) for item in actual],
            }
        )
    return smoke


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    args = parse_args()
    model, state_tensor_count, parameter_count = load_model(args.upstream, args.checkpoint)
    wrapper = make_wrapper(model)
    node_count = export_onnx(wrapper, args.output, args.opset)
    smoke = None if args.skip_smoke else smoke_onnx(wrapper, args.output)

    manifest = {
        "model": "AutoShot@F1",
        "format": "onnx",
        "opset": args.opset,
        "upstream": {
            "repository": "https://github.com/wentaozhu/AutoShot",
            "path": str(args.upstream),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "state_tensor_count": state_tensor_count,
            "parameter_count": parameter_count,
        },
        "recommended_scene_threshold": AUTOSHOT_RECOMMENDED_THRESHOLD,
        "input": {
            "name": "frames",
            "dtype": "uint8",
            "shape": ["batch", MODEL_WINDOW_FRAMES, MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNELS],
        },
        "outputs": [
            {
                "name": "single_frame",
                "dtype": "float32",
                "shape": ["batch", MODEL_OUTPUT_FRAMES_PER_WINDOW],
                "activation": "sigmoid",
            },
            {
                "name": "many_hot",
                "dtype": "float32",
                "shape": ["batch", MODEL_OUTPUT_FRAMES_PER_WINDOW],
                "activation": "sigmoid",
            },
        ],
        "export": {
            "path": str(args.output),
            "bytes": args.output.stat().st_size,
            "node_count": node_count,
            "dynamic_batch_smoke": smoke,
        },
    }
    manifest_path = args.manifest or default_manifest_path(args.output)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
