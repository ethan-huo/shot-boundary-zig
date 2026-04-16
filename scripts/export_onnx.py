#!/usr/bin/env python3
"""Export official TransNetV2 weights to the ONNX model used by the Zig Linux runtime.

Recommended invocation:

  uv run --with torch --with tensorflow==2.16.2 --with onnx --with packaging \
    scripts/export_onnx.py \
    --upstream /path/to/TransNetV2 \
    --output models/transnetv2.onnx

If a PyTorch state dict has already been generated with the upstream converter,
pass it via `--pytorch-weights` and omit TensorFlow from the uv dependency list.
If `models/transnetv2.safetensors` is available, pass it via
`--safetensors-weights` to export ONNX without TensorFlow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from export_safetensors import load_state_dict, validate_state_dict

MODEL_WIDTH = 48
MODEL_HEIGHT = 27
MODEL_CHANNELS = 3
MODEL_CONTEXT_FRAMES = 25
MODEL_WINDOW_FRAMES = 100
MODEL_OUTPUT_FRAMES_PER_WINDOW = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytorch-weights", type=Path)
    parser.add_argument("--safetensors-weights", type=Path)
    parser.add_argument("--tf-weights", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def load_export_state_dict(args: argparse.Namespace):
    if args.safetensors_weights is None:
        return load_state_dict(args)

    from safetensors.torch import load_file

    return load_file(str(args.safetensors_weights))


def load_model(args: argparse.Namespace):
    import torch
    import torch.nn.functional as F

    pytorch_dir = args.upstream / "inference-pytorch"
    sys.path.insert(0, str(pytorch_dir))

    import transnetv2_pytorch
    from transnetv2_pytorch import TransNetV2

    patch_transnetv2_export(transnetv2_pytorch, F, torch)

    raw_state_dict = load_export_state_dict(args)
    validate_state_dict(args.upstream, raw_state_dict)

    model = TransNetV2()
    missing, unexpected = model.load_state_dict(raw_state_dict, strict=False)
    bad_missing = [name for name in missing if not name.endswith("num_batches_tracked")]
    bad_unexpected = [name for name in unexpected if not name.endswith("num_batches_tracked")]
    if bad_missing or bad_unexpected:
        raise SystemExit(f"failed to load model state: missing={bad_missing}, unexpected={bad_unexpected}")

    model.eval()
    return model


def patch_transnetv2_export(transnetv2_pytorch, F, torch) -> None:
    def lookup_similarities(similarities, lookup_window):
        similarities_padded = F.pad(
            similarities,
            [(lookup_window - 1) // 2, (lookup_window - 1) // 2],
        )
        batch_size = similarities.shape[0]
        time_window = similarities.shape[1]
        time_indices = torch.arange(time_window, device=similarities.device).reshape(1, time_window, 1)
        lookup_indices = torch.arange(lookup_window, device=similarities.device).reshape(1, 1, lookup_window)
        lookup_indices = (lookup_indices + time_indices).expand(batch_size, time_window, lookup_window)
        return torch.gather(similarities_padded, 2, lookup_indices)

    def frame_similarity_forward(self, inputs):
        x = torch.cat([torch.mean(item, dim=[3, 4]) for item in inputs], dim=1)
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))
        return F.relu(self.fc(lookup_similarities(similarities, self.lookup_window)))

    def compute_color_histograms(frames):
        frames = frames.int()
        red, green, blue = frames[:, :, :, :, 0], frames[:, :, :, :, 1], frames[:, :, :, :, 2]
        binned_values = ((red >> 5) << 6) + ((green >> 5) << 3) + (blue >> 5)

        batch_size, time_window, height, width = binned_values.shape
        binned_values = binned_values.reshape(batch_size * time_window, height * width)
        frame_bin_prefix = (torch.arange(batch_size * time_window, device=frames.device) << 9).reshape(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).reshape(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        updates = torch.ones_like(binned_values, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values, updates)

        histograms = histograms.reshape(batch_size, time_window, 512).float()
        return F.normalize(histograms, p=2, dim=2)

    def color_histograms_forward(self, inputs):
        x = self.compute_color_histograms(inputs)
        similarities = torch.bmm(x, x.transpose(1, 2))
        similarities = lookup_similarities(similarities, self.lookup_window)

        if self.fc is not None:
            return F.relu(self.fc(similarities))
        return similarities

    transnetv2_pytorch.FrameSimilarity.forward = frame_similarity_forward
    transnetv2_pytorch.ColorHistograms.compute_color_histograms = staticmethod(compute_color_histograms)
    transnetv2_pytorch.ColorHistograms.forward = color_histograms_forward


class TransNetV2OnnxWrapper:
    def __init__(self, model):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, frames):
                single_frame_logits, outputs = self.wrapped(frames)
                many_hot_logits = outputs["many_hot"]
                start = MODEL_CONTEXT_FRAMES
                stop = MODEL_CONTEXT_FRAMES + MODEL_OUTPUT_FRAMES_PER_WINDOW
                single_frame = torch.sigmoid(single_frame_logits[:, start:stop, 0])
                many_hot = torch.sigmoid(many_hot_logits[:, start:stop, 0])
                return single_frame, many_hot

        self.module = Wrapper(model)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    wrapper = TransNetV2OnnxWrapper(load_model(args)).module
    wrapper.eval()
    dummy = torch.zeros(
        (1, MODEL_WINDOW_FRAMES, MODEL_HEIGHT, MODEL_WIDTH, MODEL_CHANNELS),
        dtype=torch.uint8,
    )

    torch.onnx.export(
        wrapper,
        dummy,
        str(args.output),
        export_params=True,
        opset_version=args.opset,
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

    manifest = {
        "model": "TransNetV2",
        "format": "onnx",
        "opset": args.opset,
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
    }
    manifest_path = args.manifest or args.output.with_name(f"{args.output.name}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {args.output}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
