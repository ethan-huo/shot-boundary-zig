#!/usr/bin/env python3
"""Export official TransNetV2 weights to safetensors.

Recommended invocation:

  uv run --with torch --with tensorflow==2.16.2 --with safetensors \
    scripts/export_safetensors.py \
    --upstream /Users/dio/Projects/ethan-huo/.scratch/transnetv2-rs-research/TransNetV2 \
    --output models/transnetv2.safetensors

If a PyTorch state dict has already been generated with the upstream converter,
pass it via `--pytorch-weights` and omit TensorFlow from the uv dependency list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytorch-weights", type=Path)
    parser.add_argument("--tf-weights", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def load_state_dict(args: argparse.Namespace):
    import torch

    if args.pytorch_weights is not None:
        loaded = torch.load(args.pytorch_weights, map_location="cpu")
        return loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded

    pytorch_dir = args.upstream / "inference-pytorch"
    sys.path.insert(0, str(pytorch_dir))

    from convert_weights import convert_weights

    tf_weights = args.tf_weights or args.upstream / "inference" / "transnetv2-weights"
    torch_model, _ = convert_weights(str(tf_weights))
    return torch_model.state_dict()


def validate_state_dict(upstream: Path, state_dict) -> None:
    pytorch_dir = upstream / "inference-pytorch"
    sys.path.insert(0, str(pytorch_dir))

    from transnetv2_pytorch import TransNetV2

    expected = TransNetV2().state_dict()
    missing = sorted(name for name in set(expected) - set(state_dict) if not name.endswith("num_batches_tracked"))
    extra = sorted(set(state_dict) - set(expected) - {name for name in state_dict if name.endswith("num_batches_tracked")})
    mismatched = sorted(
        name
        for name, tensor in state_dict.items()
        if name in expected and tuple(tensor.shape) != tuple(expected[name].shape)
    )

    if missing or extra or mismatched:
        raise SystemExit(
            "state dict does not match upstream PyTorch model:\n"
            f"missing={missing}\n"
            f"extra={extra}\n"
            f"mismatched={mismatched}"
        )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    raw_state_dict = load_state_dict(args)
    validate_state_dict(args.upstream, raw_state_dict)

    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in raw_state_dict.items()
        if not name.endswith("num_batches_tracked")
    }

    save_file(state_dict, str(args.output))

    manifest = {
        name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        for name, tensor in sorted(state_dict.items())
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {args.output}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
