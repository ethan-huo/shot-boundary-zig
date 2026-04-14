#!/usr/bin/env python3
"""Export AutoShot@F1 checkpoint weights to the safetensors model used by macOS MLX.

Recommended invocation:

  uv run --with torch --with einops --with numpy --with safetensors --with packaging \
    scripts/export_autoshot_safetensors.py \
    --upstream .scratch/autoshot \
    --checkpoint models/ckpt_0_200_0.pth \
    --output models/autoshot.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


AUTOSHOT_RECOMMENDED_THRESHOLD = 0.296


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True, help="Path to a shallow clone of wentaozhu/AutoShot.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to ckpt_0_200_0.pth.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
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
    import supernet_flattransf_3_8_8_8_13_12_0_16_60 as autoshot

    return autoshot


def load_state_dict(upstream: Path, checkpoint: Path) -> tuple[dict[str, Any], int, int]:
    import torch

    autoshot = import_autoshot(upstream)
    loaded = torch.load(checkpoint, map_location="cpu")
    state_dict = loaded["net"] if isinstance(loaded, dict) and "net" in loaded else loaded
    model = autoshot.TransNetV2Supernet().eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return state_dict, len(state_dict), sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    raw_state_dict, state_tensor_count, parameter_count = load_state_dict(args.upstream, args.checkpoint)
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in raw_state_dict.items()
        if not name.endswith("num_batches_tracked")
    }
    save_file(state_dict, str(args.output))

    manifest = {
        "model": "AutoShot@F1",
        "format": "safetensors",
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
        "runtime": {
            "backend": "macOS MLX-C",
            "weights": str(args.output),
        },
        "tensors": {
            name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            for name, tensor in sorted(state_dict.items())
        },
    }
    manifest_path = args.manifest or default_manifest_path(args.output)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {args.output}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
