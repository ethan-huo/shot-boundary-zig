# Model Artifacts

Model binaries in this directory are tracked with Git LFS.

Mainline artifacts:

- `autoshot.safetensors` for macOS MLX-C
- `autoshot.onnx` for Linux ONNX Runtime
- `ckpt_0_200_0.pth` upstream checkpoint used to rebuild the exported AutoShot artifacts
- `autoshot.safetensors.manifest.json` and `autoshot.onnx.manifest.json` export metadata

Explicit fallback/comparison artifact:

- `transnetv2.safetensors`

AutoShot remains the default runtime model. On macOS, TransNetV2 is selectable with `--model transnetv2` and loads from `{bin_dir}/models/transnetv2.safetensors` when `--weights` is omitted.

There is no committed `transnetv2.onnx` artifact in this checkout yet. Linux can still use an explicit TransNetV2 ONNX path once that artifact is regenerated, but the default installed Linux artifact is currently AutoShot.
