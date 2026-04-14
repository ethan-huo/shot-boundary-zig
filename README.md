# shot-boundary

A fast, native CLI for video shot boundary detection. Powered by [AutoShot](https://github.com/wentaozhu/AutoShot) (CVPR 2023) with [TransNetV2](https://github.com/soCzech/TransNetV2) as an opt-in fallback.

- **macOS**: Zig + [MLX](https://github.com/ml-explore/mlx) (Apple Silicon GPU)
- **Linux**: Zig + [ONNX Runtime](https://onnxruntime.ai/) (CPU; CUDA opt-in)

## Install

### Linux (pre-built)

```sh
curl -fsSL https://raw.githubusercontent.com/AIGC-Hackers/shot-boundary-zig/main/scripts/install.sh | sh
```

Installs to `~/.shot-boundary/` and symlinks the binary to `~/.local/bin/`. Supports x86_64 and aarch64.

Pin a version:

```sh
curl -fsSL https://raw.githubusercontent.com/AIGC-Hackers/shot-boundary-zig/main/scripts/install.sh | sh -s -- --version v0.1.2
```

Set `SHOT_BOUNDARY_HOME` to change the install directory.

### Build from source

Prerequisites: [Zig 0.15.2](https://ziglang.org/download/). On macOS, CMake is also needed (MLX-C is fetched and built automatically).

```sh
git clone https://github.com/AIGC-Hackers/shot-boundary-zig.git
cd shot-boundary-zig
git lfs pull
zig build -Doptimize=ReleaseFast
```

The binary is at `zig-out/bin/shot-boundary`; model files are at `zig-out/models/`.

## Usage

Detect scene boundaries in a video:

```sh
shot-boundary segment video.mp4
```

Output is JSON by default. Each detected scene is a `{start, end}` frame pair:

```sh
shot-boundary segment video.mp4 --format txt
```

### Options

```
shot-boundary segment <video> [options]

  --model <autoshot|transnetv2>   Model family (default: autoshot)
  --weights <path>                Model file path (auto-resolved from install prefix)
  --format <json|txt>             Output format (default: json)
  --threshold <0..1>              Scene threshold (default: 0.296 for autoshot)
  --max-frames <n>                Decode at most n frames
  --window-batch-size <n>         Inference batch size
  --runs <n>                      Repeat n times for benchmarking (default: 1)
```

### Other commands

```sh
shot-boundary env               # Print runtime environment (Zig version, OS, arch)
shot-boundary decode-smoke <v>  # Benchmark raw video decode throughput
```

## Platform support

| Platform | Runtime | Model format | GPU |
|---|---|---|---|
| macOS (Apple Silicon) | MLX-C v0.6.0 | `.safetensors` | Metal (via MLX) |
| Linux x86_64 | ONNX Runtime 1.24.4 | `.onnx` | CUDA (opt-in) |
| Linux aarch64 | ONNX Runtime 1.24.4 | `.onnx` | - |

Enable CUDA on Linux:

```sh
zig build -Doptimize=ReleaseFast -Donnxruntime-cuda=true
```

## Model

The default model is **AutoShot@F1** (Zhu et al., CVPR 2023), a NAS-derived architecture that improves on TransNetV2 in both accuracy (+4.2% F1 on the SHOT dataset) and efficiency (37 vs 41 GMACs). It holds the top position on the PapersWithCode shot boundary detection leaderboard.

Input spec: 100-frame sliding windows at 48x27 RGB. Output: per-frame `single_frame` and `many_hot` sigmoid scores; frames above the threshold mark shot boundaries.

**TransNetV2** is available as a fallback via `--model transnetv2` (default threshold: 0.02).

## Project layout

```
src/            Zig CLI, video decoder, MLX and ONNX Runtime backends
models/         Pre-exported model artifacts (Git LFS)
scripts/        Model export, Python reference inference, runtime evaluation gate
docs/           AutoShot architecture reference
```

## Development

### Tests

```sh
zig build test   # unit tests + format check + lint
```

### Re-exporting model artifacts

This is only needed when rebuilding from the upstream PyTorch checkpoint. Download `ckpt_0_200_0.pth` (sha256: `3e85290546...a6`) from the AutoShot model repository into `models/`, then:

```sh
mkdir -p .scratch
git clone --depth 1 https://github.com/wentaozhu/AutoShot .scratch/autoshot

# ONNX (Linux)
uv run --with torch --with einops --with numpy --with onnx --with onnxruntime --with packaging \
  scripts/export_autoshot_onnx.py \
  --upstream .scratch/autoshot \
  --checkpoint models/ckpt_0_200_0.pth \
  --output models/autoshot.onnx

# safetensors (macOS)
uv run --with torch --with einops --with numpy --with safetensors --with packaging \
  scripts/export_autoshot_safetensors.py \
  --upstream .scratch/autoshot \
  --checkpoint models/ckpt_0_200_0.pth \
  --output models/autoshot.safetensors
```

### Runtime evaluation gate

Validate a Zig build against the Python reference:

```sh
uv run --with torch --with einops --with numpy --with safetensors --with packaging \
  scripts/run_autoshot_reference.py \
  --upstream .scratch/autoshot \
  --weights models/autoshot.safetensors \
  --video assets/333.mp4 \
  --max-frames 20 \
  --output target/reference.json

uv run scripts/evaluate_runtime_candidate.py \
  --python target/reference.json \
  --baseline target/reference.json \
  --candidate target/candidate.json \
  --candidate-name zig-mlx \
  --require-python-fps \
  --output-json target/gate-result.json \
  --output-md target/gate-result.md
```
