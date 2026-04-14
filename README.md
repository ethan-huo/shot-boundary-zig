# shot-boundary-zig

这个仓库现在作为 shot-boundary detection 本地实现和验收工作区。

## Layout

- `src/`: Zig CLI、macOS MLX-C runtime、Linux ONNX Runtime backend。
- `assets/`: 本地验收资产。
- `scripts/`: 模型转换/导出、Python 参考/探索输出、runtime candidate gate。
- `docs/`: 当前 runtime 计划和 AutoShot 记录。
- `references/`: 构建系统和依赖管理参考。

## Current Decision

AutoShot 是当前默认模型方向；TransNetV2 作为显式 fallback/comparison 模型保留。Linux 主线是 Zig + ONNX Runtime；macOS 主线是 Zig + MLX-C。runtime backend 由编译目标决定，模型由 `--model` 选择，默认是 `autoshot`。

不要再按 `../lens/PLAN.md` 里的旧结论推进。当前方向见 `docs/PLAN.md`；构建细节见 `references/build.md`。

## Install

Linux 用户可通过 curl 一键安装（自动检测架构，下载到 `~/.shot-boundary/`，symlink 到 `~/.local/bin/`）：

```bash
curl -fsSL https://raw.githubusercontent.com/AIGC-Hackers/shot-boundary-zig/main/scripts/install.sh | sh
```

安装指定版本：

```bash
curl -fsSL https://raw.githubusercontent.com/AIGC-Hackers/shot-boundary-zig/main/scripts/install.sh | sh -s -- --version v0.1.2
```

可通过 `SHOT_BOUNDARY_HOME` 环境变量自定义安装目录。

## Zig Commands

从仓库根目录运行：

常规运行直接使用 Git LFS 拉下来的 artifacts。`zig build` 会把 repo 根目录 `models/` 下的 runtime artifacts 安装到 `{prefix}/models/`；`segment` 未传 `--weights` 时按编译目标和 `--model` 从 `{prefix}/models/` 加载模型（binary 位于 `{prefix}/bin/`，向上一级找到 prefix 根目录）。默认 `--model autoshot` 在 macOS 加载 `autoshot.safetensors`，在 Linux 加载 `autoshot.onnx`；macOS 可用 `--model transnetv2` 加载 `transnetv2.safetensors`。只有需要重建 AutoShot artifact 时，才从 AutoShot Google Drive/Baidu 模型文件夹下载 `ckpt_0_200_0.pth` 到 `models/`；当前验证过的 sha256 是 `3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6`。

```bash
mkdir -p .scratch
git clone --depth 1 https://github.com/wentaozhu/AutoShot .scratch/autoshot

uv run --with torch --with einops --with numpy --with onnx --with onnxruntime --with packaging \
  scripts/export_autoshot_onnx.py \
  --upstream .scratch/autoshot \
  --checkpoint models/ckpt_0_200_0.pth \
  --output models/autoshot.onnx

uv run --with torch --with einops --with numpy --with safetensors --with packaging \
  scripts/export_autoshot_safetensors.py \
  --upstream .scratch/autoshot \
  --checkpoint models/ckpt_0_200_0.pth \
  --output models/autoshot.safetensors

# Linux:
zig build -Doptimize=ReleaseFast run -- segment assets/333.mp4 \
  --runs 3 \
  --max-frames 20 \
  --format json > target/zig-onnx-cpu-segment-runs3-max20.json

# macOS:
zig build -Doptimize=ReleaseFast run -- segment assets/333.mp4 \
  --runs 3 \
  --max-frames 20 \
  --format json > target/zig-mlx-segment-runs3-max20.json

# macOS TransNetV2 fallback:
zig build -Doptimize=ReleaseFast run -- segment assets/333.mp4 \
  --model transnetv2 \
  --runs 3 \
  --max-frames 20 \
  --format json > target/zig-mlx-transnetv2-runs3-max20.json
```

runtime gate 用 `--baseline` 表示当前已接受的平台实现输出。Linux ONNX 和 macOS MLX 复用同一份 JSON contract。

```bash
uv run --with torch --with einops --with numpy --with safetensors --with packaging \
  scripts/run_autoshot_reference.py \
  --upstream .scratch/autoshot \
  --weights models/autoshot.safetensors \
  --video assets/333.mp4 \
  --max-frames 20 \
  --output target/autoshot-python-reference-max20.json

uv run scripts/evaluate_runtime_candidate.py \
  --python target/autoshot-python-reference-max20.json \
  --baseline target/autoshot-python-reference-max20.json \
  --candidate target/zig-mlx-segment-runs3-max20.json \
  --candidate-name zig-mlx-autoshot-max20-runs3 \
  --require-python-fps \
  --output-json target/runtime-candidate-zig-mlx-max20-runs3.json \
  --output-md target/runtime-candidate-zig-mlx-max20-runs3.md
```
