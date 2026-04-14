# shot-boundary-zig

这个仓库现在作为 shot-boundary detection 本地实现和验收工作区。

## Layout

- `src/`: Zig CLI、macOS MLX-C runtime、Linux ONNX Runtime backend。
- `assets/`: 本地验收资产。
- `scripts/`: 模型转换/导出、Python 参考/探索输出、runtime candidate gate。
- `docs/`: 当前 runtime 计划和 AutoShot 记录。
- `references/`: 构建系统和依赖管理参考。

## Current Decision

AutoShot 是当前默认模型方向。Linux 主线是 Zig + ONNX Runtime；macOS MLX-C 路径仍是平台 runtime 边界里的历史 TransNetV2 实现，后续如需 macOS parity 再单独迁移。runtime 由编译目标决定，不通过 CLI 选择 backend。

不要再按 `../lens/PLAN.md` 里的旧结论推进。当前方向见 `docs/PLAN.md`；构建细节见 `references/build.md`。

## Zig Commands

从仓库根目录运行：

先从 AutoShot Google Drive 模型文件夹下载 `ckpt_0_200_0.pth` 到 `target/models/`；当前验证过的 sha256 是 `3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6`。

```bash
git clone --depth 1 https://github.com/wentaozhu/AutoShot externals/AutoShot

uv run --with torch --with einops --with numpy --with onnx --with onnxruntime \
  scripts/export_autoshot_onnx.py \
  --upstream externals/AutoShot \
  --checkpoint target/models/ckpt_0_200_0.pth \
  --output target/models/autoshot.onnx

zig build -Doptimize=ReleaseFast run -- segment assets/333.mp4 \
  --weights target/models/autoshot.onnx \
  --window-batch-size 2 \
  --runs 3 \
  --max-frames 20 \
  --format json > target/zig-onnx-cpu-segment-runs3-max20.json
```

runtime gate 用 `--baseline` 表示当前已接受的平台实现输出。Linux ONNX 复用同一份 JSON contract。

```bash
uv run python scripts/evaluate_runtime_candidate.py \
  --python target/python-reference-max20.json \
  --baseline target/python-reference-max20.json \
  --candidate target/zig-onnx-cpu-segment-runs3-max20.json \
  --candidate-name zig-onnx-autoshot-cpu-max20-runs3 \
  --require-python-fps \
  --output-json target/runtime-candidate-zig-onnx-cpu-max20-runs3.json \
  --output-md target/runtime-candidate-zig-onnx-cpu-max20-runs3.md
```
