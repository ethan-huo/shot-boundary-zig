# Runtime Plan

## Current Decision

截至 2026-04-14，AutoShot 是默认模型方向。Linux 路径是 Zig + ONNX Runtime；macOS 路径仍是 Zig + MLX-C 的历史 TransNetV2 实现，后续如需 parity 再单独迁移。

Zig CLI 已经输出完整 `segment` JSON，并通过共享 Python correctness/performance gate。runtime 由编译目标决定，不暴露 CLI backend 选择。

## Runtime Boundary

- Zig 负责可移植 CLI、视频 decode/windowing 编排和标准化 JSON 输出。
- 平台 runtime 实现是同一 CLI contract 后面的边界：macOS 走 MLX-C，Linux 走 ONNX Runtime。
- Python 只负责模型转换、导出、reference output 生成和 acceptance gate。
- MLX-C 构建细节在 `references/build.md`；本地 MLX-C 产物放在 gitignored `externals/`，不要再放 `.scratch/`。

核心 contract 应保持小：平台 runtime 接收标准化 100-frame RGB window 输入并返回 probabilities/timings；CLI 在边界处理 I/O、validation 和 report formatting。

## Linux ONNX Runtime Status

- `scripts/export_autoshot_onnx.py` 从 upstream AutoShot@F1 PyTorch checkpoint 导出 ONNX model，并验证 batch=1/2/3 dynamic-batch smoke。
- `scripts/export_onnx.py` 保留为历史 TransNetV2 对比/回退工具。
- Linux build 只创建 ONNX Runtime artifacts，不创建 MLX smoke/build targets。
- CPU EP 是 Linux 默认构建路径。AutoShot quick check 已确认 SHOT F1 高于 TransNetV2 baseline，但当前 CPU ONNX speed 未确认更快。
- CUDA EP 可用 `-Donnxruntime-cuda=true` opt in；当前 max20 gate 性能通过但 predictions 不过阈值，需要后续对 CUDA 图执行差异做 graph/export 调整。
