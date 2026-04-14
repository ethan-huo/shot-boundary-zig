# Runtime Plan

## Current Decision

截至 2026-04-14，AutoShot 是默认模型方向，TransNetV2 作为显式 fallback/comparison 模型保留。Linux 路径是 Zig + ONNX Runtime；macOS 路径是 Zig + MLX-C。

Zig CLI 已经输出完整 `segment` JSON，并通过共享 Python correctness/performance gate。runtime 由编译目标决定，不暴露 CLI backend 选择。

## Runtime Boundary

- Zig 负责可移植 CLI、视频 decode/windowing 编排和标准化 JSON 输出。
- 平台 runtime 实现是同一 CLI contract 后面的边界：macOS 走 MLX-C，Linux 走 ONNX Runtime。
- Python 只负责模型转换、导出、reference output 生成和 acceptance gate。
- MLX-C 构建细节在 `references/build.md`；本地 MLX-C 产物放在 gitignored `externals/`，不要再放 `.scratch/`。

核心 contract 应保持小：平台 runtime 接收标准化 100-frame RGB window 输入并返回 probabilities/timings；CLI 在边界处理 I/O、validation 和 report formatting。

## Model Artifacts

- 导出完成的模型 artifact 放在 repo 根目录 `models/`，并由 Git LFS 跟踪。
- 默认模型路径按编译目标、可执行文件目录和 `--model` 选择：AutoShot 在 macOS 使用 `{bin_dir}/models/autoshot.safetensors`，在 Linux 使用 `{bin_dir}/models/autoshot.onnx`；TransNetV2 在 macOS 使用 `{bin_dir}/models/transnetv2.safetensors`，Linux 约定为 `{bin_dir}/models/transnetv2.onnx`。
- CLI 仍保留 `--weights <path>` 用于覆盖默认模型，但常规测试和本地运行不应依赖 `target/models/` 或外部冷启动导出。
- `--model` 的默认值是 `autoshot`。`transnetv2` 只作为显式 opt-in；不要让它重新成为默认主线。
- 当前 checkout 只有 `models/transnetv2.safetensors`，没有提交 `models/transnetv2.onnx`。Linux 的 TransNetV2 默认 artifact 需要后续重新生成并纳入 LFS。

## Runtime Status

- `scripts/export_autoshot_onnx.py` 从 upstream AutoShot@F1 PyTorch checkpoint 导出 ONNX model，并验证 batch=1/2/3 dynamic-batch smoke。
- `scripts/export_autoshot_safetensors.py` 从同一 checkpoint 导出 macOS MLX-C 使用的 safetensors model。
- `scripts/run_autoshot_reference.py` 生成 AutoShot PyTorch reference JSON，供 ONNX/MLX runtime candidate gate 复用。
- `scripts/export_onnx.py` 保留为 TransNetV2 ONNX artifact 再生成工具；它不是默认 runtime 路线。
- Linux build 只创建 ONNX Runtime artifacts；macOS build 只创建 MLX-C artifacts。
- CPU EP 是 Linux 默认构建路径。AutoShot quick check 已确认 SHOT F1 高于 TransNetV2 baseline，但当前 CPU ONNX speed 未确认更快。
- CUDA EP 可用 `-Donnxruntime-cuda=true` opt in；当前 max20 gate 性能通过但 predictions 不过阈值，需要后续对 CUDA 图执行差异做 graph/export 调整。

## macOS MLX-C Runtime Status

- `src/mlx_model.zig` 已切到 AutoShot@F1 拓扑：6 个独立 layer、DDCNNV2A shared spatial conv、n_dilation=5、residual+pool pairing、FrameSimilarity/ColorHistograms/head 复用同一 runtime ABI。
- AutoShot@F1 的 `Attention1D(n_layer=0)` 不参与前向；MLX runtime 加载 live `fc1_0`，忽略 checkpoint 中保留但不用的 `fc1` 权重。
- `src/mlx_model.zig` 同时恢复了历史 TransNetV2 拓扑，使用 `--model transnetv2` 显式选择；默认阈值按模型拆分，AutoShot 为 `0.296`，TransNetV2 为 quick-check baseline 使用的 `0.02`。
- 真实 checkpoint 导出的 `models/autoshot.safetensors` 已通过 Python AutoShot reference vs Zig/MLX gate：`assets/333.mp4 --max-frames 20 --runs 3` 下 single-frame max abs diff 约 `1.96e-8`，many-hot max abs diff 约 `4.17e-7`，场景输出一致；候选平均 FPS 约 `93.82`，约为 Python reference 的 `3.46x`。
