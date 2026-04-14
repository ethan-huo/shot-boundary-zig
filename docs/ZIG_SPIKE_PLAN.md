# Zig Spike Plan

日期：2026-04-14

## Goal

评估是否应该用 Zig 替换当前 Rust 实现，而不是继续在 Rust wrapper 生态上叠更多后端适配。

这个 spike 的判断标准不是语言偏好，而是能否在同一份验收资产和同一套 gate 下证明三件事：

- 输出和官方 TensorFlow Python 参考一致。
- macOS MLX 性能不低于当前 Rust MLX baseline。
- 代码边界和构建路径比 Rust wrapper 版本更简单、更直接。

## Rationale

当前真正的底层供应都是 C 或 C ABI 友好的系统边界：

- macOS: MLX-C
- Linux CPU/CUDA: ONNX Runtime C API
- 视频: FFmpeg/libav

Rust 版本已经证明了模型拆分、窗口化、后处理和验收 gate 是可行的，但它的形状、错误映射和构建封装主要服务 Rust 自身。对于一个只想把 C 供应收束成稳定工具和库的项目，Zig 可能更贴近实际边界。

## Non-Goals

- 不在 spike 通过前删除 Rust 实现。
- 不在 spike 通过前接入 `lens`。
- 不把 Zig 第一版做成完整跨平台抽象层。
- 不重写 Python 权重导出和候选 runtime gate。

Rust 现在是已验证 baseline，不是未来架构的锚点。Zig 要替换它，必须先用同一份报告证明。

## Proposed Layout

```text
.
├── assets/
├── docs/
├── rust/
├── scripts/
└── zig/
    ├── build.zig
    └── src/
```

根目录继续拥有共享资产、Python 参考脚本和 gate。`rust/` 保持当前实现；`zig/` 只承担 spike 代码。

## Shared Contract

Zig CLI 必须输出和当前 `segment` 相同的标准 JSON：

- `implementation`
- `video`
- `weights`
- `threshold`
- `environment`
- `runs[].frame_count`
- `runs[].checksum_fnv1a64`
- `runs[].predictions.single_frame`
- `runs[].predictions.many_hot`
- `runs[].scenes`
- `runs[].timings`
- `summary.mean_frames_per_second`
- `summary.mean_inference_ms`

验收命令继续使用：

```bash
uv run python scripts/evaluate_runtime_candidate.py \
  --python target/reports/python-reference.json \
  --baseline-rust target/reports/rust-mlx-segment-batch2-runs5.json \
  --candidate target/reports/zig-mlx-segment.json \
  --candidate-name zig-mlx \
  --require-python-fps \
  --output-json target/reports/runtime-candidate-zig-mlx.json \
  --output-md target/reports/runtime-candidate-zig-mlx.md
```

主验收资产仍然是 `assets/333.mp4`。

## Phase 0: Skeleton

- 建立 `zig/` 工程。
- 在 macOS 上链接 MLX-C。
- 做最小 CLI：能打印环境信息、解析 `segment <video> --weights <path> --runs N --format json`。
- 明确 allocator 所有权：CLI 顶层拥有 arena/general allocator；模型、解码器、输出 buffer 都要有清晰释放点。

## Phase 1: MLX-C Window Inference

- 读取 safetensors 权重，先允许 CPU 侧 stream 再拷贝到 MLX，避免把权重加载和 MLX device 生命周期绑死。
- 实现单个 `[1,100,27,48,3]` window inference。
- 使用当前已验证的受限卷积路径：`conv2d (1,3,3)` + `conv1d (3,1,1)`。
- 不把通用 `conv3d` 当性能路径；它 correctness 可过，但 Rust MLX 调研已经证明速度不够。
- 输出 `single_frame` 和 `many_hot` logits/probability，先用固定 window 和 Rust MLX 对齐。

## Phase 2: Video Path

- 第一版可以 shell out 到 `ffmpeg`，因为当前 Python/Rust 对齐依赖 FFmpeg 默认 vsync/scaler 语义。
- 如果改用 libav 直连，必须先证明 frame count 和 `checksum_fnv1a64` 与 Python 参考一致。
- 解码、resize、windowing 保持 effectful edge；`predictions_to_scenes` 继续作为纯后处理逻辑。

## Phase 3: End-to-End Gate

- Zig 输出完整 `segment` JSON。
- 跑 `assets/333.mp4`，至少 5 runs。
- 通过 `scripts/evaluate_runtime_candidate.py --require-python-fps`。
- 报告 correctness、FPS、decode/windowing/inference/postprocess/end-to-end 分阶段耗时。

## Phase 4: Linux Backend Spike

- Linux 不继承 macOS MLX 结论。
- 优先评估 ORT C API 和 ONNX artifact，而不是假设 safetensors 可以成为跨后端统一格式。
- CUDA 路线只在 Linux 机器上用同一套 candidate gate 证明；不要从 macOS FPS 外推。

## Decision Gate

Zig 替换 Rust 必须同时满足：

- frame count、decode checksum、scene ranges 完全一致。
- `single_frame` 和 `many_hot` 默认 `max_abs_diff <= 5e-4`。
- macOS MLX FPS 达到或超过当前 Rust MLX 5-run baseline。
- 构建和 FFI 代码没有比 Rust 版本引入更多隐藏平台耦合。
- 公共 API/CLI 的数据流更清晰，而不是只把复杂度从 Rust wrapper 换到 Zig binding 层。

如果 Zig 第一轮只在代码形态上更干净但性能低于 Rust MLX，可以继续优化；如果构建、权重加载或 MLX-C 生命周期明显更复杂，则保留 Rust baseline。

## Risks

- MLX 与 ORT 的 artifact 可能天然分裂：macOS 走 safetensors，Linux/ORT 更适合 ONNX。
- FFmpeg 默认 vsync 和 scaler 语义容易漂移；不能只看 scene 一致，必须保留 checksum gate。
- C header、rpath、framework 查找和 CI 环境可能吞掉 Zig 的表面简洁。
- MLX 路径目前还没有 Candle `--profile` 等价的内部阶段统计；Zig 版不能因此把性能报告降级成单个 total。
