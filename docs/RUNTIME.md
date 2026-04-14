# Runtime And Kernel Evaluation

日期：2026-04-13

目标不是凭感觉绑定某个 runtime，而是把替换默认路径的准入标准写死。任何 MLX、CoreML、ONNX Runtime、Accelerate、自定义 fused SDDCNN kernel，只有同时满足本页的 correctness 和 performance gate，才能替换当前默认路径。

## Gate

候选实现必须输出和 `segment` 一样的标准 JSON：

- `runs[].frame_count`
- `runs[].checksum_fnv1a64`
- `runs[].predictions.single_frame`
- `runs[].predictions.many_hot`
- `runs[].scenes`
- `summary.mean_frames_per_second`
- `summary.mean_inference_ms`

准入命令：

```bash
uv run python scripts/evaluate_runtime_candidate.py \
  --python target/reports/python-reference.json \
  --baseline-rust target/reports/rust-segment.json \
  --candidate target/reports/candidate-runtime.json \
  --candidate-name candidate-runtime \
  --require-python-fps \
  --output-json target/reports/runtime-candidate.json \
  --output-md target/reports/runtime-candidate.md
```

默认 correctness gate 使用 `max_abs_diff <= 5e-4`，并要求 frame count、decode checksum、scene ranges 完全一致。`--require-python-fps` 会要求候选实现的 mean FPS 达到或超过官方 TensorFlow Python 参考；没有这个参数时，脚本只要求候选实现不低于当前 Rust baseline，适合早期实验。

## Candle Finding

当前 profile 显示热点几乎全在 SDDCNN，也就是受限 3D 卷积的 Candle CPU 路径。`--window-batch-size 2` 可以略微提升完整视频吞吐，但完整视频 probability 差异超过严格阈值，所以不能替换默认路径。

这意味着不应该继续在 decode 或 postprocess 上花时间。macOS 方向已经转为 MLX；Linux/CUDA 方向后续单独评估。

默认库 API 仍保留 Candle 实现，但 `cli-mlx` 下的 `segment --backend auto` 已经优先使用 MLX。

## Backend Strategy Decision (2026-04-14)

Candle 作为 macOS 主推理后端的局限已经明确：GPU 支持不成熟，CNN op 覆盖薄。macOS 主线切到 MLX；Candle 保留为 fallback 和数值定位路径。

### MLX-C vs mlx-rs

选择 `mlx-rs 0.25.3`，不是直接使用 MLX-C：

- `mlx-rs` 是 MLX-C 的 Rust 封装，已经承担 CMake、MLX-C 子模块、Array 生命周期和错误处理。
- 它覆盖本模型需要的 safetensors 加载、Conv1D/Conv2D/Conv3D、matmul、mean、pad、take_along_axis、reshape/transpose。
- 直接接 MLX-C 会把 unsafe FFI、生命周期和构建细节搬进本 crate；当前没有收益。
- MLX-C 只作为 fallback：如果 `mlx-rs` 缺某个必要 primitive 或性能调度入口，再局部下探。

这个判断只适用于 Rust crate 内的后端选择。是否用 Zig 直接接 MLX-C 替换 Rust baseline，按 `docs/ZIG_SPIKE_PLAN.md` 重新评估。

注意一个容易犯的错：通用 MLX Conv3D 路径 correctness 能过，但性能没有超过 TF。最终实现用 TransNetV2 的窄形状拆分为 `conv2d (1,3,3)` + `conv1d (3,1,1)`，这才是当前 macOS 性能主路径。

### macOS Gate 结果

命令：

```bash
cargo run --manifest-path rust/Cargo.toml --release --features cli-mlx --bin transnetv2-rs -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
  --backend mlx \
  --runs 5 \
  --format json > target/reports/rust-mlx-segment-batch2-runs5.json

uv run python scripts/evaluate_runtime_candidate.py \
  --python target/reports/python-reference.json \
  --baseline-rust target/reports/rust-segment.json \
  --candidate target/reports/rust-mlx-segment-batch2-runs5.json \
  --candidate-name rust-mlx-batch2-runs5 \
  --require-python-fps \
  --output-json target/reports/runtime-candidate-mlx-batch2-runs5.json \
  --output-md target/reports/runtime-candidate-mlx-batch2-runs5.md
```

结果：

```json
{
  "passed": true,
  "single_frame_max_abs_diff": 9.547364807072078e-7,
  "many_hot_max_abs_diff": 1.0672409057610466e-6,
  "scenes_equal": true,
  "candidate_mean_fps": 338.3118280612795,
  "candidate_vs_baseline_speedup": 9.257788412306496,
  "candidate_vs_python_speedup": 2.682271655754321,
  "candidate_mean_inference_ms": 7847.7096678
}
```

### 平台分工

| 平台 | 后端 | 状态 |
|------|------|------|
| macOS | MLX via `mlx-rs` | 已实现并通过 TF correctness/performance gate |
| Linux CPU/CUDA | 待评估 | 后续在 Linux 环境重新测 ORT/CUDA 或其他成熟 runtime |

Linux 路线不要从 macOS 结论外推。CUDA 上是否用 ORT、TensorRT、libtorch 或其他 runtime，必须用同一份 gate 在 Linux 机器上重新证明。

Candle 实现保留为 fallback feature flag（`candle` feature），直到 Linux/CUDA 后端也通过 gate 后再评估是否移除。
