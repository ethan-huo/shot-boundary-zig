# Runtime And Kernel Evaluation

日期：2026-04-13

目标不是提前绑定某个 runtime，而是把替换默认路径的准入标准写死。任何 CoreML、ONNX Runtime、Accelerate、自定义 fused SDDCNN kernel，只有同时满足本页的 correctness 和 performance gate，才能替换当前 Candle 默认路径。

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
python3 scripts/evaluate_runtime_candidate.py \
  --python target/reports/python-reference.json \
  --baseline-rust target/reports/rust-segment.json \
  --candidate target/reports/candidate-runtime.json \
  --candidate-name candidate-runtime \
  --require-python-fps \
  --output-json target/reports/runtime-candidate.json \
  --output-md target/reports/runtime-candidate.md
```

默认 correctness gate 使用 `max_abs_diff <= 5e-4`，并要求 frame count、decode checksum、scene ranges 完全一致。`--require-python-fps` 会要求候选实现的 mean FPS 达到或超过官方 TensorFlow Python 参考；没有这个参数时，脚本只要求候选实现不低于当前 Rust baseline，适合早期实验。

## Current Finding

当前 profile 显示热点几乎全在 SDDCNN，也就是受限 3D 卷积的 Candle CPU 路径。`--window-batch-size 2` 可以略微提升完整视频吞吐，但完整视频 probability 差异超过严格阈值，所以不能替换默认路径。

这意味着下一步不应该继续在 decode 或 postprocess 上花时间。要接近或超过 TensorFlow，候选方向只有两类：

1. 在当前 Candle 路径内做更窄的算子优化，例如为固定形状 SDDCNN 写 fused kernel，减少 reshape/transpose/Conv2D/Conv1D 的中间张量和调度开销。
2. 增加独立 backend，把同一权重和同一预处理输入交给成熟 runtime，例如 CoreML、ONNX Runtime 或 Accelerate，然后用本页 gate 决定能不能替换默认路径。

默认库 API 仍保留 Candle 实现，直到某个候选实现同时通过 correctness 和 performance gate。
