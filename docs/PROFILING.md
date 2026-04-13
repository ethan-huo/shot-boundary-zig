# Profiling Workflow

日期：2026-04-13

第一轮优化先用内建 profile 定位瓶颈，不直接猜 runtime 或重写 kernel。

## 命令

```bash
cargo run --release --features cli --bin transnetv2-rs -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
  --runs 1 \
  --profile \
  --format json > target/reports/rust-profile.json
```

短冒烟可以加 `--max-frames 60`。

## 字段

`runs[].model_profile` 会汇总所有 100-frame windows 的模型内部耗时：

- `input_cast_ms`
- `sddcnn_ms`
- `block_ms`
- `flatten_ms`
- `frame_similarity_ms`
- `color_histograms_ms`
- `dense_ms`
- `heads_ms`
- `total_ms`

这些数据只用于定位热点。它会给每个 window 增加 `Instant` 计时开销，所以不要和无 `--profile` 的性能基线混用。

## 当前判断

已知端到端报告显示瓶颈在 Rust Candle inference。下一批优化优先处理 window 批处理和中间张量重排；如果 profile 显示 `sddcnn_ms` 仍占绝大多数，再考虑专用 kernel 或替代 runtime。
