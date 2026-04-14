# Performance Notes

日期：2026-04-13

## Decode-only 压测范围

`decode-smoke` 只压测视频边缘层：

```text
mp4 -> ffmpeg command -> RGB24 -> resize to 48x27 -> FNV-1a checksum
```

这不是 TransNetV2 推理性能。它只能说明预处理吞吐，不能替代 `segment` 的端到端性能结论。

## 测试资产

`assets/333.mp4`

```json
{
  "source_width": 576,
  "source_height": 1024,
  "source_pixel_format": "YUV420P",
  "duration_seconds": 118.04,
  "frame_count_hint": 2951,
  "avg_frame_rate": 24.98306806637318
}
```

注意：官方 Python 参考路径使用 FFmpeg 默认 output vsync，实际 rawvideo 输出是 2953 帧。Rust 的 `decode_video_rgb24` 也使用 `ffmpeg` 命令而不是 libav frame iteration，以匹配官方参考的帧重复、scaler 和 checksum 语义。

## 命令

```bash
cargo run --manifest-path rust/Cargo.toml --features cli --bin transnetv2-rs -- inspect assets/333.mp4
cargo run --manifest-path rust/Cargo.toml --release --features cli --bin transnetv2-rs -- decode-smoke assets/333.mp4 --runs 3
```

## Release 基线

```json
{
  "run_count": 3,
  "decoded_frames_per_run": 2953,
  "decoded_rgb_bytes_per_run": 11481264,
  "checksum_fnv1a64": "0526895e8cdadf99",
  "min_frames_per_second": 3120.0190402939133,
  "max_frames_per_second": 3591.080001464937,
  "mean_frames_per_second": 3326.7579104486977,
  "min_elapsed_ms": 822.315292,
  "max_elapsed_ms": 946.4685830000001,
  "mean_elapsed_ms": 890.689986
}
```

## Segment 对齐和端到端性能

完整报告输出在 `target/reports/alignment.md` 和 `target/reports/alignment.json`。2026-04-13 本地 5-run 结论：

```json
{
  "passed": true,
  "frame_count": 2953,
  "decode_checksum_fnv1a64": "0526895e8cdadf99",
  "scene_count": 57,
  "single_frame_max_abs_diff": 4.3027496332559423e-7,
  "many_hot_max_abs_diff": 3.0642425537241724e-7,
  "python_tensorflow_mean_fps": 126.12884579960185,
  "python_tensorflow_mean_total_ms": 23462.685942836106,
  "python_tensorflow_mean_inference_ms": 19417.971000541,
  "rust_candle_mean_fps": 36.54348241655181,
  "rust_candle_mean_total_ms": 80916.45457500001,
  "rust_candle_mean_inference_ms": 79811.45802340002
}
```

结论很直接：Rust Candle 输出已经和官方 TensorFlow Python 参考对齐，但 Candle CPU 推理明显慢于 TensorFlow。后续 macOS 主线不继续挖 Candle，改走 MLX。

## MLX 后端结论

2026-04-14 本地 5-run MLX gate：

```json
{
  "passed": true,
  "scene_count": 57,
  "single_frame_max_abs_diff": 9.547364807072078e-7,
  "many_hot_max_abs_diff": 1.0672409057610466e-6,
  "python_tensorflow_mean_fps": 126.12884579960185,
  "rust_candle_mean_fps": 36.54348241655181,
  "rust_mlx_mean_fps": 338.3118280612795,
  "rust_mlx_vs_tensorflow_speedup": 2.682271655754321,
  "rust_mlx_vs_candle_speedup": 9.257788412306496,
  "rust_mlx_mean_total_ms": 8730.4467502,
  "rust_mlx_mean_inference_ms": 7847.7096678
}
```

关键实现细节：不要用通用 MLX Conv3D 作为性能主路径。它 correctness 能过，但 5-run 均值约 `115 FPS`，低于 TensorFlow。当前 MLX 实现把受限 3D 卷积拆成 `conv2d (1,3,3)` + `conv1d (3,1,1)`，并把 MLX 默认 window batch 设为 `2`；这个组合通过了 `--require-python-fps` gate。

## Experimental Window Batching

`segment --window-batch-size 2` 在当前机器上能把完整视频单次 Rust FPS 从约 36.5 提到约 41.4，但这个路径不能作为默认验收路径：

```json
{
  "rust_candle_batch2_fps": 41.3992,
  "rust_candle_batch2_total_ms": 71329.842208,
  "rust_candle_batch2_inference_ms": 70492.052581,
  "scenes_equal_to_python_tensorflow": true,
  "single_frame_max_abs_diff": 0.010563135827789338,
  "many_hot_max_abs_diff": 0.007771762146759009
}
```

它说明 batch 维度能减少一部分调度开销，但 Candle CPU 的批处理卷积会改变浮点舍入路径，概率差异超过当前 `5e-4` 严格阈值。默认值保持 `1`，直到我们能证明 batch 路径同时满足 scene 一致和 probability 对齐。
