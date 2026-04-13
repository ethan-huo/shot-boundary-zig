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
cargo run --features cli --bin transnetv2-rs -- inspect assets/333.mp4
cargo run --release --features cli --bin transnetv2-rs -- decode-smoke assets/333.mp4 --runs 3
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

结论很直接：Rust 输出已经和官方 TensorFlow Python 参考对齐，但当前 Candle CPU 推理明显慢于 TensorFlow。后续优化应优先看受限 Conv3D 的 reshape/Conv1D/Conv2D 路径和 Candle CPU kernel，而不是 decode、windowing 或 scene postprocess。
