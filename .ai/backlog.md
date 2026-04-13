# Deferred Work Log

- 2026-04-13: Python/Rust alignment passes on `assets/333.mp4`, but Rust Candle CPU inference is slower than TensorFlow. Optimize the narrow Conv3D/Candle CPU path before using this for latency-sensitive `lens` workflows.
- 2026-04-13: `decode_video_rgb24` shells out to `ffmpeg` to match the official Python reference's default vsync/scaler behavior. If this crate needs library-only embedding without an `ffmpeg` binary, add an explicit frame-duplication policy to the ffmpeg-next path before removing the command dependency.
