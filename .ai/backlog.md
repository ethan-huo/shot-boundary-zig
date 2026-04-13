# Deferred Work Log

- 2026-04-13: Python/Rust alignment passes on `assets/333.mp4`, but Rust Candle CPU inference is slower than TensorFlow. Optimize the narrow Conv3D/Candle CPU path before using this for latency-sensitive `lens` workflows.
- 2026-04-13: `decode_video_rgb24` shells out to `ffmpeg` to match the official Python reference's default vsync/scaler behavior. If this crate needs library-only embedding without an `ffmpeg` binary, add an explicit frame-duplication policy to the ffmpeg-next path before removing the command dependency.
- 2026-04-13: `segment --window-batch-size 2` improves full-video Rust throughput modestly, but strict probability alignment fails even though scenes still match. Keep the default at `1` until the batched Candle path is numerically reconciled or the acceptance criteria explicitly change.
- 2026-04-13: Reaching TensorFlow-class speed likely needs a fused fixed-shape SDDCNN kernel or an alternative backend. Any candidate must emit the normalized segment JSON and pass `scripts/evaluate_runtime_candidate.py --require-python-fps` before replacing the default Candle path.
