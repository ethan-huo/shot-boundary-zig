# Deferred Work Log

- 2026-04-13: Python/Rust alignment passes on `assets/333.mp4`, but Rust Candle CPU inference is slower than TensorFlow. Optimize the narrow Conv3D/Candle CPU path before using this for latency-sensitive `lens` workflows.
- 2026-04-13: `decode_video_rgb24` shells out to `ffmpeg` to match the official Python reference's default vsync/scaler behavior. If this crate needs library-only embedding without an `ffmpeg` binary, add an explicit frame-duplication policy to the ffmpeg-next path before removing the command dependency.
- 2026-04-13: `segment --window-batch-size 2` improves full-video Rust throughput modestly, but strict probability alignment fails even though scenes still match. Keep the default at `1` until the batched Candle path is numerically reconciled or the acceptance criteria explicitly change.
- 2026-04-13: Reaching TensorFlow-class speed likely needs a fused fixed-shape SDDCNN kernel or an alternative backend. Any candidate must emit the normalized segment JSON and pass `scripts/evaluate_runtime_candidate.py --require-python-fps` before replacing the default Candle path.
- 2026-04-14: macOS MLX via `mlx-rs` now passes the TensorFlow correctness/performance gate, but MLX does not yet expose Candle-style internal model profile fields in the CLI. Keep `--profile` Candle-only until MLX stage timings are modeled separately.
- 2026-04-14: Linux CPU/CUDA remains unevaluated. Do not assume the macOS MLX result transfers; rerun the same runtime candidate gate on Linux before choosing ORT/CUDA/TensorRT/libtorch or another backend.
- 2026-04-14: Zig spike is pending. Keep the Rust implementation as the verified baseline until Zig emits the same segment JSON and passes the shared Python/Rust candidate gate.
