# Local Alignment Workflow

日期：2026-04-13

目标是在同一台机器上比较官方 TensorFlow Python 版本和 Rust MLX 版本：

- decode checksum 必须一致。
- scene ranges 必须完全一致。
- `single_frame` 和 `many_hot` prediction 默认 `max_abs_diff <= 5e-4`。
- 报告必须包含两边的分阶段性能数据。

## 1. 导出权重

```bash
uv run --with torch --with tensorflow==2.16.2 --with safetensors \
  scripts/export_safetensors.py \
  --upstream /Users/dio/Projects/ethan-huo/.scratch/transnetv2-rs-research/TransNetV2 \
  --output target/models/transnetv2.safetensors
```

## 2. 运行 Python TensorFlow 参考

```bash
uv run --with tensorflow==2.16.2 --with numpy \
  scripts/run_python_reference.py \
  --upstream /Users/dio/Projects/ethan-huo/.scratch/transnetv2-rs-research/TransNetV2 \
  --video assets/333.mp4 \
  --output target/reports/python-reference.json \
  --runs 5
```

## 3. 运行 Rust MLX 版本

```bash
cargo run --manifest-path rust/Cargo.toml --release --features cli-mlx --bin transnetv2-rs -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
  --backend mlx \
  --runs 5 \
  --format json > target/reports/rust-mlx-segment-batch2-runs5.json
```

## 4. 生成对齐报告

MLX 作为候选后端必须同时超过 Candle baseline 和官方 TensorFlow Python FPS：

如果还没有 Candle baseline，先生成一次：

```bash
cargo run --manifest-path rust/Cargo.toml --release --features cli --bin transnetv2-rs -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
  --backend candle \
  --runs 5 \
  --format json > target/reports/rust-segment.json
```

```bash
uv run python scripts/evaluate_runtime_candidate.py \
  --python target/reports/python-reference.json \
  --baseline-rust target/reports/rust-segment.json \
  --candidate target/reports/rust-mlx-segment-batch2-runs5.json \
  --candidate-name rust-mlx-batch2-runs5 \
  --require-python-fps \
  --output-json target/reports/runtime-candidate-mlx-batch2-runs5.json \
  --output-md target/reports/runtime-candidate-mlx-batch2-runs5.md
```

如果这个命令返回非零退出码，不要接 `lens`。先看报告里的 `checksums_equal`、`scenes_equal`、`single_frame.max_abs_diff` 和 `many_hot.max_abs_diff`。

## 5. 评估候选 Runtime

如果新增 Linux/CUDA runtime 或自定义 kernel 路径，先让候选路径输出同样的标准 JSON，再运行同一个 gate：

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

这个 gate 同时检查 Python/Rust 数值一致性和候选实现是否达到官方 TensorFlow Python 的 mean FPS。失败时不要替换默认 backend。

## 输出约定

Python 和 Rust 输出都使用同一组核心字段：

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
- `summary`

性能数据里的 `total_ms` 包含权重加载、decode、windowing、inference 和 postprocess。`decode-smoke` 仍然只是视频边缘层压测，不能替代本报告。
