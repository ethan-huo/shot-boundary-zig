# Local Alignment Workflow

日期：2026-04-13

目标是在同一台机器上比较官方 TensorFlow Python 版本和 Rust Candle 版本：

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

## 3. 运行 Rust Candle 版本

```bash
cargo run --release --features cli --bin transnetv2-rs -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
  --runs 5 \
  --format json > target/reports/rust-segment.json
```

## 4. 生成对齐报告

```bash
python3 scripts/compare_segment_outputs.py \
  --python target/reports/python-reference.json \
  --rust target/reports/rust-segment.json \
  --output-json target/reports/alignment.json \
  --output-md target/reports/alignment.md
```

如果这个命令返回非零退出码，不要接 `lens`。先看报告里的 `checksums_equal`、`scenes_equal`、`single_frame.max_abs_diff` 和 `many_hot.max_abs_diff`。

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
