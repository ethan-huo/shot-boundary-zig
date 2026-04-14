# transnetv2

这个仓库现在作为 TransNetV2 本地实现和验收工作区，不再默认等同于 Rust crate。

## Layout

- `rust/`: 已验证的 Rust 实现，包含 Candle fallback 和 macOS MLX backend。
- `assets/`: Rust/Zig 后端共享的本地验收资产。
- `scripts/`: 共享权重导出、Python TF 参考输出、runtime candidate gate。
- `docs/`: 共享决策记录、性能记录和 Zig spike 计划。

## Current Decision

macOS 主线现在是 MLX via `mlx-rs`。Rust 版保留为已验证 baseline 和当前可用实现；Zig 是否替换它，需要用同一份输出 JSON 和同一套 gate 证明。

不要再按 `../lens/PLAN.md` 里的旧 Rust/Candle 结论推进。当前方向见 `docs/RUNTIME.md` 和 `docs/ZIG_SPIKE_PLAN.md`。

## Rust Commands

从仓库根目录运行：

```bash
cargo run --manifest-path rust/Cargo.toml --release --features cli-mlx --bin transnetv2-rs -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
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
