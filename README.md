# shot-boundary-zig

这个仓库现在作为 TransNetV2 本地实现和验收工作区，不再默认等同于 Rust crate。

## Layout

- `src/`: Zig CLI 和 macOS MLX-C backend。
- `rust/`: 历史 Rust 实现；确认外部引用后删除。
- `assets/`: Rust/Zig 后端共享的本地验收资产。
- `scripts/`: 模型转换/导出、Python TF 参考输出、runtime candidate gate。
- `docs/`: 当前 runtime 计划和后续 AutoShot 记录。
- `references/`: 构建系统和依赖管理参考。

## Current Decision

macOS 主线现在是 Zig + MLX-C。Zig 版已经输出完整 `segment` JSON，并通过同一套 Python correctness/performance gate；迁移期对 Rust MLX batch2/runs5 基线也通过性能 gate。

Rust 版不再是默认推进方向。删除 `rust/` 前只需要确认没有外部脚本还依赖 Rust-only 子命令或报告文件名。

不要再按 `../lens/PLAN.md` 里的旧结论推进。当前方向见 `docs/PLAN.md`；构建细节见 `references/build.md`。

## Zig Commands

从仓库根目录运行：

```bash
zig build -Doptimize=ReleaseFast run -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
  --backend mlx \
  --window-batch-size 2 \
  --runs 5 \
  --format json > target/reports/zig-mlx-segment-runs5.json
```

迁移期 gate 脚本仍要求传入 `--baseline-rust`。Linux ONNX 阶段应先把这个参数泛化，再让 benchmark 文档脱离 Rust 报告文件名。

```bash
uv run python scripts/evaluate_runtime_candidate.py \
  --python target/reports/python-reference.json \
  --baseline-rust target/reports/rust-mlx-segment-batch2-runs5.json \
  --candidate target/reports/zig-mlx-segment-runs5.json \
  --candidate-name zig-mlx-runs5 \
  --require-python-fps \
  --output-json target/reports/runtime-candidate-zig-mlx-runs5.json \
  --output-md target/reports/runtime-candidate-zig-mlx-runs5.md
```
