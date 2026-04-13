# Porting Status

日期：2026-04-13

## 当前边界

已经开始 TransNetV2 模型移植，并已把 `segment(video)` 暴露给 CLI 作为本地对齐入口。当前边界是：可以跑，但只有 Python/Rust 对齐报告通过后，才能把它当成可接入 `lens` 的稳定结果。

- `TransNetV2::load_from_safetensors`
- `TransNetV2::forward`
- 受限 3D 空间卷积：`(1,3,3)` 通过 Conv2D reshape 实现
- 受限 3D 时间卷积：`(3,1,1)` 通过 Conv1D reshape 实现
- 受限 3D 空间池化：`(1,2,2)` 通过 2D pooling reshape 实现
- FrameSimilarity
- ColorHistograms

## 权重导出

用 `uv` 执行导出脚本，不要用 `pip`：

```bash
uv run --with torch --with tensorflow==2.16.2 --with safetensors \
  scripts/export_safetensors.py \
  --upstream /Users/dio/Projects/ethan-huo/.scratch/transnetv2-rs-research/TransNetV2 \
  --output target/models/transnetv2.safetensors
```

如果已经有官方 PyTorch converter 生成的 `.pth`：

```bash
uv run --with torch --with safetensors \
  scripts/export_safetensors.py \
  --upstream /Users/dio/Projects/ethan-huo/.scratch/transnetv2-rs-research/TransNetV2 \
  --pytorch-weights transnetv2-pytorch-weights.pth \
  --output target/models/transnetv2.safetensors
```

导出脚本会验证 state dict 的 key 和 shape 是否匹配上游 PyTorch 模型，并生成 manifest。

## 本地对齐

完整命令见 `docs/ALIGNMENT.md`。对齐必须同时看：

1. Python/Rust decode checksum 是否一致。
2. Python/Rust scenes 是否完全一致。
3. `single_frame` 和 `many_hot` prediction 的 `max_abs_diff` 是否小于阈值。
4. 两边 release/CPU 性能数据是否在同一报告中。

不要只看 scene 结果。scene 后处理可能掩盖 logits 或 probability 的小范围错位。这里最容易出错的是权重命名、卷积 reshape 顺序、FrameSimilarity 的局部窗口 gather、ColorHistograms 的 512-bin 归一化，以及 FFmpeg scaler 参数。
