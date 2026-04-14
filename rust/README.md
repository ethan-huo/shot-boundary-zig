# transnetv2-rs

`transnetv2-rs` 是 TransNetV2 的独立 Rust 落地点。它不属于 `lens` 应用层；`lens` 之后应该把它当成库或 CLI 工具来消费。

## 边界

- 核心库负责：模型输入规格、权重加载、前向推理、预测值到场景区间的纯后处理。
- 可选边缘层负责：视频解码、resize、CLI、与 `lens` 的集成适配。
- `lens` 负责：agent-facing 工作流、缓存策略、宫格图、OCR/VLM/音频等应用编排。

这个边界能避免把 TransNetV2 的权重格式、张量算子和视频解码细节锁进 `lens` 的产品目录。

## 初评结论

可行，但 `lens/PLAN.md` 里的原计划偏乐观：

- 官方仓库不只有 TensorFlow inference；还有 PyTorch inference 和 TF -> PyTorch 权重转换脚本。第一阶段应从 PyTorch 对齐和 safetensors 导出开始。
- Candle 0.10.2 有 Conv2D、BatchNorm、Linear、cat/stack/mean 等基础能力，但没有通用 Conv3D/AvgPool3D/MaxPool3D API。
- TransNetV2 的 3D 操作形状受限：`(1,3,3)` 空间卷积、`(3,1,1)` 时间卷积、`(1,2,2)` 空间池化。可以用 Conv2D + reshape/transpose 或专用窄算子实现，不需要写完整通用 Conv3D。
- `ffmpeg-next` 应是可选特性；模型核心不应该依赖视频解码。

## 当前状态

已落地：

- crate 骨架和依赖特性开关
- `predictions_to_scenes` 纯后处理函数
- `TransNetV2::load_from_safetensors` / `TransNetV2::forward`
- `MlxTransNetV2::load_from_safetensors` / `MlxTransNetV2::forward`
- `segment_frames` / `segment_video` 窗口化推理入口
- CLI 工具：`inspect`、`decode-smoke`、`scenes`、`segment --backend auto|mlx|candle`
- 可行性评估文档：`../docs/ASSESSMENT.md`
- 当前视频解码压测基线：`../docs/PERF.md`
- 模型移植入口、权重导出脚本和本地对齐脚本：`../docs/PORTING.md`、`../docs/ALIGNMENT.md`
- runtime/backend 替换准入流程：`../docs/RUNTIME.md`

macOS 主线现在使用 MLX；Candle 保留为 fallback 和数值定位路径。只有在 Python/Rust 对齐报告通过后，才应该接入 `lens`：

1. 用官方 `inference-pytorch/convert_weights.py` 路径生成 PyTorch state dict。
2. 导出为 safetensors，并固定权重命名。
3. 用同一个视频对齐官方 TensorFlow Python 输出和 Rust MLX 输出。
4. 报告 scenes 是否一致、prediction 误差和两边性能数据。

## CLI

从仓库根目录运行：

```bash
cargo run --manifest-path rust/Cargo.toml --features cli --bin transnetv2-rs -- inspect assets/333.mp4
cargo run --manifest-path rust/Cargo.toml --release --features cli --bin transnetv2-rs -- decode-smoke assets/333.mp4 --runs 3
cargo run --manifest-path rust/Cargo.toml --features cli --bin transnetv2-rs -- scenes video.predictions.txt --column 0 --threshold 0.5
cargo run --manifest-path rust/Cargo.toml --release --features cli-mlx --bin transnetv2-rs -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
  --runs 5 \
  --format json
cargo run --manifest-path rust/Cargo.toml --release --features cli --bin transnetv2-rs -- segment assets/333.mp4 \
  --weights target/models/transnetv2.safetensors \
  --backend candle \
  --runs 1 \
  --profile \
  --format json > target/reports/rust-profile.json
```

`decode-smoke` 也有 `bench-decode` alias。它只统计视频解码并缩放到模型输入尺寸 `48x27 RGB` 的吞吐，不能代表最终模型推理速度。真正的端到端结论使用 `segment` 和 `scripts/evaluate_runtime_candidate.py`。`--features cli-mlx` 下 `--backend auto` 默认走 MLX，未指定 `--window-batch-size` 时 MLX 默认用 `2`，Candle 默认仍是 `1`。
