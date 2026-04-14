# AutoShot: TransNetV2 的 NAS 后继者

> 论文: *AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection* (CVPR 2023)
> 作者: Wentao Zhu et al. (Kuaishou + UT Austin)
> 代码: https://github.com/wentaozhu/AutoShot

## 核心结论

AutoShot 不是全新架构，而是在 TransNetV2 的构件上做 NAS 搜索得到的变体。
最终搜出的最优架构仍然是**纯 (2+1)D 分解卷积 CNN**——Transformer 被搜索淘汰 (`n_layer=0`)。

- FLOPs: 37 GMACs (TransNetV2: 41 GMACs)，少 ~10%，推理更快
- F1: 在 ClipShots 上 +1.1%，在短视频数据集 SHOT 上 +4.2%
- PapersWithCode SBD 排行榜 Top-1 (截至 2025)

## 架构对比

| 维度 | TransNetV2 | AutoShot@F1 |
|---|---|---|
| 核心卷积 | SeparableConv3d (conv2d+conv1d) | 完全相同 |
| DDCNN blocks | 3 层 × 2 blocks, n_dilation=4 | 6 blocks, n_dilation=4~5 |
| 空间卷积 | 每个膨胀分支独立 2D conv | 层 1-3 共享一个 2D conv (DDCNNV2A) |
| Transformer | 无 | SuperNet 设计了, NAS 搜掉了 |
| FrameSimilarity | 有 | 完全保留 |
| ColorHistograms | 有 | 完全保留 |
| 分类头 (fc1 → cls) | 有 | 完全保留 |

## NAS 搜索空间与最终架构

SuperNet 包含 4 种 block 变体:

- **DDCNNV2**: 原始 TransNetV2 风格, 每个膨胀分支有独立的 2D 空间卷积
- **DDCNNV2A**: 所有膨胀分支共享一个 2D 空间卷积, 时序 1D conv 各自独立
- **DDCNNV2B**: (论文中另一变体)
- **DDCNNV2C**: (论文中另一变体)

### AutoShot@F1 (最优 F1)

```
Block 0: DDCNNV2   (n_c=4F,  n_d=4)   ← 原始 TransNetV2 风格
Block 1: DDCNNV2A  (n_c=4F,  n_d=5)   ← 共享 2D conv
Block 2: DDCNNV2A  (n_c=4F,  n_d=5)
Block 3: DDCNNV2A  (n_c=4F,  n_d=5)
Block 4: DDCNNV2   (n_c=12F, n_d=5)
Block 5: DDCNNV2   (n_c=8F,  n_d=5)
Block 6: Attention1D, n_layer=0        ← NAS 选择不用 Transformer
```

### AutoShot@Precision (精度优先)

```
Block 0: DDCNNV2  (n_c=12F, n_d=4)
Block 1: DDCNNV2  (n_c=8F,  n_d=4)
Block 2: DDCNNV2B (n_d=4)
Block 3: DDCNNV2C (n_d=4)
Block 4: DDCNNV2B (n_d=5)
Block 5: DDCNNV2B (n_d=4)
Block 6: n_layer=0
```

## 从现有 TransNetV2 实现迁移所需改动

### 可完全复用 (零改动)

- SeparableConv3d (conv2d + conv1d 分解)
- FrameSimilarity 模块
- ColorHistograms 模块
- 分类头 (fc1 → cls_layer1/cls_layer2)
- 窗口滑动 / 场景后处理 / ffmpeg 解码
- correctness gate 框架

### 需要新增

1. **DDCNNV2A block 变体**: 多个膨胀分支共享一个 2D 空间卷积, 时序 1D conv 各自独立。约 30 行代码。
2. **n_dilation=5 支持**: 现有硬编码 `[1,2,4,8]`, 加一个 `dilation=16` 分支。
3. **可配置层拓扑**: 从固定 `3层×2blocks` 改为 6 个独立配置的 block, 每个指定类型和参数。
4. **权重导出脚本**: `export_autoshot_safetensors.py`, 从 PyTorch checkpoint 导出, 命名规则对齐现有 safetensors 格式。

### 风险点

- 预训练权重: GitHub repo 里的 `supernet_best_f1.pickle` 不是模型权重, 而是 200 个测试视频的预测结果 dict。真正权重在 README 里叫 `ckpt_0_200_0.pth`, 需要从外部网盘下载后再转换。
- DDCNNV2B/C 的具体实现细节需要从 `linear.py` / `utils.py` 确认。
- 架构配置编码在文件名中 (如 `supernet_flattransf_3_8_8_8_13_12_0_1...`), 需对照推理代码解码各数字含义。

## ONNX backend 可行性评估

结论: **可行, 且建议优先走 ONNX 导出路径**, 不建议先在 Zig/MLX 侧手写 AutoShot 算子。

当前 Linux ONNX backend 的 runtime contract 已经足够小:

- 输入: `frames`, `uint8 [batch, 100, 27, 48, 3]`
- 输出: `single_frame`, `float32 [batch, 50]`
- 输出: `many_hot`, `float32 [batch, 50]`

AutoShot 上游 PyTorch 模型原生使用 `N,C,T,H,W`, 但可以在导出 wrapper 里做 `frames.permute(0, 4, 1, 2, 3).float()`。这样导出的 ONNX 模型可以保持现有 Zig ONNX ABI, Zig 侧不需要理解 AutoShot 内部拓扑。

### 已验证

浅克隆 `https://github.com/wentaozhu/AutoShot` 后检查到:

- HEAD: `77c82ff826a9301bb173d9be786297a49d73d081`
- 最优 F1 架构在 `supernet_flattransf_3_8_8_8_13_12_0_16_60.py`
- `TransNetV2Supernet()` 参数量约 `14,299,202`
- 原始模型 forward 输入为 `float/uint8-like [B,3,100,27,48]`, 输出为两个 logits `[B,100,1]`
- 包一层 ONNX wrapper 后, 可得到两个 sigmoid 后的中心窗口概率 `[B,50]`
- 用随机权重导出 opset 18 ONNX 后, ONNX Runtime CPU 可跑通, 与 PyTorch 最大绝对误差约 `1.8e-7`

随机权重验证不能替代真实 checkpoint correctness gate, 但它说明模型结构和 ONNX Runtime CPU 算子覆盖没有根本障碍。

### 关键发现: 直接导出有 dynamic batch bug

上游 `FrameSimilarity` / `ColorHistograms` 复用的 `gather_nd` 实现会执行:

```python
indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
output = params[indices]
```

这会让 TorchScript ONNX export 把索引 trace 成常量。用 batch=1 dummy 直接导出的模型, batch=1 可以跑, 但当前 CLI 默认 `--window-batch-size 2` 会在 ORT 中失败, 典型错误是 `color_hist_layer` reshape 试图把 batch=1 的 gather 结果 reshape 成 batch=2。

可行修复是在导出脚本中不要使用这个 `gather_nd`, 改为 ONNX-friendly 的 `torch.gather`:

```python
similarities_padded = F.pad(similarities, [50, 50, 0, 0, 0, 0])
time_indices = torch.arange(time_window, device=similarities.device).reshape(1, time_window, 1)
lookup_indices = torch.arange(lookup_window, device=similarities.device).reshape(1, 1, lookup_window) + time_indices
lookup_indices = lookup_indices.expand(batch_size, time_window, lookup_window)
similarities = torch.gather(similarities_padded, 2, lookup_indices)
```

用这个 shim 重新导出的随机权重 ONNX, batch=1/2/3 都能在 ONNX Runtime CPU 跑通, 和 PyTorch 的最大绝对误差约 `1.8e-7`。这应作为 `export_autoshot_onnx.py` 的硬性实现要求。

### 影响范围

**Zig ONNX runtime: 低影响。**

如果 `export_autoshot_onnx.py` 保持现有 ABI, `src/onnx_model.zig` 的输入名、输出名、dtype、rank、预分配输出 buffer 都可以复用。最多需要做命名层面的清理, 例如把 `TransNetV2` 类型名泛化成 `RuntimeModel`, 但这不是功能必需项。

**decode/windowing/postprocess: 低影响。**

AutoShot 仍使用 100 帧窗口、25 帧上下文、中心 50 帧输出、48x27 RGB 输入。`segment_core.windowSourceIndices`, `runtime_segment.buildWindowBatch`, ffmpeg decode 都可以复用。

**CLI default threshold: 已切换。**

AutoShot 上游 SHOT F1 最优阈值记录为 `0.296`。在决定以精度收益推进完整迁移后, CLI 全局默认阈值已切到 `0.296`。如果使用历史 TransNetV2 模型做对比, 需要显式传 `--threshold 0.5` 或该模型对应的阈值。

**模型资产/导出脚本: 高影响, 是主工作量。**

已新增 `scripts/export_autoshot_onnx.py`, 职责包括:

- 加载上游 `TransNetV2Supernet`
- 加载 `ckpt_0_200_0.pth` 中的 `pretrained_dict["net"]`
- 只接受 strict 或近 strict 的 state dict 校验, 避免静默漏权重
- 包装 NHWC uint8 输入到 NCTHW float 模型输入
- 替换 `gather_nd` 为 dynamic-batch-safe 的 `torch.gather` shim
- 输出和现有 ONNX backend 一样的 `frames` / `single_frame` / `many_hot`
- 生成 manifest, 写明模型名、opset、输入输出、推荐阈值 `0.296`

**acceptance gate: 中等影响。**

现有 `scripts/evaluate_runtime_candidate.py` 可以复用, 但需要新增 AutoShot Python reference 输出脚本, 或在导出脚本旁补一个 `run_autoshot_reference.py`。真实 checkpoint 到手后, gate 应至少覆盖:

- Python AutoShot reference vs ONNX Runtime 输出概率差异
- batch=1 和 batch=2/3 的 dynamic batch correctness
- `assets/333.mp4` smoke
- 如果拿得到 SHOT 测试数据, 再跑官方阈值 `0.296` 的 F1 sanity check

### 推荐下一步

1. 用 `scripts/export_autoshot_onnx.py` 导出并固定 release 用 ONNX artifact。
2. 用 `assets/333.mp4 --window-batch-size 2` 跑 Zig ONNX segment smoke。
3. 后续如要追速度, 再单独评估 ORT graph optimization / CUDA, 不阻塞 AutoShot 精度迁移。
