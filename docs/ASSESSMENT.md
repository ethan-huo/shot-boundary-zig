# TransNetV2 Rust Port Assessment

日期：2026-04-13

## 判断

计划可行，且应该从 `lens` 拆成独立 Rust crate。原因很简单：TransNetV2 是可复用的 shot-boundary detection 引擎，不是 `lens` 的应用编排逻辑。把它放在独立目录里，`lens` 以后只需要消费它的库 API 或 CLI JSON 输出。

## 已核对事实

- 上游仓库：`soCzech/TransNetV2`
  - license 是 MIT。
  - `inference/` 提供 TensorFlow SavedModel 权重和 Python 推理。
  - `inference-pytorch/` 提供 PyTorch inference 和 `convert_weights.py`。
  - 模型输入是 `[batch, frames, 27, 48, 3]` 的 RGB `uint8`，窗口大小 100 帧，中间 50 帧作为有效输出。
- Candle 当前版本：
  - crates.io 上 `candle-core` / `candle-nn` 最新为 `0.10.2`。
  - `candle-nn` 有 Conv2D、BatchNorm、Linear。
  - `candle-core` 有 2D pooling、cat、stack、mean 等张量操作。
  - 没有通用 Conv3D/Pool3D API。计划里“Conv3D/BN/Dense 均有现成实现”的说法不准确。
- `ffmpeg-next` 当前 crates.io 最新为 `8.1.0`，应作为可选 `video-io` 特性，而不是核心依赖。
- `safetensors` 当前 crates.io 最新为 `0.7.0`，适合承载导出的模型权重。

## 架构建议

核心层：

- `model`：TransNetV2 前向传播。
- `ops`：只实现本模型需要的窄算子，不做通用 ML 框架。
- `weights`：从 safetensors 加载权重，固定命名和形状校验。
- `scenes`：预测值转场景区间，保持纯函数。

边缘层：

- `video-io` feature：用 `ffmpeg-next` 解码为 RGB 48x27 帧。
- `cli` feature 或单独 bin：读取视频/帧目录，输出 JSON/CSV。
- `lens` 集成：只调用 CLI 或库 API，不拥有模型实现。

已落地的 CLI 先覆盖冒烟测试：`inspect` 读取视频元数据，`decode-smoke` 压测解码+resize，`scenes` 把 prediction 文件转换为 scene ranges。它还不是 `segment` 命令，不应该把当前压测结果解读为 TransNetV2 推理速度。

## 主要风险

1. Candle 没有通用 Conv3D/Pool3D，不能照计划直接调用 `Conv3d`。应实现受限形状：
   - `(1,3,3)` 空间卷积：把 `[B,C,T,H,W]` reshape 为 `[B*T,C,H,W]` 后用 Conv2D。
   - `(3,1,1)` 时间卷积：按空间点 reshape 成 1D/2D 等价计算，或写小型专用循环/张量实现。
   - `(1,2,2)` pooling：同样 reshape 后用 2D pooling。
2. FrameSimilarity 和 ColorHistograms 不是普通 CNN 层，必须单独对齐：
   - FrameSimilarity 包含 mean、linear projection、L2 normalize、batch matmul、局部窗口 gather。
   - ColorHistograms 包含 512-bin RGB histogram、L2 normalize、batch matmul、局部窗口 gather。
3. 权重转换仍需要 Python/PyTorch 作为一次性工具链。运行时可以是纯 Rust，但转换阶段不值得强行纯 Rust。
4. ffmpeg resize 和像素格式细节会影响最终预测。需要固定 `rgb24`、`48x27`，并用官方 Python 输出做端到端校验。

## 建议里程碑

1. 权重导出：TF SavedModel -> PyTorch state dict -> safetensors，输出权重 manifest。
2. 纯张量单元测试：空间卷积、时间卷积、池化、BatchNorm、Dense 与 PyTorch 对齐。
3. 模型 logits 对齐：固定随机 `uint8` 输入，比较 single/many-hot logits。
4. 视频端到端对齐：同一视频输出 predictions/scenes 与官方 Python 实现差异在可解释阈值内。
5. 再接 `lens`：只暴露 `segment(video)` 或 JSON CLI，不把模型实现搬进 `lens`。
