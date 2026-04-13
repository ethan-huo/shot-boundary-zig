use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor, bail};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Linear, Module,
    ModuleT, VarBuilder, batch_norm, linear,
};

use crate::{MODEL_INPUT_CHANNELS, ModelInputSpec};

const BASE_FILTERS: usize = 16;
const LAYERS: usize = 3;
const BLOCKS_PER_LAYER: usize = 2;
const DENSE_DIM: usize = 1024;
const LOOKUP_WINDOW: usize = 101;
const SIMILARITY_DIM: usize = 128;
const AUX_OUTPUT_DIM: usize = 128;
const BATCH_NORM_EPS: f64 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransNetV2Config {
    pub base_filters: usize,
    pub layers: usize,
    pub blocks_per_layer: usize,
    pub dense_dim: usize,
    pub lookup_window: usize,
    pub use_frame_similarity: bool,
    pub use_color_histograms: bool,
}

impl Default for TransNetV2Config {
    fn default() -> Self {
        Self {
            base_filters: BASE_FILTERS,
            layers: LAYERS,
            blocks_per_layer: BLOCKS_PER_LAYER,
            dense_dim: DENSE_DIM,
            lookup_window: LOOKUP_WINDOW,
            use_frame_similarity: true,
            use_color_histograms: true,
        }
    }
}

#[derive(Debug)]
pub struct TransNetV2 {
    blocks: Vec<StackedDdcnn>,
    frame_similarity: Option<FrameSimilarity>,
    color_histograms: Option<ColorHistograms>,
    fc1: Linear,
    cls_layer1: Linear,
    cls_layer2: Linear,
}

#[derive(Debug)]
pub struct TransNetV2Output {
    pub single_frame_logits: Tensor,
    pub many_hot_logits: Tensor,
}

impl TransNetV2 {
    pub fn load_from_safetensors(path: impl AsRef<Path>, device: &Device) -> Result<TransNetV2> {
        let data = fs::read(path)?;
        let vb = VarBuilder::from_buffered_safetensors(data, DType::F32, device)?;
        Self::load(TransNetV2Config::default(), vb)
    }

    pub fn load(config: TransNetV2Config, vb: VarBuilder<'_>) -> Result<TransNetV2> {
        validate_config(config)?;

        let mut blocks = Vec::with_capacity(config.layers);
        let mut in_filters = MODEL_INPUT_CHANNELS;
        for layer_index in 0..config.layers {
            let filters = config.base_filters * (1 << layer_index);
            blocks.push(StackedDdcnn::load(
                in_filters,
                config.blocks_per_layer,
                filters,
                vb.pp("SDDCNN").pp(layer_index),
            )?);
            in_filters = filters * 4;
        }

        let frame_similarity = config
            .use_frame_similarity
            .then(|| {
                let in_filters = (0..config.layers)
                    .map(|layer_index| config.base_filters * (1 << layer_index) * 4)
                    .sum();
                FrameSimilarity::load(in_filters, config.lookup_window, vb.pp("frame_sim_layer"))
            })
            .transpose()?;

        let color_histograms = config
            .use_color_histograms
            .then(|| ColorHistograms::load(config.lookup_window, vb.pp("color_hist_layer")))
            .transpose()?;

        let cnn_output_dim = config.base_filters * (1 << (config.layers - 1)) * 4 * 3 * 6;
        let mut output_dim = cnn_output_dim;
        if config.use_frame_similarity {
            output_dim += AUX_OUTPUT_DIM;
        }
        if config.use_color_histograms {
            output_dim += AUX_OUTPUT_DIM;
        }

        Ok(Self {
            blocks,
            frame_similarity,
            color_histograms,
            fc1: linear(output_dim, config.dense_dim, vb.pp("fc1"))?,
            cls_layer1: linear(config.dense_dim, 1, vb.pp("cls_layer1"))?,
            cls_layer2: linear(config.dense_dim, 1, vb.pp("cls_layer2"))?,
        })
    }

    pub fn forward(&self, inputs: &Tensor) -> Result<TransNetV2Output> {
        validate_input_window(inputs)?;

        let mut x = (inputs.permute((0, 4, 1, 2, 3))?.to_dtype(DType::F32)? / 255.0)?;
        let mut block_features = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            x = block.forward(&x)?;
            block_features.push(x.clone());
        }

        let (batch, channels, frames, height, width) = x.dims5()?;
        let mut features =
            x.permute((0, 2, 3, 4, 1))?
                .reshape((batch, frames, height * width * channels))?;

        if let Some(frame_similarity) = &self.frame_similarity {
            features = Tensor::cat(&[frame_similarity.forward(&block_features)?, features], 2)?;
        }

        if let Some(color_histograms) = &self.color_histograms {
            features = Tensor::cat(&[color_histograms.forward(inputs)?, features], 2)?;
        }

        let hidden = self.fc1.forward(&features)?.relu()?;

        Ok(TransNetV2Output {
            single_frame_logits: self.cls_layer1.forward(&hidden)?,
            many_hot_logits: self.cls_layer2.forward(&hidden)?,
        })
    }
}

fn validate_config(config: TransNetV2Config) -> Result<()> {
    if config.base_filters == 0 || config.layers == 0 || config.blocks_per_layer == 0 {
        bail!("TransNetV2 dimensions must be non-zero: {config:?}");
    }

    if config.lookup_window == 0 || config.lookup_window % 2 == 0 {
        bail!(
            "TransNetV2 lookup_window must be a positive odd integer, got {}",
            config.lookup_window
        );
    }

    Ok(())
}

fn validate_input_window(inputs: &Tensor) -> Result<()> {
    let spec = ModelInputSpec::default();
    let (_, frames, height, width, channels) = inputs.dims5()?;

    if inputs.dtype() != DType::U8 {
        bail!(
            "TransNetV2 input dtype must be U8, got {:?}",
            inputs.dtype()
        );
    }

    if (frames, height, width, channels)
        != (spec.window_frames, spec.height, spec.width, spec.channels)
    {
        bail!(
            "TransNetV2 input must be [batch, {}, {}, {}, {}], got {:?}",
            spec.window_frames,
            spec.height,
            spec.width,
            spec.channels,
            inputs.dims()
        );
    }

    Ok(())
}

#[derive(Debug)]
struct StackedDdcnn {
    blocks: Vec<DilatedDdcnn>,
}

impl StackedDdcnn {
    fn load(
        in_filters: usize,
        n_blocks: usize,
        filters: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(n_blocks);
        for block_index in 0..n_blocks {
            blocks.push(DilatedDdcnn::load(
                if block_index == 0 {
                    in_filters
                } else {
                    filters * 4
                },
                filters,
                block_index + 1 != n_blocks,
                vb.pp("DDCNN").pp(block_index),
            )?);
        }

        Ok(Self { blocks })
    }

    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let mut x = inputs.clone();
        let mut shortcut = None;

        for block in &self.blocks {
            x = block.forward(&x)?;
            if shortcut.is_none() {
                shortcut = Some(x.clone());
            }
        }

        let x = x.relu()?;
        let shortcut = shortcut.expect("StackedDdcnn always has at least one block");
        let x = (x + shortcut)?;

        avg_pool3d_spatial_2x2(&x)
    }
}

#[derive(Debug)]
struct DilatedDdcnn {
    convs: Vec<SeparableConv3d>,
    bn: BatchNorm,
    activate: bool,
}

impl DilatedDdcnn {
    fn load(in_filters: usize, filters: usize, activate: bool, vb: VarBuilder<'_>) -> Result<Self> {
        let convs = [1, 2, 4, 8]
            .into_iter()
            .map(|dilation| {
                SeparableConv3d::load(
                    in_filters,
                    filters,
                    dilation,
                    vb.pp(format!("Conv3D_{dilation}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let bn = batch_norm(
            filters * 4,
            BatchNormConfig {
                eps: BATCH_NORM_EPS,
                ..BatchNormConfig::default()
            },
            vb.pp("bn"),
        )?;

        Ok(Self {
            convs,
            bn,
            activate,
        })
    }

    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let convs = self
            .convs
            .iter()
            .map(|conv| conv.forward(inputs))
            .collect::<Result<Vec<_>>>()?;
        let refs = convs.iter().collect::<Vec<_>>();
        let x = Tensor::cat(&refs, 1)?;
        let x = self.bn.forward_t(&x, false)?;

        if self.activate { x.relu() } else { Ok(x) }
    }
}

#[derive(Debug)]
struct SeparableConv3d {
    spatial: Conv2d,
    temporal: Conv1d,
}

impl SeparableConv3d {
    fn load(
        in_filters: usize,
        filters: usize,
        temporal_dilation: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let spatial_weight = vb
            .pp("layers")
            .pp(0)
            .get((2 * filters, in_filters, 1, 3, 3), "weight")?
            .squeeze(2)?;
        let temporal_vb = vb.pp("layers").pp(1);
        let temporal_weight = temporal_vb
            .get((filters, 2 * filters, 3, 1, 1), "weight")?
            .squeeze(4)?
            .squeeze(3)?;
        let temporal_bias = if temporal_vb.contains_tensor("bias") {
            Some(temporal_vb.get(filters, "bias")?)
        } else {
            None
        };

        Ok(Self {
            spatial: Conv2d::new(
                spatial_weight,
                None,
                Conv2dConfig {
                    padding: 1,
                    ..Conv2dConfig::default()
                },
            ),
            temporal: Conv1d::new(
                temporal_weight,
                temporal_bias,
                Conv1dConfig {
                    padding: temporal_dilation,
                    dilation: temporal_dilation,
                    ..Conv1dConfig::default()
                },
            ),
        })
    }

    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let x = conv3d_spatial_1x3x3(inputs, &self.spatial)?;
        conv3d_temporal_3x1x1(&x, &self.temporal)
    }
}

#[derive(Debug)]
struct FrameSimilarity {
    projection: Linear,
    fc: Linear,
    lookup_window: usize,
}

impl FrameSimilarity {
    fn load(in_filters: usize, lookup_window: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            projection: linear(in_filters, SIMILARITY_DIM, vb.pp("projection"))?,
            fc: linear(lookup_window, AUX_OUTPUT_DIM, vb.pp("fc"))?,
            lookup_window,
        })
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let pooled = inputs
            .iter()
            .map(|x| x.mean((3, 4)))
            .collect::<Result<Vec<_>>>()?;
        let pooled_refs = pooled.iter().collect::<Vec<_>>();
        let x = Tensor::cat(&pooled_refs, 1)?.transpose(1, 2)?;
        let x = l2_normalize_last_dim(&self.projection.forward(&x)?)?;
        let similarities = x.matmul(&x.transpose(1, 2)?)?;
        let windows = local_similarity_windows(&similarities, self.lookup_window)?;

        self.fc.forward(&windows)?.relu()
    }
}

#[derive(Debug)]
struct ColorHistograms {
    fc: Linear,
    lookup_window: usize,
}

impl ColorHistograms {
    fn load(lookup_window: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            fc: linear(lookup_window, AUX_OUTPUT_DIM, vb.pp("fc"))?,
            lookup_window,
        })
    }

    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let histograms = compute_color_histograms(inputs)?;
        let similarities = histograms.matmul(&histograms.transpose(1, 2)?)?;
        let windows = local_similarity_windows(&similarities, self.lookup_window)?;

        self.fc.forward(&windows)?.relu()
    }
}

fn conv3d_spatial_1x3x3(inputs: &Tensor, conv: &Conv2d) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = inputs.dims5()?;
    let x = inputs
        .permute((0, 2, 1, 3, 4))?
        .reshape((batch * frames, channels, height, width))?;
    let y = conv.forward(&x)?;
    let (_, out_channels, out_height, out_width) = y.dims4()?;

    y.reshape((batch, frames, out_channels, out_height, out_width))?
        .permute((0, 2, 1, 3, 4))
}

fn conv3d_temporal_3x1x1(inputs: &Tensor, conv: &Conv1d) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = inputs.dims5()?;
    let x = inputs
        .permute((0, 3, 4, 1, 2))?
        .reshape((batch * height * width, channels, frames))?;
    let y = conv.forward(&x)?;
    let (_, out_channels, out_frames) = y.dims3()?;

    y.reshape((batch, height, width, out_channels, out_frames))?
        .permute((0, 3, 4, 1, 2))
}

fn avg_pool3d_spatial_2x2(inputs: &Tensor) -> Result<Tensor> {
    let (batch, channels, frames, height, width) = inputs.dims5()?;
    let x = inputs
        .permute((0, 2, 1, 3, 4))?
        .reshape((batch * frames, channels, height, width))?;
    let y = x.avg_pool2d((2, 2))?;
    let (_, _, out_height, out_width) = y.dims4()?;

    y.reshape((batch, frames, channels, out_height, out_width))?
        .permute((0, 2, 1, 3, 4))
}

fn l2_normalize_last_dim(inputs: &Tensor) -> Result<Tensor> {
    inputs.broadcast_div(&(inputs.sqr()?.sum_keepdim(2)? + 1e-12)?.sqrt()?)
}

fn local_similarity_windows(similarities: &Tensor, lookup_window: usize) -> Result<Tensor> {
    let (_, frames, _) = similarities.dims3()?;
    let radius = lookup_window / 2;
    let padded = similarities.pad_with_zeros(2, radius, radius)?;
    let mut windows = Vec::with_capacity(frames);

    for frame_index in 0..frames {
        windows.push(
            padded
                .narrow(1, frame_index, 1)?
                .narrow(2, frame_index, lookup_window)?
                .squeeze(1)?,
        );
    }

    Tensor::stack(&windows, 1)
}

fn compute_color_histograms(inputs: &Tensor) -> Result<Tensor> {
    let (batch, frames, height, width, channels) = inputs.dims5()?;
    if channels != MODEL_INPUT_CHANNELS {
        bail!(
            "color histogram expects RGB inputs, got {} channels",
            channels
        );
    }
    if inputs.dtype() != DType::U8 {
        bail!(
            "color histogram expects U8 inputs, got {:?}",
            inputs.dtype()
        );
    }

    let data = inputs.contiguous()?.flatten_all()?.to_vec1::<u8>()?;
    let frame_pixels = height * width;
    let frame_bytes = frame_pixels * channels;
    let mut histograms = vec![0f32; batch * frames * 512];

    for frame_index in 0..(batch * frames) {
        let frame_offset = frame_index * frame_bytes;
        let histogram_offset = frame_index * 512;

        for pixel in data[frame_offset..frame_offset + frame_bytes].chunks_exact(3) {
            let r = usize::from(pixel[0] >> 5);
            let g = usize::from(pixel[1] >> 5);
            let b = usize::from(pixel[2] >> 5);
            histograms[histogram_offset + (r << 6) + (g << 3) + b] += 1.0;
        }

        let norm = histograms[histogram_offset..histogram_offset + 512]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 {
            for value in &mut histograms[histogram_offset..histogram_offset + 512] {
                *value /= norm;
            }
        }
    }

    Tensor::from_vec(histograms, (batch, frames, 512), inputs.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MODEL_WINDOW_FRAMES;
    use candle_core::Device;

    #[test]
    fn rejects_wrong_input_shape() {
        let input = Tensor::zeros(
            (1, MODEL_WINDOW_FRAMES - 1, 27, 48, 3),
            DType::U8,
            &Device::Cpu,
        )
        .unwrap();

        assert!(validate_input_window(&input).is_err());
    }

    #[test]
    fn color_histograms_are_l2_normalized() {
        let input =
            Tensor::zeros((1, MODEL_WINDOW_FRAMES, 27, 48, 3), DType::U8, &Device::Cpu).unwrap();

        let histograms = compute_color_histograms(&input).unwrap();
        assert_eq!(histograms.dims(), &[1, MODEL_WINDOW_FRAMES, 512]);

        let first = histograms.narrow(1, 0, 1).unwrap().flatten_all().unwrap();
        let values = first.to_vec1::<f32>().unwrap();
        assert_eq!(values[0], 1.0);
        assert!(values[1..].iter().all(|value| *value == 0.0));
    }
}
