use std::collections::HashMap;
use std::path::Path;

use mlx_rs::error::{AsSliceError, Exception, IoError};
use mlx_rs::ops::indexing::{TryIndexOp, take_along_axis};
use mlx_rs::{Array, Device, Dtype, StreamOrDevice, ops};
use thiserror::Error;

use crate::{MODEL_INPUT_CHANNELS, ModelInputSpec};

const BASE_FILTERS: usize = 16;
const LAYERS: usize = 3;
const BLOCKS_PER_LAYER: usize = 2;
const DENSE_DIM: usize = 1024;
const LOOKUP_WINDOW: usize = 101;
const SIMILARITY_DIM: usize = 128;
const AUX_OUTPUT_DIM: usize = 128;
const BATCH_NORM_EPS: f32 = 1e-3;

#[derive(Debug, Error)]
pub enum MlxModelError {
    #[error("mlx error: {0}")]
    Mlx(#[from] Exception),

    #[error("mlx io error: {0}")]
    Io(#[from] IoError),

    #[error("mlx slice error: {0}")]
    Slice(#[from] AsSliceError),

    #[error("missing MLX weight {0}")]
    MissingWeight(String),

    #[error("invalid shape for {name}: expected {expected:?}, got {actual:?}")]
    InvalidShape {
        name: String,
        expected: Vec<i32>,
        actual: Vec<i32>,
    },

    #[error("invalid rank for {name}: expected {expected}, got shape {actual:?}")]
    InvalidRank {
        name: String,
        expected: usize,
        actual: Vec<i32>,
    },

    #[error("TransNetV2 dimensions must be non-zero: {0:?}")]
    InvalidConfig(MlxTransNetV2Config),

    #[error("TransNetV2 lookup_window must be a positive odd integer, got {0}")]
    InvalidLookupWindow(usize),

    #[error("TransNetV2 input dtype must be U8, got {0:?}")]
    InvalidInputDtype(Dtype),

    #[error("color histogram expects RGB inputs, got {0} channels")]
    InvalidColorHistogramChannels(i32),
}

type MlxResult<T> = Result<T, MlxModelError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxTransNetV2Config {
    pub base_filters: usize,
    pub layers: usize,
    pub blocks_per_layer: usize,
    pub dense_dim: usize,
    pub lookup_window: usize,
    pub use_frame_similarity: bool,
    pub use_color_histograms: bool,
}

impl Default for MlxTransNetV2Config {
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
pub struct MlxTransNetV2 {
    blocks: Vec<MlxStackedDdcnn>,
    frame_similarity: Option<MlxFrameSimilarity>,
    color_histograms: Option<MlxColorHistograms>,
    fc1: MlxLinear,
    cls_layer1: MlxLinear,
    cls_layer2: MlxLinear,
}

#[derive(Debug)]
pub struct MlxTransNetV2Output {
    pub single_frame_logits: Array,
    pub many_hot_logits: Array,
}

impl MlxTransNetV2 {
    pub fn load_from_safetensors(path: impl AsRef<Path>) -> MlxResult<Self> {
        // MLX-C requires safetensors I/O on CPU; compute defaults back to GPU after load.
        let mut weights = Array::load_safetensors_device(path, StreamOrDevice::cpu())?;
        let gpu = Device::gpu();
        Device::set_default(&gpu);
        Self::load(MlxTransNetV2Config::default(), &mut weights)
    }

    pub fn load(
        config: MlxTransNetV2Config,
        weights: &mut HashMap<String, Array>,
    ) -> MlxResult<Self> {
        validate_config(config)?;

        let mut blocks = Vec::with_capacity(config.layers);
        let mut in_filters = MODEL_INPUT_CHANNELS;
        for layer_index in 0..config.layers {
            let filters = config.base_filters * (1 << layer_index);
            blocks.push(MlxStackedDdcnn::load(
                in_filters,
                config.blocks_per_layer,
                filters,
                &format!("SDDCNN.{layer_index}"),
                weights,
            )?);
            in_filters = filters * 4;
        }

        let frame_similarity = config
            .use_frame_similarity
            .then(|| {
                let in_filters = (0..config.layers)
                    .map(|layer_index| config.base_filters * (1 << layer_index) * 4)
                    .sum();
                MlxFrameSimilarity::load(
                    in_filters,
                    config.lookup_window,
                    "frame_sim_layer",
                    weights,
                )
            })
            .transpose()?;

        let color_histograms = config
            .use_color_histograms
            .then(|| MlxColorHistograms::load(config.lookup_window, "color_hist_layer", weights))
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
            fc1: MlxLinear::load(output_dim, config.dense_dim, "fc1", weights)?,
            cls_layer1: MlxLinear::load(config.dense_dim, 1, "cls_layer1", weights)?,
            cls_layer2: MlxLinear::load(config.dense_dim, 1, "cls_layer2", weights)?,
        })
    }

    pub fn forward(&self, inputs: &Array) -> MlxResult<MlxTransNetV2Output> {
        validate_input_window(inputs)?;

        let mut x = inputs
            .as_dtype(Dtype::Float32)?
            .divide(&Array::from_f32(255.0))?;
        let mut block_features = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            x = block.forward(&x)?;
            block_features.push(x.clone());
        }

        let (batch, frames, height, width, channels) = dims5("sddcnn_output", &x)?;
        let mut features = x.reshape(&[batch, frames, height * width * channels])?;

        if let Some(frame_similarity) = &self.frame_similarity {
            features =
                ops::concatenate_axis(&[frame_similarity.forward(&block_features)?, features], 2)?;
        }

        if let Some(color_histograms) = &self.color_histograms {
            features = ops::concatenate_axis(&[color_histograms.forward(inputs)?, features], 2)?;
        }

        let hidden = relu(&self.fc1.forward(&features)?)?;

        Ok(MlxTransNetV2Output {
            single_frame_logits: self.cls_layer1.forward(&hidden)?,
            many_hot_logits: self.cls_layer2.forward(&hidden)?,
        })
    }
}

fn validate_config(config: MlxTransNetV2Config) -> MlxResult<()> {
    if config.base_filters == 0 || config.layers == 0 || config.blocks_per_layer == 0 {
        return Err(MlxModelError::InvalidConfig(config));
    }

    if config.lookup_window == 0 || config.lookup_window % 2 == 0 {
        return Err(MlxModelError::InvalidLookupWindow(config.lookup_window));
    }

    Ok(())
}

fn validate_input_window(inputs: &Array) -> MlxResult<()> {
    let spec = ModelInputSpec::default();
    let expected = [
        -1,
        spec.window_frames as i32,
        spec.height as i32,
        spec.width as i32,
        spec.channels as i32,
    ];
    let actual = inputs.shape();

    if inputs.dtype() != Dtype::Uint8 {
        return Err(MlxModelError::InvalidInputDtype(inputs.dtype()));
    }
    if actual.len() != expected.len()
        || actual[1..]
            != [
                spec.window_frames as i32,
                spec.height as i32,
                spec.width as i32,
                spec.channels as i32,
            ]
    {
        return Err(MlxModelError::InvalidShape {
            name: "inputs".to_string(),
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }

    Ok(())
}

#[derive(Debug)]
struct MlxStackedDdcnn {
    blocks: Vec<MlxDilatedDdcnn>,
}

impl MlxStackedDdcnn {
    fn load(
        in_filters: usize,
        n_blocks: usize,
        filters: usize,
        prefix: &str,
        weights: &mut HashMap<String, Array>,
    ) -> MlxResult<Self> {
        let mut blocks = Vec::with_capacity(n_blocks);
        for block_index in 0..n_blocks {
            blocks.push(MlxDilatedDdcnn::load(
                if block_index == 0 {
                    in_filters
                } else {
                    filters * 4
                },
                filters,
                block_index + 1 != n_blocks,
                &format!("{prefix}.DDCNN.{block_index}"),
                weights,
            )?);
        }

        Ok(Self { blocks })
    }

    fn forward(&self, inputs: &Array) -> MlxResult<Array> {
        let mut x = inputs.clone();
        let mut shortcut = None;

        for block in &self.blocks {
            x = block.forward(&x)?;
            if shortcut.is_none() {
                shortcut = Some(x.clone());
            }
        }

        let x = relu(&x)?;
        let shortcut = shortcut.expect("MlxStackedDdcnn always has at least one block");
        let x = x.add(&shortcut)?;

        avg_pool3d_spatial_2x2(&x)
    }
}

#[derive(Debug)]
struct MlxDilatedDdcnn {
    convs: Vec<MlxSeparableConv3d>,
    bn: MlxBatchNorm,
    activate: bool,
}

impl MlxDilatedDdcnn {
    fn load(
        in_filters: usize,
        filters: usize,
        activate: bool,
        prefix: &str,
        weights: &mut HashMap<String, Array>,
    ) -> MlxResult<Self> {
        let convs = [1, 2, 4, 8]
            .into_iter()
            .map(|dilation| {
                MlxSeparableConv3d::load(
                    in_filters,
                    filters,
                    dilation,
                    &format!("{prefix}.Conv3D_{dilation}"),
                    weights,
                )
            })
            .collect::<MlxResult<Vec<_>>>()?;

        Ok(Self {
            convs,
            bn: MlxBatchNorm::load(filters * 4, &format!("{prefix}.bn"), weights)?,
            activate,
        })
    }

    fn forward(&self, inputs: &Array) -> MlxResult<Array> {
        let convs = self
            .convs
            .iter()
            .map(|conv| conv.forward(inputs))
            .collect::<MlxResult<Vec<_>>>()?;
        let x = ops::concatenate_axis(&convs, 4)?;
        let x = self.bn.forward(&x)?;

        if self.activate { relu(&x) } else { Ok(x) }
    }
}

#[derive(Debug)]
struct MlxSeparableConv3d {
    spatial_weight: Array,
    temporal_weight: Array,
    temporal_dilation: i32,
}

impl MlxSeparableConv3d {
    fn load(
        in_filters: usize,
        filters: usize,
        temporal_dilation: i32,
        prefix: &str,
        weights: &mut HashMap<String, Array>,
    ) -> MlxResult<Self> {
        let spatial_name = format!("{prefix}.layers.0.weight");
        let spatial_weight = take_weight(weights, &spatial_name)?;
        validate_shape(
            &spatial_name,
            &spatial_weight,
            &[(2 * filters) as i32, in_filters as i32, 1, 3, 3],
        )?;

        let temporal_name = format!("{prefix}.layers.1.weight");
        let temporal_weight = take_weight(weights, &temporal_name)?;
        validate_shape(
            &temporal_name,
            &temporal_weight,
            &[filters as i32, (2 * filters) as i32, 3, 1, 1],
        )?;

        // This model only needs separable 3D conv; on Apple Silicon, conv2d+conv1d is
        // materially faster than routing these narrow kernels through generic conv3d.
        Ok(Self {
            spatial_weight: spatial_weight.transpose_axes(&[0, 3, 4, 1, 2])?.reshape(&[
                (2 * filters) as i32,
                3,
                3,
                in_filters as i32,
            ])?,
            temporal_weight: temporal_weight
                .transpose_axes(&[0, 2, 1, 3, 4])?
                .reshape(&[filters as i32, 3, (2 * filters) as i32])?,
            temporal_dilation,
        })
    }

    fn forward(&self, inputs: &Array) -> MlxResult<Array> {
        let (batch, frames, height, width, channels) = dims5("conv3d_spatial_input", inputs)?;
        let spatial_input = inputs.reshape(&[batch * frames, height, width, channels])?;
        let spatial = ops::conv2d(
            &spatial_input,
            &self.spatial_weight,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )?;
        let (_, out_height, out_width, out_channels) = dims4("conv3d_spatial_output", &spatial)?;
        let temporal_input = spatial
            .reshape(&[batch, frames, out_height, out_width, out_channels])?
            .transpose_axes(&[0, 2, 3, 1, 4])?
            .reshape(&[batch * out_height * out_width, frames, out_channels])?;
        let temporal = ops::conv1d(
            &temporal_input,
            &self.temporal_weight,
            1,
            self.temporal_dilation,
            self.temporal_dilation,
            1,
        )?;
        let (_, out_frames, out_channels) = dims3("conv3d_temporal_output", &temporal)?;

        Ok(temporal
            .reshape(&[batch, out_height, out_width, out_frames, out_channels])?
            .transpose_axes(&[0, 3, 1, 2, 4])?)
    }
}

#[derive(Debug)]
struct MlxBatchNorm {
    weight: Array,
    bias: Array,
    running_mean: Array,
    running_var: Array,
}

impl MlxBatchNorm {
    fn load(
        channels: usize,
        prefix: &str,
        weights: &mut HashMap<String, Array>,
    ) -> MlxResult<Self> {
        let expected = [channels as i32];
        let weight = take_weight(weights, &format!("{prefix}.weight"))?;
        validate_shape(&format!("{prefix}.weight"), &weight, &expected)?;
        let bias = take_weight(weights, &format!("{prefix}.bias"))?;
        validate_shape(&format!("{prefix}.bias"), &bias, &expected)?;
        let running_mean = take_weight(weights, &format!("{prefix}.running_mean"))?;
        validate_shape(&format!("{prefix}.running_mean"), &running_mean, &expected)?;
        let running_var = take_weight(weights, &format!("{prefix}.running_var"))?;
        validate_shape(&format!("{prefix}.running_var"), &running_var, &expected)?;

        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
        })
    }

    fn forward(&self, inputs: &Array) -> MlxResult<Array> {
        let scale = self
            .running_var
            .add(&Array::from_f32(BATCH_NORM_EPS))?
            .rsqrt()?
            .multiply(&self.weight)?;

        inputs
            .subtract(&self.running_mean)?
            .multiply(&scale)?
            .add(&self.bias)
            .map_err(Into::into)
    }
}

#[derive(Debug)]
struct MlxFrameSimilarity {
    projection: MlxLinear,
    fc: MlxLinear,
    lookup_window: usize,
}

impl MlxFrameSimilarity {
    fn load(
        in_filters: usize,
        lookup_window: usize,
        prefix: &str,
        weights: &mut HashMap<String, Array>,
    ) -> MlxResult<Self> {
        Ok(Self {
            projection: MlxLinear::load(
                in_filters,
                SIMILARITY_DIM,
                &format!("{prefix}.projection"),
                weights,
            )?,
            fc: MlxLinear::load(
                lookup_window,
                AUX_OUTPUT_DIM,
                &format!("{prefix}.fc"),
                weights,
            )?,
            lookup_window,
        })
    }

    fn forward(&self, inputs: &[Array]) -> MlxResult<Array> {
        let pooled = inputs
            .iter()
            .map(|x| x.mean_axes(&[2, 3], None))
            .collect::<Result<Vec<_>, _>>()?;
        let x = ops::concatenate_axis(&pooled, 2)?;
        let x = l2_normalize_last_dim(&self.projection.forward(&x)?)?;
        let similarities = x.matmul(&x.transpose_axes(&[0, 2, 1])?)?;
        let windows = local_similarity_windows(&similarities, self.lookup_window)?;

        relu(&self.fc.forward(&windows)?)
    }
}

#[derive(Debug)]
struct MlxColorHistograms {
    fc: MlxLinear,
    lookup_window: usize,
}

impl MlxColorHistograms {
    fn load(
        lookup_window: usize,
        prefix: &str,
        weights: &mut HashMap<String, Array>,
    ) -> MlxResult<Self> {
        Ok(Self {
            fc: MlxLinear::load(
                lookup_window,
                AUX_OUTPUT_DIM,
                &format!("{prefix}.fc"),
                weights,
            )?,
            lookup_window,
        })
    }

    fn forward(&self, inputs: &Array) -> MlxResult<Array> {
        let histograms = compute_color_histograms(inputs)?;
        let similarities = histograms.matmul(&histograms.transpose_axes(&[0, 2, 1])?)?;
        let windows = local_similarity_windows(&similarities, self.lookup_window)?;

        relu(&self.fc.forward(&windows)?)
    }
}

#[derive(Debug)]
struct MlxLinear {
    weight: Array,
    bias: Array,
}

impl MlxLinear {
    fn load(
        in_dim: usize,
        out_dim: usize,
        prefix: &str,
        weights: &mut HashMap<String, Array>,
    ) -> MlxResult<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = take_weight(weights, &weight_name)?;
        validate_shape(&weight_name, &weight, &[out_dim as i32, in_dim as i32])?;
        let bias_name = format!("{prefix}.bias");
        let bias = take_weight(weights, &bias_name)?;
        validate_shape(&bias_name, &bias, &[out_dim as i32])?;

        Ok(Self { weight, bias })
    }

    fn forward(&self, inputs: &Array) -> MlxResult<Array> {
        inputs
            .matmul(&self.weight.transpose()?)?
            .add(&self.bias)
            .map_err(Into::into)
    }
}

fn avg_pool3d_spatial_2x2(inputs: &Array) -> MlxResult<Array> {
    let (batch, frames, height, width, channels) = dims5("avg_pool_input", inputs)?;
    let out_height = height / 2;
    let out_width = width / 2;
    let cropped = inputs.try_index((.., .., 0..(out_height * 2), 0..(out_width * 2), ..))?;
    let grouped = cropped.reshape(&[batch * frames, out_height, 2, out_width, 2, channels])?;
    let pooled = grouped.mean_axes(&[2, 4], None)?;

    Ok(pooled.reshape(&[batch, frames, out_height, out_width, channels])?)
}

fn l2_normalize_last_dim(inputs: &Array) -> MlxResult<Array> {
    let norm = inputs
        .square()?
        .sum_axis(-1, true)?
        .add(&Array::from_f32(1e-12))?
        .sqrt()?;

    Ok(inputs.divide(&norm)?)
}

fn local_similarity_windows(similarities: &Array, lookup_window: usize) -> MlxResult<Array> {
    let (_, frames, _) = dims3("similarities", similarities)?;
    let lookup_window = lookup_window as i32;
    let radius = lookup_window / 2;
    let pads = [(0, 0), (0, 0), (radius, radius)];
    let padded = ops::pad(
        similarities,
        &pads,
        Some(Array::from_f32(0.0)),
        None::<ops::PadMode>,
    )?;
    let frame_indices =
        Array::arange::<i32, i32>(0, frames, None::<i32>)?.reshape(&[1, frames, 1])?;
    let offsets = Array::arange::<i32, i32>(0, lookup_window, None::<i32>)?.reshape(&[
        1,
        1,
        lookup_window,
    ])?;
    let indices = frame_indices.add(&offsets)?;

    Ok(take_along_axis(&padded, &indices, 2)?)
}

fn compute_color_histograms(inputs: &Array) -> MlxResult<Array> {
    let (batch, frames, height, width, channels) = dims5("color_histogram_input", inputs)?;
    if channels != MODEL_INPUT_CHANNELS as i32 {
        return Err(MlxModelError::InvalidColorHistogramChannels(channels));
    }
    if inputs.dtype() != Dtype::Uint8 {
        return Err(MlxModelError::InvalidInputDtype(inputs.dtype()));
    }

    let data = inputs.try_as_slice::<u8>()?;
    let frame_pixels = height as usize * width as usize;
    let frame_bytes = frame_pixels * channels as usize;
    let mut histograms = vec![0f32; batch as usize * frames as usize * 512];

    for frame_index in 0..(batch as usize * frames as usize) {
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

    Ok(Array::from_slice(&histograms, &[batch, frames, 512]))
}

fn relu(inputs: &Array) -> MlxResult<Array> {
    Ok(ops::maximum(inputs, &Array::from_f32(0.0))?)
}

fn take_weight(weights: &mut HashMap<String, Array>, name: &str) -> MlxResult<Array> {
    weights
        .remove(name)
        .ok_or_else(|| MlxModelError::MissingWeight(name.to_string()))
}

fn validate_shape(name: &str, array: &Array, expected: &[i32]) -> MlxResult<()> {
    if array.shape() != expected {
        return Err(MlxModelError::InvalidShape {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: array.shape().to_vec(),
        });
    }

    Ok(())
}

fn dims5(name: &str, array: &Array) -> MlxResult<(i32, i32, i32, i32, i32)> {
    let shape = array.shape();
    if shape.len() != 5 {
        return Err(MlxModelError::InvalidRank {
            name: name.to_string(),
            expected: 5,
            actual: shape.to_vec(),
        });
    }

    Ok((shape[0], shape[1], shape[2], shape[3], shape[4]))
}

fn dims3(name: &str, array: &Array) -> MlxResult<(i32, i32, i32)> {
    let shape = array.shape();
    if shape.len() != 3 {
        return Err(MlxModelError::InvalidRank {
            name: name.to_string(),
            expected: 3,
            actual: shape.to_vec(),
        });
    }

    Ok((shape[0], shape[1], shape[2]))
}

fn dims4(name: &str, array: &Array) -> MlxResult<(i32, i32, i32, i32)> {
    let shape = array.shape();
    if shape.len() != 4 {
        return Err(MlxModelError::InvalidRank {
            name: name.to_string(),
            expected: 4,
            actual: shape.to_vec(),
        });
    }

    Ok((shape[0], shape[1], shape[2], shape[3]))
}
