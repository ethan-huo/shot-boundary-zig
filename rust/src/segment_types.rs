use serde::Serialize;

use crate::{DEFAULT_SCENE_THRESHOLD, Scene};

pub const DEFAULT_WINDOW_BATCH_SIZE: usize = 1;

#[derive(Debug, Clone, Copy)]
pub struct SegmentOptions {
    pub threshold: f32,
    pub collect_model_profile: bool,
    pub window_batch_size: usize,
}

impl Default for SegmentOptions {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_SCENE_THRESHOLD,
            collect_model_profile: false,
            window_batch_size: DEFAULT_WINDOW_BATCH_SIZE,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentPredictions {
    pub single_frame: Vec<f32>,
    pub many_hot: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SegmentFramesTimings {
    pub windowing_ms: f64,
    pub inference_ms: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SegmentModelProfileSummary {
    pub window_count: usize,
    pub batch_count: usize,
    pub window_batch_size: usize,
    pub total_ms: f64,
    pub input_cast_ms: f64,
    pub sddcnn_ms: f64,
    pub block_ms: Vec<f64>,
    pub flatten_ms: f64,
    pub frame_similarity_ms: f64,
    pub color_histograms_ms: f64,
    pub dense_ms: f64,
    pub heads_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentFramesReport {
    pub frame_count: usize,
    pub predictions: SegmentPredictions,
    pub scenes: Vec<Scene>,
    pub timings: SegmentFramesTimings,
    pub model_profile: Option<SegmentModelProfileSummary>,
}

#[cfg(feature = "video-io")]
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SegmentVideoTimings {
    pub decode_ms: f64,
    pub windowing_ms: f64,
    pub inference_ms: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
}

#[cfg(feature = "video-io")]
#[derive(Debug, Clone, Serialize)]
pub struct SegmentVideoReport {
    pub source: crate::video::VideoInfo,
    pub frame_count: usize,
    pub target_width: usize,
    pub target_height: usize,
    pub checksum_fnv1a64: String,
    pub limited_by_max_frames: bool,
    pub predictions: SegmentPredictions,
    pub scenes: Vec<Scene>,
    pub timings: SegmentVideoTimings,
    pub model_profile: Option<SegmentModelProfileSummary>,
}
