#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

#[cfg(feature = "mlx")]
pub mod mlx_model;
#[cfg(feature = "mlx")]
pub mod mlx_segment;
#[cfg(feature = "candle")]
pub mod model;
mod scenes;
#[cfg(feature = "candle")]
pub mod segment;
mod segment_types;
#[cfg(feature = "video-io")]
pub mod video;

pub use scenes::{FrameIndex, Scene, SceneDetectionError, predictions_to_scenes};
pub use segment_types::{
    DEFAULT_WINDOW_BATCH_SIZE, SegmentFramesReport, SegmentFramesTimings,
    SegmentModelProfileSummary, SegmentOptions, SegmentPredictions,
};
#[cfg(feature = "video-io")]
pub use segment_types::{SegmentVideoReport, SegmentVideoTimings};

pub const MODEL_INPUT_WIDTH: usize = 48;
pub const MODEL_INPUT_HEIGHT: usize = 27;
pub const MODEL_INPUT_CHANNELS: usize = 3;
pub const MODEL_WINDOW_FRAMES: usize = 100;
pub const MODEL_CONTEXT_FRAMES: usize = 25;
pub const MODEL_OUTPUT_FRAMES_PER_WINDOW: usize = 50;
pub const DEFAULT_SCENE_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ModelInputSpec {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub window_frames: usize,
    pub context_frames: usize,
    pub output_frames_per_window: usize,
}

impl Default for ModelInputSpec {
    fn default() -> Self {
        Self {
            width: MODEL_INPUT_WIDTH,
            height: MODEL_INPUT_HEIGHT,
            channels: MODEL_INPUT_CHANNELS,
            window_frames: MODEL_WINDOW_FRAMES,
            context_frames: MODEL_CONTEXT_FRAMES,
            output_frames_per_window: MODEL_OUTPUT_FRAMES_PER_WINDOW,
        }
    }
}
