use std::time::Instant;

use mlx_rs::Array;
use mlx_rs::error::{AsSliceError, Exception};
use mlx_rs::ops;
use mlx_rs::ops::indexing::TryIndexOp;
use thiserror::Error;

use crate::mlx_model::{MlxModelError, MlxTransNetV2};
use crate::segment_types::{
    SegmentFramesReport, SegmentFramesTimings, SegmentOptions, SegmentPredictions,
};
#[cfg(feature = "video-io")]
use crate::segment_types::{SegmentVideoReport, SegmentVideoTimings};
use crate::{
    MODEL_CONTEXT_FRAMES, MODEL_INPUT_CHANNELS, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
    MODEL_OUTPUT_FRAMES_PER_WINDOW, MODEL_WINDOW_FRAMES, SceneDetectionError,
    predictions_to_scenes,
};

const FRAME_BYTES: usize = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNELS;
pub const DEFAULT_MLX_WINDOW_BATCH_SIZE: usize = 2;

#[derive(Debug, Error)]
pub enum MlxSegmentError {
    #[error("mlx model error: {0}")]
    Model(#[from] MlxModelError),

    #[error("mlx error: {0}")]
    Mlx(#[from] Exception),

    #[error("mlx slice error: {0}")]
    Slice(#[from] AsSliceError),

    #[error("scene detection error: {0}")]
    SceneDetection(#[from] SceneDetectionError),

    #[cfg(feature = "video-io")]
    #[error("video error: {0}")]
    Video(#[from] crate::video::VideoError),

    #[error("segment input cannot be empty")]
    EmptyFrames,

    #[error(
        "segment input must be {expected_width}x{expected_height} RGB frames, got {width}x{height}x{channels}"
    )]
    UnexpectedFrameShape {
        expected_width: usize,
        expected_height: usize,
        width: usize,
        height: usize,
        channels: usize,
    },

    #[error("frame buffer length {byte_len} is not divisible by frame size {frame_bytes}")]
    InvalidFrameBuffer { byte_len: usize, frame_bytes: usize },

    #[error("window_batch_size must be greater than zero")]
    InvalidWindowBatchSize,

    #[error("MLX backend does not support Candle model profile fields yet")]
    UnsupportedModelProfile,
}

pub fn segment_frames(
    model: &MlxTransNetV2,
    frames_rgb24: &[u8],
    width: usize,
    height: usize,
    threshold: f32,
) -> Result<SegmentFramesReport, MlxSegmentError> {
    segment_frames_with_options(
        model,
        frames_rgb24,
        width,
        height,
        SegmentOptions {
            threshold,
            ..SegmentOptions::default()
        },
    )
}

pub fn segment_frames_with_options(
    model: &MlxTransNetV2,
    frames_rgb24: &[u8],
    width: usize,
    height: usize,
    options: SegmentOptions,
) -> Result<SegmentFramesReport, MlxSegmentError> {
    validate_frame_shape(width, height, MODEL_INPUT_CHANNELS)?;
    if options.collect_model_profile {
        return Err(MlxSegmentError::UnsupportedModelProfile);
    }
    if options.window_batch_size == 0 {
        return Err(MlxSegmentError::InvalidWindowBatchSize);
    }

    let frame_count = frame_count_from_bytes(frames_rgb24.len(), FRAME_BYTES)?;
    let mut single_frame = Vec::new();
    let mut many_hot = Vec::new();
    let mut windowing_ms = 0.0;
    let mut inference_ms = 0.0;
    let windows = window_source_indices(frame_count)?;
    let started_at = Instant::now();

    for window_batch in windows.chunks(options.window_batch_size) {
        let window_started_at = Instant::now();
        let window = build_window_batch(frames_rgb24, FRAME_BYTES, window_batch);
        let batch_size = window_batch.len();
        let input = Array::from_slice(
            &window,
            &[
                batch_size as i32,
                MODEL_WINDOW_FRAMES as i32,
                MODEL_INPUT_HEIGHT as i32,
                MODEL_INPUT_WIDTH as i32,
                MODEL_INPUT_CHANNELS as i32,
            ],
        );
        windowing_ms += window_started_at.elapsed().as_secs_f64() * 1_000.0;

        let inference_started_at = Instant::now();
        let output = model.forward(&input)?;
        single_frame.extend(center_probabilities(&output.single_frame_logits)?);
        many_hot.extend(center_probabilities(&output.many_hot_logits)?);
        inference_ms += inference_started_at.elapsed().as_secs_f64() * 1_000.0;
    }

    single_frame.truncate(frame_count);
    many_hot.truncate(frame_count);

    let postprocess_started_at = Instant::now();
    let scenes = predictions_to_scenes(&single_frame, options.threshold)?;
    let postprocess_ms = postprocess_started_at.elapsed().as_secs_f64() * 1_000.0;

    Ok(SegmentFramesReport {
        frame_count,
        predictions: SegmentPredictions {
            single_frame,
            many_hot,
        },
        scenes,
        timings: SegmentFramesTimings {
            windowing_ms,
            inference_ms,
            postprocess_ms,
            total_ms: started_at.elapsed().as_secs_f64() * 1_000.0,
        },
        model_profile: None,
    })
}

#[cfg(feature = "video-io")]
pub fn segment_video(
    model: &MlxTransNetV2,
    video: impl AsRef<std::path::Path>,
    threshold: f32,
    options: crate::video::DecodeSmokeOptions,
) -> Result<SegmentVideoReport, MlxSegmentError> {
    segment_video_with_options(
        model,
        video,
        options,
        SegmentOptions {
            threshold,
            ..SegmentOptions::default()
        },
    )
}

#[cfg(feature = "video-io")]
pub fn segment_video_with_options(
    model: &MlxTransNetV2,
    video: impl AsRef<std::path::Path>,
    decode_options: crate::video::DecodeSmokeOptions,
    segment_options: SegmentOptions,
) -> Result<SegmentVideoReport, MlxSegmentError> {
    let decoded = crate::video::decode_video_rgb24(video, decode_options)?;
    let frame_report = segment_frames_with_options(
        model,
        &decoded.data,
        decoded.target_width,
        decoded.target_height,
        segment_options,
    )?;

    Ok(SegmentVideoReport {
        source: decoded.source,
        frame_count: frame_report.frame_count,
        target_width: decoded.target_width,
        target_height: decoded.target_height,
        checksum_fnv1a64: decoded.checksum_fnv1a64,
        limited_by_max_frames: decoded.limited_by_max_frames,
        predictions: frame_report.predictions,
        scenes: frame_report.scenes,
        model_profile: frame_report.model_profile,
        timings: SegmentVideoTimings {
            decode_ms: decoded.elapsed_ms,
            windowing_ms: frame_report.timings.windowing_ms,
            inference_ms: frame_report.timings.inference_ms,
            postprocess_ms: frame_report.timings.postprocess_ms,
            total_ms: decoded.elapsed_ms + frame_report.timings.total_ms,
        },
    })
}

fn validate_frame_shape(
    width: usize,
    height: usize,
    channels: usize,
) -> Result<(), MlxSegmentError> {
    if (width, height, channels) != (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, MODEL_INPUT_CHANNELS) {
        return Err(MlxSegmentError::UnexpectedFrameShape {
            expected_width: MODEL_INPUT_WIDTH,
            expected_height: MODEL_INPUT_HEIGHT,
            width,
            height,
            channels,
        });
    }

    Ok(())
}

fn frame_count_from_bytes(byte_len: usize, frame_bytes: usize) -> Result<usize, MlxSegmentError> {
    if byte_len == 0 {
        return Err(MlxSegmentError::EmptyFrames);
    }
    if byte_len % frame_bytes != 0 {
        return Err(MlxSegmentError::InvalidFrameBuffer {
            byte_len,
            frame_bytes,
        });
    }

    Ok(byte_len / frame_bytes)
}

fn window_source_indices(frame_count: usize) -> Result<Vec<Vec<usize>>, MlxSegmentError> {
    if frame_count == 0 {
        return Err(MlxSegmentError::EmptyFrames);
    }

    let padded_start = MODEL_CONTEXT_FRAMES;
    let remainder = frame_count % MODEL_OUTPUT_FRAMES_PER_WINDOW;
    let padded_end = MODEL_CONTEXT_FRAMES + MODEL_OUTPUT_FRAMES_PER_WINDOW
        - if remainder == 0 {
            MODEL_OUTPUT_FRAMES_PER_WINDOW
        } else {
            remainder
        };
    let padded_count = padded_start + frame_count + padded_end;
    let mut windows = Vec::new();
    let mut ptr = 0;

    while ptr + MODEL_WINDOW_FRAMES <= padded_count {
        let mut indices = Vec::with_capacity(MODEL_WINDOW_FRAMES);
        for padded_index in ptr..ptr + MODEL_WINDOW_FRAMES {
            let source_index = if padded_index < padded_start {
                0
            } else if padded_index < padded_start + frame_count {
                padded_index - padded_start
            } else {
                frame_count - 1
            };
            indices.push(source_index);
        }
        windows.push(indices);
        ptr += MODEL_OUTPUT_FRAMES_PER_WINDOW;
    }

    Ok(windows)
}

fn build_window_batch(
    frames_rgb24: &[u8],
    frame_bytes: usize,
    window_batch: &[Vec<usize>],
) -> Vec<u8> {
    let mut batch = Vec::with_capacity(window_batch.len() * MODEL_WINDOW_FRAMES * frame_bytes);
    for indices in window_batch {
        for &index in indices {
            let start = index * frame_bytes;
            let end = start + frame_bytes;
            batch.extend_from_slice(&frames_rgb24[start..end]);
        }
    }
    batch
}

fn center_probabilities(logits: &Array) -> Result<Vec<f32>, MlxSegmentError> {
    let probs = ops::sigmoid(logits)?;
    let batch = probs.shape()[0];
    let channels = probs.shape()[2];
    let center = probs.try_index((
        ..,
        MODEL_CONTEXT_FRAMES as i32..(MODEL_CONTEXT_FRAMES + MODEL_OUTPUT_FRAMES_PER_WINDOW) as i32,
        ..,
    ))?;
    let flattened = center.reshape(&[batch * MODEL_OUTPUT_FRAMES_PER_WINDOW as i32 * channels])?;

    Ok(flattened.try_as_slice::<f32>()?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_windows_match_upstream_center_stride_and_trim_policy() {
        let windows = window_source_indices(51).unwrap();

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0][0], 0);
        assert_eq!(windows[0][MODEL_CONTEXT_FRAMES], 0);
        assert_eq!(windows[0][MODEL_CONTEXT_FRAMES + 49], 49);
        assert_eq!(windows[1][0], 25);
        assert_eq!(windows[1][MODEL_CONTEXT_FRAMES], 50);
        assert_eq!(windows[1][MODEL_CONTEXT_FRAMES + 25], 50);
        assert_eq!(windows[1][MODEL_WINDOW_FRAMES - 1], 50);
    }
}
