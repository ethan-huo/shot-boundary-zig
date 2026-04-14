use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ffmpeg::media::Type;
use ffmpeg_next as ffmpeg;
use serde::Serialize;
use thiserror::Error;

use crate::{MODEL_INPUT_CHANNELS, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, ModelInputSpec};

const AV_TIME_BASE_SECONDS: f64 = 1_000_000.0;

#[derive(Debug, Error)]
pub enum VideoError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error("no video stream found in {path}")]
    VideoStreamNotFound { path: PathBuf },

    #[error("target frame size must be non-zero RGB dimensions, got {width}x{height}")]
    InvalidTargetSize { width: usize, height: usize },

    #[error("ffmpeg command failed with status {status}: {stderr}")]
    FfmpegCommandFailed { status: String, stderr: String },

    #[error("io error while running ffmpeg: {0}")]
    Io(#[from] std::io::Error),

    #[error("decoded rawvideo byte length {byte_len} is not divisible by frame size {frame_bytes}")]
    InvalidRawVideoLength { byte_len: usize, frame_bytes: usize },
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct RationalInfo {
    pub numerator: i32,
    pub denominator: i32,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VideoInfo {
    pub path: PathBuf,
    pub stream_index: usize,
    pub source_width: u32,
    pub source_height: u32,
    pub source_pixel_format: String,
    pub bit_rate: Option<i64>,
    pub duration_seconds: Option<f64>,
    pub stream_duration_seconds: Option<f64>,
    pub frame_count_hint: Option<i64>,
    pub avg_frame_rate: RationalInfo,
    pub time_base: RationalInfo,
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeSmokeOptions {
    pub target_width: usize,
    pub target_height: usize,
    pub max_frames: Option<usize>,
}

impl Default for DecodeSmokeOptions {
    fn default() -> Self {
        Self {
            target_width: MODEL_INPUT_WIDTH,
            target_height: MODEL_INPUT_HEIGHT,
            max_frames: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeSmokeReport {
    pub source: VideoInfo,
    pub model_input: ModelInputSpec,
    pub target_width: usize,
    pub target_height: usize,
    pub decoded_frames: usize,
    pub decoded_rgb_bytes: usize,
    pub checksum_fnv1a64: String,
    pub elapsed_ms: f64,
    pub frames_per_second: f64,
    pub limited_by_max_frames: bool,
}

#[derive(Debug, Clone)]
pub struct DecodedVideoFrames {
    pub source: VideoInfo,
    pub target_width: usize,
    pub target_height: usize,
    pub data: Vec<u8>,
    pub checksum_fnv1a64: String,
    pub elapsed_ms: f64,
    pub limited_by_max_frames: bool,
}

impl DecodedVideoFrames {
    pub fn frame_bytes(&self) -> usize {
        self.target_width * self.target_height * MODEL_INPUT_CHANNELS
    }

    pub fn frame_count(&self) -> usize {
        let frame_bytes = self.frame_bytes();
        if frame_bytes == 0 {
            0
        } else {
            self.data.len() / frame_bytes
        }
    }
}

pub fn inspect_video(path: impl AsRef<Path>) -> Result<VideoInfo, VideoError> {
    ffmpeg::init()?;

    let path = path.as_ref();
    let ictx = ffmpeg::format::input(path)?;
    let stream =
        ictx.streams()
            .best(Type::Video)
            .ok_or_else(|| VideoError::VideoStreamNotFound {
                path: path.to_path_buf(),
            })?;
    let decoder_context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
    let decoder = decoder_context.decoder().video()?;
    let info = video_info_from_parts(path, &ictx, &stream, &decoder);

    Ok(info)
}

pub fn decode_smoke(
    path: impl AsRef<Path>,
    options: DecodeSmokeOptions,
) -> Result<DecodeSmokeReport, VideoError> {
    let decoded = decode_video_rgb24(path, options)?;
    let elapsed_seconds = decoded.elapsed_ms / 1_000.0;
    let decoded_frames = decoded.frame_count();
    let decoded_rgb_bytes = decoded.data.len();
    let frames_per_second = if elapsed_seconds > 0.0 {
        decoded_frames as f64 / elapsed_seconds
    } else {
        0.0
    };

    Ok(DecodeSmokeReport {
        source: decoded.source,
        model_input: ModelInputSpec::default(),
        target_width: decoded.target_width,
        target_height: decoded.target_height,
        decoded_frames,
        decoded_rgb_bytes,
        checksum_fnv1a64: decoded.checksum_fnv1a64,
        elapsed_ms: decoded.elapsed_ms,
        frames_per_second,
        limited_by_max_frames: decoded.limited_by_max_frames,
    })
}

pub fn decode_video_rgb24(
    path: impl AsRef<Path>,
    options: DecodeSmokeOptions,
) -> Result<DecodedVideoFrames, VideoError> {
    if options.target_width == 0 || options.target_height == 0 {
        return Err(VideoError::InvalidTargetSize {
            width: options.target_width,
            height: options.target_height,
        });
    }

    let path = path.as_ref();
    let source = inspect_video(path)?;
    let started_at = Instant::now();
    let raw = run_ffmpeg_rawvideo(path, options)?;
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    let frame_bytes = options.target_width * options.target_height * MODEL_INPUT_CHANNELS;
    if raw.is_empty() || raw.len() % frame_bytes != 0 {
        return Err(VideoError::InvalidRawVideoLength {
            byte_len: raw.len(),
            frame_bytes,
        });
    }
    let frame_count = raw.len() / frame_bytes;
    let checksum = fnv1a64(&raw);

    Ok(DecodedVideoFrames {
        source,
        target_width: options.target_width,
        target_height: options.target_height,
        checksum_fnv1a64: format!("{checksum:016x}"),
        elapsed_ms: elapsed_seconds * 1_000.0,
        data: raw,
        limited_by_max_frames: options
            .max_frames
            .is_some_and(|max_frames| frame_count >= max_frames),
    })
}

fn run_ffmpeg_rawvideo(path: &Path, options: DecodeSmokeOptions) -> Result<Vec<u8>, VideoError> {
    let mut command = Command::new("ffmpeg");
    command
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(path);
    if let Some(max_frames) = options.max_frames {
        command.arg("-frames:v").arg(max_frames.to_string());
    }
    // Use the ffmpeg command rather than libav frame iteration here because the official
    // TensorFlow reference relies on ffmpeg's default output vsync and scaler semantics.
    command
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("rgb24")
        .arg("-s")
        .arg(format!(
            "{}x{}",
            options.target_width, options.target_height
        ))
        .arg("-");

    let output = command.output()?;
    if !output.status.success() {
        return Err(VideoError::FfmpegCommandFailed {
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    Ok(output.stdout)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut checksum = 0xcbf29ce484222325;
    for byte in bytes {
        checksum ^= u64::from(*byte);
        checksum = checksum.wrapping_mul(0x100000001b3);
    }
    checksum
}

fn video_info_from_parts(
    path: &Path,
    ictx: &ffmpeg::format::context::Input,
    stream: &ffmpeg::Stream<'_>,
    decoder: &ffmpeg::decoder::Video,
) -> VideoInfo {
    VideoInfo {
        path: path.to_path_buf(),
        stream_index: stream.index(),
        source_width: decoder.width(),
        source_height: decoder.height(),
        source_pixel_format: format!("{:?}", decoder.format()),
        bit_rate: positive_i64(ictx.bit_rate()),
        duration_seconds: format_duration_seconds(ictx.duration()),
        stream_duration_seconds: stream_duration_seconds(stream.duration(), stream.time_base()),
        frame_count_hint: positive_i64(stream.frames()),
        avg_frame_rate: rational_info(stream.avg_frame_rate()),
        time_base: rational_info(stream.time_base()),
    }
}

fn rational_info(rational: ffmpeg::Rational) -> RationalInfo {
    let numerator = rational.numerator();
    let denominator = rational.denominator();

    RationalInfo {
        numerator,
        denominator,
        value: if denominator != 0 {
            Some(numerator as f64 / denominator as f64)
        } else {
            None
        },
    }
}

fn format_duration_seconds(duration: i64) -> Option<f64> {
    positive_i64(duration).map(|duration| duration as f64 / AV_TIME_BASE_SECONDS)
}

fn stream_duration_seconds(duration: i64, time_base: ffmpeg::Rational) -> Option<f64> {
    let denominator = time_base.denominator();
    if duration <= 0 || denominator == 0 {
        return None;
    }

    Some(duration as f64 * time_base.numerator() as f64 / denominator as f64)
}

fn positive_i64(value: i64) -> Option<i64> {
    (value > 0).then_some(value)
}
