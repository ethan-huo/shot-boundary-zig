use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use transnetv2_rs::model::TransNetV2;
use transnetv2_rs::segment::{
    SegmentModelProfileSummary, SegmentOptions, SegmentPredictions, segment_video_with_options,
};
use transnetv2_rs::video::{DecodeSmokeOptions, DecodeSmokeReport, decode_smoke, inspect_video};
use transnetv2_rs::{DEFAULT_SCENE_THRESHOLD, Scene, predictions_to_scenes};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect {
        video: PathBuf,
    },

    #[command(alias = "bench-decode")]
    DecodeSmoke {
        video: PathBuf,

        #[arg(long, default_value_t = 1)]
        runs: usize,

        #[arg(long)]
        max_frames: Option<usize>,
    },

    Scenes {
        predictions: PathBuf,

        #[arg(long, default_value_t = DEFAULT_SCENE_THRESHOLD)]
        threshold: f32,

        #[arg(long, default_value_t = 0)]
        column: usize,
    },

    Segment {
        video: PathBuf,

        #[arg(long)]
        weights: PathBuf,

        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,

        #[arg(long, default_value_t = DEFAULT_SCENE_THRESHOLD)]
        threshold: f32,

        #[arg(long, default_value_t = 1)]
        runs: usize,

        #[arg(long)]
        max_frames: Option<usize>,

        #[arg(long)]
        profile: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Json,
    Txt,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Inspect { video } => {
            let info = inspect_video(video)?;
            print_json(&info)?;
        }
        Command::DecodeSmoke {
            video,
            runs,
            max_frames,
        } => {
            let output = run_decode_smoke(video, runs, max_frames)?;
            print_json(&output)?;
        }
        Command::Scenes {
            predictions,
            threshold,
            column,
        } => {
            let output = run_scenes(predictions, threshold, column)?;
            print_json(&output)?;
        }
        Command::Segment {
            video,
            weights,
            format,
            threshold,
            runs,
            max_frames,
            profile,
        } => {
            let output = run_segment(video, weights, threshold, runs, max_frames, profile)?;
            match format {
                OutputFormat::Json => print_json(&output)?,
                OutputFormat::Txt => print_segment_text(&output)?,
            }
        }
    }

    Ok(())
}

fn run_decode_smoke(
    video: PathBuf,
    runs: usize,
    max_frames: Option<usize>,
) -> Result<DecodeSmokeOutput, Box<dyn std::error::Error>> {
    if runs == 0 {
        return Err("runs must be greater than zero".into());
    }

    let options = DecodeSmokeOptions {
        max_frames,
        ..DecodeSmokeOptions::default()
    };

    let mut reports = Vec::with_capacity(runs);
    for _ in 0..runs {
        reports.push(decode_smoke(&video, options)?);
    }

    let summary = DecodeSmokeSummary::from_reports(&reports);

    Ok(DecodeSmokeOutput {
        runs: reports,
        summary,
    })
}

fn run_scenes(
    predictions: PathBuf,
    threshold: f32,
    column: usize,
) -> Result<ScenesOutput, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(&predictions)?;
    let values = parse_prediction_column(&content, column)?;
    let scenes = predictions_to_scenes(&values, threshold)?
        .into_iter()
        .map(SceneOutput::from_scene)
        .collect::<Vec<_>>();

    Ok(ScenesOutput {
        path: predictions,
        threshold,
        column,
        prediction_count: values.len(),
        scenes,
    })
}

fn run_segment(
    video: PathBuf,
    weights: PathBuf,
    threshold: f32,
    runs: usize,
    max_frames: Option<usize>,
    profile: bool,
) -> Result<SegmentCliOutput, Box<dyn std::error::Error>> {
    if runs == 0 {
        return Err("runs must be greater than zero".into());
    }

    let device = Device::Cpu;
    let options = DecodeSmokeOptions {
        max_frames,
        ..DecodeSmokeOptions::default()
    };
    let mut run_outputs = Vec::with_capacity(runs);

    for run_index in 0..runs {
        let load_started_at = Instant::now();
        let model = TransNetV2::load_from_safetensors(&weights, &device)?;
        let load_model_ms = load_started_at.elapsed().as_secs_f64() * 1_000.0;
        let report = segment_video_with_options(
            &model,
            &video,
            &device,
            options,
            SegmentOptions {
                threshold,
                collect_model_profile: profile,
            },
        )?;
        let total_ms = load_model_ms + report.timings.total_ms;
        let frames_per_second = if total_ms > 0.0 {
            report.frame_count as f64 / (total_ms / 1_000.0)
        } else {
            0.0
        };

        run_outputs.push(SegmentRunOutput {
            run_index,
            source: report.source,
            frame_count: report.frame_count,
            target_width: report.target_width,
            target_height: report.target_height,
            checksum_fnv1a64: report.checksum_fnv1a64,
            limited_by_max_frames: report.limited_by_max_frames,
            predictions: report.predictions,
            scenes: report
                .scenes
                .into_iter()
                .map(SceneOutput::from_scene)
                .collect(),
            model_profile: report.model_profile,
            timings: SegmentRunTimings {
                load_model_ms,
                decode_ms: report.timings.decode_ms,
                windowing_ms: report.timings.windowing_ms,
                inference_ms: report.timings.inference_ms,
                postprocess_ms: report.timings.postprocess_ms,
                total_ms,
            },
            frames_per_second,
        });
    }

    let summary = SegmentSummary::from_runs(&run_outputs);

    Ok(SegmentCliOutput {
        implementation: "rust-candle",
        video,
        weights,
        threshold,
        environment: EnvironmentOutput::current(),
        runs: run_outputs,
        summary,
    })
}

fn parse_prediction_column(content: &str, column: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();

    for (line_index, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let fields = trimmed
            .split(|ch: char| ch == ',' || ch.is_ascii_whitespace())
            .filter(|field| !field.is_empty())
            .collect::<Vec<_>>();

        let raw = fields.get(column).ok_or_else(|| {
            format!(
                "line {} has {} columns, cannot read column {}",
                line_index + 1,
                fields.len(),
                column
            )
        })?;

        let value = raw.parse::<f32>().map_err(|error| {
            format!("line {} has invalid float {raw:?}: {error}", line_index + 1)
        })?;
        values.push(value);
    }

    if values.is_empty() {
        return Err("prediction file did not contain any values".to_string());
    }

    Ok(values)
}

fn print_json<T: Serialize>(value: &T) -> Result<(), serde_json::Error> {
    serde_json::to_writer_pretty(io::stdout(), value)?;
    println!();
    Ok(())
}

fn print_segment_text(output: &SegmentCliOutput) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    let last_run = output
        .runs
        .last()
        .expect("segment output always contains at least one run");

    writeln!(stdout, "implementation: {}", output.implementation)?;
    writeln!(stdout, "video: {}", output.video.display())?;
    writeln!(stdout, "weights: {}", output.weights.display())?;
    writeln!(stdout, "threshold: {}", output.threshold)?;
    writeln!(stdout, "runs: {}", output.runs.len())?;
    writeln!(stdout, "frame_count: {}", last_run.frame_count)?;
    writeln!(stdout, "checksum_fnv1a64: {}", last_run.checksum_fnv1a64)?;
    writeln!(
        stdout,
        "mean_frames_per_second: {:.6}",
        output.summary.mean_frames_per_second
    )?;
    writeln!(stdout, "mean_total_ms: {:.6}", output.summary.mean_total_ms)?;
    writeln!(stdout)?;

    writeln!(stdout, "# scenes")?;
    for scene in &last_run.scenes {
        writeln!(stdout, "{} {}", scene.start, scene.end)?;
    }
    writeln!(stdout)?;

    writeln!(stdout, "# predictions")?;
    writeln!(stdout, "frame single_frame many_hot")?;
    for (index, (single_frame, many_hot)) in last_run
        .predictions
        .single_frame
        .iter()
        .zip(&last_run.predictions.many_hot)
        .enumerate()
    {
        writeln!(stdout, "{index} {single_frame:.8} {many_hot:.8}")?;
    }

    Ok(())
}

#[derive(Debug, Serialize)]
struct DecodeSmokeOutput {
    runs: Vec<DecodeSmokeReport>,
    summary: DecodeSmokeSummary,
}

#[derive(Debug, Serialize)]
struct DecodeSmokeSummary {
    run_count: usize,
    min_frames_per_second: f64,
    max_frames_per_second: f64,
    mean_frames_per_second: f64,
    min_elapsed_ms: f64,
    max_elapsed_ms: f64,
    mean_elapsed_ms: f64,
}

impl DecodeSmokeSummary {
    fn from_reports(reports: &[DecodeSmokeReport]) -> Self {
        let run_count = reports.len();
        let mut min_fps = f64::INFINITY;
        let mut max_fps = f64::NEG_INFINITY;
        let mut sum_fps = 0.0;
        let mut min_elapsed_ms = f64::INFINITY;
        let mut max_elapsed_ms = f64::NEG_INFINITY;
        let mut sum_elapsed_ms = 0.0;

        for report in reports {
            min_fps = min_fps.min(report.frames_per_second);
            max_fps = max_fps.max(report.frames_per_second);
            sum_fps += report.frames_per_second;
            min_elapsed_ms = min_elapsed_ms.min(report.elapsed_ms);
            max_elapsed_ms = max_elapsed_ms.max(report.elapsed_ms);
            sum_elapsed_ms += report.elapsed_ms;
        }

        Self {
            run_count,
            min_frames_per_second: min_fps,
            max_frames_per_second: max_fps,
            mean_frames_per_second: sum_fps / run_count as f64,
            min_elapsed_ms,
            max_elapsed_ms,
            mean_elapsed_ms: sum_elapsed_ms / run_count as f64,
        }
    }
}

#[derive(Debug, Serialize)]
struct SegmentCliOutput {
    implementation: &'static str,
    video: PathBuf,
    weights: PathBuf,
    threshold: f32,
    environment: EnvironmentOutput,
    runs: Vec<SegmentRunOutput>,
    summary: SegmentSummary,
}

#[derive(Debug, Serialize)]
struct EnvironmentOutput {
    crate_version: &'static str,
    rust_profile: &'static str,
    os: &'static str,
    arch: &'static str,
    family: &'static str,
}

impl EnvironmentOutput {
    fn current() -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION"),
            rust_profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            family: std::env::consts::FAMILY,
        }
    }
}

#[derive(Debug, Serialize)]
struct SegmentRunOutput {
    run_index: usize,
    source: transnetv2_rs::video::VideoInfo,
    frame_count: usize,
    target_width: usize,
    target_height: usize,
    checksum_fnv1a64: String,
    limited_by_max_frames: bool,
    predictions: SegmentPredictions,
    scenes: Vec<SceneOutput>,
    model_profile: Option<SegmentModelProfileSummary>,
    timings: SegmentRunTimings,
    frames_per_second: f64,
}

#[derive(Debug, Serialize)]
struct SegmentRunTimings {
    load_model_ms: f64,
    decode_ms: f64,
    windowing_ms: f64,
    inference_ms: f64,
    postprocess_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Serialize)]
struct SegmentSummary {
    run_count: usize,
    min_frames_per_second: f64,
    max_frames_per_second: f64,
    mean_frames_per_second: f64,
    min_total_ms: f64,
    max_total_ms: f64,
    mean_total_ms: f64,
    mean_load_model_ms: f64,
    mean_decode_ms: f64,
    mean_windowing_ms: f64,
    mean_inference_ms: f64,
    mean_postprocess_ms: f64,
}

impl SegmentSummary {
    fn from_runs(runs: &[SegmentRunOutput]) -> Self {
        let run_count = runs.len();
        let mut min_fps = f64::INFINITY;
        let mut max_fps = f64::NEG_INFINITY;
        let mut sum_fps = 0.0;
        let mut min_total_ms = f64::INFINITY;
        let mut max_total_ms = f64::NEG_INFINITY;
        let mut sum_total_ms = 0.0;
        let mut sum_load_model_ms = 0.0;
        let mut sum_decode_ms = 0.0;
        let mut sum_windowing_ms = 0.0;
        let mut sum_inference_ms = 0.0;
        let mut sum_postprocess_ms = 0.0;

        for run in runs {
            min_fps = min_fps.min(run.frames_per_second);
            max_fps = max_fps.max(run.frames_per_second);
            sum_fps += run.frames_per_second;
            min_total_ms = min_total_ms.min(run.timings.total_ms);
            max_total_ms = max_total_ms.max(run.timings.total_ms);
            sum_total_ms += run.timings.total_ms;
            sum_load_model_ms += run.timings.load_model_ms;
            sum_decode_ms += run.timings.decode_ms;
            sum_windowing_ms += run.timings.windowing_ms;
            sum_inference_ms += run.timings.inference_ms;
            sum_postprocess_ms += run.timings.postprocess_ms;
        }

        Self {
            run_count,
            min_frames_per_second: min_fps,
            max_frames_per_second: max_fps,
            mean_frames_per_second: sum_fps / run_count as f64,
            min_total_ms,
            max_total_ms,
            mean_total_ms: sum_total_ms / run_count as f64,
            mean_load_model_ms: sum_load_model_ms / run_count as f64,
            mean_decode_ms: sum_decode_ms / run_count as f64,
            mean_windowing_ms: sum_windowing_ms / run_count as f64,
            mean_inference_ms: sum_inference_ms / run_count as f64,
            mean_postprocess_ms: sum_postprocess_ms / run_count as f64,
        }
    }
}

#[derive(Debug, Serialize)]
struct ScenesOutput {
    path: PathBuf,
    threshold: f32,
    column: usize,
    prediction_count: usize,
    scenes: Vec<SceneOutput>,
}

#[derive(Debug, Serialize)]
struct SceneOutput {
    start: usize,
    end: usize,
}

impl SceneOutput {
    fn from_scene(scene: Scene) -> Self {
        Self {
            start: scene.start(),
            end: scene.end(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prediction_column_from_whitespace_and_csv() {
        let values = parse_prediction_column("0.1 0.9\n0.2,0.8\n", 1).unwrap();

        assert_eq!(values, vec![0.9, 0.8]);
    }

    #[test]
    fn rejects_missing_prediction_column() {
        let error = parse_prediction_column("0.1\n", 1).unwrap_err();

        assert!(error.contains("cannot read column 1"));
    }
}
