//! Platform runtime-backed segmentation orchestration.

const std = @import("std");
const builtin = @import("builtin");
const segment_core = @import("segment_core.zig");
const spec = @import("spec");
const time_util = @import("time_util.zig");

pub const default_window_batch_size: usize = switch (builtin.target.os.tag) {
    .macos => 2,
    else => 1,
};

pub const SegmentOptions = struct {
    threshold: f32 = spec.default_scene_threshold,
    window_batch_size: usize = default_window_batch_size,
};

pub const SegmentPredictions = struct {
    single_frame: []f32,
    many_hot: []f32,

    pub fn deinit(self: SegmentPredictions, allocator: std.mem.Allocator) void {
        allocator.free(self.single_frame);
        allocator.free(self.many_hot);
    }
};

pub const SegmentTimings = struct {
    windowing_ms: f64,
    inference_ms: f64,
    postprocess_ms: f64,
    total_ms: f64,
};

pub const SegmentFramesReport = struct {
    frame_count: usize,
    predictions: SegmentPredictions,
    scenes: []const segment_core.Scene,
    timings: SegmentTimings,

    pub fn deinit(self: SegmentFramesReport, allocator: std.mem.Allocator) void {
        self.predictions.deinit(allocator);
        allocator.free(self.scenes);
    }
};

pub fn segmentFrames(
    allocator: std.mem.Allocator,
    model: anytype,
    frames_rgb24: []const u8,
    options: SegmentOptions,
) !SegmentFramesReport {
    if (frames_rgb24.len == 0) return error.EmptyFrames;
    if (frames_rgb24.len % spec.frameBytes() != 0) return error.InvalidInput;
    if (options.window_batch_size == 0) return error.InvalidInput;

    const started_at = try std.time.Instant.now();
    const frame_count = frames_rgb24.len / spec.frameBytes();
    const windows = try segment_core.windowSourceIndices(allocator, frame_count);
    defer allocator.free(windows);

    var single_frame: std.ArrayList(f32) = .empty;
    errdefer single_frame.deinit(allocator);
    var many_hot: std.ArrayList(f32) = .empty;
    errdefer many_hot.deinit(allocator);

    var windowing_ms: f64 = 0.0;
    var inference_ms: f64 = 0.0;
    var window_index: usize = 0;
    while (window_index < windows.len) {
        const batch_end = @min(window_index + options.window_batch_size, windows.len);
        const batch = windows[window_index..batch_end];

        const window_started_at = try std.time.Instant.now();
        const window_data = try buildWindowBatch(allocator, frames_rgb24, batch);
        defer allocator.free(window_data);
        windowing_ms += time_util.elapsedMs(window_started_at);

        const inference_started_at = try std.time.Instant.now();
        const predictions = try model.predictBatch(allocator, window_data, batch.len);
        defer predictions.deinit(allocator);
        try single_frame.appendSlice(allocator, predictions.single_frame);
        try many_hot.appendSlice(allocator, predictions.many_hot);
        inference_ms += time_util.elapsedMs(inference_started_at);

        window_index = batch_end;
    }

    single_frame.shrinkRetainingCapacity(frame_count);
    many_hot.shrinkRetainingCapacity(frame_count);

    const postprocess_started_at = try std.time.Instant.now();
    const scenes = try segment_core.predictionsToScenes(allocator, single_frame.items, options.threshold);
    errdefer allocator.free(scenes);
    const postprocess_ms = time_util.elapsedMs(postprocess_started_at);

    const single_frame_output = try single_frame.toOwnedSlice(allocator);
    errdefer allocator.free(single_frame_output);
    const many_hot_output = try many_hot.toOwnedSlice(allocator);
    errdefer allocator.free(many_hot_output);

    return .{
        .frame_count = frame_count,
        .predictions = .{
            .single_frame = single_frame_output,
            .many_hot = many_hot_output,
        },
        .scenes = scenes,
        .timings = .{
            .windowing_ms = windowing_ms,
            .inference_ms = inference_ms,
            .postprocess_ms = postprocess_ms,
            .total_ms = time_util.elapsedMs(started_at),
        },
    };
}

fn buildWindowBatch(
    allocator: std.mem.Allocator,
    frames_rgb24: []const u8,
    windows: []const [spec.window_frames]usize,
) ![]u8 {
    const frame_bytes = spec.frameBytes();
    const output = try allocator.alloc(u8, windows.len * spec.window_frames * frame_bytes);
    var cursor: usize = 0;
    for (windows) |indices| {
        for (indices) |index| {
            const start = index * frame_bytes;
            const end = start + frame_bytes;
            @memcpy(output[cursor .. cursor + frame_bytes], frames_rgb24[start..end]);
            cursor += frame_bytes;
        }
    }
    return output;
}
