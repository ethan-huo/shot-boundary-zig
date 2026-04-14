//! Application entry point.

const std = @import("std");
const builtin = @import("builtin");
const clap = @import("clap");
const runtime_model = @import("runtime_model.zig");
const runtime_segment = @import("runtime_segment.zig");
const segment_core = @import("segment_core.zig");
const spec = @import("spec");
const video = @import("video.zig");

const default_scene_threshold: f32 = spec.default_scene_threshold;

const usage =
    \\usage:
    \\  shot_boundary_zig env
    \\  shot_boundary_zig decode-smoke <video> [options]
    \\  shot_boundary_zig segment <video> --weights <path> [options]
    \\
    \\decode-smoke options:
    \\  --format <json|txt>          output format, default json
    \\  --max-frames <n>             optional decode limit, must be > 0
    \\
    \\segment options:
    \\  --weights <path>             required platform model path (.onnx on Linux, safetensors on macOS)
    \\  --format <json|txt>          output format, default json
    \\  --threshold <0..1>           scene threshold, default 0.296
    \\  --runs <n>                   run count, must be > 0, default 1
    \\  --max-frames <n>             optional decode limit, must be > 0
    \\  --window-batch-size <n>      optional inference batch size, must be > 0
    \\  --profile                    request model profiling
    \\
;

const CliError = error{
    MissingCommand,
    UnknownCommand,
    MissingVideo,
    MissingWeights,
    MissingValue,
    DuplicateOption,
    UnknownOption,
    UnexpectedArgument,
    UnsupportedProfile,
    InvalidFormat,
    InvalidRuns,
    InvalidMaxFrames,
    InvalidWindowBatchSize,
    InvalidThreshold,
};

const Command = union(enum) {
    decode_smoke: DecodeSmokeOptions,
    environment,
    help,
    segment: SegmentOptions,
};

pub const OutputFormat = enum {
    json,
    txt,

    pub fn jsonStringify(self: OutputFormat, jw: *std.json.Stringify) std.json.Stringify.Error!void {
        try jw.write(@tagName(self));
    }
};

const SegmentOptions = struct {
    video: []const u8,
    weights: []const u8,
    format: OutputFormat = .json,
    threshold: f32 = default_scene_threshold,
    runs: usize = 1,
    max_frames: ?usize = null,
    window_batch_size: ?usize = null,
    profile: bool = false,
};

const DecodeSmokeOptions = struct {
    video: []const u8,
    format: OutputFormat = .json,
    max_frames: ?usize = null,
};

const decode_smoke_params = clap.parseParamsComptime(
    \\-h, --help                   Display this help and exit.
    \\    --format <format>...     Output format: json or txt.
    \\    --max-frames <usize>...  Optional decode limit, must be > 0.
    \\<video>
    \\<extra>...
    \\
);

const decode_smoke_parsers = .{
    .format = clap.parsers.enumeration(OutputFormat),
    .usize = clap.parsers.int(usize, 10),
    .video = clap.parsers.string,
    .extra = clap.parsers.string,
};

const segment_params = clap.parseParamsComptime(
    \\-h, --help                   Display this help and exit.
    \\    --weights <str>...       Required platform model path.
    \\    --format <format>...     Output format: json or txt.
    \\    --threshold <f32>...     Scene threshold in [0, 1].
    \\    --runs <usize>...        Run count, must be > 0.
    \\    --max-frames <usize>...  Optional decode limit, must be > 0.
    \\    --window-batch-size <usize>... Optional inference batch size, must be > 0.
    \\    --profile                Request model profiling.
    \\<video>
    \\<extra>...
    \\
);

const segment_parsers = .{
    .str = clap.parsers.string,
    .format = clap.parsers.enumeration(OutputFormat),
    .f32 = clap.parsers.float(f32),
    .usize = clap.parsers.int(usize, 10),
    .video = clap.parsers.string,
    .extra = clap.parsers.string,
};

const EnvironmentOutput = struct {
    zig_version: []const u8,
    optimize_mode: []const u8,
    os: []const u8,
    arch: []const u8,
    abi: []const u8,

    fn current() EnvironmentOutput {
        return .{
            .zig_version = builtin.zig_version_string,
            .optimize_mode = @tagName(builtin.mode),
            .os = @tagName(builtin.os.tag),
            .arch = @tagName(builtin.cpu.arch),
            .abi = @tagName(builtin.abi),
        };
    }
};

const EnvCommandOutput = struct {
    implementation: []const u8 = "zig-phase0",
    command: []const u8 = "env",
    environment: EnvironmentOutput,
};

const DecodeSmokeOutput = struct {
    implementation: []const u8 = "zig-phase2",
    command: []const u8 = "decode-smoke",
    report: video.DecodeReport,
    environment: EnvironmentOutput,
};

const SegmentPredictionsOutput = struct {
    single_frame: []const f32,
    many_hot: []const f32,
};

const SegmentSourceOutput = struct {
    path: []const u8,
};

const SegmentModelProfileOutput = struct {};

const SegmentRunTimings = struct {
    load_model_ms: f64,
    decode_ms: f64,
    windowing_ms: f64,
    inference_ms: f64,
    postprocess_ms: f64,
    total_ms: f64,
};

const SegmentRunOutput = struct {
    run_index: usize,
    source: SegmentSourceOutput,
    frame_count: usize,
    target_width: usize,
    target_height: usize,
    checksum_fnv1a64: []const u8,
    limited_by_max_frames: bool,
    predictions: SegmentPredictionsOutput,
    scenes: []const segment_core.Scene,
    model_profile: ?SegmentModelProfileOutput = null,
    timings: SegmentRunTimings,
    frames_per_second: f64,

    fn deinit(self: SegmentRunOutput, allocator: std.mem.Allocator) void {
        allocator.free(self.checksum_fnv1a64);
        allocator.free(self.predictions.single_frame);
        allocator.free(self.predictions.many_hot);
        allocator.free(self.scenes);
    }
};

const SegmentSummary = struct {
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

    fn fromRuns(runs: []const SegmentRunOutput) SegmentSummary {
        var min_fps = std.math.inf(f64);
        var max_fps = -std.math.inf(f64);
        var sum_fps: f64 = 0.0;
        var min_total_ms = std.math.inf(f64);
        var max_total_ms = -std.math.inf(f64);
        var sum_total_ms: f64 = 0.0;
        var sum_load_model_ms: f64 = 0.0;
        var sum_decode_ms: f64 = 0.0;
        var sum_windowing_ms: f64 = 0.0;
        var sum_inference_ms: f64 = 0.0;
        var sum_postprocess_ms: f64 = 0.0;

        for (runs) |run| {
            min_fps = @min(min_fps, run.frames_per_second);
            max_fps = @max(max_fps, run.frames_per_second);
            sum_fps += run.frames_per_second;
            min_total_ms = @min(min_total_ms, run.timings.total_ms);
            max_total_ms = @max(max_total_ms, run.timings.total_ms);
            sum_total_ms += run.timings.total_ms;
            sum_load_model_ms += run.timings.load_model_ms;
            sum_decode_ms += run.timings.decode_ms;
            sum_windowing_ms += run.timings.windowing_ms;
            sum_inference_ms += run.timings.inference_ms;
            sum_postprocess_ms += run.timings.postprocess_ms;
        }

        const run_count_float: f64 = @floatFromInt(runs.len);
        return .{
            .run_count = runs.len,
            .min_frames_per_second = min_fps,
            .max_frames_per_second = max_fps,
            .mean_frames_per_second = sum_fps / run_count_float,
            .min_total_ms = min_total_ms,
            .max_total_ms = max_total_ms,
            .mean_total_ms = sum_total_ms / run_count_float,
            .mean_load_model_ms = sum_load_model_ms / run_count_float,
            .mean_decode_ms = sum_decode_ms / run_count_float,
            .mean_windowing_ms = sum_windowing_ms / run_count_float,
            .mean_inference_ms = sum_inference_ms / run_count_float,
            .mean_postprocess_ms = sum_postprocess_ms / run_count_float,
        };
    }
};

const SegmentCliOutput = struct {
    implementation: []const u8 = runtime_model.implementation,
    video: []const u8,
    weights: []const u8,
    threshold: f32,
    environment: EnvironmentOutput,
    runs: []const SegmentRunOutput,
    summary: SegmentSummary,
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const command = parseCli(allocator, args[1..]) catch |err| {
        try writeCliError(err);
        std.process.exit(2);
    };

    switch (command) {
        .decode_smoke => |options| try runDecodeSmoke(allocator, options),
        .environment => try writeJson(EnvCommandOutput{ .environment = .current() }),
        .help => try writeUsage(),
        .segment => |options| try runSegment(allocator, options),
    }
}

fn parseCli(allocator: std.mem.Allocator, args: []const []const u8) CliError!Command {
    if (args.len == 0) return error.MissingCommand;

    if (std.mem.eql(u8, args[0], "env")) {
        if (args.len != 1) return error.UnexpectedArgument;
        return .environment;
    }

    if (std.mem.eql(u8, args[0], "decode-smoke") or std.mem.eql(u8, args[0], "bench-decode")) {
        return parseDecodeSmokeCommand(allocator, args[1..]);
    }

    if (std.mem.eql(u8, args[0], "help") or
        std.mem.eql(u8, args[0], "--help") or
        std.mem.eql(u8, args[0], "-h"))
    {
        if (args.len != 1) return error.UnexpectedArgument;
        return .help;
    }

    if (std.mem.eql(u8, args[0], "segment")) {
        return parseSegmentCommand(allocator, args[1..]);
    }

    return error.UnknownCommand;
}

fn parseDecodeSmokeCommand(allocator: std.mem.Allocator, args: []const []const u8) CliError!Command {
    var iter: clap.args.SliceIterator = .{ .args = args };
    var diag: clap.Diagnostic = .{};
    var res = clap.parseEx(clap.Help, &decode_smoke_params, decode_smoke_parsers, &iter, .{
        .diagnostic = &diag,
        .allocator = allocator,
        .assignment_separators = "=",
    }) catch |err| return mapDecodeSmokeClapError(err, diag);
    defer res.deinit();

    if (res.args.help != 0) return .help;
    if (res.args.help > 1) return error.DuplicateOption;
    if (res.positionals[1].len != 0) return error.UnexpectedArgument;

    return .{
        .decode_smoke = .{
            .video = res.positionals[0] orelse return error.MissingVideo,
            .format = (try singleOptional(OutputFormat, res.args.format)) orelse OutputFormat.json,
            .max_frames = try positiveOptionalUsizeOption(@field(res.args, "max-frames"), error.InvalidMaxFrames),
        },
    };
}

fn parseSegmentCommand(allocator: std.mem.Allocator, args: []const []const u8) CliError!Command {
    var iter: clap.args.SliceIterator = .{ .args = args };
    var diag: clap.Diagnostic = .{};
    var res = clap.parseEx(clap.Help, &segment_params, segment_parsers, &iter, .{
        .diagnostic = &diag,
        .allocator = allocator,
        .assignment_separators = "=",
    }) catch |err| return mapSegmentClapError(err, diag);
    defer res.deinit();

    if (res.args.help != 0) return .help;
    if (res.args.help > 1) return error.DuplicateOption;
    if (res.positionals[1].len != 0) return error.UnexpectedArgument;

    const video_path = res.positionals[0] orelse return error.MissingVideo;
    const weights = (try singleOptional([]const u8, res.args.weights)) orelse return error.MissingWeights;
    const format = (try singleOptional(OutputFormat, res.args.format)) orelse OutputFormat.json;
    const threshold = try thresholdOption(res.args.threshold);
    const runs = try positiveUsizeOption(res.args.runs, 1, error.InvalidRuns);
    const max_frames = try positiveOptionalUsizeOption(@field(res.args, "max-frames"), error.InvalidMaxFrames);
    const window_batch_size = try positiveOptionalUsizeOption(
        @field(res.args, "window-batch-size"),
        error.InvalidWindowBatchSize,
    );

    if (res.args.profile > 1) return error.DuplicateOption;

    return .{
        .segment = .{
            .video = video_path,
            .weights = weights,
            .format = format,
            .threshold = threshold,
            .runs = runs,
            .max_frames = max_frames,
            .window_batch_size = window_batch_size,
            .profile = res.args.profile != 0,
        },
    };
}

fn mapDecodeSmokeClapError(err: anyerror, diag: clap.Diagnostic) CliError {
    return switch (err) {
        error.MissingValue => error.MissingValue,
        error.InvalidArgument => if (diag.name.long != null or diag.name.short != null)
            error.UnknownOption
        else
            error.UnexpectedArgument,
        error.NameNotPartOfEnum => if (diagLongName(diag, "format"))
            error.InvalidFormat
        else
            error.UnexpectedArgument,
        error.InvalidCharacter, error.Overflow => if (diagLongName(diag, "max-frames"))
            error.InvalidMaxFrames
        else
            error.UnexpectedArgument,
        else => error.UnexpectedArgument,
    };
}

fn singleOptional(comptime T: type, values: []const T) CliError!?T {
    if (values.len == 0) return null;
    if (values.len > 1) return error.DuplicateOption;
    return values[0];
}

fn thresholdOption(values: []const f32) CliError!f32 {
    const threshold = (try singleOptional(f32, values)) orelse default_scene_threshold;
    if (!std.math.isFinite(threshold) or threshold < 0.0 or threshold > 1.0) return error.InvalidThreshold;
    return threshold;
}

fn positiveUsizeOption(values: []const usize, default: usize, invalid: CliError) CliError!usize {
    const value = (try singleOptional(usize, values)) orelse default;
    if (value == 0) return invalid;
    return value;
}

fn positiveOptionalUsizeOption(values: []const usize, invalid: CliError) CliError!?usize {
    const value = (try singleOptional(usize, values)) orelse return null;
    if (value == 0) return invalid;
    return value;
}

fn mapSegmentClapError(err: anyerror, diag: clap.Diagnostic) CliError {
    return switch (err) {
        error.MissingValue => error.MissingValue,
        error.InvalidArgument => if (diag.name.long != null or diag.name.short != null)
            error.UnknownOption
        else
            error.UnexpectedArgument,
        error.NameNotPartOfEnum => if (diagLongName(diag, "format")) error.InvalidFormat else error.UnexpectedArgument,
        error.InvalidCharacter, error.Overflow => if (diagLongName(diag, "threshold"))
            error.InvalidThreshold
        else if (diagLongName(diag, "runs"))
            error.InvalidRuns
        else if (diagLongName(diag, "max-frames"))
            error.InvalidMaxFrames
        else if (diagLongName(diag, "window-batch-size"))
            error.InvalidWindowBatchSize
        else
            error.UnexpectedArgument,
        else => error.UnexpectedArgument,
    };
}

fn diagLongName(diag: clap.Diagnostic, expected: []const u8) bool {
    const actual = diag.name.long orelse return false;
    return std.mem.eql(u8, actual, expected);
}

fn runSegment(allocator: std.mem.Allocator, options: SegmentOptions) !void {
    if (options.profile) return writeRuntimeCliError(error.UnsupportedProfile);

    const window_batch_size = options.window_batch_size orelse runtime_segment.default_window_batch_size;
    var runs: std.ArrayList(SegmentRunOutput) = .empty;
    defer {
        for (runs.items) |run| run.deinit(allocator);
        runs.deinit(allocator);
    }

    for (0..options.runs) |run_index| {
        const load_started_at = try std.time.Instant.now();
        var model = try runtime_model.TransNetV2.load(allocator, options.weights);
        defer model.deinit();
        const load_model_ms = elapsedMs(load_started_at);

        const decoded = try video.decodeRgb24(allocator, options.video, .{ .max_frames = options.max_frames });
        defer decoded.deinit(allocator);

        const report = try runtime_segment.segmentFrames(allocator, &model, decoded.data, .{
            .threshold = options.threshold,
            .window_batch_size = window_batch_size,
        });
        var report_owned = true;
        errdefer if (report_owned) report.deinit(allocator);

        const total_ms = load_model_ms + decoded.elapsed_ms + report.timings.total_ms;
        const frames_per_second = if (total_ms > 0.0)
            @as(f64, @floatFromInt(report.frame_count)) / (total_ms / 1_000.0)
        else
            0.0;

        const checksum = try allocator.dupe(u8, &decoded.checksum_fnv1a64);
        var checksum_owned = true;
        errdefer if (checksum_owned) allocator.free(checksum);

        const run: SegmentRunOutput = .{
            .run_index = run_index,
            .source = .{ .path = decoded.path },
            .frame_count = report.frame_count,
            .target_width = decoded.target_width,
            .target_height = decoded.target_height,
            .checksum_fnv1a64 = checksum,
            .limited_by_max_frames = decoded.limited_by_max_frames,
            .predictions = .{
                .single_frame = report.predictions.single_frame,
                .many_hot = report.predictions.many_hot,
            },
            .scenes = report.scenes,
            .timings = .{
                .load_model_ms = load_model_ms,
                .decode_ms = decoded.elapsed_ms,
                .windowing_ms = report.timings.windowing_ms,
                .inference_ms = report.timings.inference_ms,
                .postprocess_ms = report.timings.postprocess_ms,
                .total_ms = total_ms,
            },
            .frames_per_second = frames_per_second,
        };
        try runs.append(allocator, run);
        report_owned = false;
        checksum_owned = false;
    }

    const output: SegmentCliOutput = .{
        .video = options.video,
        .weights = options.weights,
        .threshold = options.threshold,
        .environment = .current(),
        .runs = runs.items,
        .summary = .fromRuns(runs.items),
    };

    switch (options.format) {
        .json => try writeJson(output),
        .txt => try writeSegmentText(output),
    }
}

fn runDecodeSmoke(allocator: std.mem.Allocator, options: DecodeSmokeOptions) !void {
    const report = try video.decodeReport(allocator, options.video, .{ .max_frames = options.max_frames });
    defer allocator.free(report.checksum_fnv1a64);
    const output: DecodeSmokeOutput = .{
        .report = report,
        .environment = .current(),
    };

    switch (options.format) {
        .json => try writeJson(output),
        .txt => try writeDecodeSmokeText(output),
    }
}

fn writeDecodeSmokeText(output: DecodeSmokeOutput) !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    defer stdout_writer.interface.flush() catch {};
    const stdout = &stdout_writer.interface;

    try stdout.print("implementation: {s}\n", .{output.implementation});
    try stdout.print("video: {s}\n", .{output.report.video});
    try stdout.print("decoded_frames: {d}\n", .{output.report.decoded_frames});
    try stdout.print("decoded_rgb_bytes: {d}\n", .{output.report.decoded_rgb_bytes});
    try stdout.print("checksum_fnv1a64: {s}\n", .{output.report.checksum_fnv1a64});
    try stdout.print("frames_per_second: {d}\n", .{output.report.frames_per_second});
    try stdout.print("limited_by_max_frames: {}\n", .{output.report.limited_by_max_frames});
}

fn writeJson(value: anytype) !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    defer stdout_writer.interface.flush() catch {};

    var jw: std.json.Stringify = .{
        .writer = &stdout_writer.interface,
        .options = .{ .whitespace = .indent_2 },
    };
    try jw.write(value);
    try stdout_writer.interface.writeByte('\n');
}

fn writeSegmentText(output: SegmentCliOutput) !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    defer stdout_writer.interface.flush() catch {};
    const stdout = &stdout_writer.interface;
    const last_run = output.runs[output.runs.len - 1];

    try stdout.print("implementation: {s}\n", .{output.implementation});
    try stdout.print("video: {s}\n", .{output.video});
    try stdout.print("weights: {s}\n", .{output.weights});
    try stdout.print("threshold: {d}\n", .{output.threshold});
    try stdout.print("runs: {d}\n", .{output.runs.len});
    try stdout.print("frame_count: {d}\n", .{last_run.frame_count});
    try stdout.print("checksum_fnv1a64: {s}\n", .{last_run.checksum_fnv1a64});
    try stdout.print("mean_frames_per_second: {d:.6}\n", .{output.summary.mean_frames_per_second});
    try stdout.print("mean_total_ms: {d:.6}\n", .{output.summary.mean_total_ms});
    try stdout.print("zig_version: {s}\n", .{output.environment.zig_version});
    try stdout.print("target: {s}-{s}-{s}\n", .{
        output.environment.arch,
        output.environment.os,
        output.environment.abi,
    });
    try stdout.writeAll("\n# scenes\n");
    for (last_run.scenes) |scene| {
        try stdout.print("{d} {d}\n", .{ scene.start, scene.end });
    }
    try stdout.writeAll("\n# predictions\nframe single_frame many_hot\n");
    for (last_run.predictions.single_frame, last_run.predictions.many_hot, 0..) |single, many, index| {
        try stdout.print("{d} {d:.8} {d:.8}\n", .{ index, single, many });
    }
}

fn writeUsage() !void {
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    defer stdout_writer.interface.flush() catch {};

    try stdout_writer.interface.writeAll(usage);
}

fn writeCliError(err: CliError) !void {
    var stderr_buf: [4096]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buf);
    defer stderr_writer.interface.flush() catch {};
    const stderr = &stderr_writer.interface;

    try stderr.print("error: {s}\n\n", .{cliErrorMessage(err)});
    try stderr.writeAll(usage);
}

fn cliErrorMessage(err: CliError) []const u8 {
    return switch (err) {
        error.MissingCommand => "missing command",
        error.UnknownCommand => "unknown command",
        error.MissingVideo => "segment requires a video path",
        error.MissingWeights => "segment requires --weights <path>",
        error.MissingValue => "option requires a value",
        error.DuplicateOption => "duplicate option",
        error.UnknownOption => "unknown option",
        error.UnexpectedArgument => "unexpected positional argument",
        error.UnsupportedProfile => "runtime profiling is not supported yet",
        error.InvalidFormat => "format must be json or txt",
        error.InvalidRuns => "runs must be a positive integer",
        error.InvalidMaxFrames => "max-frames must be a positive integer",
        error.InvalidWindowBatchSize => "window-batch-size must be a positive integer",
        error.InvalidThreshold => "threshold must be a finite probability in [0, 1]",
    };
}

fn writeRuntimeCliError(err: CliError) !void {
    try writeCliError(err);
    std.process.exit(2);
}

fn elapsedMs(started_at: std.time.Instant) f64 {
    const now = std.time.Instant.now() catch unreachable;
    return @as(f64, @floatFromInt(now.since(started_at))) / std.time.ns_per_ms;
}

test "parse segment command from README shape" {
    const command = try parseCli(std.testing.allocator, &.{
        "segment",
        "assets/333.mp4",
        "--weights",
        "target/models/transnetv2.safetensors",
        "--runs",
        "5",
        "--format",
        "json",
    });

    const options = command.segment;
    try std.testing.expectEqualStrings("assets/333.mp4", options.video);
    try std.testing.expectEqualStrings("target/models/transnetv2.safetensors", options.weights);
    try std.testing.expectEqual(@as(usize, 5), options.runs);
    try std.testing.expectEqual(OutputFormat.json, options.format);
    try std.testing.expectEqual(default_scene_threshold, options.threshold);
}

test "parse segment accepts runtime options" {
    const command = try parseCli(std.testing.allocator, &.{
        "segment",
        "assets/333.mp4",
        "--weights",
        "target/models/transnetv2.safetensors",
        "--format",
        "txt",
        "--threshold",
        "0.35",
        "--runs",
        "2",
        "--max-frames",
        "100",
        "--window-batch-size",
        "2",
        "--profile",
    });

    const options = command.segment;
    try std.testing.expectEqual(OutputFormat.txt, options.format);
    try std.testing.expectEqual(@as(f32, 0.35), options.threshold);
    try std.testing.expectEqual(@as(usize, 2), options.runs);
    try std.testing.expectEqual(@as(?usize, 100), options.max_frames);
    try std.testing.expectEqual(@as(?usize, 2), options.window_batch_size);
    try std.testing.expect(options.profile);
}

test "parse segment requires weights" {
    try std.testing.expectError(error.MissingWeights, parseCli(std.testing.allocator, &.{
        "segment",
        "assets/333.mp4",
    }));
}

test "parse segment rejects invalid numeric options" {
    try std.testing.expectError(error.InvalidRuns, parseCli(std.testing.allocator, &.{
        "segment",
        "assets/333.mp4",
        "--weights",
        "target/models/transnetv2.safetensors",
        "--runs",
        "0",
    }));

    try std.testing.expectError(error.InvalidThreshold, parseCli(std.testing.allocator, &.{
        "segment",
        "assets/333.mp4",
        "--weights",
        "target/models/transnetv2.safetensors",
        "--threshold",
        "1.1",
    }));
}

test "parse segment rejects duplicate and unknown options" {
    try std.testing.expectError(error.DuplicateOption, parseCli(std.testing.allocator, &.{
        "segment",
        "assets/333.mp4",
        "--weights",
        "one.safetensors",
        "--weights",
        "two.safetensors",
    }));

    try std.testing.expectError(error.UnknownOption, parseCli(std.testing.allocator, &.{
        "segment",
        "assets/333.mp4",
        "--weights",
        "target/models/transnetv2.safetensors",
        "--unknown",
        "value",
    }));
}

test "parse decode-smoke accepts ffmpeg edge options" {
    const command = try parseCli(std.testing.allocator, &.{
        "decode-smoke",
        "assets/333.mp4",
        "--format",
        "txt",
        "--max-frames",
        "10",
    });

    const options = command.decode_smoke;
    try std.testing.expectEqualStrings("assets/333.mp4", options.video);
    try std.testing.expectEqual(OutputFormat.txt, options.format);
    try std.testing.expectEqual(@as(?usize, 10), options.max_frames);
}

test "imported module tests are reachable" {
    std.testing.refAllDecls(runtime_segment);
    std.testing.refAllDecls(segment_core);
    std.testing.refAllDecls(video);
}
