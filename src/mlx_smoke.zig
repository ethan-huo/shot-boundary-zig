const std = @import("std");

const c = @cImport({
    @cInclude("mlx/c/mlx.h");
});

const MlxError = error{MlxCallFailed};

pub fn main() !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    var version = c.mlx_string_new();
    defer _ = c.mlx_string_free(version);
    try check(c.mlx_version(&version), "mlx_version");

    var metal_available = false;
    try check(c.mlx_metal_is_available(&metal_available), "mlx_metal_is_available");

    var gpu_count: c_int = 0;
    try check(c.mlx_device_count(&gpu_count, c.MLX_GPU), "mlx_device_count");

    const stream = c.mlx_default_cpu_stream_new();
    defer _ = c.mlx_stream_free(stream);

    var data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]c_int{ 2, 3 };
    const arr = c.mlx_array_new_data(&data[0], &shape[0], shape.len, c.MLX_FLOAT32);
    defer _ = c.mlx_array_free(arr);

    const two = c.mlx_array_new_float32(2);
    defer _ = c.mlx_array_free(two);

    var half = c.mlx_array_new();
    defer _ = c.mlx_array_free(half);
    try check(c.mlx_divide(&half, arr, two, stream), "mlx_divide");
    try check(c.mlx_synchronize(stream), "mlx_synchronize");

    var rendered = c.mlx_string_new();
    defer _ = c.mlx_string_free(rendered);
    try check(c.mlx_array_tostring(&rendered, half), "mlx_array_tostring");

    try stdout.print(
        \\MLX version: {s}
        \\Metal available: {}
        \\GPU count: {d}
        \\Zig MLX-C result:
        \\{s}
        \\
    , .{
        std.mem.span(c.mlx_string_data(version)),
        metal_available,
        gpu_count,
        std.mem.span(c.mlx_string_data(rendered)),
    });
    try stdout.flush();
}

fn check(rc: c_int, context: []const u8) MlxError!void {
    if (rc == 0) return;
    std.debug.print("MLX-C call failed: {s} returned {d}\n", .{ context, rc });
    return error.MlxCallFailed;
}
