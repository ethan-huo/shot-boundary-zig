//! Small time helpers shared by effectful orchestration edges.

const std = @import("std");

pub fn elapsedMs(started_at: std.time.Instant) f64 {
    const now = std.time.Instant.now() catch unreachable;
    return @as(f64, @floatFromInt(now.since(started_at))) / std.time.ns_per_ms;
}
