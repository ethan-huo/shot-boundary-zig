//! Platform-selected shot-boundary model edge.

const runtime_model_impl = @import("runtime_model_impl");

pub const Predictions = runtime_model_impl.Predictions;
pub const AutoShot = runtime_model_impl.AutoShot;
pub const TransNetV2 = runtime_model_impl.TransNetV2;

pub const implementation: []const u8 = runtime_model_impl.implementation;
