//! Shared shot-boundary runtime model dimensions.

pub const input_width: usize = 48;
pub const input_height: usize = 27;
pub const input_channels: usize = 3;
pub const window_frames: usize = 100;
pub const context_frames: usize = 25;
pub const output_frames_per_window: usize = 50;
pub const autoshot_scene_threshold: f32 = 0.296;
pub const transnetv2_scene_threshold: f32 = 0.02;
pub const default_scene_threshold: f32 = autoshot_scene_threshold;

pub const ModelInputSpec = struct {
    width: usize = input_width,
    height: usize = input_height,
    channels: usize = input_channels,
    window_frames: usize = window_frames,
    context_frames: usize = context_frames,
    output_frames_per_window: usize = output_frames_per_window,
};

pub fn frameBytes() usize {
    return input_width * input_height * input_channels;
}
