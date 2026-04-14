//! Compile-time placeholder for unsupported platform targets.

comptime {
    @compileError("shot-boundary runtime is only implemented for Linux/ONNX and macOS/MLX");
}
