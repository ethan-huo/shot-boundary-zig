//! ONNX Runtime AutoShot/TransNetV2-compatible model edge.

const std = @import("std");
const runtime_options = @import("runtime_options");
const spec = @import("spec");

const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

const input_name = "frames";
const output_single_frame_name = "single_frame";
const output_many_hot_name = "many_hot";

pub const implementation: []const u8 = if (runtime_options.onnxruntime_cuda)
    "zig-onnxruntime-cuda"
else
    "zig-onnxruntime";

pub const OnnxModelError = error{
    OrtApiUnavailable,
    OrtCallFailed,
    InvalidInput,
    InvalidShape,
    InvalidRank,
    InvalidDtype,
    NullData,
};

pub const Predictions = struct {
    single_frame: []f32,
    many_hot: []f32,

    pub fn deinit(self: Predictions, allocator: std.mem.Allocator) void {
        allocator.free(self.single_frame);
        allocator.free(self.many_hot);
    }
};

pub const TransNetV2 = struct {
    api: *const c.OrtApi,
    env: *c.OrtEnv,
    session: *c.OrtSession,
    memory_info: *c.OrtMemoryInfo,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !TransNetV2 {
        const api = getApi() orelse return error.OrtApiUnavailable;
        try validateApi(api);

        var env: ?*c.OrtEnv = null;
        try check(api, api.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "shot-boundary", &env), "CreateEnv");
        errdefer api.ReleaseEnv.?(env);

        var session_options: ?*c.OrtSessionOptions = null;
        try check(api, api.CreateSessionOptions.?(&session_options), "CreateSessionOptions");
        defer api.ReleaseSessionOptions.?(session_options);

        var cuda_options: ?*c.OrtCUDAProviderOptionsV2 = null;
        if (runtime_options.onnxruntime_cuda) {
            try check(api, api.CreateCUDAProviderOptions.?(&cuda_options), "CreateCUDAProviderOptions");
            defer api.ReleaseCUDAProviderOptions.?(cuda_options);

            try check(
                api,
                api.SessionOptionsAppendExecutionProvider_CUDA_V2.?(session_options, cuda_options),
                "SessionOptionsAppendExecutionProvider_CUDA_V2",
            );
        }

        var memory_info: ?*c.OrtMemoryInfo = null;
        try check(
            api,
            api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &memory_info),
            "CreateCpuMemoryInfo",
        );
        errdefer api.ReleaseMemoryInfo.?(memory_info);

        const z_path = try allocator.dupeZ(u8, path);
        defer allocator.free(z_path);

        var session: ?*c.OrtSession = null;
        try check(api, api.CreateSession.?(env, z_path.ptr, session_options, &session), "CreateSession");
        errdefer api.ReleaseSession.?(session);

        try validateIo(api, session.?);

        return .{
            .api = api,
            .env = env.?,
            .session = session.?,
            .memory_info = memory_info.?,
        };
    }

    pub fn deinit(self: *TransNetV2) void {
        self.api.ReleaseMemoryInfo.?(self.memory_info);
        self.api.ReleaseSession.?(self.session);
        self.api.ReleaseEnv.?(self.env);
        self.* = undefined;
    }

    pub fn predictBatch(
        self: *const TransNetV2,
        allocator: std.mem.Allocator,
        window_batch_rgb24: []const u8,
        batch_size: usize,
    ) !Predictions {
        if (batch_size == 0) return error.InvalidInput;
        const expected_len = batch_size * spec.window_frames * spec.frameBytes();
        if (window_batch_rgb24.len != expected_len) return error.InvalidInput;

        var shape = [_]i64{
            @intCast(batch_size),
            @intCast(spec.window_frames),
            @intCast(spec.input_height),
            @intCast(spec.input_width),
            @intCast(spec.input_channels),
        };

        var input_value: ?*c.OrtValue = null;
        try check(
            self.api,
            self.api.CreateTensorWithDataAsOrtValue.?(
                self.memory_info,
                @constCast(window_batch_rgb24.ptr),
                window_batch_rgb24.len,
                &shape,
                shape.len,
                c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                &input_value,
            ),
            "CreateTensorWithDataAsOrtValue",
        );
        defer self.api.ReleaseValue.?(input_value);

        const expected_output_len = batch_size * spec.output_frames_per_window;
        const single_frame = try allocator.alloc(f32, expected_output_len);
        errdefer allocator.free(single_frame);
        const many_hot = try allocator.alloc(f32, expected_output_len);
        errdefer allocator.free(many_hot);
        var output_shape = [_]i64{
            @intCast(batch_size),
            @intCast(spec.output_frames_per_window),
        };

        var single_frame_value: ?*c.OrtValue = null;
        try check(
            self.api,
            self.api.CreateTensorWithDataAsOrtValue.?(
                self.memory_info,
                single_frame.ptr,
                single_frame.len * @sizeOf(f32),
                &output_shape,
                output_shape.len,
                c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &single_frame_value,
            ),
            "CreateTensorWithDataAsOrtValue(single_frame)",
        );
        defer self.api.ReleaseValue.?(single_frame_value);

        var many_hot_value: ?*c.OrtValue = null;
        try check(
            self.api,
            self.api.CreateTensorWithDataAsOrtValue.?(
                self.memory_info,
                many_hot.ptr,
                many_hot.len * @sizeOf(f32),
                &output_shape,
                output_shape.len,
                c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &many_hot_value,
            ),
            "CreateTensorWithDataAsOrtValue(many_hot)",
        );
        defer self.api.ReleaseValue.?(many_hot_value);

        const input_names = [_][*:0]const u8{input_name};
        const input_values = [_]?*const c.OrtValue{input_value};
        const output_names = [_][*:0]const u8{ output_single_frame_name, output_many_hot_name };
        var outputs = [_]?*c.OrtValue{ single_frame_value, many_hot_value };

        try check(
            self.api,
            self.api.Run.?(
                self.session,
                null,
                @ptrCast(&input_names),
                @ptrCast(&input_values),
                input_names.len,
                @ptrCast(&output_names),
                output_names.len,
                @ptrCast(&outputs),
            ),
            "Run",
        );

        return .{ .single_frame = single_frame, .many_hot = many_hot };
    }
};

fn getApi() ?*const c.OrtApi {
    const base = c.OrtGetApiBase();
    if (base == null) return null;
    const get_api = base[0].GetApi orelse return null;
    const api = get_api(c.ORT_API_VERSION);
    if (api == null) return null;
    return api;
}

fn validateApi(api: *const c.OrtApi) !void {
    if (api.CreateEnv == null or
        api.ReleaseEnv == null or
        api.CreateSessionOptions == null or
        api.ReleaseSessionOptions == null or
        api.CreateCpuMemoryInfo == null or
        api.ReleaseMemoryInfo == null or
        api.CreateSession == null or
        api.ReleaseSession == null or
        api.CreateTensorWithDataAsOrtValue == null or
        api.ReleaseValue == null or
        api.Run == null or
        api.SessionGetInputCount == null or
        api.SessionGetOutputCount == null or
        api.SessionGetInputTypeInfo == null or
        api.SessionGetOutputTypeInfo == null or
        api.ReleaseTypeInfo == null or
        api.CastTypeInfoToTensorInfo == null or
        api.GetTensorElementType == null or
        api.GetDimensionsCount == null or
        api.GetTensorTypeAndShape == null or
        api.ReleaseTensorTypeAndShapeInfo == null or
        api.GetDimensions == null or
        api.GetTensorShapeElementCount == null or
        api.GetTensorMutableData == null or
        api.ReleaseStatus == null or
        api.GetErrorMessage == null)
    {
        return error.OrtApiUnavailable;
    }

    if (runtime_options.onnxruntime_cuda and
        (api.CreateCUDAProviderOptions == null or
            api.ReleaseCUDAProviderOptions == null or
            api.SessionOptionsAppendExecutionProvider_CUDA_V2 == null))
    {
        return error.OrtApiUnavailable;
    }
}

fn validateIo(api: *const c.OrtApi, session: *c.OrtSession) !void {
    var input_count: usize = 0;
    try check(api, api.SessionGetInputCount.?(session, &input_count), "SessionGetInputCount");
    if (input_count != 1) return error.InvalidShape;

    var output_count: usize = 0;
    try check(api, api.SessionGetOutputCount.?(session, &output_count), "SessionGetOutputCount");
    if (output_count != 2) return error.InvalidShape;

    try validateTensor(api, session, .input, 0, c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 5);
    try validateTensor(api, session, .output, 0, c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 2);
    try validateTensor(api, session, .output, 1, c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 2);
}

const TensorSide = enum {
    input,
    output,
};

fn validateTensor(
    api: *const c.OrtApi,
    session: *c.OrtSession,
    side: TensorSide,
    index: usize,
    expected_dtype: c.ONNXTensorElementDataType,
    expected_rank: usize,
) !void {
    var type_info: ?*c.OrtTypeInfo = null;
    const status = switch (side) {
        .input => api.SessionGetInputTypeInfo.?(session, index, &type_info),
        .output => api.SessionGetOutputTypeInfo.?(session, index, &type_info),
    };
    try check(api, status, "SessionGetTypeInfo");
    defer api.ReleaseTypeInfo.?(type_info);

    var tensor_info: ?*const c.OrtTensorTypeAndShapeInfo = null;
    try check(
        api,
        api.CastTypeInfoToTensorInfo.?(type_info, &tensor_info),
        "CastTypeInfoToTensorInfo",
    );
    if (tensor_info == null) return error.InvalidDtype;

    var dtype: c.ONNXTensorElementDataType = undefined;
    try check(api, api.GetTensorElementType.?(tensor_info, &dtype), "GetTensorElementType");
    if (dtype != expected_dtype) return error.InvalidDtype;

    var rank: usize = 0;
    try check(api, api.GetDimensionsCount.?(tensor_info, &rank), "GetDimensionsCount");
    if (rank != expected_rank) return error.InvalidRank;
}

fn copyOutputFlat(
    allocator: std.mem.Allocator,
    api: *const c.OrtApi,
    value: *c.OrtValue,
    expected_len: usize,
) ![]f32 {
    var tensor_info: ?*c.OrtTensorTypeAndShapeInfo = null;
    try check(api, api.GetTensorTypeAndShape.?(value, &tensor_info), "GetTensorTypeAndShape");
    defer api.ReleaseTensorTypeAndShapeInfo.?(tensor_info);

    var dtype: c.ONNXTensorElementDataType = undefined;
    try check(api, api.GetTensorElementType.?(tensor_info, &dtype), "GetTensorElementType");
    if (dtype != c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return error.InvalidDtype;

    var rank: usize = 0;
    try check(api, api.GetDimensionsCount.?(tensor_info, &rank), "GetDimensionsCount");
    if (rank != 2) return error.InvalidRank;

    var dims: [2]i64 = undefined;
    try check(api, api.GetDimensions.?(tensor_info, &dims, dims.len), "GetDimensions");
    if (dims[1] != spec.output_frames_per_window) return error.InvalidShape;

    var element_count: usize = 0;
    try check(api, api.GetTensorShapeElementCount.?(tensor_info, &element_count), "GetTensorShapeElementCount");
    if (element_count != expected_len) return error.InvalidShape;

    var data: ?*anyopaque = null;
    try check(api, api.GetTensorMutableData.?(value, &data), "GetTensorMutableData");
    const ptr: [*]const f32 = @ptrCast(@alignCast(data orelse return error.NullData));
    return allocator.dupe(f32, ptr[0..element_count]);
}

fn check(api: *const c.OrtApi, status: ?*c.OrtStatus, context: []const u8) OnnxModelError!void {
    if (status == null) return;
    defer api.ReleaseStatus.?(status);
    const message = api.GetErrorMessage.?(status);
    std.debug.print("ONNX Runtime call failed: {s}: {s}\n", .{ context, std.mem.span(message) });
    return error.OrtCallFailed;
}
