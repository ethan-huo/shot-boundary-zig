//! MLX-C AutoShot@F1 model edge.

const std = @import("std");
const spec = @import("spec");

const c = @cImport({
    @cInclude("mlx/c/mlx.h");
});

const base_filters = 16;
const layer_count = 6;
const transnet_layer_count = 3;
const transnet_blocks_per_layer = 2;
const transnet_dilation_count = 4;
const block_feature_count = 3;
const max_dilations = 5;
const dense_dim = 1024;
const lookup_window = 101;
const similarity_dim = 128;
const aux_output_dim = 128;
const batch_norm_eps: f32 = 1e-3;
const hist_bins = 512;
const frame_similarity_in_filters = 448;
const cnn_output_dim = 4608;
const fc1_input_dim = cnn_output_dim + aux_output_dim + aux_output_dim;

pub const implementation: []const u8 = "zig-mlx";

pub const MlxModelError = error{
    MlxCallFailed,
    MissingWeight,
    InvalidShape,
    InvalidRank,
    InvalidDtype,
    InvalidInput,
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

pub const AutoShot = struct {
    stream: c.mlx_stream,
    layers: [layer_count]AutoShotLayer,
    frame_similarity: FrameSimilarity,
    color_histograms: ColorHistograms,
    fc1: Linear,
    cls_layer1: Linear,
    cls_layer2: Linear,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !AutoShot {
        const cpu_stream = c.mlx_default_cpu_stream_new();
        defer _ = c.mlx_stream_free(cpu_stream);

        var data = c.mlx_map_string_to_array_new();
        defer _ = c.mlx_map_string_to_array_free(data);
        var metadata = c.mlx_map_string_to_string_new();
        defer _ = c.mlx_map_string_to_string_free(metadata);

        const z_path = try allocator.dupeZ(u8, path);
        defer allocator.free(z_path);
        try check(c.mlx_load_safetensors(&data, &metadata, z_path.ptr, cpu_stream), "mlx_load_safetensors");

        const gpu = c.mlx_device_new_type(c.MLX_GPU, 0);
        defer _ = c.mlx_device_free(gpu);
        try check(c.mlx_set_default_device(gpu), "mlx_set_default_device");

        var model = try loadFromMap(data);
        errdefer model.deinit();
        return model;
    }

    fn loadFromMap(data: c.mlx_map_string_to_array) !AutoShot {
        var model_layers: [layer_count]AutoShotLayer = undefined;
        var initialized_layers: usize = 0;
        errdefer {
            for (model_layers[0..initialized_layers]) |*layer| layer.deinit();
        }

        for (&model_layers, layer_configs) |*layer, config| {
            layer.* = try AutoShotLayer.load(data, config);
            initialized_layers += 1;
        }

        const frame_similarity = try FrameSimilarity.load(
            data,
            frame_similarity_in_filters,
            lookup_window,
            "frame_sim_layer",
        );
        errdefer frame_similarity.deinit();
        const color_histograms = try ColorHistograms.load(data, lookup_window, "color_hist_layer");
        errdefer color_histograms.deinit();

        // AutoShot@F1 has Attention1D(n_layer=0), so the transformer branch is absent.
        // The live classifier input is fc1_0, while fc1 is kept only in upstream weights.
        const fc1 = try Linear.load(data, fc1_input_dim, dense_dim, "fc1_0");
        errdefer fc1.deinit();
        const cls_layer1 = try Linear.load(data, dense_dim, 1, "cls_layer1");
        errdefer cls_layer1.deinit();
        const cls_layer2 = try Linear.load(data, dense_dim, 1, "cls_layer2");
        errdefer cls_layer2.deinit();

        return .{
            .stream = c.mlx_default_gpu_stream_new(),
            .layers = model_layers,
            .frame_similarity = frame_similarity,
            .color_histograms = color_histograms,
            .fc1 = fc1,
            .cls_layer1 = cls_layer1,
            .cls_layer2 = cls_layer2,
        };
    }

    pub fn deinit(self: *AutoShot) void {
        self.cls_layer2.deinit();
        self.cls_layer1.deinit();
        self.fc1.deinit();
        self.color_histograms.deinit();
        self.frame_similarity.deinit();
        for (&self.layers) |*layer| layer.deinit();
        _ = c.mlx_stream_free(self.stream);
        self.* = undefined;
    }

    pub fn predictBatch(
        self: *const AutoShot,
        allocator: std.mem.Allocator,
        window_batch_rgb24: []const u8,
        batch_size: usize,
    ) !Predictions {
        if (batch_size == 0) return error.InvalidInput;
        const expected_len = batch_size * spec.window_frames * spec.frameBytes();
        if (window_batch_rgb24.len != expected_len) return error.InvalidInput;

        var input_shape = [_]c_int{
            @intCast(batch_size),
            @intCast(spec.window_frames),
            @intCast(spec.input_height),
            @intCast(spec.input_width),
            @intCast(spec.input_channels),
        };
        const inputs = c.mlx_array_new_data(
            window_batch_rgb24.ptr,
            &input_shape,
            @intCast(input_shape.len),
            c.MLX_UINT8,
        );
        defer freeArray(inputs);

        const output = try self.forward(allocator, inputs, window_batch_rgb24, batch_size);
        defer output.deinit();

        const single_frame = try centerProbabilities(allocator, output.single_frame_logits, self.stream);
        errdefer allocator.free(single_frame);
        const many_hot = try centerProbabilities(allocator, output.many_hot_logits, self.stream);

        return .{
            .single_frame = single_frame,
            .many_hot = many_hot,
        };
    }

    fn forward(
        self: *const AutoShot,
        allocator: std.mem.Allocator,
        inputs: c.mlx_array,
        window_batch_rgb24: []const u8,
        batch_size: usize,
    ) !ModelOutput {
        validateInputWindow(inputs);

        const as_float = try astype(inputs, c.MLX_FLOAT32, self.stream);
        defer freeArray(as_float);
        const divisor = c.mlx_array_new_float32(255.0);
        defer freeArray(divisor);
        var x = try binaryOp(c.mlx_divide, as_float, divisor, self.stream, "mlx_divide");
        defer freeArray(x);

        var block_features: [block_feature_count]c.mlx_array = undefined;
        var initialized_features: usize = 0;
        defer {
            for (block_features[0..initialized_features]) |feature| freeArray(feature);
        }
        var shortcut: ?c.mlx_array = null;
        defer if (shortcut) |array| freeArray(array);

        for (&self.layers, 0..) |*layer, index| {
            const next = try layer.forward(x, self.stream);
            freeArray(x);
            x = next;

            if (index == 0 or index == 2 or index == 4) {
                if (shortcut) |array| freeArray(array);
                shortcut = try cloneArray(x);
            } else {
                const shortcut_array = shortcut orelse return error.InvalidShape;
                shortcut = null;
                defer freeArray(shortcut_array);

                const added = try binaryOp(c.mlx_add, x, shortcut_array, self.stream, "mlx_add");
                defer freeArray(added);
                const pooled = try avgPool3dSpatial2x2(added, self.stream);
                freeArray(x);
                x = pooled;

                block_features[initialized_features] = try cloneArray(x);
                initialized_features += 1;
            }
        }

        const dims = try dims5("sddcnn_output", x);
        var feature_shape = [_]c_int{ dims[0], dims[1], dims[2] * dims[3] * dims[4] };
        var features = try reshape(x, &feature_shape, self.stream);
        defer freeArray(features);

        if (initialized_features != block_feature_count) return error.InvalidShape;
        const sim_features = try self.frame_similarity.forward(block_features[0..initialized_features], self.stream);
        defer freeArray(sim_features);
        const features_with_similarity = try concatenatePrefix(features, sim_features, 2, self.stream);
        freeArray(features);
        features = features_with_similarity;

        const color_features = try self.color_histograms.forward(
            allocator,
            window_batch_rgb24,
            batch_size,
            self.stream,
        );
        defer freeArray(color_features);
        const features_with_color = try concatenatePrefix(features, color_features, 2, self.stream);
        freeArray(features);
        features = features_with_color;

        const fc = try self.fc1.forward(features, self.stream);
        defer freeArray(fc);
        const hidden = try relu(fc, self.stream);
        defer freeArray(hidden);

        const single_frame_logits = try self.cls_layer1.forward(hidden, self.stream);
        errdefer freeArray(single_frame_logits);
        const many_hot_logits = try self.cls_layer2.forward(hidden, self.stream);

        return .{
            .single_frame_logits = single_frame_logits,
            .many_hot_logits = many_hot_logits,
        };
    }
};

pub const TransNetV2 = struct {
    stream: c.mlx_stream,
    blocks: [transnet_layer_count]StackedDdcnn,
    frame_similarity: FrameSimilarity,
    color_histograms: ColorHistograms,
    fc1: Linear,
    cls_layer1: Linear,
    cls_layer2: Linear,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !TransNetV2 {
        const cpu_stream = c.mlx_default_cpu_stream_new();
        defer _ = c.mlx_stream_free(cpu_stream);

        var data = c.mlx_map_string_to_array_new();
        defer _ = c.mlx_map_string_to_array_free(data);
        var metadata = c.mlx_map_string_to_string_new();
        defer _ = c.mlx_map_string_to_string_free(metadata);

        const z_path = try allocator.dupeZ(u8, path);
        defer allocator.free(z_path);
        try check(c.mlx_load_safetensors(&data, &metadata, z_path.ptr, cpu_stream), "mlx_load_safetensors");

        const gpu = c.mlx_device_new_type(c.MLX_GPU, 0);
        defer _ = c.mlx_device_free(gpu);
        try check(c.mlx_set_default_device(gpu), "mlx_set_default_device");

        var model = try loadFromMap(data);
        errdefer model.deinit();
        return model;
    }

    fn loadFromMap(data: c.mlx_map_string_to_array) !TransNetV2 {
        var blocks: [transnet_layer_count]StackedDdcnn = undefined;
        var initialized_blocks: usize = 0;
        errdefer {
            for (blocks[0..initialized_blocks]) |*block| block.deinit();
        }

        var in_filters = spec.input_channels;
        for (&blocks, 0..) |*block, layer_index| {
            const filters = base_filters * (@as(usize, 1) << @intCast(layer_index));
            var prefix_buf: [32]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&prefix_buf, "SDDCNN.{d}", .{layer_index});
            block.* = try StackedDdcnn.load(data, in_filters, filters, prefix);
            initialized_blocks += 1;
            in_filters = filters * 4;
        }

        const frame_similarity = try FrameSimilarity.loadPlainLinear(
            data,
            frame_similarity_in_filters,
            lookup_window,
            "frame_sim_layer",
        );
        errdefer frame_similarity.deinit();
        const color_histograms = try ColorHistograms.loadPlainLinear(data, lookup_window, "color_hist_layer");
        errdefer color_histograms.deinit();

        const fc1 = try Linear.loadPlain(data, fc1_input_dim, dense_dim, "fc1");
        errdefer fc1.deinit();
        const cls_layer1 = try Linear.loadPlain(data, dense_dim, 1, "cls_layer1");
        errdefer cls_layer1.deinit();
        const cls_layer2 = try Linear.loadPlain(data, dense_dim, 1, "cls_layer2");
        errdefer cls_layer2.deinit();

        return .{
            .stream = c.mlx_default_gpu_stream_new(),
            .blocks = blocks,
            .frame_similarity = frame_similarity,
            .color_histograms = color_histograms,
            .fc1 = fc1,
            .cls_layer1 = cls_layer1,
            .cls_layer2 = cls_layer2,
        };
    }

    pub fn deinit(self: *TransNetV2) void {
        self.cls_layer2.deinit();
        self.cls_layer1.deinit();
        self.fc1.deinit();
        self.color_histograms.deinit();
        self.frame_similarity.deinit();
        for (&self.blocks) |*block| block.deinit();
        _ = c.mlx_stream_free(self.stream);
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

        var input_shape = [_]c_int{
            @intCast(batch_size),
            @intCast(spec.window_frames),
            @intCast(spec.input_height),
            @intCast(spec.input_width),
            @intCast(spec.input_channels),
        };
        const inputs = c.mlx_array_new_data(
            window_batch_rgb24.ptr,
            &input_shape,
            @intCast(input_shape.len),
            c.MLX_UINT8,
        );
        defer freeArray(inputs);

        const output = try self.forward(allocator, inputs, window_batch_rgb24, batch_size);
        defer output.deinit();

        const single_frame = try centerProbabilities(allocator, output.single_frame_logits, self.stream);
        errdefer allocator.free(single_frame);
        const many_hot = try centerProbabilities(allocator, output.many_hot_logits, self.stream);

        return .{
            .single_frame = single_frame,
            .many_hot = many_hot,
        };
    }

    fn forward(
        self: *const TransNetV2,
        allocator: std.mem.Allocator,
        inputs: c.mlx_array,
        window_batch_rgb24: []const u8,
        batch_size: usize,
    ) !ModelOutput {
        validateInputWindow(inputs);

        const as_float = try astype(inputs, c.MLX_FLOAT32, self.stream);
        defer freeArray(as_float);
        const divisor = c.mlx_array_new_float32(255.0);
        defer freeArray(divisor);
        var x = try binaryOp(c.mlx_divide, as_float, divisor, self.stream, "mlx_divide");
        defer freeArray(x);

        var block_features: [block_feature_count]c.mlx_array = undefined;
        var initialized_features: usize = 0;
        defer {
            for (block_features[0..initialized_features]) |feature| freeArray(feature);
        }

        for (&self.blocks, 0..) |*block, index| {
            const next = try block.forward(x, self.stream);
            freeArray(x);
            x = next;
            block_features[index] = try cloneArray(x);
            initialized_features += 1;
        }

        const dims = try dims5("sddcnn_output", x);
        var feature_shape = [_]c_int{ dims[0], dims[1], dims[2] * dims[3] * dims[4] };
        var features = try reshape(x, &feature_shape, self.stream);
        defer freeArray(features);

        if (initialized_features != block_feature_count) return error.InvalidShape;
        const sim_features = try self.frame_similarity.forward(block_features[0..initialized_features], self.stream);
        defer freeArray(sim_features);
        const features_with_similarity = try concatenatePrefix(features, sim_features, 2, self.stream);
        freeArray(features);
        features = features_with_similarity;

        const color_features = try self.color_histograms.forward(
            allocator,
            window_batch_rgb24,
            batch_size,
            self.stream,
        );
        defer freeArray(color_features);
        const features_with_color = try concatenatePrefix(features, color_features, 2, self.stream);
        freeArray(features);
        features = features_with_color;

        const fc = try self.fc1.forward(features, self.stream);
        defer freeArray(fc);
        const hidden = try relu(fc, self.stream);
        defer freeArray(hidden);

        const single_frame_logits = try self.cls_layer1.forward(hidden, self.stream);
        errdefer freeArray(single_frame_logits);
        const many_hot_logits = try self.cls_layer2.forward(hidden, self.stream);

        return .{
            .single_frame_logits = single_frame_logits,
            .many_hot_logits = many_hot_logits,
        };
    }
};

const ModelOutput = struct {
    single_frame_logits: c.mlx_array,
    many_hot_logits: c.mlx_array,

    fn deinit(self: ModelOutput) void {
        freeArray(self.single_frame_logits);
        freeArray(self.many_hot_logits);
    }
};

const LayerKind = enum {
    standard,
    shared_spatial_a,
};

const LayerConfig = struct {
    prefix: []const u8,
    in_filters: usize,
    filters: usize,
    multiplier: usize,
    dilation_count: usize,
    kind: LayerKind,

    fn outputChannels(self: LayerConfig) usize {
        return self.filters * 4;
    }

    fn midFilters(self: LayerConfig) usize {
        return self.filters * self.multiplier;
    }
};

const layer_configs = [_]LayerConfig{
    .{
        .prefix = "Layer_0_3",
        .in_filters = spec.input_channels,
        .filters = base_filters,
        .multiplier = 1,
        .dilation_count = 4,
        .kind = .standard,
    },
    .{
        .prefix = "Layer_1_8",
        .in_filters = base_filters * 4,
        .filters = base_filters,
        .multiplier = 4,
        .dilation_count = 5,
        .kind = .shared_spatial_a,
    },
    .{
        .prefix = "Layer_2_8",
        .in_filters = base_filters * 4,
        .filters = base_filters * 2,
        .multiplier = 4,
        .dilation_count = 5,
        .kind = .shared_spatial_a,
    },
    .{
        .prefix = "Layer_3_8",
        .in_filters = base_filters * 8,
        .filters = base_filters * 2,
        .multiplier = 4,
        .dilation_count = 5,
        .kind = .shared_spatial_a,
    },
    .{
        .prefix = "Layer_4_13",
        .in_filters = base_filters * 8,
        .filters = base_filters * 4,
        .multiplier = 3,
        .dilation_count = 5,
        .kind = .standard,
    },
    .{
        .prefix = "Layer_5_12",
        .in_filters = base_filters * 16,
        .filters = base_filters * 4,
        .multiplier = 2,
        .dilation_count = 5,
        .kind = .standard,
    },
};

const StackedDdcnn = struct {
    blocks: [transnet_blocks_per_layer]DilatedDdcnn,

    fn load(
        data: c.mlx_map_string_to_array,
        in_filters: usize,
        filters: usize,
        prefix: []const u8,
    ) !StackedDdcnn {
        var blocks: [transnet_blocks_per_layer]DilatedDdcnn = undefined;
        var initialized: usize = 0;
        errdefer {
            for (blocks[0..initialized]) |*block| block.deinit();
        }

        for (&blocks, 0..) |*block, block_index| {
            var block_prefix_buf: [64]u8 = undefined;
            const block_prefix = try std.fmt.bufPrint(
                &block_prefix_buf,
                "{s}.DDCNN.{d}",
                .{ prefix, block_index },
            );
            block.* = try DilatedDdcnn.load(
                data,
                if (block_index == 0) in_filters else filters * 4,
                filters,
                block_index + 1 != transnet_blocks_per_layer,
                block_prefix,
            );
            initialized += 1;
        }

        return .{ .blocks = blocks };
    }

    fn deinit(self: *StackedDdcnn) void {
        for (&self.blocks) |*block| block.deinit();
        self.* = undefined;
    }

    fn forward(self: *const StackedDdcnn, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        var x = try cloneArray(inputs);
        defer freeArray(x);
        var shortcut: ?c.mlx_array = null;
        defer if (shortcut) |array| freeArray(array);

        for (&self.blocks) |*block| {
            const next = try block.forward(x, stream);
            freeArray(x);
            x = next;
            // Upstream TransNetV2 uses the first block output as the residual shortcut.
            if (shortcut == null) shortcut = try cloneArray(x);
        }

        const activated = try relu(x, stream);
        defer freeArray(activated);
        const shortcut_array = shortcut orelse return error.InvalidShape;
        const added = try binaryOp(c.mlx_add, activated, shortcut_array, stream, "mlx_add");
        defer freeArray(added);
        return avgPool3dSpatial2x2(added, stream);
    }
};

const DilatedDdcnn = struct {
    convs: [transnet_dilation_count]ConfigurableConv3d,
    bn: BatchNorm,
    activate: bool,

    fn load(
        data: c.mlx_map_string_to_array,
        in_filters: usize,
        filters: usize,
        activate: bool,
        prefix: []const u8,
    ) !DilatedDdcnn {
        const dilations = [_]c_int{ 1, 2, 4, 8 };
        var convs: [transnet_dilation_count]ConfigurableConv3d = undefined;
        var initialized: usize = 0;
        errdefer {
            for (convs[0..initialized]) |*conv| conv.deinit();
        }

        for (&convs, dilations) |*conv, dilation| {
            var conv_prefix_buf: [96]u8 = undefined;
            const conv_prefix = try std.fmt.bufPrint(
                &conv_prefix_buf,
                "{s}.Conv3D_{d}.layers",
                .{ prefix, dilation },
            );
            conv.* = try ConfigurableConv3d.loadSeparable(
                data,
                in_filters,
                filters * 2,
                filters,
                dilation,
                conv_prefix,
            );
            initialized += 1;
        }

        var bn_prefix_buf: [80]u8 = undefined;
        const bn_prefix = try std.fmt.bufPrint(&bn_prefix_buf, "{s}.bn", .{prefix});
        const bn = try BatchNorm.load(data, filters * 4, bn_prefix);
        errdefer bn.deinit();

        return .{
            .convs = convs,
            .bn = bn,
            .activate = activate,
        };
    }

    fn deinit(self: *DilatedDdcnn) void {
        self.bn.deinit();
        for (&self.convs) |*conv| conv.deinit();
        self.* = undefined;
    }

    fn forward(self: *const DilatedDdcnn, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        var conv_outputs: [transnet_dilation_count]c.mlx_array = undefined;
        var initialized: usize = 0;
        defer {
            for (conv_outputs[0..initialized]) |array| freeArray(array);
        }

        for (&self.convs, 0..) |*conv, index| {
            conv_outputs[index] = try conv.forward(inputs, stream);
            initialized += 1;
        }

        const concatenated = try concatenate(conv_outputs[0..initialized], 4, stream);
        defer freeArray(concatenated);
        const normalized = try self.bn.forward(concatenated, stream);
        if (!self.activate) return normalized;
        defer freeArray(normalized);
        return relu(normalized, stream);
    }
};

const AutoShotLayer = struct {
    shared_spatial_weight: ?c.mlx_array,
    convs: [max_dilations]ConfigurableConv3d,
    conv_count: usize,
    bn: BatchNorm,

    fn load(data: c.mlx_map_string_to_array, config: LayerConfig) !AutoShotLayer {
        std.debug.assert(config.dilation_count <= max_dilations);
        const output_channels = config.outputChannels();
        const mid_filters = config.midFilters();

        var shared_spatial_weight: ?c.mlx_array = null;
        errdefer if (shared_spatial_weight) |array| freeArray(array);
        if (config.kind == .shared_spatial_a) {
            var share_name_buf: [128]u8 = undefined;
            const share_name = try std.fmt.bufPrintZ(&share_name_buf, "{s}.share.weight", .{config.prefix});
            shared_spatial_weight = try loadSpatialWeight(data, share_name, config.in_filters, mid_filters);
        }

        var convs: [max_dilations]ConfigurableConv3d = undefined;
        var initialized: usize = 0;
        errdefer {
            for (convs[0..initialized]) |*conv| conv.deinit();
        }

        const regular_branch_filters = output_channels / config.dilation_count;
        for (convs[0..config.dilation_count], 0..) |*conv, dilation_index| {
            const branch_filters = if (dilation_index + 1 == config.dilation_count)
                output_channels - regular_branch_filters * (config.dilation_count - 1)
            else
                regular_branch_filters;
            const dilation: c_int = @as(c_int, 1) << @intCast(dilation_index);
            var conv_prefix_buf: [128]u8 = undefined;
            const conv_prefix = try std.fmt.bufPrint(
                &conv_prefix_buf,
                "{s}.conv_blocks.{d}.layers",
                .{ config.prefix, dilation_index },
            );
            conv.* = switch (config.kind) {
                .standard => try ConfigurableConv3d.loadSeparable(
                    data,
                    config.in_filters,
                    mid_filters,
                    branch_filters,
                    dilation,
                    conv_prefix,
                ),
                .shared_spatial_a => try ConfigurableConv3d.loadTemporalOnly(
                    data,
                    mid_filters,
                    branch_filters,
                    dilation,
                    conv_prefix,
                ),
            };
            initialized += 1;
        }

        var bn_prefix_buf: [80]u8 = undefined;
        const bn_prefix = try std.fmt.bufPrint(&bn_prefix_buf, "{s}.batch_norm", .{config.prefix});
        const bn = try BatchNorm.load(data, output_channels, bn_prefix);
        errdefer bn.deinit();

        return .{
            .shared_spatial_weight = shared_spatial_weight,
            .convs = convs,
            .conv_count = config.dilation_count,
            .bn = bn,
        };
    }

    fn deinit(self: *AutoShotLayer) void {
        self.bn.deinit();
        for (self.convs[0..self.conv_count]) |*conv| conv.deinit();
        if (self.shared_spatial_weight) |array| freeArray(array);
        self.* = undefined;
    }

    fn forward(self: *const AutoShotLayer, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        var shared: ?c.mlx_array = null;
        defer if (shared) |array| freeArray(array);
        if (self.shared_spatial_weight) |weight| {
            shared = try convSpatial(inputs, weight, stream);
        }

        var conv_outputs: [max_dilations]c.mlx_array = undefined;
        var initialized: usize = 0;
        defer {
            for (conv_outputs[0..initialized]) |array| freeArray(array);
        }

        for (self.convs[0..self.conv_count], 0..) |*conv, index| {
            conv_outputs[index] = if (shared) |array|
                try conv.forwardTemporal(array, stream)
            else
                try conv.forward(inputs, stream);
            initialized += 1;
        }

        const concatenated = try concatenate(conv_outputs[0..initialized], 4, stream);
        defer freeArray(concatenated);
        const normalized = try self.bn.forward(concatenated, stream);
        defer freeArray(normalized);
        return relu(normalized, stream);
    }
};

const ConfigurableConv3d = struct {
    spatial_weight: ?c.mlx_array,
    temporal_weight: c.mlx_array,
    temporal_dilation: c_int,

    fn loadSeparable(
        data: c.mlx_map_string_to_array,
        in_filters: usize,
        mid_filters: usize,
        out_filters: usize,
        temporal_dilation: c_int,
        prefix: []const u8,
    ) !ConfigurableConv3d {
        var spatial_name_buf: [128]u8 = undefined;
        const spatial_name = try std.fmt.bufPrintZ(&spatial_name_buf, "{s}.0.weight", .{prefix});
        const spatial_weight = try loadSpatialWeight(data, spatial_name, in_filters, mid_filters);
        errdefer freeArray(spatial_weight);

        var temporal_name_buf: [128]u8 = undefined;
        const temporal_name = try std.fmt.bufPrintZ(&temporal_name_buf, "{s}.1.weight", .{prefix});
        const temporal_weight = try loadTemporalWeight(data, temporal_name, mid_filters, out_filters);
        errdefer freeArray(temporal_weight);

        return .{
            .spatial_weight = spatial_weight,
            .temporal_weight = temporal_weight,
            .temporal_dilation = temporal_dilation,
        };
    }

    fn loadTemporalOnly(
        data: c.mlx_map_string_to_array,
        in_filters: usize,
        out_filters: usize,
        temporal_dilation: c_int,
        prefix: []const u8,
    ) !ConfigurableConv3d {
        var temporal_name_buf: [128]u8 = undefined;
        const temporal_name = try std.fmt.bufPrintZ(&temporal_name_buf, "{s}.0.weight", .{prefix});
        const temporal_weight = try loadTemporalWeight(data, temporal_name, in_filters, out_filters);
        errdefer freeArray(temporal_weight);
        return .{
            .spatial_weight = null,
            .temporal_weight = temporal_weight,
            .temporal_dilation = temporal_dilation,
        };
    }

    fn deinit(self: *ConfigurableConv3d) void {
        freeArray(self.temporal_weight);
        if (self.spatial_weight) |array| freeArray(array);
        self.* = undefined;
    }

    fn forward(self: *const ConfigurableConv3d, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        const spatial_weight = self.spatial_weight orelse return error.InvalidShape;
        const spatial = try convSpatial(inputs, spatial_weight, stream);
        defer freeArray(spatial);
        return self.forwardTemporal(spatial, stream);
    }

    fn forwardTemporal(self: *const ConfigurableConv3d, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        const dims = try dims5("conv3d_temporal_input", inputs);
        const batch = dims[0];
        const frames = dims[1];
        const height = dims[2];
        const width = dims[3];
        const channels = dims[4];
        const temporal_input = try reshapeThenTransposeThenReshape(
            inputs,
            &.{ batch, frames, height, width, channels },
            &.{ 0, 2, 3, 1, 4 },
            &.{ batch * height * width, frames, channels },
            stream,
        );
        defer freeArray(temporal_input);
        const temporal = try conv1d(
            temporal_input,
            self.temporal_weight,
            1,
            self.temporal_dilation,
            self.temporal_dilation,
            1,
            stream,
        );
        defer freeArray(temporal);

        const temporal_dims = try dims3("conv3d_temporal_output", temporal);
        const out_frames = temporal_dims[1];
        const temporal_channels = temporal_dims[2];
        return reshapeThenTransposeThenReshape(
            temporal,
            &.{ batch, height, width, out_frames, temporal_channels },
            &.{ 0, 3, 1, 2, 4 },
            &.{ batch, out_frames, height, width, temporal_channels },
            stream,
        );
    }
};

const BatchNorm = struct {
    weight: c.mlx_array,
    bias: c.mlx_array,
    running_mean: c.mlx_array,
    running_var: c.mlx_array,

    fn load(data: c.mlx_map_string_to_array, channels: usize, prefix: []const u8) !BatchNorm {
        const expected = [_]c_int{@intCast(channels)};
        const weight = try takeNamedVector(data, prefix, "weight", &expected);
        errdefer freeArray(weight);
        const bias = try takeNamedVector(data, prefix, "bias", &expected);
        errdefer freeArray(bias);
        const running_mean = try takeNamedVector(data, prefix, "running_mean", &expected);
        errdefer freeArray(running_mean);
        const running_var = try takeNamedVector(data, prefix, "running_var", &expected);
        errdefer freeArray(running_var);
        return .{
            .weight = weight,
            .bias = bias,
            .running_mean = running_mean,
            .running_var = running_var,
        };
    }

    fn deinit(self: BatchNorm) void {
        freeArray(self.running_var);
        freeArray(self.running_mean);
        freeArray(self.bias);
        freeArray(self.weight);
    }

    fn forward(self: *const BatchNorm, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        const eps = c.mlx_array_new_float32(batch_norm_eps);
        defer freeArray(eps);
        const variance = try binaryOp(c.mlx_add, self.running_var, eps, stream, "mlx_add");
        defer freeArray(variance);
        const inv_std = try unaryOp(c.mlx_rsqrt, variance, stream, "mlx_rsqrt");
        defer freeArray(inv_std);
        const scale = try binaryOp(c.mlx_multiply, inv_std, self.weight, stream, "mlx_multiply");
        defer freeArray(scale);
        const centered = try binaryOp(c.mlx_subtract, inputs, self.running_mean, stream, "mlx_subtract");
        defer freeArray(centered);
        const scaled = try binaryOp(c.mlx_multiply, centered, scale, stream, "mlx_multiply");
        defer freeArray(scaled);
        return binaryOp(c.mlx_add, scaled, self.bias, stream, "mlx_add");
    }
};

const FrameSimilarity = struct {
    projection: Linear,
    fc: Linear,
    lookup_window: usize,

    fn load(data: c.mlx_map_string_to_array, in_filters: usize, window: usize, prefix: []const u8) !FrameSimilarity {
        var projection_prefix_buf: [80]u8 = undefined;
        const projection_prefix = try std.fmt.bufPrint(&projection_prefix_buf, "{s}.projection", .{prefix});
        const projection = try Linear.load(data, in_filters, similarity_dim, projection_prefix);
        errdefer projection.deinit();
        var fc_prefix_buf: [80]u8 = undefined;
        const fc_prefix = try std.fmt.bufPrint(&fc_prefix_buf, "{s}.fc", .{prefix});
        const fc = try Linear.load(data, window, aux_output_dim, fc_prefix);
        errdefer fc.deinit();
        return .{ .projection = projection, .fc = fc, .lookup_window = window };
    }

    fn loadPlainLinear(
        data: c.mlx_map_string_to_array,
        in_filters: usize,
        window: usize,
        prefix: []const u8,
    ) !FrameSimilarity {
        var projection_prefix_buf: [80]u8 = undefined;
        const projection_prefix = try std.fmt.bufPrint(&projection_prefix_buf, "{s}.projection", .{prefix});
        const projection = try Linear.loadPlain(data, in_filters, similarity_dim, projection_prefix);
        errdefer projection.deinit();
        var fc_prefix_buf: [80]u8 = undefined;
        const fc_prefix = try std.fmt.bufPrint(&fc_prefix_buf, "{s}.fc", .{prefix});
        const fc = try Linear.loadPlain(data, window, aux_output_dim, fc_prefix);
        errdefer fc.deinit();
        return .{ .projection = projection, .fc = fc, .lookup_window = window };
    }

    fn deinit(self: FrameSimilarity) void {
        self.fc.deinit();
        self.projection.deinit();
    }

    fn forward(self: *const FrameSimilarity, inputs: []const c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        var pooled: [block_feature_count]c.mlx_array = undefined;
        var initialized: usize = 0;
        defer {
            for (pooled[0..initialized]) |array| freeArray(array);
        }

        for (inputs, 0..) |input, index| {
            pooled[index] = try meanAxes(input, &.{ 2, 3 }, false, stream);
            initialized += 1;
        }

        const concatenated = try concatenate(pooled[0..initialized], 2, stream);
        defer freeArray(concatenated);
        const projected = try self.projection.forward(concatenated, stream);
        defer freeArray(projected);
        const normalized = try l2NormalizeLastDim(projected, stream);
        defer freeArray(normalized);
        const transposed = try transposeAxes(normalized, &.{ 0, 2, 1 }, stream);
        defer freeArray(transposed);
        const similarities = try binaryOp(c.mlx_matmul, normalized, transposed, stream, "mlx_matmul");
        defer freeArray(similarities);
        const windows = try localSimilarityWindows(similarities, self.lookup_window, stream);
        defer freeArray(windows);
        const fc_output = try self.fc.forward(windows, stream);
        defer freeArray(fc_output);
        return relu(fc_output, stream);
    }
};

const ColorHistograms = struct {
    fc: Linear,
    lookup_window: usize,

    fn load(data: c.mlx_map_string_to_array, window: usize, prefix: []const u8) !ColorHistograms {
        var fc_prefix_buf: [80]u8 = undefined;
        const fc_prefix = try std.fmt.bufPrint(&fc_prefix_buf, "{s}.fc", .{prefix});
        return .{
            .fc = try Linear.load(data, window, aux_output_dim, fc_prefix),
            .lookup_window = window,
        };
    }

    fn loadPlainLinear(data: c.mlx_map_string_to_array, window: usize, prefix: []const u8) !ColorHistograms {
        var fc_prefix_buf: [80]u8 = undefined;
        const fc_prefix = try std.fmt.bufPrint(&fc_prefix_buf, "{s}.fc", .{prefix});
        return .{
            .fc = try Linear.loadPlain(data, window, aux_output_dim, fc_prefix),
            .lookup_window = window,
        };
    }

    fn deinit(self: ColorHistograms) void {
        self.fc.deinit();
    }

    fn forward(
        self: *const ColorHistograms,
        allocator: std.mem.Allocator,
        window_batch_rgb24: []const u8,
        batch_size: usize,
        stream: c.mlx_stream,
    ) !c.mlx_array {
        const histograms = try computeColorHistograms(allocator, window_batch_rgb24, batch_size);
        defer freeArray(histograms);
        const transposed = try transposeAxes(histograms, &.{ 0, 2, 1 }, stream);
        defer freeArray(transposed);
        const similarities = try binaryOp(c.mlx_matmul, histograms, transposed, stream, "mlx_matmul");
        defer freeArray(similarities);
        const windows = try localSimilarityWindows(similarities, self.lookup_window, stream);
        defer freeArray(windows);
        const fc_output = try self.fc.forward(windows, stream);
        defer freeArray(fc_output);
        return relu(fc_output, stream);
    }
};

const Linear = struct {
    weight: c.mlx_array,
    bias: c.mlx_array,

    fn load(data: c.mlx_map_string_to_array, in_dim: usize, out_dim: usize, prefix: []const u8) !Linear {
        const weight = try takeNamedVector(data, prefix, "linear.weight", &.{ @intCast(out_dim), @intCast(in_dim) });
        errdefer freeArray(weight);
        const bias = try takeNamedVector(data, prefix, "linear.bias", &.{@intCast(out_dim)});
        errdefer freeArray(bias);
        return .{ .weight = weight, .bias = bias };
    }

    fn loadPlain(data: c.mlx_map_string_to_array, in_dim: usize, out_dim: usize, prefix: []const u8) !Linear {
        const weight = try takeNamedVector(data, prefix, "weight", &.{ @intCast(out_dim), @intCast(in_dim) });
        errdefer freeArray(weight);
        const bias = try takeNamedVector(data, prefix, "bias", &.{@intCast(out_dim)});
        errdefer freeArray(bias);
        return .{ .weight = weight, .bias = bias };
    }

    fn deinit(self: Linear) void {
        freeArray(self.bias);
        freeArray(self.weight);
    }

    fn forward(self: *const Linear, inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
        const weight_t = try transpose(self.weight, stream);
        defer freeArray(weight_t);
        const product = try binaryOp(c.mlx_matmul, inputs, weight_t, stream, "mlx_matmul");
        defer freeArray(product);
        return binaryOp(c.mlx_add, product, self.bias, stream, "mlx_add");
    }
};

fn loadSpatialWeight(
    data: c.mlx_map_string_to_array,
    name: [:0]const u8,
    in_filters: usize,
    out_filters: usize,
) !c.mlx_array {
    const raw = try takeWeight(data, name);
    defer freeArray(raw);
    try validateShape(name, raw, &.{
        @intCast(out_filters),
        @intCast(in_filters),
        1,
        3,
        3,
    });
    const stream = c.mlx_default_gpu_stream_new();
    defer _ = c.mlx_stream_free(stream);
    const transposed = try transposeAxes(raw, &.{ 0, 3, 4, 1, 2 }, stream);
    defer freeArray(transposed);
    return reshape(transposed, &.{
        @intCast(out_filters),
        3,
        3,
        @intCast(in_filters),
    }, stream);
}

fn loadTemporalWeight(
    data: c.mlx_map_string_to_array,
    name: [:0]const u8,
    in_filters: usize,
    out_filters: usize,
) !c.mlx_array {
    const raw = try takeWeight(data, name);
    defer freeArray(raw);
    try validateShape(name, raw, &.{
        @intCast(out_filters),
        @intCast(in_filters),
        3,
        1,
        1,
    });
    const stream = c.mlx_default_gpu_stream_new();
    defer _ = c.mlx_stream_free(stream);
    const transposed = try transposeAxes(raw, &.{ 0, 2, 1, 3, 4 }, stream);
    defer freeArray(transposed);
    return reshape(transposed, &.{
        @intCast(out_filters),
        3,
        @intCast(in_filters),
    }, stream);
}

fn takeNamedVector(
    data: c.mlx_map_string_to_array,
    prefix: []const u8,
    suffix: []const u8,
    expected: []const c_int,
) !c.mlx_array {
    var name_buf: [128]u8 = undefined;
    const name = try std.fmt.bufPrintZ(&name_buf, "{s}.{s}", .{ prefix, suffix });
    const weight = try takeWeight(data, name);
    errdefer freeArray(weight);
    try validateShape(name, weight, expected);
    return weight;
}

fn takeWeight(data: c.mlx_map_string_to_array, name: [:0]const u8) !c.mlx_array {
    var array = c.mlx_array_new();
    errdefer freeArray(array);
    const rc = c.mlx_map_string_to_array_get(&array, data, name.ptr);
    if (rc == 2) return error.MissingWeight;
    try check(rc, "mlx_map_string_to_array_get");
    return array;
}

fn centerProbabilities(allocator: std.mem.Allocator, logits: c.mlx_array, stream: c.mlx_stream) ![]f32 {
    const probs = try unaryOp(c.mlx_sigmoid, logits, stream, "mlx_sigmoid");
    defer freeArray(probs);
    const shape = c.mlx_array_shape(probs);
    if (c.mlx_array_ndim(probs) != 3) return error.InvalidRank;
    const batch = shape[0];
    const channels = shape[2];
    const start = [_]c_int{ 0, @intCast(spec.context_frames), 0 };
    const stop = [_]c_int{
        batch,
        @intCast(spec.context_frames + spec.output_frames_per_window),
        channels,
    };
    const strides = [_]c_int{ 1, 1, 1 };
    const center = try slice(probs, &start, &stop, &strides, stream);
    defer freeArray(center);
    const flattened_len = batch * @as(c_int, @intCast(spec.output_frames_per_window)) * channels;
    const flat = try reshape(center, &.{flattened_len}, stream);
    defer freeArray(flat);
    return copyFloat32(allocator, flat);
}

fn computeColorHistograms(
    allocator: std.mem.Allocator,
    window_batch_rgb24: []const u8,
    batch_size: usize,
) !c.mlx_array {
    const frame_bytes = spec.frameBytes();
    const frame_count = batch_size * spec.window_frames;
    if (window_batch_rgb24.len != frame_count * frame_bytes) return error.InvalidInput;

    const histogram_values = try allocator.alloc(f32, frame_count * hist_bins);
    defer allocator.free(histogram_values);
    @memset(histogram_values, 0.0);

    const frame_pixels = spec.input_height * spec.input_width;
    for (0..frame_count) |frame_index| {
        const frame_start = frame_index * frame_bytes;
        const histogram_start = frame_index * hist_bins;
        const frame = window_batch_rgb24[frame_start .. frame_start + frame_bytes];
        for (0..frame_pixels) |pixel_index| {
            const pixel_start = pixel_index * spec.input_channels;
            const red = @as(usize, frame[pixel_start] >> 5);
            const green = @as(usize, frame[pixel_start + 1] >> 5);
            const blue = @as(usize, frame[pixel_start + 2] >> 5);
            histogram_values[histogram_start + (red << 6) + (green << 3) + blue] += 1.0;
        }

        var norm: f32 = 0.0;
        for (histogram_values[histogram_start .. histogram_start + hist_bins]) |value| {
            norm += value * value;
        }
        norm = @sqrt(norm);
        if (norm > 0.0) {
            for (histogram_values[histogram_start .. histogram_start + hist_bins]) |*value| {
                value.* /= norm;
            }
        }
    }

    var shape = [_]c_int{
        @intCast(batch_size),
        @intCast(spec.window_frames),
        @intCast(hist_bins),
    };
    return c.mlx_array_new_data(&histogram_values[0], &shape, @intCast(shape.len), c.MLX_FLOAT32);
}

fn localSimilarityWindows(similarities: c.mlx_array, window: usize, stream: c.mlx_stream) !c.mlx_array {
    const dims = try dims3("similarities", similarities);
    const frames: usize = @intCast(dims[1]);
    const radius: c_int = @intCast(window / 2);
    const pad_value = c.mlx_array_new_float32(0.0);
    defer freeArray(pad_value);
    const axes = [_]c_int{ 0, 1, 2 };
    const low = [_]c_int{ 0, 0, radius };
    const high = [_]c_int{ 0, 0, radius };
    const padded = try pad(similarities, &axes, &low, &high, pad_value, "constant", stream);
    defer freeArray(padded);

    if (frames > spec.window_frames or window > lookup_window) return error.InvalidShape;
    var index_storage: [spec.window_frames * lookup_window]c_int = undefined;
    const indices_values = index_storage[0 .. frames * window];
    for (0..frames) |frame| {
        for (0..window) |offset| {
            indices_values[frame * window + offset] = @intCast(frame + offset);
        }
    }
    var shape = [_]c_int{ 1, @intCast(frames), @intCast(window) };
    const indices = c.mlx_array_new_data(&indices_values[0], &shape, @intCast(shape.len), c.MLX_INT32);
    defer freeArray(indices);

    return takeAlongAxis(padded, indices, 2, stream);
}

fn avgPool3dSpatial2x2(inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
    const dims = try dims5("avg_pool_input", inputs);
    const batch = dims[0];
    const frames = dims[1];
    const height = dims[2];
    const width = dims[3];
    const channels = dims[4];
    const out_height = @divTrunc(height, 2);
    const out_width = @divTrunc(width, 2);

    const start = [_]c_int{ 0, 0, 0, 0, 0 };
    const stop = [_]c_int{ batch, frames, out_height * 2, out_width * 2, channels };
    const strides = [_]c_int{ 1, 1, 1, 1, 1 };
    const cropped = try slice(inputs, &start, &stop, &strides, stream);
    defer freeArray(cropped);
    const grouped = try reshape(cropped, &.{ batch * frames, out_height, 2, out_width, 2, channels }, stream);
    defer freeArray(grouped);
    const pooled = try meanAxes(grouped, &.{ 2, 4 }, false, stream);
    defer freeArray(pooled);
    return reshape(pooled, &.{ batch, frames, out_height, out_width, channels }, stream);
}

fn l2NormalizeLastDim(inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
    const squared = try unaryOp(c.mlx_square, inputs, stream, "mlx_square");
    defer freeArray(squared);
    const summed = try sumAxis(squared, -1, true, stream);
    defer freeArray(summed);
    const epsilon = c.mlx_array_new_float32(1e-12);
    defer freeArray(epsilon);
    const safe_sum = try binaryOp(c.mlx_add, summed, epsilon, stream, "mlx_add");
    defer freeArray(safe_sum);
    const norm = try unaryOp(c.mlx_sqrt, safe_sum, stream, "mlx_sqrt");
    defer freeArray(norm);
    return binaryOp(c.mlx_divide, inputs, norm, stream, "mlx_divide");
}

fn relu(inputs: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
    const zero = c.mlx_array_new_float32(0.0);
    defer freeArray(zero);
    return binaryOp(c.mlx_maximum, inputs, zero, stream, "mlx_maximum");
}

fn concatenatePrefix(
    current: c.mlx_array,
    prefix: c.mlx_array,
    axis: c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    const arrays = [_]c.mlx_array{ prefix, current };
    return concatenate(&arrays, axis, stream);
}

fn reshapeThenTransposeThenReshape(
    input: c.mlx_array,
    shape: []const c_int,
    axes: []const c_int,
    final_shape: []const c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    const reshaped = try reshape(input, shape, stream);
    defer freeArray(reshaped);
    const transposed = try transposeAxes(reshaped, axes, stream);
    defer freeArray(transposed);
    return reshape(transposed, final_shape, stream);
}

fn convSpatial(inputs: c.mlx_array, weight: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
    const dims = try dims5("conv3d_spatial_input", inputs);
    const batch = dims[0];
    const frames = dims[1];
    const height = dims[2];
    const width = dims[3];
    const channels = dims[4];
    const spatial_input = try reshape(inputs, &.{ batch * frames, height, width, channels }, stream);
    defer freeArray(spatial_input);
    const spatial = try conv2d(spatial_input, weight, 1, 1, 1, 1, 1, 1, stream);
    defer freeArray(spatial);
    const spatial_dims = try dims4("conv3d_spatial_output", spatial);
    const out_height = spatial_dims[1];
    const out_width = spatial_dims[2];
    const out_channels = spatial_dims[3];
    return reshape(spatial, &.{ batch, frames, out_height, out_width, out_channels }, stream);
}

fn validateInputWindow(inputs: c.mlx_array) void {
    std.debug.assert(c.mlx_array_dtype(inputs) == c.MLX_UINT8);
    std.debug.assert(c.mlx_array_ndim(inputs) == 5);
    const shape = c.mlx_array_shape(inputs);
    std.debug.assert(shape[1] == spec.window_frames);
    std.debug.assert(shape[2] == spec.input_height);
    std.debug.assert(shape[3] == spec.input_width);
    std.debug.assert(shape[4] == spec.input_channels);
}

fn validateShape(name: []const u8, array: c.mlx_array, expected: []const c_int) !void {
    const ndim = c.mlx_array_ndim(array);
    if (ndim != expected.len) {
        std.debug.print("MLX tensor {s}: expected rank {d}, got {d}\n", .{ name, expected.len, ndim });
        return error.InvalidRank;
    }
    const actual = c.mlx_array_shape(array);
    for (expected, 0..) |dim, index| {
        if (actual[index] != dim) {
            std.debug.print(
                "MLX tensor {s}: dimension {d} expected {d}, got {d}\n",
                .{ name, index, dim, actual[index] },
            );
            return error.InvalidShape;
        }
    }
}

fn dims3(name: []const u8, array: c.mlx_array) ![3]c_int {
    _ = name;
    if (c.mlx_array_ndim(array) != 3) return error.InvalidRank;
    const shape = c.mlx_array_shape(array);
    return .{ shape[0], shape[1], shape[2] };
}

fn dims4(name: []const u8, array: c.mlx_array) ![4]c_int {
    _ = name;
    if (c.mlx_array_ndim(array) != 4) return error.InvalidRank;
    const shape = c.mlx_array_shape(array);
    return .{ shape[0], shape[1], shape[2], shape[3] };
}

fn dims5(name: []const u8, array: c.mlx_array) ![5]c_int {
    _ = name;
    if (c.mlx_array_ndim(array) != 5) return error.InvalidRank;
    const shape = c.mlx_array_shape(array);
    return .{ shape[0], shape[1], shape[2], shape[3], shape[4] };
}

fn astype(input: c.mlx_array, dtype: c.mlx_dtype, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_astype(&result, input, dtype, stream), "mlx_astype");
    return result;
}

fn reshape(input: c.mlx_array, shape: []const c_int, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_reshape(&result, input, shape.ptr, shape.len, stream), "mlx_reshape");
    return result;
}

fn transpose(input: c.mlx_array, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_transpose(&result, input, stream), "mlx_transpose");
    return result;
}

fn transposeAxes(input: c.mlx_array, axes: []const c_int, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_transpose_axes(&result, input, axes.ptr, axes.len, stream), "mlx_transpose_axes");
    return result;
}

fn slice(
    input: c.mlx_array,
    start: []const c_int,
    stop: []const c_int,
    strides: []const c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(
        c.mlx_slice(&result, input, start.ptr, start.len, stop.ptr, stop.len, strides.ptr, strides.len, stream),
        "mlx_slice",
    );
    return result;
}

fn pad(
    input: c.mlx_array,
    axes: []const c_int,
    low: []const c_int,
    high: []const c_int,
    value: c.mlx_array,
    mode: [*:0]const u8,
    stream: c.mlx_stream,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(
        c.mlx_pad(&result, input, axes.ptr, axes.len, low.ptr, low.len, high.ptr, high.len, value, mode, stream),
        "mlx_pad",
    );
    return result;
}

fn concatenate(inputs: []const c.mlx_array, axis: c_int, stream: c.mlx_stream) !c.mlx_array {
    const vector = c.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = c.mlx_vector_array_free(vector);
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_concatenate_axis(&result, vector, axis, stream), "mlx_concatenate_axis");
    return result;
}

fn conv1d(
    input: c.mlx_array,
    weight: c.mlx_array,
    stride: c_int,
    padding: c_int,
    dilation: c_int,
    groups: c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_conv1d(&result, input, weight, stride, padding, dilation, groups, stream), "mlx_conv1d");
    return result;
}

fn conv2d(
    input: c.mlx_array,
    weight: c.mlx_array,
    stride_0: c_int,
    stride_1: c_int,
    padding_0: c_int,
    padding_1: c_int,
    dilation_0: c_int,
    dilation_1: c_int,
    stream: c.mlx_stream,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(
        c.mlx_conv2d(
            &result,
            input,
            weight,
            stride_0,
            stride_1,
            padding_0,
            padding_1,
            dilation_0,
            dilation_1,
            1,
            stream,
        ),
        "mlx_conv2d",
    );
    return result;
}

fn meanAxes(input: c.mlx_array, axes: []const c_int, keepdims: bool, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_mean_axes(&result, input, axes.ptr, axes.len, keepdims, stream), "mlx_mean_axes");
    return result;
}

fn sumAxis(input: c.mlx_array, axis: c_int, keepdims: bool, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_sum_axis(&result, input, axis, keepdims, stream), "mlx_sum_axis");
    return result;
}

fn takeAlongAxis(input: c.mlx_array, indices: c.mlx_array, axis: c_int, stream: c.mlx_stream) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(c.mlx_take_along_axis(&result, input, indices, axis, stream), "mlx_take_along_axis");
    return result;
}

fn unaryOp(
    comptime func: fn (*c.mlx_array, c.mlx_array, c.mlx_stream) callconv(.c) c_int,
    input: c.mlx_array,
    stream: c.mlx_stream,
    context: []const u8,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(func(&result, input, stream), context);
    return result;
}

fn binaryOp(
    comptime func: fn (*c.mlx_array, c.mlx_array, c.mlx_array, c.mlx_stream) callconv(.c) c_int,
    left: c.mlx_array,
    right: c.mlx_array,
    stream: c.mlx_stream,
    context: []const u8,
) !c.mlx_array {
    var result = c.mlx_array_new();
    errdefer freeArray(result);
    try check(func(&result, left, right, stream), context);
    return result;
}

fn copyFloat32(allocator: std.mem.Allocator, input: c.mlx_array) ![]f32 {
    try check(c.mlx_array_eval(input), "mlx_array_eval");
    const len = c.mlx_array_size(input);
    const ptr = c.mlx_array_data_float32(input);
    if (ptr == null) return error.NullData;
    return allocator.dupe(f32, ptr[0..len]);
}

fn cloneArray(input: c.mlx_array) !c.mlx_array {
    var cloned = c.mlx_array_new();
    errdefer freeArray(cloned);
    try check(c.mlx_array_set(&cloned, input), "mlx_array_set");
    return cloned;
}

fn freeArray(array: c.mlx_array) void {
    _ = c.mlx_array_free(array);
}

fn check(rc: c_int, context: []const u8) MlxModelError!void {
    if (rc == 0) return;
    std.debug.print("MLX-C call failed: {s} returned {d}\n", .{ context, rc });
    return error.MlxCallFailed;
}
