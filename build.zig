const std = @import("std");
const ziglint = @import("ziglint");

const mlx_c_version = "v0.6.0";
const mlx_c_root_dir = "externals/mlx-c";
const mlx_c_src_dir = mlx_c_root_dir ++ "/src";
const mlx_c_build_dir = mlx_c_root_dir ++ "/build";
const mlx_c_install_dir = mlx_c_root_dir ++ "/install";
const mlx_c_cmake_cache = mlx_c_build_dir ++ "/CMakeCache.txt";
const mlx_c_build_lib = mlx_c_build_dir ++ "/libmlxc.dylib";
const mlx_build_lib = mlx_c_build_dir ++ "/_deps/mlx-build/libmlx.dylib";
const mlx_c_install_lib = mlx_c_install_dir ++ "/lib/libmlxc.dylib";
const mlx_install_lib = mlx_c_install_dir ++ "/lib/libmlx.dylib";
const mlx_c_install_header = mlx_c_install_dir ++ "/include/mlx/c/mlx.h";

const ort_version = "1.24.4";
const ort_root_dir = "externals/onnxruntime";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const target_os = target.result.os.tag;
    const target_arch = target.result.cpu.arch;
    const onnxruntime_cuda = b.option(
        bool,
        "onnxruntime-cuda",
        "Enable ONNX Runtime CUDA execution provider on Linux",
    ) orelse false;
    const platform_runtime: PlatformRuntime = switch (target_os) {
        .macos => .{ .mlx = setupMlx(b) },
        .linux => .{ .ort = setupOrt(b, target_arch, onnxruntime_cuda) },
        else => .unsupported,
    };
    const runtime_options = b.addOptions();
    runtime_options.addOption(bool, "onnxruntime_cuda", onnxruntime_cuda);
    const spec_module = b.createModule(.{
        .root_source_file = b.path("src/spec.zig"),
        .target = target,
        .optimize = optimize,
    });
    const runtime_impl = createRuntimeModule(b, target, optimize, platform_runtime, spec_module, runtime_options);

    const exe = b.addExecutable(.{
        .name = "shot_boundary_zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const clap_dep = b.dependency("clap", .{ .target = target, .optimize = optimize });
    exe.root_module.addImport("clap", clap_dep.module("clap"));
    exe.root_module.addImport("spec", spec_module);
    exe.root_module.addImport("runtime_model_impl", runtime_impl);
    addRuntimeLink(exe.root_module, platform_runtime);
    if (runtimeBuildStep(platform_runtime)) |s| exe.step.dependOn(s);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run unit tests");
    const exe_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe_tests.root_module.addImport("clap", clap_dep.module("clap"));
    exe_tests.root_module.addImport("spec", spec_module);
    exe_tests.root_module.addImport("runtime_model_impl", runtime_impl);
    addRuntimeLink(exe_tests.root_module, platform_runtime);
    if (runtimeBuildStep(platform_runtime)) |s| exe_tests.step.dependOn(s);
    test_step.dependOn(&b.addRunArtifact(exe_tests).step);

    const fmt_step = b.step("fmt", "Check code formatting");
    const fmt_check = b.addFmt(.{ .paths = &.{ "src", "build.zig", "build.zig.zon" }, .check = true });
    fmt_step.dependOn(&fmt_check.step);
    test_step.dependOn(fmt_step);

    const lint_step = b.step("lint", "Run ziglint");
    const ziglint_dep = b.dependency("ziglint", .{ .optimize = .ReleaseFast });
    lint_step.dependOn(ziglint.addLint(b, ziglint_dep, &.{ b.path("src"), b.path("build.zig") }));
    test_step.dependOn(lint_step);

    switch (platform_runtime) {
        .mlx => |mlx| {
            const mlx_smoke = b.addExecutable(.{
                .name = "mlx_smoke",
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/mlx_smoke.zig"),
                    .target = target,
                    .optimize = optimize,
                }),
            });
            addMlxLink(mlx_smoke.root_module, mlx.paths);
            if (mlx.build_step) |s| mlx_smoke.step.dependOn(s);

            const mlx_smoke_run = b.addRunArtifact(mlx_smoke);
            const mlx_smoke_step = b.step("mlx-smoke", "Run a Zig -> MLX-C link and execution smoke test");
            mlx_smoke_step.dependOn(&mlx_smoke_run.step);
        },
        .ort, .unsupported => {},
    }

    const setup_step = b.step("setup", "Fetch and build the platform runtime dependency");
    switch (platform_runtime) {
        .mlx => |mlx| if (mlx.build_step) |s| setup_step.dependOn(s),
        .ort => |ort| if (ort.build_step) |s| setup_step.dependOn(s),
        .unsupported => {},
    }
}

const PlatformRuntime = union(enum) {
    mlx: MlxSetup,
    ort: OrtSetup,
    unsupported,
};

const MlxPaths = struct {
    include_dir: []const u8,
    mlxc_lib_dir: []const u8,
    mlx_c_build_dir: []const u8,
    mlx_lib_dir: []const u8,
};

const MlxSetup = struct {
    paths: MlxPaths,
    /// Non-null when MLX-C must be fetched and built from source.
    build_step: ?*std.Build.Step,
};

const OrtPaths = struct {
    include_dir: []const u8,
    lib_dir: []const u8,
};

const OrtSetup = struct {
    paths: OrtPaths,
    /// Non-null when ONNX Runtime must be downloaded and unpacked.
    build_step: ?*std.Build.Step,
};

const OrtRelease = struct {
    archive_name: []const u8,
    install_dir_name: []const u8,
    sha256: []const u8,
};

fn createRuntimeModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    runtime: PlatformRuntime,
    spec_module: *std.Build.Module,
    runtime_options: *std.Build.Step.Options,
) *std.Build.Module {
    const root_source_file = switch (runtime) {
        .mlx => b.path("src/mlx_model.zig"),
        .ort => b.path("src/onnx_model.zig"),
        .unsupported => b.path("src/unsupported_model.zig"),
    };
    const module = b.createModule(.{
        .root_source_file = root_source_file,
        .target = target,
        .optimize = optimize,
    });
    module.addImport("spec", spec_module);
    module.addOptions("runtime_options", runtime_options);
    addRuntimeLink(module, runtime);
    return module;
}

fn setupMlx(b: *std.Build) MlxSetup {
    const user_prefix = b.option([]const u8, "mlx-c-prefix", "Path to a pre-built MLX-C install prefix");
    const user_build_dir = b.option([]const u8, "mlx-c-build-dir", "Path to an MLX-C CMake build directory");

    const prefix = user_prefix orelse mlx_c_install_dir;
    const build_dir = user_build_dir orelse mlx_c_build_dir;

    const paths: MlxPaths = .{
        .include_dir = b.pathJoin(&.{ prefix, "include" }),
        .mlxc_lib_dir = b.pathJoin(&.{ prefix, "lib" }),
        .mlx_c_build_dir = build_dir,
        .mlx_lib_dir = b.pathJoin(&.{ build_dir, "_deps", "mlx-build" }),
    };

    // User provided explicit paths; skip automatic fetch/build.
    if (user_prefix != null or user_build_dir != null) {
        return .{ .paths = paths, .build_step = null };
    }

    const fetch = b.addSystemCommand(&.{
        "sh", "-c",
        "test -d " ++ mlx_c_src_dir ++ "/.git || " ++
            "git clone --depth 1 --branch " ++ mlx_c_version ++
            " https://github.com/ml-explore/mlx-c.git " ++ mlx_c_src_dir,
    });

    const configure = b.addSystemCommand(&.{
        "sh", "-c",
        "test -f " ++ mlx_c_cmake_cache ++ " || " ++
            "cmake -S " ++ mlx_c_src_dir ++
            " -B " ++ mlx_c_build_dir ++
            " -DCMAKE_BUILD_TYPE=Release" ++
            " -DBUILD_SHARED_LIBS=ON" ++
            " -DMLX_C_BUILD_EXAMPLES=OFF",
    });
    configure.step.dependOn(&fetch.step);

    const cmake_build = b.addSystemCommand(&.{
        "sh", "-c",
        "test -f " ++ mlx_c_build_lib ++
            " && test -f " ++ mlx_build_lib ++
            " || cmake --build " ++ mlx_c_build_dir ++ " -j",
    });
    cmake_build.step.dependOn(&configure.step);

    // CMake install is not fully idempotent on macOS rpaths, so skip it once
    // the install prefix has both libraries and public headers.
    const install = b.addSystemCommand(&.{
        "sh", "-c",
        "test -f " ++ mlx_c_install_lib ++
            " && test -f " ++ mlx_install_lib ++
            " && test -f " ++ mlx_c_install_header ++
            " || cmake --install " ++ mlx_c_build_dir ++ " --prefix " ++ mlx_c_install_dir,
    });
    install.step.dependOn(&cmake_build.step);

    return .{ .paths = paths, .build_step = &install.step };
}

fn setupOrt(b: *std.Build, target_arch: std.Target.Cpu.Arch, enable_cuda: bool) OrtSetup {
    const user_prefix = b.option([]const u8, "onnxruntime-prefix", "Path to a pre-built ONNX Runtime install prefix");
    const release = ortRelease(target_arch, enable_cuda);
    const install_dir = b.pathJoin(&.{ ort_root_dir, release.install_dir_name });
    const prefix = user_prefix orelse install_dir;
    const paths: OrtPaths = .{
        .include_dir = b.pathJoin(&.{ prefix, "include" }),
        .lib_dir = b.pathJoin(&.{ prefix, "lib" }),
    };

    if (user_prefix != null) {
        return .{ .paths = paths, .build_step = null };
    }

    const archive_path = b.pathJoin(&.{ ort_root_dir, release.archive_name });
    const archive_url = b.fmt(
        "https://github.com/microsoft/onnxruntime/releases/download/v{s}/{s}",
        .{ ort_version, release.archive_name },
    );
    const install_header = b.pathJoin(&.{ install_dir, "include", "onnxruntime_c_api.h" });
    const install_lib = b.pathJoin(&.{ install_dir, "lib", "libonnxruntime.so" });
    const fetch_cmd = b.fmt(
        "mkdir -p {s} && test -f {s} || curl -L {s} -o {s}",
        .{ ort_root_dir, archive_path, archive_url, archive_path },
    );
    const verify_cmd = b.fmt(
        "printf '%s  %s\\n' '{s}' '{s}' | sha256sum --status -c -",
        .{ release.sha256, archive_path },
    );
    const unpack_cmd = b.fmt(
        "test -f {s} && test -f {s} || tar -xzf {s} -C {s}",
        .{ install_header, install_lib, archive_path, ort_root_dir },
    );
    const fetch = b.addSystemCommand(&.{
        "sh",
        "-c",
        fetch_cmd,
    });

    const verify = b.addSystemCommand(&.{
        "sh",
        "-c",
        verify_cmd,
    });
    verify.step.dependOn(&fetch.step);

    const unpack = b.addSystemCommand(&.{
        "sh",
        "-c",
        unpack_cmd,
    });
    unpack.step.dependOn(&verify.step);

    return .{ .paths = paths, .build_step = &unpack.step };
}

fn ortRelease(target_arch: std.Target.Cpu.Arch, enable_cuda: bool) OrtRelease {
    if (enable_cuda) {
        if (target_arch != .x86_64) {
            @panic("ONNX Runtime CUDA release is only wired for Linux x86_64");
        }
        return .{
            .archive_name = "onnxruntime-linux-x64-gpu_cuda13-" ++ ort_version ++ ".tgz",
            .install_dir_name = "onnxruntime-linux-x64-gpu-" ++ ort_version,
            .sha256 = "fdc6eb18317b4eaeda8b3b86595e5da7e853f72bac67ccac9b04ffc20c9f7fe0",
        };
    }

    return switch (target_arch) {
        .x86_64 => .{
            .archive_name = "onnxruntime-linux-x64-" ++ ort_version ++ ".tgz",
            .install_dir_name = "onnxruntime-linux-x64-" ++ ort_version,
            .sha256 = "3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747",
        },
        .aarch64 => .{
            .archive_name = "onnxruntime-linux-aarch64-" ++ ort_version ++ ".tgz",
            .install_dir_name = "onnxruntime-linux-aarch64-" ++ ort_version,
            .sha256 = "866109a9248d057671a039b9d725be4bd86888e3754140e6701ec621be9d4d7e",
        },
        else => @panic("ONNX Runtime CPU release is only wired for Linux x86_64 and aarch64"),
    };
}

fn addRuntimeLink(module: *std.Build.Module, runtime: PlatformRuntime) void {
    switch (runtime) {
        .mlx => |mlx| addMlxLink(module, mlx.paths),
        .ort => |ort| addOrtLink(module, ort.paths),
        .unsupported => {},
    }
}

fn runtimeBuildStep(runtime: PlatformRuntime) ?*std.Build.Step {
    return switch (runtime) {
        .mlx => |mlx| mlx.build_step,
        .ort => |ort| ort.build_step,
        .unsupported => null,
    };
}

fn addMlxLink(module: *std.Build.Module, paths: MlxPaths) void {
    const b = module.owner;
    module.addIncludePath(lazyPath(b, paths.include_dir));
    module.addLibraryPath(lazyPath(b, paths.mlxc_lib_dir));
    module.addLibraryPath(lazyPath(b, paths.mlx_c_build_dir));
    module.addLibraryPath(lazyPath(b, paths.mlx_lib_dir));
    module.addRPath(lazyPath(b, paths.mlxc_lib_dir));
    module.addRPath(lazyPath(b, paths.mlx_c_build_dir));
    module.addRPath(lazyPath(b, paths.mlx_lib_dir));
    module.linkSystemLibrary("c++", .{});
    module.linkSystemLibrary("mlxc", .{ .needed = true, .use_pkg_config = .no });
    module.linkSystemLibrary("mlx", .{ .needed = true, .use_pkg_config = .no });
}

fn addOrtLink(module: *std.Build.Module, paths: OrtPaths) void {
    const b = module.owner;
    module.link_libc = true;
    module.addIncludePath(lazyPath(b, paths.include_dir));
    module.addLibraryPath(lazyPath(b, paths.lib_dir));
    module.addRPathSpecial("$ORIGIN/../lib");
    module.addRPath(lazyPath(b, paths.lib_dir));
    module.linkSystemLibrary("onnxruntime", .{ .needed = true, .use_pkg_config = .no });
}

fn lazyPath(b: *std.Build, path: []const u8) std.Build.LazyPath {
    return if (std.fs.path.isAbsolute(path)) .{ .cwd_relative = path } else b.path(path);
}
