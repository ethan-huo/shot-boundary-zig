# Build System Reference

## Quick Start

```bash
zig build          # fetch platform runtime dependency + compile
zig build run      # build and run the application
zig build test     # fmt + lint + unit tests
zig build setup    # fetch/build the platform runtime dependency
```

On macOS, `zig build mlx-smoke` is also available. Linux builds do not create MLX targets.

## Platform Runtime Selection

The runtime is selected by compile target:

| Target OS | Runtime | Model path |
|-----------|---------|------------|
| macOS | MLX-C | `.safetensors` |
| Linux | ONNX Runtime | `.onnx` |

There is no CLI backend flag.

## Linux ONNX Runtime

Linux builds use ONNX Runtime v1.24.4. By default the build downloads the CPU package for the target architecture into gitignored `externals/onnxruntime/`; currently `x86_64` and `aarch64` Linux CPU packages are wired.

```bash
zig build -Doptimize=ReleaseFast
```

To point at a prebuilt ORT prefix:

```bash
zig build -Donnxruntime-prefix=/path/to/onnxruntime
```

CUDA EP is intentionally not part of the default Linux target. To opt into the x86_64 CUDA EP while investigating CUDA graph differences:

```bash
zig build -Doptimize=ReleaseFast -Donnxruntime-cuda=true
```

CUDA EP requires the matching CUDA/cuDNN runtime libraries on `LD_LIBRARY_PATH` or installed in the system loader path. Do not hard-code host-specific CUDA library paths in source.

## macOS MLX-C Dependency

MLX-C is a C wrapper around Apple's MLX framework. It is fetched and built automatically via CMake on first macOS `zig build`.

Each external stage has an artifact guard:
- fetch is skipped when `externals/mlx-c/src/.git` exists
- configure is skipped when `externals/mlx-c/build/CMakeCache.txt` exists
- build is skipped when both build-local `libmlxc.dylib` and `libmlx.dylib` exist
- install is skipped when the install prefix has both dylibs and the public `mlx/c/mlx.h` header

### Dependency chain

```
transnetv2_zig (Zig legacy binary name) -> MLX-C v0.6.0 (C++, CMake) -> MLX v0.31.1
```

### Overriding with pre-built MLX-C

```bash
zig build -Dmlx-c-prefix=/path/to/install -Dmlx-c-build-dir=/path/to/build
```

## Other Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| `clap` | CLI argument parsing | 0.11.0 |
| `ziglint` | Zig linter | 0.5.2 |

## Build Steps

| Step | Linux | macOS |
|------|-------|-------|
| `zig build` | ORT setup + compile | MLX-C setup + compile |
| `zig build run` | ORT setup + run | MLX-C setup + run |
| `zig build test` | fmt + lint + tests | fmt + lint + tests |
| `zig build fmt` | format check | format check |
| `zig build lint` | ziglint | ziglint |
| `zig build mlx-smoke` | unavailable | MLX-C smoke test |
| `zig build setup` | ORT setup | MLX-C setup |

## Linux Release Artifacts

GitHub Actions builds Linux CLI release archives for:
- `x86_64-linux-gnu`
- `aarch64-linux-gnu`

The release archive includes `bin/transnetv2_zig`, ONNX Runtime shared libraries under `lib/`, and ONNX Runtime notices under `third_party/onnxruntime/`. The executable is linked with `$ORIGIN/../lib` rpath so the bundled libraries are found after unpacking.

Versioning is tag-first:
- Pushing a tag like `v0.1.0` builds both Linux archives and publishes a GitHub Release with those assets.
- Manual `workflow_dispatch` builds artifacts without publishing a release. If the optional `version` input is empty, artifact names use `YYYYMMDD-shortsha`.

macOS release artifacts are intentionally not wired in GitHub Actions yet; that path should be confirmed separately because the MLX-C/MLX stack depends on Apple's macOS toolchain and Metal environment.
