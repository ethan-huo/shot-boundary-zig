# Shot Boundary Web

Browser prototype for TransNetV2 shot-boundary detection.

## Setup

```sh
cd web
bun install
bun run dev
```

The app serves model files through `web/public/models`, which is a symlink to the repository root `models/` directory. The default model URL is:

```text
/models/transnetv2.onnx
```

## Commands

```sh
bun run typecheck
bun test
bun run build
bun run smoke:model public/models/transnetv2.onnx
```

## Pipeline

- Decode video samples with Mediabunny `BlobSource` and `CanvasSink`.
- Resize frames to `48x27`.
- Pack RGB24 bytes into `uint8 [batch, 100, 27, 48, 3]`.
- Run ONNX inference with `onnxruntime-web`.
- Convert `single_frame` predictions into scenes.
- Render a thumbnail timeline with markers at scene boundaries.

## Notes

The thumbnail timeline is an editor-style visualization, not a parity guarantee. For strict parity, compare the RGB window bytes against the native ffmpeg path.
