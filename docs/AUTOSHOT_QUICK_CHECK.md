# AutoShot Quick Check

Branch: `explore/autoshot-runtime`

This is the fast decision check for whether AutoShot should replace the current TransNetV2 ONNX path.

## Inputs

- AutoShot upstream shallow clone: `/tmp/autoshot-codex`
- Upstream commit: `77c82ff826a9301bb173d9be786297a49d73d081`
- Checkpoint: `target/models/ckpt_0_200_0.pth`
- Checkpoint sha256: `3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6`
- Exported ONNX: `target/models/autoshot.onnx`
- Current baseline ONNX: `target/models/transnetv2.onnx`
- Local speed sample: `assets/333.mp4`

The checkpoint loads cleanly into `TransNetV2Supernet`:

- state tensors: `90`
- missing keys: `0`
- unexpected keys: `0`
- model parameters: `14,299,202`

## Accuracy

Using the official SHOT test-set prediction pickles from the AutoShot Google Drive folder:

| Model | F1 | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|
| TransNetV2 baseline | 0.799290 | 0.904165 | 0.716216 | 0.02 |
| AutoShot@F1 | 0.840545 | 0.847334 | 0.833863 | 0.296 |

Result: AutoShot improves F1 by `+0.041255` on the official 200-video SHOT test prediction set. This confirms the upstream accuracy claim well enough for the quick check.

## ONNX Export

The exported AutoShot ONNX keeps the current runtime ABI:

- input: `frames`, `uint8 [batch, 100, 27, 48, 3]`
- outputs: `single_frame`, `many_hot`, `float32 [batch, 50]`

Dynamic batch smoke against PyTorch passed for batch sizes 1, 2, and 3:

| Batch | Max abs diff |
|---:|---:|
| 1 | ~1.0e-7 |
| 2 | ~9.1e-8 |
| 3 | ~8.6e-8 |

The export must keep the `torch.gather` shim for `FrameSimilarity` and `ColorHistograms`; direct export of upstream `gather_nd(...).tolist()` produces a batch=1-only graph.

## Speed

CPU ONNX Runtime speed on the first 8 windows of `assets/333.mp4`, batch size 1, 3 runs:

| Model | Mean inference ms | Effective FPS |
|---|---:|---:|
| AutoShot ONNX | 1945.261 | 205.628 |
| TransNetV2 ONNX | 1445.245 | 276.770 |

AutoShot speedup vs TransNetV2: `0.743x`.

AutoShot-only batch size 2 on the same 8 windows was slower:

| Model | Batch | Mean inference ms | Effective FPS |
|---|---:|---:|---:|
| AutoShot ONNX | 2 | 3965.192 | 100.878 |

The current `target/models/transnetv2.onnx` failed under Python ONNX Runtime with batch size 2 in `ScatterElements`, so the directly comparable speed number is batch size 1.

## Decision

AutoShot accuracy is confirmed higher, but speed is not confirmed higher on the current CPU ONNX backend. We are proceeding with the full implementation because the accuracy gain is the deciding factor:

- Proceed with AutoShot as the default model direction.
- Do not switch solely on the expectation that AutoShot is faster in our ONNX Runtime CPU path.
- Before a full transition, rerun speed on the intended release hardware and optionally test ORT graph optimization / CUDA separately.
