# AutoShot Architecture Reference

> Paper: *AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection* (CVPR 2023)
> Authors: Wentao Zhu et al. (Kuaishou + UT Austin)
> Code: https://github.com/wentaozhu/AutoShot

AutoShot is not a novel architecture — it is a NAS-searched variant built on TransNetV2's building blocks.
The best architecture is a pure (2+1)D factored-convolution CNN; the Transformer was eliminated by NAS (`n_layer=0`).

- 37 GMACs (TransNetV2: 41 GMACs)
- F1: +1.1% on ClipShots, +4.2% on SHOT dataset
- PapersWithCode SBD leaderboard Top-1 (as of 2025)

## Architecture comparison

| | TransNetV2 | AutoShot@F1 |
|---|---|---|
| Core conv | SeparableConv3d (conv2d+conv1d) | identical |
| DDCNN blocks | 3 layers x 2 blocks, n_dilation=4 | 6 blocks, n_dilation=4-5 |
| Spatial conv | Independent per dilation branch | Blocks 1-3 share one 2D conv (DDCNNV2A) |
| Transformer | none | designed in SuperNet, eliminated by NAS |
| FrameSimilarity | yes | yes |
| ColorHistograms | yes | yes |
| Classification head | fc1 -> cls | identical |

## AutoShot@F1 block layout

```
Block 0: DDCNNV2   (n_c=4F,  n_d=4)   <- original TransNetV2 style
Block 1: DDCNNV2A  (n_c=4F,  n_d=5)   <- shared spatial conv
Block 2: DDCNNV2A  (n_c=4F,  n_d=5)
Block 3: DDCNNV2A  (n_c=4F,  n_d=5)
Block 4: DDCNNV2   (n_c=12F, n_d=5)
Block 5: DDCNNV2   (n_c=8F,  n_d=5)
Block 6: Attention1D, n_layer=0        <- NAS chose no Transformer
```

## ONNX export: gather_nd workaround

Upstream `FrameSimilarity` / `ColorHistograms` share a `gather_nd` that calls `.tolist()` on indices, causing TorchScript to trace them as constants. A model exported with batch=1 dummy input will fail at batch>1 in ONNX Runtime (`ScatterElements` shape error in `color_hist_layer`).

The export scripts replace `gather_nd` with a `torch.gather`-based shim:

```python
similarities_padded = F.pad(similarities, [50, 50, 0, 0, 0, 0])
time_indices = torch.arange(time_window, device=similarities.device).reshape(1, time_window, 1)
lookup_indices = torch.arange(lookup_window, device=similarities.device).reshape(1, 1, lookup_window) + time_indices
lookup_indices = lookup_indices.expand(batch_size, time_window, lookup_window)
similarities = torch.gather(similarities_padded, 2, lookup_indices)
```

This is a hard requirement for dynamic-batch ONNX export. If the export scripts are ever rewritten, this shim must be preserved.
