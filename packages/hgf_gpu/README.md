# hgf_gpu

Batched HGF filtering on the GPU. `HgfKernel` compiles an `hgf` `Network`
into a WGSL compute shader. Each GPU thread filters one parameter set,
with the update schedule unrolled and its state in registers. `GpuBatch`
runs that shader on WebGPU in the browser, or on wgpu-native on desktop.
Both use the same host code.

```dart
final kernel = HgfKernel(network);                 // TotalGaussianSurprise
final backend = await WebGpuBackend.create();      // or WgpuBackend.create()
final gpu = await GpuBatch.create(backend!, kernel, omegas.length)
  ..setInitial(1, NodeField.tonicVolatility, omegas);
final nll = await gpu.surprise(series);            // one total per ω
```

Status: phase 7 probe (see `docs/ROADMAP.md`).
- Supported: continuous, binary and constant nodes; standard, eHGF and
  unbounded updates; named couplings; missing observations; time steps.
- Precision: f32, as accurate as pyhgf's float32 mode.
- Not yet: vector node kinds, custom couplings, gradients.

- Web: `import 'package:hgf_gpu/web.dart'`. On Linux, Chromium needs
  `--enable-unsafe-webgpu` and Vulkan.
- Desktop: `import 'package:hgf_gpu/native.dart'`. Needs `libwgpu_native`
  v29: install it, or run `tool/fetch_wgpu.sh` and set `WGPU_LIB`.
  Regenerate the bindings with
  `dart run ffigen --config ffigen.yaml --compiler-opts -I/usr/lib/clang/<v>/include`.
- Tests: `melos run test:gpu`. Benchmarks: `melos run bench:gpu-web`,
  `melos run bench:gpu-native`.
