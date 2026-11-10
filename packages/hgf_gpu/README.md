# hgf_gpu

Batched HGF filtering and fitting on the GPU. `HgfKernel` compiles an `hgf`
`Network` into a WGSL compute shader. Each GPU thread filters one parameter
set, with the update schedule unrolled and its state in registers.
`GpuBatch` runs that shader on WebGPU in the browser, or on wgpu-native on
desktop. Both use the same host code.

```dart
final kernel = HgfKernel(network);                 // TotalGaussianSurprise
final backend = await WebGpuBackend.create();      // or WgpuBackend.create()
final gpu = await GpuBatch.create(backend!, kernel, omegas.length)
  ..setInitial(1, NodeField.tonicVolatility, omegas);
final nll = await gpu.surprise(series);            // one total per ω
```

For fitting, `GpuFitter` answers `hgf_fit`'s batched passes on the GPU.
`FitKernel` compiles an `HgfModel`'s likelihood, and with forward-mode
tangents its exact gradient. Every `hgf_fit` driver runs on it:

```dart
final gpu = GpuFitter(backend!, cpu: workers);     // workers: optional
final map = await gpu.fitMap(model, subjects);
final post = await gpu.sampleNuts(model, subjects);
final lap = await gpu.laplace(model, map);         // f64, on the CPU
```

Passes that need f64 (finite differences, Hessians, reported estimates)
and models without a kernel run on the CPU.

For `hgf_deep`'s `DeepNetwork` (HGF neural networks), `GpuDeepNetwork`
runs batched learning and prediction. Each layer is one fused kernel: a
tiled GEMM with the layer's belief updates as its epilogue. Weights,
optimiser moments and carried beliefs stay on the GPU:

```dart
final gpu = await GpuDeepNetwork.create(backend!, net, capacity: 1024);
await gpu.trainEpoch(x, y, batchSize: 1024, optimiser: Adam(1e-3));
final yHat = await gpu.predict(xTest);
await gpu.download(net);                           // weights + beliefs back
```

`GpuDeepEnsemble` runs the sequential `fit` (one sample at a time, as
pyhgf's) for many networks of one shape at once: one workgroup per network
runs the whole sample loop in a single kernel. Members can differ in
weights, beliefs and data (one matrix for all, or one per member):

```dart
final nets = [for (var s = 0; s < 1024; s++) build()..weightInitialisation(const XavierInit(), seed: s)];
final ens = await GpuDeepEnsemble.create(backend!, nets);
final predictions = await ens.fit([x], [y], optimiser: Adam(1e-3));
await ens.downloadAll(nets);
```

A single network is no faster than the CPU (each sample is a chain of
small dependent steps); 1024 medium networks train ~200× faster than one
core runs them (`melos run bench:gpu-ensemble`).

Status: see Phase 7 in `docs/ROADMAP.md`.
- Supported: continuous, binary and constant nodes; standard, eHGF and
  unbounded updates; named couplings; missing observations; time steps.
- Responses: observation surprises (`TotalGaussianSurprise`,
  `FirstLevelGaussianSurprise`, `FirstLevelBinarySurprise`) and readouts
  with fitted parameters (`GaussianReadout`, `BinarySoftmaxReadout`).
- Precision: f32. Totals match f64 to ~1e-6 relative and gradients to ~1e-5.
- Not yet: vector node kinds, custom couplings or response models.
- Deep networks: `batchUpdate`, `trainEpoch` (SGD, Adam) and `predict`,
  every layer kind, volatility update and setting `hgf_deep` has; f32,
  matching f64 to ~1e-6. Sequential `fit` for ensembles
  (`GpuDeepEnsemble`), matching f64 `fit` to ~1e-5 until the f64 run itself
  leaves f32's range (pyhgf's runaway, P18); no trajectory recording.

- Web: `import 'package:hgf_gpu/web.dart'`. On Linux, Chromium needs
  `--enable-unsafe-webgpu` and Vulkan.
- Desktop: `import 'package:hgf_gpu/native.dart'`. Needs `libwgpu_native`
  v29: install it, or run `tool/fetch_wgpu.sh` and set `WGPU_LIB`.
  Regenerate the bindings with
  `dart run ffigen --config ffigen.yaml --compiler-opts -I/usr/lib/clang/<v>/include`.
- Tests: `melos run test:gpu`. Benchmarks: `melos run bench:gpu-web`,
  `melos run bench:gpu-native`, `melos run bench:gpu-fit`,
  `melos run bench:gpu-deep`.
