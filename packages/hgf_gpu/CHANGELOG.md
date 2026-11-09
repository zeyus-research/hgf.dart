## Unreleased

- Phase 7 probe: `HgfKernel` (Network → WGSL compute shader, one thread
  per parameter set), `GpuBatch` (per-element parameters, GPU-side reset,
  dirty-slot uploads), `WebGpuBackend` (`dart:js_interop`) and
  `WgpuBackend` (wgpu-native v29 through `dart:ffi`).
- GPU fitting: `FitKernel` compiles an `HgfModel`'s likelihood and, with
  forward-mode tangents, its exact gradient; `GpuFitter` is a
  `hgf_fit` `PassRunner`, so `fitMap`, `sampleNuts`, `sampleHierarchical`,
  `laplace` and `fitEmpiricalBayes` run on the GPU.
- The code generator is built on an expression IR (`ir.dart`) with
  derivative rules; `HgfKernel` and `FitKernel` share the trial compiler.
- Responses with parameters and response data (`GpuGaussianReadout`,
  `GpuBinarySoftmax`, `GpuBatch.setResponseParameter`, `responses:`).
- Binary nodes carry `1 − μ̂` as `π̂ / μ̂` instead of subtracting: binary
  fixtures that drifted 5e-4 to 1e-2 in f32 now match f64 to ~3e-6.
- `GpuBackend.writeUints`.
- `GpuFitter.cancel()` (jobs fail with `FitCancelled`) and
  `GpuFitter.yieldInterval`.
- `WgpuIsolateFitter` (`native.dart`): a `GpuFitter` on wgpu-native in a
  background isolate, for UIs (wgpu-native's calls block). The drivers stay
  on the calling isolate; the model and subjects cross once, then each
  pass's points and results. Same results as `GpuFitter`, ~3% slower.
