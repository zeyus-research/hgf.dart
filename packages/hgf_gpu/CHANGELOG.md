## Unreleased

- `GpuDeepEnsemble`: hgf_deep's sequential `fit` for many networks of one
  shape at once (one workgroup per member runs the sample loop in one
  kernel, `SequentialFitKernel.sequentialFit` on
  `DeepKernels.sequential`); shared or per-member data, SGD/Adam (bias
  corrections from a host table), `upload`/`download`/`downloadAll`,
  dispatches sized to ~50 ms so drivers don't reset the device.
- `DeepKernels`' epilogues are factored out for reuse (the batch kernels'
  WGSL is byte-identical); `DeepBinding.biasCorrection`.
- `GpuDeepNetwork`: hgf_deep's `DeepNetwork` on the GPU (`batchUpdate`,
  `trainEpoch`, `predict`, `upload`/`download`), from `DeepKernels`, which
  generates one fused GEMM-plus-epilogue kernel per layer, a split-K weight
  gradient fused with the SGD/Adam step, and the carried-belief reduction.
- `GpuBackend` records copies and dispatches into one command buffer (one
  compute pass for consecutive dispatches) and submits at the next write,
  read, `idle` or the new `flush`; bind groups are cached. `dispatch` takes
  a second grid dimension and `null` bindings (left out of the group).

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
