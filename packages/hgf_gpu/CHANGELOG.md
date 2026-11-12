## Unreleased

- Reverse mode in `GpuFitter`: `FitKernel(reverse: true)` (√T
  checkpoints, split across dispatches like the forward pass) and
  `GpuFitter(gradientMode:, reverseAbove:)`, `auto` by default (reverse
  above 150 parameters × carried slots: 2–19× faster where forward mode's
  tangents spill). The unbounded update now has an SSA form, and adjoints
  stay finite through branches not taken. The IR moved to
  `package:hgf_fit/ir.dart` (`src/ir.dart` and `src/trial.dart` re-export
  it); `gpuResponseOf` is `irResponseOf`. `GpuDeepNetwork` and
  `GpuDeepEnsemble` reject hgf_deep's new learning options for now.

- `GpuSoftmax`: hgf_fit's `SoftmaxReadout` on the GPU (`GpuFitter`), with
  tangents; f32 within 2e-7 of f64 on the IGT and bandit cases.
- `GpuDeepEnsemble` keeps each member's working state in workgroup memory
  where that was measured faster (`DeepKernels.sequential(workgroupState:)`):
  always for small states, below 128 members for medium ones (more
  members lose occupancy). Small networks 10–40 % faster, medium 12–16 %
  up to 16 members. `workgroupParams` puts the weights and optimiser
  moments there too (opt-in; no consistent gain measured).
  `benchmark/bin/deep_ensemble.dart --state` compares them.
- `WebGpuBackend` polls the GPU queue while `read` and `idle` wait (an
  empty `queue.submit` every 8th event-loop task through a
  `MessageChannel`, then once per timer tick after 20 ms; `pokeQueue =
  false` turns it off). Firefox resolves `mapAsync` only on a 100 ms
  timer otherwise, so every readback took 100 ms; now 0.25 ms. Chromium's
  round trip went from 1.8–2.5 ms to 0.25–0.35 ms.
- `WebGpuBackend.description` names the adapter even when the browser
  hides `GPUAdapterInfo` (Firefox leaves every field empty): "a hardware
  adapter (Firefox doesn't name it)", or "a software adapter".
- `tool/firefox_webgpu.sh`: Firefox with WebGPU for `dart test -p firefox`
  (`melos run test:gpu-firefox`); every GPU parity suite passes there.

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
