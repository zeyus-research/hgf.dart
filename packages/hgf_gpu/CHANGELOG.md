## Unreleased

- dp-state nodes in the kernels (`HgfKernel`, `FitKernel`, so `GpuFitter`
  runs `DpSurprise` fits on the GPU; Nelder–Mead, as the likelihood has no
  gradients). A network whose trial runs the dp steps compiles to a
  cooperative kernel (`cooperative`): one workgroup of 256 invocations per
  element (`workgroups(n)`, along x then y), which all run the trial and
  share pyhgf's candidate search over its 20 000 proposals (bound after
  the other buffers from `dpProposals(backend)`, uploaded once per
  backend). `GpuBatch.readVector` reads a dp node's cluster counts,
  expected means and sigmas, `read` its surprise; `setInitial` sets α
  and the sensory precision per element. All six dp fixtures match f64
  within 4.1e-7 (Metal and SwiftShader). A likelihood that reads only
  other nodes skips the dp steps (and stays one invocation per element).
  `benchmark/bin/fit_dp_gpu.dart`.
- categorical-state nodes in the kernels (`HgfKernel`, `FitKernel`, so
  `GpuFitter` runs categorical models on the GPU), with
  `CategoricalSurprise` and `CategoricalSoftmaxReadout` (exact gradients,
  forward and reverse mode). `GpuBatch.readVector` / `setInitialVector`
  read and set a categorical node's α and probabilities (and ef ξ);
  `read` takes its `klDivergence` and `surprise`. Unobserved, the kernel
  keeps α (WGSL has no infinities); the surprise follows
  `NetworkFixes.categoricalMissing`. All seven categorical fixtures match
  f64 on Metal: totals, probabilities and surprise within 3e-6, α within
  6.7e-6, the KL divergence within 1.9e-6 of `ln Γ(Σα)`.
- ef-state nodes in the kernels (`HgfKernel`, `FitKernel`, so `GpuFitter`
  no longer falls back to the CPU for them): fixed and HGF learning, any
  dimension, beside scalar inputs. A trial's data row is
  `Network.nInputColumns` wide (`HgfKernel.nColumns`) and observed flags
  are one per input node (`nInputs`), as on the CPU; `GpuFitter` and
  `GpuBatch` take subjects and masks in that layout. `GpuBatch` reads and
  sets ξ (`readXis`, `setInitialXis`) and ν (`NodeField.nus`). ν's δ/Δ
  is taken as 1/K, K the gain of the update of the node that scales it
  (in f32 an observation often rounds to its prediction, δ = Δ = 0). All
  eleven ef fixtures match f64 on Metal (totals and ξ ~3e-6, ν ~2e-6).
- hgf_deep's newer learning options on the GPU, in `GpuDeepNetwork`
  (`batchUpdate`, `trainEpoch`) and `GpuDeepEnsemble.fit`: the weight-belief
  rule (`LearningKind.synapticUncertainty` with `synapticUncertainty:`; the
  evidence carried up the layers, the curvature as a second GEMM
  accumulator, the belief step fused into the gradient kernel; weight beliefs
  upload and download with the network), `DeepSettings.updateInputLayer`
  (the root prediction and the top layer's precision-only posterior), and
  `batchUpdate(sampleWeight:)`. All 16 fixtures that use them now run on
  the GPU too (f32, ~1e-7 batched, ~1e-5 sequential). An ensemble needs
  `create(weightBeliefs: true)` for the rule, so other ensembles' parameters
  don't grow. The learning request is checked by hgf_deep's
  `checkLearningRequest`, as on the CPU.
- GEMM tiles halve their depth until they fit WebGPU's default 16 KiB of
  workgroup storage (two-accumulator tiles over wide batches could exceed
  it).
- macOS: the build hook makes room in wgpu-native's Mach-O header for the
  absolute install name that `dart test`/`dart run` (Dart 3.14) and
  `flutter test` write (`hook/macho_headroom.dart`: drops
  `LC_SOURCE_VERSION` and an empty `LC_DATA_IN_CODE`, then re-signs ad
  hoc), so the user-define `bundle: false` is no longer needed there.
- pyhgf #478's time-step changes in the kernels, through the IR.
- Noise parents of input nodes (pyhgf #477) in the kernels, through the
  IR; the WGSL `lambertW0` no longer overflows (pyhgf #475).
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
