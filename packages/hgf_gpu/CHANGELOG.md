## Unreleased

- Phase 7 probe: `HgfKernel` (Network → WGSL compute shader, one thread
  per parameter set), `GpuBatch` (per-element parameters, GPU-side reset,
  dirty-slot uploads), `WebGpuBackend` (`dart:js_interop`) and
  `WgpuBackend` (wgpu-native v29 through `dart:ffi`).
