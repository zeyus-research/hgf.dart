/// GPU kernels for batched hgf networks.
///
/// [HgfKernel] compiles a `Network` into a WGSL compute shader; a
/// [GpuBatch] runs it on a [GpuBackend] (WebGPU in the browser, wgpu-native
/// on desktop). For fitting, [GpuFitter] answers hgf_fit's batched passes
/// on the GPU ([FitKernel]: the model's likelihood and its exact gradient),
/// so `fitMap`, `sampleNuts`, `sampleHierarchical` and friends run there:
/// `gpu.sampleNuts(model, subjects)`.
library;

export 'src/batch.dart';
export 'src/fit/fit_kernel.dart' show FitKernel, gpuResponseOf;
export 'src/fit/gpu_fitter.dart';
export 'src/wgsl.dart';
