/// GPU kernels for batched hgf networks (phase 7 probe).
///
/// [HgfKernel] compiles a `Network` into a WGSL compute shader; a
/// [GpuBatch] runs it on a [GpuBackend] (WebGPU in the browser, wgpu-native
/// on desktop).
library;

export 'src/batch.dart';
export 'src/wgsl.dart';
