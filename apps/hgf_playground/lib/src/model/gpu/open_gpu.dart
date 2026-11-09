import 'package:hgf_fit/hgf_fit.dart';

import 'open_gpu_stub.dart'
    if (dart.library.io) 'open_gpu_native.dart'
    if (dart.library.js_interop) 'open_gpu_web.dart'
    as platform;

/// A GPU the fits can run on.
abstract interface class GpuDevice {
  /// The adapter, for the settings panel.
  String get description;

  /// A runner answering passes on this GPU, the precise ones (Laplace's
  /// Hessian, reported values) on [cpu].
  GpuRunner runner(FitWorkers cpu);

  void close();
}

/// A [PassRunner] on a [GpuDevice].
abstract interface class GpuRunner implements PassRunner {
  /// Passes answered on the GPU so far.
  int get gpuPasses;

  /// Why the last model that fell back to the CPU has no GPU kernel.
  String? get fallbackReason;

  /// Stops the running job ([FitCancelled]).
  void cancel();
}

/// The GPU, or why there is none.
typedef GpuOpened = ({GpuDevice? device, String? reason});

/// Opens this platform's GPU: WebGPU in the browser; on desktop,
/// wgpu-native (`libwgpu_native` on the library path, or `$WGPU_LIB`) in a
/// background isolate, since its calls block.
Future<GpuOpened> openGpu() async {
  try {
    return await platform.openGpu();
  } catch (e) {
    return (device: null, reason: '$e');
  }
}
