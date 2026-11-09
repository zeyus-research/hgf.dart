import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/web.dart';

import 'open_gpu.dart';

Future<GpuOpened> openGpu() async {
  final b = await WebGpuBackend.create();
  return b == null
      ? (
          device: null,
          reason:
              'This browser has no WebGPU (or no adapter). Chrome and Edge '
              'have it; Firefox and Safari in recent versions.',
        )
      : (device: _WebGpu(b), reason: null);
}

/// WebGPU: its calls are asynchronous, so the fitter runs on the page's
/// thread and frames draw while it waits.
final class _WebGpu implements GpuDevice {
  _WebGpu(this._backend);

  final WebGpuBackend _backend;
  _WebRunner? _runner;

  @override
  String get description => _backend.description;

  @override
  GpuRunner runner(FitWorkers cpu) {
    // A fresh fitter per fit (each fit compiles its own model's kernel
    // anyway): the last one's buffers and pipelines are released.
    _runner?._fitter.dispose();
    return _runner = _WebRunner(
      GpuFitter(_backend, cpu: cpu)
        // The host work between passes is short, but a long job still
        // gives way now and then.
        ..yieldInterval = const Duration(milliseconds: 12),
    );
  }

  @override
  void close() => _runner?._fitter.dispose();
}

final class _WebRunner implements GpuRunner {
  _WebRunner(this._fitter);

  final GpuFitter _fitter;

  @override
  Future<T> run<T>(Program<T> program) => _fitter.run(program);

  @override
  int get streams => _fitter.streams;

  @override
  int get gpuPasses => _fitter.gpuPasses;

  @override
  String? get fallbackReason => _fitter.fallbackReason;

  @override
  void cancel() => _fitter.cancel();
}
