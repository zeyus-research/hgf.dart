import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/native.dart';

import 'open_gpu.dart';

Future<GpuOpened> openGpu() async {
  try {
    final fitter = await WgpuIsolateFitter.spawn();
    return (device: _NativeGpu(fitter), reason: null);
  } on StateError catch (e) {
    return (device: null, reason: '${e.message}.');
  }
}

/// wgpu-native in a background isolate: one fitter, whose CPU side follows
/// the current pool.
final class _NativeGpu implements GpuDevice, GpuRunner {
  _NativeGpu(this._fitter);

  final WgpuIsolateFitter _fitter;

  @override
  String get description => _fitter.description;

  @override
  GpuRunner runner(FitWorkers cpu) {
    _fitter.cpu = cpu;
    return this;
  }

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

  @override
  void close() => _fitter.close();
}
