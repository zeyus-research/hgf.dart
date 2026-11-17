import 'dart:typed_data';

import 'package:hgf_playground/src/model/fit_engine.dart';
import 'package:hgf_playground/src/model/fit_jobs.dart';
import 'package:hgf_playground/src/model/fit_wire.dart';

/// Jobs, events and results through the web fit host's wire format (as the
/// page and the worker exchange them), run on a [LocalFitEngine].
final class WireEngine implements FitEngine {
  WireEngine(this.inner);

  final LocalFitEngine inner;
  var jobs = 0, events = 0;

  @override
  Future<GpuInfo> probeGpu() async =>
      decodeGpuInfo(encodeGpuInfo(await inner.probeGpu()));

  @override
  Future<R> run<R>(FitJobSpec<R> job, void Function(JobEvent) onEvent) async {
    final m = encodeJob(job);
    final there = decodeJob(m.json, _copied(m.arrays));
    jobs++;
    final result = await inner.run(there, (e) {
      events++;
      onEvent(decodeEvent(encodeEvent(e)));
    });
    final r = encodeResult(result);
    return decodeResult(job, r.json, _copied(r.arrays));
  }

  // Structured clone: copies.
  static List<Float64List> _copied(List<Float64List> a) => [
    for (final x in a) Float64List.fromList(x),
  ];

  @override
  void cancel() => inner.cancel();

  @override
  Future<void> close() => inner.close();
}
