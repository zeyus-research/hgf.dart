/// The phase 7 probe, shared by the browser page (`web/gpu_bench.dart`) and
/// the native binary (`bin/gpu_native.dart`): for each scenario, accuracy of
/// the f32 GPU totals against `BatchNetwork` (f64) on 1000 parameter sets,
/// then GPU time against batch size.
library;

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_gpu/hgf_gpu.dart';

Float64List _linspace(double a, double b, int n) => Float64List.fromList([
  for (var k = 0; k < n; k++) a + (b - a) * k / (n - 1 == 0 ? 1 : n - 1),
]);

double _median(List<double> xs) {
  final s = [...xs]..sort();
  return s[s.length ~/ 2];
}

/// Runs the probe on [backend]: [inputs] is the benchmark's `inputs.json`,
/// [q] the options (`sizes`, `trials`, `chunk`, `repeat`, `cpuRepeat`,
/// `scenarios`), [emit] receives each result.
Future<void> runProbe(
  GpuBackend backend,
  Map<String, Object?> inputs,
  Map<String, String> q,
  void Function(Map<String, Object?>) emit,
) async {
  final sizes = (q['sizes'] ?? '1000,4000,16000,64000,256000,1000000')
      .split(',')
      .map(int.parse)
      .toList();
  final trials = int.parse(q['trials'] ?? '1000');
  final chunk = int.parse(q['chunk'] ?? '0');
  final repeat = int.parse(q['repeat'] ?? '5');
  final cpuRepeat = int.parse(q['cpuRepeat'] ?? '3');

  emit({
    'backend': backend.description,
    'maxStorage': backend.maxStorageBufferBytes,
  });

  final specs = inputs['specs'] as Map<String, Object?>;
  Float64List floats(Object? o) => Float64List.fromList([
    for (final x in o! as List<Object?>) (x! as num).toDouble(),
  ]);
  final continuous = Float64List.sublistView(
    floats(inputs['continuous']),
    0,
    trials,
  );
  final binary = Float64List.sublistView(floats(inputs['binary']), 0, trials);

  Network net(String name, String mode, {bool meanField = false}) {
    final s = jsonDecode(jsonEncode(specs[name])) as Map<String, Object?>;
    final nw = s['network'] as Map<String, Object?>;
    nw['volatility_updates'] = mode;
    nw['mean_field_updates'] = meanField;
    if (name == 'continuous_3level') {
      for (final i in [2, 3]) {
        final node = (s['nodes'] as List<Object?>)[i] as Map<String, Object?>;
        (node['params'] as Map<String, Object?>)['tonic_volatility'] = -6.0;
      }
    }
    return NetworkSpec.fromJson(s).build();
  }

  final all =
      <String, (Network Function(), Float64List, double, double, GpuResponse)>{
        for (final mode in ['standard', 'eHGF', 'unbounded']) ...{
          'c2_$mode': (
            () => net('continuous_2level', mode),
            continuous,
            -10.0,
            -4.0,
            GpuResponse.totalGaussianSurprise,
          ),
          'c3_$mode': (
            () => net('continuous_3level', mode),
            continuous,
            -10.0,
            -4.0,
            GpuResponse.totalGaussianSurprise,
          ),
        },
        for (final mode in ['standard', 'eHGF'])
          'b3_$mode': (
            () => net('binary_3level', mode),
            binary,
            -6.0,
            -1.0,
            GpuResponse.firstLevelBinarySurprise,
          ),
      };
  final chosen = (q['scenarios']?.split(',') ?? all.keys.toList());

  for (final name in chosen) {
    final (make, data, lo, hi, response) = all[name]!;
    final network = make();
    final kernel = HgfKernel(network, response: response);
    final cpuResponse = response == GpuResponse.totalGaussianSurprise
        ? const TotalGaussianSurprise()
        : const FirstLevelBinarySurprise() as ResponseFunction;

    // --- accuracy: 1000 parameter sets, f64 CPU vs f32 GPU ---
    const nAcc = 1000;
    final omegas = _linspace(lo, hi, nAcc);
    final cpu = BatchNetwork(make(), nAcc)
      ..setInitial(1, NodeField.tonicVolatility, omegas);
    final ref = cpu.surprise(data, cpuResponse);
    final cpuTimes = <double>[];
    for (var r = 0; r < cpuRepeat; r++) {
      final sw = Stopwatch()..start();
      cpu.surprise(data, cpuResponse);
      cpuTimes.add(sw.elapsedMicroseconds / 1e3);
    }
    final gpuAcc = await GpuBatch.create(backend, kernel, nAcc);
    gpuAcc.setInitial(1, NodeField.tonicVolatility, omegas);
    final got = await gpuAcc.surprise(data, trialsPerDispatch: chunk);
    gpuAcc.dispose();
    final rel = <double>[];
    var infBoth = 0, infMismatch = 0;
    for (var k = 0; k < nAcc; k++) {
      final a = ref[k], b = got[k];
      if (!a.isFinite || !b.isFinite) {
        if (a.isFinite == b.isFinite) {
          infBoth++;
        } else {
          infMismatch++;
        }
        continue;
      }
      rel.add((a - b).abs() / math.max(a.abs(), 1.0));
    }
    rel.sort();
    // Where the minimum (MAP over this grid) lands.
    int argmin(Float64List xs) {
      var best = 0;
      for (var k = 1; k < xs.length; k++) {
        if (xs[k] < xs[best]) best = k;
      }
      return best;
    }

    emit({
      'scenario': name,
      'slots': kernel.nSlots,
      'loaded': kernel.loaded,
      'stored': kernel.stored,
      'cpuMs1000': _median(cpuTimes),
      'relErrMedian': rel.isEmpty ? null : rel[rel.length ~/ 2],
      'relErrP99': rel.isEmpty ? null : rel[(rel.length * 0.99).floor()],
      'relErrMax': rel.isEmpty ? null : rel.last,
      'nonFiniteBoth': infBoth,
      'nonFiniteMismatch': infMismatch,
      'argminCpu': omegas[argmin(ref)],
      'argminGpu': omegas[argmin(got)],
      'sample': [
        for (final k in [0, 250, 500, 750, 999]) [ref[k], got[k]],
      ],
    });

    // --- scaling ---
    for (final b in sizes) {
      final GpuBatch gpu;
      try {
        gpu = await GpuBatch.create(backend, kernel, b);
      } on ArgumentError catch (e) {
        emit({'scenario': name, 'batch': b, 'skipped': '$e'});
        continue;
      }
      final grid = _linspace(lo, hi, b);
      gpu.setInitial(1, NodeField.tonicVolatility, grid);
      await gpu.surprise(data, trialsPerDispatch: chunk); // warm-up
      final timings = <GpuTimings>[];
      for (var r = 0; r < repeat; r++) {
        // As in a fitting loop: new parameters every call.
        gpu.setInitial(1, NodeField.tonicVolatility, grid);
        await gpu.surprise(data, trialsPerDispatch: chunk);
        timings.add(gpu.lastTimings!);
      }
      gpu.dispose();
      final total = _median([for (final t in timings) t.total]);
      final gpuMs = _median([for (final t in timings) t.gpu]);
      emit({
        'scenario': name,
        'batch': b,
        'trials': data.length,
        'totalMs': total,
        'gpuMs': gpuMs,
        'uploadMs': _median([for (final t in timings) t.upload]),
        'readbackMs': _median([for (final t in timings) t.readback]),
        'nsPerElementTrial': total * 1e6 / (b * data.length),
        'cpuNsPerElementTrial': _median(cpuTimes) * 1e6 / (nAcc * data.length),
      });
    }
  }
}
