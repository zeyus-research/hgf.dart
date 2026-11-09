import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:hgf/hgf.dart';

import 'model/fitting.dart';
import 'model/playground.dart';

/// The trial count of a scripted benchmark run, if one was asked for:
/// `?bench=10000` on the web, `--dart-define=HGF_BENCH=10000` elsewhere.
int? benchTrials() {
  const defined = String.fromEnvironment('HGF_BENCH');
  final raw = defined.isNotEmpty
      ? defined
      : (kIsWeb ? Uri.base.queryParameters['bench'] : null);
  if (raw == null) return null;
  return int.tryParse(raw) ?? 10000;
}

/// Drag a slider in code: change ω₂ every frame for [frames] frames (each
/// change re-filters the whole series and redraws every plot), then print
/// the frame and filter timings as one `HGF_BENCH {json}` line.
Future<Map<String, Object>> runBench(
  Playground pg, {
  int frames = 300,
  String label = '',
}) async {
  final binding = SchedulerBinding.instance;
  final timings = <FrameTiming>[];
  void collect(List<FrameTiming> t) => timings.addAll(t);

  // Let the first frames (shader warm-up, font loading) pass.
  for (var f = 0; f < 30; f++) {
    binding.scheduleFrame();
    await binding.endOfFrame;
  }
  binding.addTimingsCallback(collect);
  final filter = <double>[], wall = <double>[];
  final sw = Stopwatch()..start();
  var last = 0;
  for (var f = 0; f < frames; f++) {
    pg.setParam(
      1,
      NodeField.tonicVolatility,
      -5 + 3 * math.sin(f / 20), // a slider dragged back and forth
    );
    await binding.endOfFrame;
    filter.add(pg.filterMicros / 1000); // re-filtered at the frame's start
    final now = sw.elapsedMicroseconds;
    wall.add((now - last) / 1000);
    last = now;
  }
  // Timings arrive in batches; wait for the last ones.
  await Future<void>.delayed(const Duration(milliseconds: 1500));
  binding.removeTimingsCallback(collect);

  Map<String, double> stats(List<double> xs) {
    if (xs.isEmpty) return const {};
    final s = [...xs]..sort();
    double q(double p) => s[math.min(s.length - 1, (p * s.length).floor())];
    return {
      'median': double.parse(q(0.5).toStringAsFixed(3)),
      'p95': double.parse(q(0.95).toStringAsFixed(3)),
      'max': double.parse(s.last.toStringAsFixed(3)),
    };
  }

  final result = <String, Object>{
    'label': label,
    'target': kIsWeb
        ? (const bool.fromEnvironment('dart.tool.dart2wasm') ? 'wasm' : 'js')
        : defaultTargetPlatform.name,
    'mode': kReleaseMode ? 'release' : (kProfileMode ? 'profile' : 'debug'),
    'preset': pg.preset.id,
    'trials': pg.observations.length,
    'frames': frames,
    'filter_ms': stats(filter),
    'frame_interval_ms': stats(wall.skip(1).toList()),
    'build_ms': stats([
      for (final t in timings) t.buildDuration.inMicroseconds / 1000,
    ]),
    'raster_ms': stats([
      for (final t in timings) t.rasterDuration.inMicroseconds / 1000,
    ]),
    'timed_frames': timings.length,
  };
  // ignore: avoid_print
  print('HGF_BENCH ${jsonEncode(result)}');
  return result;
}

/// Whether a scripted fitting benchmark was asked for: `?fitbench` on the
/// web (`&workers=0` fits on the main thread), `--dart-define=HGF_FITBENCH=1`
/// elsewhere.
/// `?fitbench&idle` / `HGF_FITBENCH=idle`: don't force a frame every
/// vsync while fitting (frame stats are then not recorded).
final bool _idle =
    const String.fromEnvironment('HGF_FITBENCH') == 'idle' ||
    (kIsWeb && Uri.base.queryParameters.containsKey('idle'));

bool fitBenchRequested() =>
    const String.fromEnvironment('HGF_FITBENCH').isNotEmpty ||
    (kIsWeb && Uri.base.queryParameters.containsKey('fitbench'));

/// Fit simulated agents' responses with MAP and NUTS on two presets,
/// recording each fit's wall time and the frame intervals while it ran
/// (the page must keep drawing), then print one `HGF_FITBENCH {json}` line.
Future<void> runFitBench(Playground pg, Fitting fit) async {
  final binding = SchedulerBinding.instance;
  final runs = <Map<String, Object?>>[];
  // Spawn the workers and warm up, untimed.
  pg.setPreset('binary3');
  fit
    ..simulateAgent(seed: 1)
    ..setMethod(FitMethod.map);
  await fit.fit();
  for (final preset in ['binary3', 'continuous3']) {
    pg.setPreset(preset);
    pg.updateData((d) => d.nTrials = 400);
    fit.simulateAgent(seed: 1);
    for (final method in FitMethod.values) {
      fit.setMethod(method);
      final frames = <double>[];
      final sw = Stopwatch()..start();
      final job = fit.fit();
      var last = 0;
      while (fit.running) {
        if (_idle) {
          // Only the frames the app asks for itself (progress updates).
          await Future<void>.delayed(const Duration(milliseconds: 50));
          continue;
        }
        binding.scheduleFrame();
        await binding.endOfFrame;
        final now = sw.elapsedMicroseconds;
        frames.add((now - last) / 1000);
        last = now;
      }
      await job;
      frames.sort();
      final r = fit.result;
      runs.add({
        'preset': preset,
        'method': method.name,
        'error': fit.error,
        'ms': r?.millis,
        'where': r?.where,
        'frames': frames.length,
        'frame_median_ms': frames.isEmpty ? null : frames[frames.length ~/ 2],
        'frame_max_ms': frames.isEmpty ? null : frames.last,
        'estimates': {
          for (final e in r?.estimates ?? const <ParamEstimate>[])
            e.param.id: [e.estimate, e.lower, e.upper, e.truth],
        },
      });
    }
  }
  final target = kIsWeb
      ? (kIsWasm ? 'wasm' : 'js')
      : defaultTargetPlatform.name;
  // ignore: avoid_print
  print('HGF_FITBENCH ${jsonEncode({'target': target, 'runs': runs})}');
}
