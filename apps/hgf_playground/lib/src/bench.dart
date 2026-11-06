import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:hgf/hgf.dart';

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
