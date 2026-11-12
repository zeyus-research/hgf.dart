import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_gpu/hgf_gpu.dart';

// The fixture reader of hgf_deep's own parity tests.
import '../../../hgf_deep/test/src/deep_check.dart' show buildNetwork, matrix;

/// f32 against f64, relative to max(1, |expected|).
const double tolerance = 2e-4;

/// Values that may be non-finite on the GPU, per fixture and quantity, for
/// a reason found by hand (f32 range; JAX's float32 mode fails the same
/// way). Operations that start from a state f32 can't hold are skipped by
/// `checkFixture` itself.
const allowedLost = {
  // Samples 3 and 8 of op 1 send layer 1's volatility mean from 4.5 to
  // 130 and 137 (the P18 runaway); eHGF's re-prediction exp(ω + μ_vol)
  // overflows f32, so that node's batch-mean precision_vol is NaN.
  'mean_field_eHGF': {'op 1 (batch_update) precision_vol 1': 1},
};

/// The [results] of fixture [name] beyond [tolerance] or [allowedLost].
List<String> failures(String name, List<(String, double, int)> results) {
  final allowed = allowedLost[name] ?? const {};
  return [
    for (final r in results)
      if (r.$2 > tolerance || r.$3 > (allowed[r.$1] ?? 0))
        '${r.$1}: ${r.$2.toStringAsExponential(2)}, ${r.$3} lost',
  ];
}

/// The largest error of [actual] against [expected], relative to
/// `max(1, |expected|)`, over the entries where [expected] is finite; and
/// how many of those are not finite in [actual].
({double err, int lost}) compare(List<double> expected, List<double> actual) {
  var err = 0.0;
  var lost = 0;
  for (var i = 0; i < expected.length; i++) {
    final e = expected[i], a = actual[i];
    if (!e.isFinite) continue;
    if (!a.isFinite) {
      lost++;
      continue;
    }
    err = math.max(err, (e - a).abs() / math.max(1.0, e.abs()));
  }
  return (err: err, lost: lost);
}

/// The first belief or weight of [net] that f32 can't hold (finite but
/// beyond f32's normal range), or `null`. Only the weights count unless
/// [beliefs] (`predict` reads nothing else).
String? outsideF32(DeepNetwork net, {bool beliefs = true}) {
  bool bad(double v) =>
      v.isFinite && v != 0 && (v.abs() < 1.17549435e-38 || v.abs() > 3.4e38);
  for (var l = 0; l < net.nLayers; l++) {
    for (final x in net.weights[l]?.data ?? const <double>[]) {
      if (bad(x)) return 'weights $l: $x';
    }
    if (!beliefs) continue;
    for (final f in LayerField.values) {
      final v = net.state(l).field(f);
      if (v == null) continue;
      for (final x in v) {
        if (bad(x)) return '${f.pyhgfName} $l = $x';
      }
    }
    // The predicted variance 1/π + exp(μ_vol + 1/(2π_vol) + ω): beyond
    // √(f32 max) ≈ 1.8e19 the posterior's mean shift (~ variance) squares
    // to infinity in f32.
    final s = net.state(l);
    final mv = s.meanVol, pv = s.precisionVol;
    if (mv == null || pv == null) continue;
    final tonic = net.debugLayers[l].tonicVolatility;
    for (var i = 0; i < mv.length; i++) {
      final e = mv[i] + 1 / (2 * pv[i]) + (tonic?[i] ?? 0);
      final variance = 1 / s.precision[i] + math.exp(e);
      if (variance > 1.8e19) {
        return 'predicted variance ${variance.toStringAsExponential(1)} '
            '(layer $l)';
      }
    }
  }
  return null;
}

/// A network with [src]'s weights and beliefs (same fixture).
DeepNetwork cloneFrom(Map<String, Object?> fixture, DeepNetwork src) {
  final net = buildNetwork(fixture);
  net.setWeights([for (final w in src.weights.skip(1)) w!.copy()]);
  for (var l = 0; l < src.nLayers; l++) {
    net.state(l).setFrom(src.state(l));
  }
  return net;
}

Optimiser? freshOptimiser(Object? spec) {
  if (spec == null) return null;
  final m = spec as Map<String, Object?>;
  final lr = (m['lr'] as num).toDouble();
  return m['kind'] == 'sgd' ? Sgd(lr) : Adam(lr);
}

/// Every quantity a GPU step changes, CPU vs GPU: `(what, error, lost)`.
Future<List<(String, double, int)>> compareNetworks(
  GpuDeepNetwork gpu,
  DeepNetwork cpu,
  DeepNetwork scratch,
) async {
  await gpu.download(scratch);
  final out = <(String, double, int)>[];
  for (var l = 1; l < cpu.nLayers; l++) {
    final c = compare(cpu.weights[l]!.data, scratch.weights[l]!.data);
    out.add(('weights $l', c.err, c.lost));
  }
  for (var l = 0; l < cpu.nLayers; l++) {
    for (final f in const [
      LayerField.precision,
      LayerField.meanVol,
      LayerField.precisionVol,
    ]) {
      final e = cpu.state(l).field(f);
      if (e == null) continue;
      final c = compare(e, scratch.state(l).field(f)!);
      out.add(('${f.pyhgfName} $l', c.err, c.lost));
    }
  }
  return out;
}

/// Whether the GPU kernels cover every feature [fixture] uses: not
/// `update_input_layer`, the weight-belief rule (`synaptic_uncertainty`),
/// weighted batches or the gradient-health probe (CPU only so far).
bool gpuDeepSupported(Map<String, Object?> fixture) {
  final settings = fixture['settings'] as Map<String, Object?>;
  if (settings['update_input_layer'] == true) return false;
  for (final o in fixture['ops'] as List<Object?>) {
    final op = o as Map<String, Object?>;
    if (op['op'] == 'health' ||
        op['learning_kind'] == 'synaptic_uncertainty' ||
        op['sample_weight'] != null) {
      return false;
    }
  }
  return true;
}

/// Replays [fixture]'s operations on the CPU; before each `batch_update`
/// and `predict`, the GPU takes the CPU's current state and runs the same
/// operation (with a fresh optimiser on both sides). Returns one line per
/// compared quantity; an operation that starts from a state f32 can't hold
/// is skipped, with a line whose error is NaN.
Future<List<(String, double, int)>> checkFixture(
  GpuBackend backend,
  Map<String, Object?> fixture,
) async {
  final cpu = buildNetwork(fixture);
  final ops = fixture['ops'] as List<Object?>;
  final capacity = [
    for (final o in ops) ((o as Map)['x'] as List).length,
  ].reduce(math.max);
  final gpu = await GpuDeepNetwork.create(backend, cpu, capacity: capacity);
  final results = <(String, double, int)>[];
  final optimisers = <int, Optimiser>{};
  try {
    for (var i = 0; i < ops.length; i++) {
      final op = ops[i] as Map<String, Object?>;
      final kind = op['op'] as String;
      final tag = 'op $i ($kind)';
      final x = matrix(op['x']);
      final timeStep = (op['time_step'] as num?)?.toDouble() ?? 1.0;
      final learningKind = LearningKind.parse(
        op['learning_kind'] as String? ?? 'precision_weighted',
      );
      final range = kind == 'fit'
          ? null
          : outsideF32(cpu, beliefs: kind == 'batch_update');
      if (range != null) {
        results.add(('$tag skipped: $range is outside f32', double.nan, 0));
      }
      switch (kind) {
        case 'fit':
          final spec = op['optimiser'] as Map<String, Object?>?;
          cpu.fit(
            x,
            matrix(op['y']),
            optimiser: spec == null
                ? null
                : optimisers.putIfAbsent(
                    spec['id'] as int,
                    () => freshOptimiser(spec)!,
                  ),
            learningKind: learningKind,
            weightUpdate: op['weight_update'] as bool? ?? true,
            timeStep: timeStep,
            updatePrecisions: op['update_precisions'] as bool? ?? true,
          );
        case 'batch_update':
          final y = matrix(op['y']);
          final updatePrecisions = op['update_precisions'] as bool? ?? true;
          if (range != null) {
            _replay(cpu, op, x, y, optimisers, learningKind, timeStep);
            continue;
          }
          final ref = cloneFrom(fixture, cpu);
          gpu.upload(ref);
          final cpuErr = ref.batchUpdate(
            x,
            y,
            optimiser: freshOptimiser(op['optimiser']),
            learningKind: learningKind,
            timeStep: timeStep,
            updatePrecisions: updatePrecisions,
          );
          final gpuErr = await gpu.batchUpdate(
            x,
            y,
            optimiser: freshOptimiser(op['optimiser']),
            learningKind: learningKind,
            timeStep: timeStep,
            updatePrecisions: updatePrecisions,
          );
          final c = compare(cpuErr.data, gpuErr!.data);
          results.add(('$tag input errors', c.err, c.lost));
          for (final r in await compareNetworks(
            gpu,
            ref,
            buildNetwork(fixture),
          )) {
            results.add(('$tag ${r.$1}', r.$2, r.$3));
          }
          _replay(cpu, op, x, y, optimisers, learningKind, timeStep);
        case 'predict':
          if (range != null) continue;
          gpu.upload(cpu);
          final c = compare(cpu.predict(x).data, (await gpu.predict(x)).data);
          results.add(('$tag predictions', c.err, c.lost));
      }
    }
  } finally {
    gpu.dispose();
  }
  return results;
}

/// The fixture's own `batch_update` on the CPU (its optimiser sequence).
void _replay(
  DeepNetwork cpu,
  Map<String, Object?> op,
  Matrix x,
  Matrix y,
  Map<int, Optimiser> optimisers,
  LearningKind learningKind,
  double timeStep,
) {
  final spec = op['optimiser'] as Map<String, Object?>?;
  cpu.batchUpdate(
    x,
    y,
    optimiser: spec == null
        ? null
        : optimisers.putIfAbsent(
            spec['id'] as int,
            () => freshOptimiser(spec)!,
          ),
    learningKind: learningKind,
    timeStep: timeStep,
    updatePrecisions: op['update_precisions'] as bool? ?? true,
  );
}

Matrix randn(int rows, int cols, math.Random rng, [double scale = 1]) {
  final m = Matrix(rows, cols);
  for (var k = 0; k < m.data.length; k++) {
    final u1 = 1 - rng.nextDouble(), u2 = rng.nextDouble();
    m.data[k] =
        scale * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
  }
  return m;
}

/// A random [sizes] network (bottom first) with scaled Gaussian weights.
DeepNetwork randomNetwork(
  List<int> sizes,
  int seed, {
  CouplingFn coupling = CouplingFn.tanh,
  DeepVolatilityUpdate update = DeepVolatilityUpdate.unbounded,
  bool carryPrecision = true,
}) {
  final net = DeepNetwork(
    coupling: coupling,
    volatilityUpdates: update,
    carryPrecision: carryPrecision,
  );
  for (final s in sizes) {
    net.addLayer(s);
  }
  final rng = math.Random(seed);
  net.setWeights([
    for (var l = 1; l < sizes.length; l++)
      randn(sizes[l - 1], sizes[l] + 1, rng, 1 / math.sqrt(sizes[l])),
  ]);
  return net;
}

Float64List flat(Matrix m) => m.data;

/// Replays [fixture]'s operations on the CPU; before each `fit`, a
/// two-member [GpuDeepEnsemble] takes the CPU's current state (both
/// members) and runs the same fit, with a fresh optimiser on both sides.
/// Compares both members' predictions, weights and carried beliefs with
/// the CPU's. Lines as [checkFixture]'s.
///
/// A fit whose CPU trajectory leaves f32's range (the P18 runaway: means
/// of 1e105 with precisions of 1e-106, whose product the gradient needs)
/// is compared only on the predictions up to the row that does, with a
/// line whose error is NaN saying so.
Future<List<(String, double, int)>> checkFitOps(
  GpuBackend backend,
  Map<String, Object?> fixture,
) async {
  final cpu = buildNetwork(fixture);
  final ops = fixture['ops'] as List<Object?>;
  final results = <(String, double, int)>[];
  final optimisers = <int, Optimiser>{};
  for (var i = 0; i < ops.length; i++) {
    final op = ops[i] as Map<String, Object?>;
    final kind = op['op'] as String;
    final x = matrix(op['x']);
    final timeStep = (op['time_step'] as num?)?.toDouble() ?? 1.0;
    final learningKind = LearningKind.parse(
      op['learning_kind'] as String? ?? 'precision_weighted',
    );
    switch (kind) {
      case 'fit':
        final y = matrix(op['y']);
        final spec = op['optimiser'] as Map<String, Object?>?;
        final weightUpdate = op['weight_update'] as bool? ?? true;
        final updatePrecisions = op['update_precisions'] as bool? ?? true;
        final tag = 'op $i (fit)';
        final range = outsideF32(cpu);
        if (range != null) {
          results.add(('$tag skipped: $range is outside f32', double.nan, 0));
        } else {
          final ref = cloneFrom(fixture, cpu);
          final gpu = await GpuDeepEnsemble.create(backend, [
            cloneFrom(fixture, cpu),
            cloneFrom(fixture, cpu),
          ]);
          try {
            final cpuPred = ref.fit(
              x,
              y,
              optimiser: freshOptimiser(spec),
              learningKind: learningKind,
              weightUpdate: weightUpdate,
              timeStep: timeStep,
              updatePrecisions: updatePrecisions,
              record: LayerField.values.toSet(),
            );
            final leaves = firstRowOutsideF32(ref.trajectories!);
            final gpuPred = (await gpu.fit(
              [x],
              [y],
              optimiser: freshOptimiser(spec),
              learningKind: learningKind,
              weightUpdate: weightUpdate,
              timeStep: timeStep,
              updatePrecisions: updatePrecisions,
              rowsPerDispatch: 7,
            ))!;
            if (leaves != null) {
              results.add((
                '$tag: row ${leaves.$1} leaves f32 or turns NaN (${leaves.$2}); '
                    'predictions compared before it',
                double.nan,
                0,
              ));
            }
            final upTo = leaves?.$1 ?? x.rows;
            for (var e = 0; e < 2; e++) {
              final n = upTo * cpuPred.cols;
              final c = compare(
                cpuPred.data.sublist(0, n),
                gpuPred[e].data.sublist(0, n),
              );
              results.add(('$tag member $e predictions', c.err, c.lost));
              if (leaves != null) continue;
              final scratch = buildNetwork(fixture);
              await gpu.download(e, scratch);
              for (final r in compareStates(ref, scratch)) {
                results.add(('$tag member $e ${r.$1}', r.$2, r.$3));
              }
            }
          } finally {
            gpu.dispose();
          }
        }
        cpu.fit(
          x,
          y,
          optimiser: spec == null
              ? null
              : optimisers.putIfAbsent(
                  spec['id'] as int,
                  () => freshOptimiser(spec)!,
                ),
          learningKind: learningKind,
          weightUpdate: weightUpdate,
          timeStep: timeStep,
          updatePrecisions: updatePrecisions,
        );
      case 'batch_update':
        _replay(
          cpu,
          op,
          x,
          matrix(op['y']),
          optimisers,
          learningKind,
          timeStep,
        );
      case 'predict':
    }
  }
  return results;
}

/// Weights and carried beliefs of [actual] against [expected].
List<(String, double, int)> compareStates(
  DeepNetwork expected,
  DeepNetwork actual,
) {
  final out = <(String, double, int)>[];
  for (var l = 1; l < expected.nLayers; l++) {
    final c = compare(expected.weights[l]!.data, actual.weights[l]!.data);
    out.add(('weights $l', c.err, c.lost));
  }
  for (var l = 0; l < expected.nLayers; l++) {
    for (final f in const [
      LayerField.precision,
      LayerField.meanVol,
      LayerField.precisionVol,
    ]) {
      final e = expected.state(l).field(f);
      if (e == null) continue;
      final c = compare(e, actual.state(l).field(f)!);
      out.add(('${f.pyhgfName} $l', c.err, c.lost));
    }
  }
  return out;
}

/// The first recorded row whose beliefs hold a finite value beyond f32's
/// normal range, a prediction error beyond √(f32 max) ≈ 1.8e19 (its
/// square, in the volatility update and Adam's second moment, overflows),
/// or a NaN, and which; or `null`.
///
/// NaN counts because its onset is a threshold: the guards zero a NaN
/// node's gradients, so a GPU that reaches the NaN one row later moves
/// those weights by a whole optimiser step (seen on SwiftShader).
(int, String)? firstRowOutsideF32(DeepTrajectories traj) {
  bool bad(double v) =>
      !v.isFinite || v != 0 && (v.abs() < 1.17549435e-38 || v.abs() > 3.4e38);
  for (var t = 0; t < traj.steps; t++) {
    for (var l = 0; ; l++) {
      final Float64List? mean, expected;
      try {
        mean = traj.of(LayerField.mean, l);
        expected = traj.of(LayerField.expectedMean, l);
      } on RangeError {
        break;
      }
      final n = mean!.length ~/ traj.steps;
      for (var i = 0; i < n; i++) {
        final d = mean[t * n + i] - expected![t * n + i];
        if (d.isFinite && d.abs() > 1.8e19) {
          return (
            t,
            'prediction error ${d.toStringAsExponential(1)}, layer $l',
          );
        }
      }
    }
    for (final f in traj.fields) {
      for (var l = 0; ; l++) {
        final Float64List? data;
        try {
          data = traj.of(f, l);
        } on RangeError {
          break;
        }
        if (data == null) continue;
        final n = data.length ~/ traj.steps;
        for (var i = 0; i < n; i++) {
          final v = data[t * n + i];
          if (bad(v)) return (t, '${f.pyhgfName} $l = $v');
        }
      }
    }
  }
  return null;
}
