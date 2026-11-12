import 'dart:typed_data';

import 'package:hgf_deep/hgf_deep.dart';

/// Relative tolerance against the float64 JAX reference.
const double relTol = 1e-9;

/// Absolute floor: values within this of each other count as equal.
const double absTol = 1e-12;

/// Decode a fixture number (`"NaN"`, `"Infinity"`, `"-Infinity"` or a num).
double decode(Object? v) => switch (v) {
  num() => v.toDouble(),
  'NaN' => double.nan,
  'Infinity' => double.infinity,
  '-Infinity' => double.negativeInfinity,
  _ => throw FormatException('Bad fixture number: $v'),
};

List<double> series(Object? v) => [
  for (final x in v as List<Object?>) decode(x),
];

Matrix matrix(Object? rows) =>
    Matrix.fromRows([for (final r in rows as List<Object?>) series(r)]);

/// The first element where [actual] differs from [expected] beyond
/// tolerance (NaN matches NaN), with the max relative error; `null` when
/// everything matches.
String? firstMismatch(
  List<double> expected,
  List<double> actual, {
  double rel = relTol,
  double abs = absTol,
}) {
  if (expected.length != actual.length) {
    return 'length ${actual.length} != ${expected.length}';
  }
  int? first;
  var maxRel = 0.0;
  for (var t = 0; t < expected.length; t++) {
    final e = expected[t], a = actual[t];
    if (e.isNaN && a.isNaN) continue;
    if (e == a) continue;
    final diff = (e - a).abs();
    if (diff <= abs) continue;
    final r = diff / (e.abs() > a.abs() ? e.abs() : a.abs());
    if (r.isNaN || r > rel) first ??= t;
    if (r > maxRel || r.isNaN) maxRel = r;
  }
  if (first == null) return null;
  return '[$first] expected ${expected[first]} got ${actual[first]} '
      '(max rel err ${maxRel.toStringAsExponential(2)})';
}

/// The corrections a fixture was generated with: none (released pyhgf)
/// unless its settings name some (a patched reference, see
/// `tool/parity/gen_deep_fixtures.py`).
DeepFixes fixesOf(Map<String, Object?> settings) {
  final f = settings['fixes'] as Map<String, Object?>? ?? const {};
  return DeepFixes(
    curvatureFloor: f['curvature_floor'] as bool? ?? false,
    logSpaceVolatility: f['log_space_volatility'] as bool? ?? false,
  );
}

/// Build the fixture's network with its starting weights ([fixes] overrides
/// the fixture's own).
DeepNetwork buildNetwork(Map<String, Object?> fixture, {DeepFixes? fixes}) {
  final s = fixture['settings'] as Map<String, Object?>;
  final net = DeepNetwork(
    coupling: CouplingFn.parse(s['coupling'] as String? ?? 'linear'),
    volatilityUpdates: DeepVolatilityUpdate.parse(
      s['volatility_updates'] as String? ?? 'unbounded',
    ),
    maxPosteriorPrecision:
        (s['max_posterior_precision'] as num?)?.toDouble() ?? 1e10,
    precisionClippingValue:
        (s['precision_clipping_value'] as num?)?.toDouble() ?? 1e-6,
    predictPrecision: s['predict_precision'] as bool? ?? true,
    feedforwardUncertainty: s['feedforward_uncertainty'] as bool? ?? false,
    tonicVolatility: s['tonic_volatility'] as bool? ?? false,
    meanFieldUpdates: s['mean_field_updates'] as bool? ?? false,
    carryPrecision: s['carry_precision'] as bool? ?? true,
    updateInputLayer: s['update_input_layer'] as bool? ?? false,
    fixes: fixes ?? fixesOf(s),
  );
  for (final raw in fixture['layers'] as List<Object?>) {
    final layer = raw as Map<String, Object?>;
    final overrides = Map<String, Object?>.from(
      layer['overrides'] as Map? ?? const {},
    );
    final tonic = (overrides.remove('tonic_volatility') as num?)?.toDouble();
    final tonicVol = (overrides.remove('tonic_volatility_vol') as num?)
        ?.toDouble();
    net.addLayer(
      layer['size'] as int,
      kind: LayerKind.parse(layer['kind'] as String? ?? 'volatile'),
      addConstantInput: layer['add_constant_input'] as bool? ?? true,
      fullyConnected: layer['fully_connected'] as bool? ?? true,
      coupling: layer['coupling'] == null
          ? null
          : CouplingFn.parse(layer['coupling'] as String),
      volatilityParent: layer['volatility_parent'] as bool?,
      tonicVolatility: tonic,
      tonicVolatilityVol: tonicVol ?? -4.0,
      initial: {
        for (final e in overrides.entries)
          LayerField.parse(e.key): (e.value as num).toDouble(),
      },
    );
  }
  net.setWeights([
    for (final w in fixture['weights'] as List<Object?>) matrix(w),
  ]);
  return net;
}

Optimiser? _optimiser(Map<int, Optimiser> cache, Object? spec) {
  if (spec == null) return null;
  final m = spec as Map<String, Object?>;
  return cache.putIfAbsent(m['id'] as int, () {
    final lr = (m['lr'] as num).toDouble();
    return m['kind'] == 'sgd' ? Sgd(lr) : Adam(lr);
  });
}

/// Whether any value in [v] (nested lists of decoded numbers) is not finite.
bool _anyNonFinite(Object? v) => switch (v) {
  null => false,
  List<Object?>() => v.any(_anyNonFinite),
  Map<String, Object?>() => v.values.any(_anyNonFinite),
  num() => !v.isFinite,
  'NaN' || 'Infinity' || '-Infinity' => true,
  _ => false,
};

/// The first sample at which any recorded reference trajectory is not
/// finite (`null` if none is).
int? _firstNonFiniteRow(Map<String, Object?>? traj) {
  int? first;
  traj?.forEach((_, perLayer) {
    for (final layer in perLayer as List<Object?>) {
      if (layer == null) continue;
      final rows = layer as List<Object?>;
      for (var t = 0; t < rows.length; t++) {
        if (_anyNonFinite(rows[t])) {
          if (first == null || t < first!) first = t;
          break;
        }
      }
    }
  });
  return first;
}

int _currentSamples = 1;

/// The op's weight-belief settings (`learning_kwargs`), or `null`.
SynapticUncertainty? synapticOf(Map<String, Object?> op) {
  final kw = op['learning_kwargs'] as Map<String, Object?>?;
  if (kw == null) return null;
  double? d(String k) => (kw[k] as num?)?.toDouble();
  return SynapticUncertainty(
    window: d('window')!,
    priorVariance: d('prior_variance') ?? 1.0,
    priorMean: d('prior_mean') ?? 0.0,
    learningRate: d('learning_rate') ?? 1.0,
    incrementScale: d('increment_scale') ?? 1.0,
  );
}

/// Run every operation of [fixture] and compare against the reference.
/// Returns the mismatches (empty when everything agrees).
///
/// With [finiteOnly] the comparison stops where the reference first holds a
/// non-finite value: within a `fit` it covers the samples before it, and no
/// later operation is compared (for corrections that change only what
/// pyhgf cannot represent).
List<String> checkDeepFixture(
  Map<String, Object?> fixture, {
  double rel = relTol,
  double abs = absTol,
  DeepFixes? fixes,
  bool finiteOnly = false,
}) {
  final failures = <String>[];
  var rows = -1; // compare only the first `rows` samples (−1: all)
  void cmp(String what, List<double> expected, List<double> actual) {
    if (rows >= 0 && expected.isNotEmpty) {
      final width = expected.length ~/ _currentSamples;
      expected = expected.sublist(0, rows * width);
      actual = actual.sublist(0, rows * width);
    }
    final m = firstMismatch(expected, actual, rel: rel, abs: abs);
    if (m != null) failures.add('$what: $m');
  }

  final net = buildNetwork(fixture, fixes: fixes);
  final optimisers = <int, Optimiser>{};
  final ops = fixture['ops'] as List<Object?>;
  for (var i = 0; i < ops.length; i++) {
    final op = ops[i] as Map<String, Object?>;
    final kind = op['op'] as String;
    final tag = 'op $i ($kind)';
    final timeStep = (op['time_step'] as num?)?.toDouble() ?? 1.0;
    final learningKind = LearningKind.parse(
      op['learning_kind'] as String? ?? 'precision_weighted',
    );
    if (finiteOnly) {
      _currentSamples = (op['x'] as List<Object?>).length;
      final bad = kind == 'fit'
          ? (_firstNonFiniteRow(op['trajectories'] as Map<String, Object?>?) ??
                (_anyNonFinite(op['predictions']) || _anyNonFinite(op['after'])
                    ? 0
                    : null))
          : (_anyNonFinite(op) ? 0 : null);
      if (bad != null) {
        if (kind != 'fit' || bad == 0) break;
        rows = bad;
      }
    }
    switch (kind) {
      case 'fit':
        final record = {
          for (final f in op['record'] as List<Object?>? ?? const [])
            LayerField.parse(f as String),
        };
        final pred = net.fit(
          matrix(op['x']),
          matrix(op['y']),
          optimiser: _optimiser(optimisers, op['optimiser']),
          learningKind: learningKind,
          synapticUncertainty: synapticOf(op),
          weightUpdate: op['weight_update'] as bool? ?? true,
          timeStep: timeStep,
          updatePrecisions: op['update_precisions'] as bool? ?? true,
          record: record,
        );
        cmp('$tag predictions', matrix(op['predictions']).data, pred.data);
        final traj = op['trajectories'] as Map<String, Object?>?;
        traj?.forEach((name, perLayer) {
          final field = LayerField.parse(name);
          final layers = perLayer as List<Object?>;
          for (var l = 0; l < layers.length; l++) {
            final actual = net.trajectories!.of(field, l);
            if (layers[l] == null || actual == null) {
              if ((layers[l] == null) != (actual == null)) {
                failures.add('$tag $name layer $l: presence differs');
              }
              continue;
            }
            cmp(
              '$tag trajectory $name layer $l',
              matrix(layers[l]).data,
              actual,
            );
          }
        });
      case 'batch_update':
        final errors = net.batchUpdate(
          matrix(op['x']),
          matrix(op['y']),
          optimiser: _optimiser(optimisers, op['optimiser']),
          learningKind: learningKind,
          synapticUncertainty: synapticOf(op),
          timeStep: timeStep,
          updatePrecisions: op['update_precisions'] as bool? ?? true,
          sampleWeight: op['sample_weight'] == null
              ? null
              : series(op['sample_weight']),
        );
        cmp('$tag input errors', matrix(op['input_errors']).data, errors.data);
      case 'health':
        final h = net.checkGradientHealth(
          series(op['x']),
          series(op['y']),
          learningKind: learningKind,
        );
        final mags = op['magnitudes'] as List<Object?>;
        cmp(
          '$tag magnitudes',
          [for (final m in mags.skip(1)) decode(m)],
          [for (final m in h.magnitudes.skip(1)) m!],
        );
        if (h.magnitudes.first != null) failures.add('$tag magnitude 0');
        if ('${h.dead}' != '${op['dead']}') {
          failures.add('$tag dead ${h.dead} != ${op['dead']}');
        }
        if ('${h.quantised}' != '${op['quantised']}') {
          failures.add('$tag quantised ${h.quantised} != ${op['quantised']}');
        }
        continue;
      case 'predict':
        final pred = net.predict(matrix(op['x']));
        cmp('$tag predictions', matrix(op['predictions']).data, pred.data);
      default:
        throw FormatException('Unknown op $kind');
    }
    if (rows >= 0) break; // the reference went non-finite in this fit
    final after = op['after'] as Map<String, Object?>;
    final weights = after['weights'] as List<Object?>;
    final deltas = after['weights_precision_delta'] as List<Object?>?;
    final ours = net.weightPrecisionDeltas;
    for (var l = 0; l < weights.length; l++) {
      final d = deltas?[l];
      if ((d == null) != (ours[l + 1] == null)) {
        failures.add('$tag weights_precision_delta ${l + 1}: presence differs');
      } else if (d != null) {
        cmp(
          '$tag weights_precision_delta ${l + 1}',
          matrix(d).data,
          ours[l + 1]!.data,
        );
      }
    }
    for (var l = 0; l < weights.length; l++) {
      cmp(
        '$tag weights ${l + 1}',
        matrix(weights[l]).data,
        net.weights[l + 1]!.data,
      );
    }
    (after['state'] as Map<String, Object?>).forEach((name, perLayer) {
      final field = LayerField.parse(name);
      final layers = perLayer as List<Object?>;
      for (var l = 0; l < layers.length; l++) {
        final Float64List? actual = net.state(l).field(field);
        if (layers[l] == null || actual == null) {
          if ((layers[l] == null) != (actual == null)) {
            failures.add('$tag state $name layer $l: presence differs');
          }
          continue;
        }
        cmp('$tag state $name layer $l', series(layers[l]), actual);
      }
    });
  }
  return failures;
}
