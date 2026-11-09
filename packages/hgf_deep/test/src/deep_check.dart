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

/// Build the fixture's network with its starting weights.
DeepNetwork buildNetwork(Map<String, Object?> fixture) {
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

/// Run every operation of [fixture] and compare against the reference.
/// Returns the mismatches (empty when everything agrees).
List<String> checkDeepFixture(
  Map<String, Object?> fixture, {
  double rel = relTol,
  double abs = absTol,
}) {
  final failures = <String>[];
  void cmp(String what, List<double> expected, List<double> actual) {
    final m = firstMismatch(expected, actual, rel: rel, abs: abs);
    if (m != null) failures.add('$what: $m');
  }

  final net = buildNetwork(fixture);
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
          timeStep: timeStep,
          updatePrecisions: op['update_precisions'] as bool? ?? true,
        );
        cmp('$tag input errors', matrix(op['input_errors']).data, errors.data);
      case 'predict':
        final pred = net.predict(matrix(op['x']));
        cmp('$tag predictions', matrix(op['predictions']).data, pred.data);
      default:
        throw FormatException('Unknown op $kind');
    }
    final after = op['after'] as Map<String, Object?>;
    final weights = after['weights'] as List<Object?>;
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
