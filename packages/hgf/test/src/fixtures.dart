import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

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

/// Row-major flattening of a `[[...], ...]` matrix.
Float64List flatten(Object? rows) => Float64List.fromList([
  for (final row in rows as List<Object?>) ...series(row),
]);

/// Describes the first element where [actual] differs from [expected]
/// beyond tolerance (NaN matches NaN), plus the max relative error; `null`
/// when everything matches.
String? firstMismatch(
  List<double> expected,
  List<double> actual,
  double relTol,
  double absTol,
) {
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
    if (diff <= absTol) continue;
    final rel = diff / [e.abs(), a.abs()].reduce((x, y) => x > y ? x : y);
    if (rel.isNaN || rel > relTol) {
      first ??= t;
    }
    if (rel > maxRel || rel.isNaN) maxRel = rel;
  }
  if (first == null) return null;
  return 't=$first expected ${expected[first]} got ${actual[first]} '
      '(max rel err ${maxRel.toStringAsExponential(2)})';
}

/// The custom coupling functions the fixtures use by name (JAX differentiates
/// them with autodiff; here they come with closed-form derivatives).
final Map<String, Coupling> testCouplings = {
  'sin': CustomCoupling(
    'sin',
    g: math.sin,
    dg: math.cos,
    d2g: (x) => -math.sin(x),
  ),
  'softplus': CustomCoupling(
    'softplus',
    g: (x) => math.max(x, 0.0) + math.log(1.0 + math.exp(-x.abs())),
    dg: sigmoid,
    d2g: (x) {
      final s = sigmoid(x);
      return s * (1.0 - s);
    },
  ),
};

/// Mismatches of the fixture's ef-state vector trajectories
/// (`vector_trajectories`: `{node: {field: [[...] per trial]}}`).
List<String> vectorMismatches(
  Map<String, Object?> fixture,
  Trajectories traj,
  double relTol,
  double absTol,
) {
  final failures = <String>[];
  final ref = fixture['vector_trajectories'] as Map<String, Object?>?;
  ref?.forEach((nodeKey, raw) {
    final node = int.parse(nodeKey);
    (raw as Map<String, Object?>).forEach((fieldName, rows) {
      final field = VectorField.tryParse(fieldName);
      if (field == null) {
        failures.add('node $node: unknown vector field $fieldName');
        return;
      }
      final mismatch = firstMismatch(
        flatten(rows),
        traj.vectorSeries(node, field),
        relTol,
        absTol,
      );
      if (mismatch != null) failures.add('node $node $fieldName: $mismatch');
    });
  });
  return failures;
}

/// The fixture's JAX update sequence (`"node:function"`), and this
/// network's, by [UpdateStep.pyhgfName]. Predictions are sorted: JAX
/// schedules them a pass at a time and rshgf (which hgf.dart follows) one
/// node at a time, so independent nodes come in a different order, which
/// doesn't change any value. The update steps must match in order.
(List<String> expected, List<String> actual) jaxSequences(
  Map<String, Object?> fixture,
  Network network,
) {
  final ref = fixture['update_sequence'] as Map<String, Object?>;
  List<String> names(Iterable<Object?> steps) => [
    for (final s in steps) '${(s as List)[0]}:${s[1]}',
  ];
  final seq = network.updateSequence;
  List<String> ours(List<(int, UpdateStep)> steps) => [
    for (final (i, step) in steps) '$i:${step.pyhgfName}',
  ];
  return (
    [
      ...names(ref['prediction'] as List<Object?>)..sort(),
      ...names(ref['update'] as List<Object?>),
    ],
    [...ours(seq.predictions)..sort(), ...ours(seq.updates)],
  );
}
