import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'fixtures.dart';
import 'surprise_checks.dart';

const double relTol = 1e-9;
const double absTol = 1e-12;
const double surpriseAbsTol = 1e-10;

/// Rebuild [fixture]'s network, run it with every field recorded, and list
/// how its trajectories, surprise and response functions differ from pyhgf
/// (empty when they match). Shared by `web_parity_test.dart` and the
/// standalone `tool/parity_check.dart` (which can be compiled with any
/// flags, e.g. `dart compile wasm -O4`).
///
/// The tolerance is [relTol], or twice the JAX↔rshgf gap where the two
/// pyhgf backends disagree by more.
List<String> fixtureMismatches(Map<String, Object?> fixture) {
  final name = fixture['name'] as String;
  final network = NetworkSpec.fromJson(
    fixture['spec'] as Map<String, Object?>,
  ).build(couplings: testCouplings);
  final traj = network.inputData(
    flatten(fixture['inputs']),
    observed: fixture['observed'] == null ? null : flatten(fixture['observed']),
    timeSteps: fixture['time_steps'] == null
        ? null
        : Float64List.fromList(series(fixture['time_steps'])),
    fields: NodeField.values,
  )!;
  final rsDiff = (fixture['rshgf_max_rel_diff'] as num?)?.toDouble() ?? 0;
  final tol = rsDiff > relTol ? 2 * rsDiff : relTol;
  final failures = <String>[];
  (fixture['trajectories'] as Map<String, Object?>).forEach((key, raw) {
    (raw as Map<String, Object?>).forEach((field, values) {
      final mismatch = firstMismatch(
        series(values),
        traj.series(int.parse(key), NodeField.parse(field)),
        tol,
        absTol,
      );
      if (mismatch != null) failures.add('$name node $key $field: $mismatch');
    });
  });
  failures.addAll(
    vectorMismatches(fixture, traj, tol, absTol).map((m) => '$name $m'),
  );
  final (expectedSeq, actualSeq) = jaxSequences(fixture, network);
  if (expectedSeq.join(',') != actualSeq.join(',')) {
    failures.add('$name update sequence: $actualSeq != $expectedSeq');
  }
  failures.addAll(
    surpriseMismatches(
      fixture,
      network,
      traj,
      rsDiff > relTol ? 10 * tol : relTol,
      surpriseAbsTol,
    ).map((m) => '$name $m'),
  );
  return failures;
}
