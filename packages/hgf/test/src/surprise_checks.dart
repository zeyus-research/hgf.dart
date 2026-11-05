import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'fixtures.dart';

/// The fixture's run arguments.
({Float64List data, Float64List? observed, Float64List? timeSteps}) runArgs(
  Map<String, Object?> fixture,
) => (
  data: flatten(fixture['inputs']),
  observed: fixture['observed'] == null ? null : flatten(fixture['observed']),
  timeSteps: fixture['time_steps'] == null
      ? null
      : Float64List.fromList(series(fixture['time_steps'])),
);

/// The response models a fixture has references for, by pyhgf name.
Map<String, (ResponseFunction, List<double>)> fixtureResponses(
  Map<String, Object?> fixture,
) {
  final ref = fixture['surprise'] as Map<String, Object?>;
  final responses = ref['responses'] == null
      ? null
      : Float64List.fromList(series(ref['responses']));
  final out = <String, (ResponseFunction, List<double>)>{};
  (ref['response_functions'] as Map<String, Object?>).forEach((name, raw) {
    switch (name) {
      case 'first_level_gaussian_surprise':
        out[name] = (const FirstLevelGaussianSurprise(), series(raw));
      case 'total_gaussian_surprise':
        out[name] = (const TotalGaussianSurprise(), series(raw));
      case 'first_level_binary_surprise':
        out[name] = (const FirstLevelBinarySurprise(), series(raw));
      case 'binary_softmax':
        out[name] = (BinarySoftmax(responses!), series(raw));
      case 'binary_softmax_inverse_temperature':
        final m = raw as Map<String, Object?>;
        out[name] = (
          BinarySoftmaxInverseTemperature(
            responses!,
            inverseTemperature: (m['inverse_temperature'] as num).toDouble(),
          ),
          series(m['surprise']),
        );
      default:
        throw StateError('No Dart response function for $name');
    }
  });
  return out;
}

/// Compare per-node / total surprise (from [traj]) and every response model
/// (run on [network], through both `inputData(response:)` and
/// `Network.surprise`) against the fixture. Returns the mismatches.
List<String> surpriseMismatches(
  Map<String, Object?> fixture,
  Network network,
  Trajectories traj,
  double relTol,
  double absTol,
) {
  final failures = <String>[];
  final ref = fixture['surprise'] as Map<String, Object?>;
  void check(String what, List<double> expected, List<double> actual) {
    final m = firstMismatch(expected, actual, relTol, absTol);
    if (m != null) failures.add('$what: $m');
  }

  (ref['node_surprise'] as Map<String, Object?>).forEach((key, raw) {
    check('node $key surprise', series(raw), traj.nodeSurprise(int.parse(key)));
  });
  check('total surprise', series(ref['total_surprise']), traj.totalSurprise());

  final args = runArgs(fixture);
  fixtureResponses(fixture).forEach((name, entry) {
    final (fn, expected) = entry;
    final perTrial = Float64List(expected.length);
    final total = network.surprise(
      args.data,
      fn,
      observed: args.observed,
      timeSteps: args.timeSteps,
      perTrial: perTrial,
    );
    // pyhgf's total_gaussian_surprise is +inf on *every* trial once node 1's
    // mean is NaN anywhere; the trial-wise port is +inf from that trial on.
    // Only the totals are comparable then.
    final allInf = expected.every((v) => v == double.infinity);
    if (!allInf) check(name, expected, perTrial);
    final expectedTotal = expected.fold(0.0, (a, b) => a + b);
    check('$name (total)', [expectedTotal], [total]);
    final recorded = network.inputData(
      args.data,
      observed: args.observed,
      timeSteps: args.timeSteps,
      fields: const [],
      response: fn,
    )!;
    if (!allInf) {
      check('$name (inputData)', expected, recorded.responseSurprise!);
    }
  });
  return failures;
}
