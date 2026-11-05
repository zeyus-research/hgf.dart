@TestOn('vm')
@Tags(['parity'])
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

import 'src/fixtures.dart';
import 'src/surprise_checks.dart';

/// Relative tolerance against the float64 JAX reference.
const double relTol = 1e-9;

/// Absolute floor: values within this of each other count as equal (values
/// near 0, where one ulp reads as a large relative error).
const double absTol = 1e-12;

/// Absolute floor for surprise values, which are sums of O(1-10) terms and
/// cross zero, so a relative error near a crossing is meaningless.
const double surpriseAbsTol = 1e-10;

void expectTrajectories(
  Trajectories traj,
  Map<String, Object?> reference,
  double tol,
) {
  final failures = <String>[];
  var compared = 0;
  reference.forEach((nodeKey, raw) {
    final node = int.parse(nodeKey);
    (raw as Map<String, Object?>).forEach((fieldName, values) {
      final expected = series(values);
      final actual = traj.series(node, NodeField.parse(fieldName));
      compared += expected.length;
      final mismatch = firstMismatch(expected, actual, tol, absTol);
      if (mismatch != null) failures.add('node $node $fieldName: $mismatch');
    });
  });
  expect(compared, greaterThan(0));
  expect(failures, isEmpty, reason: failures.take(10).join('\n'));
}

void main() {
  final dir = Directory('test/fixtures');
  final files =
      dir
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.json'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

  test('fixtures exist', () {
    expect(files, isNotEmpty, reason: 'Run `melos run fixtures` first');
  });

  for (final file in files) {
    final fixture = jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
    final name = fixture['name'] as String;

    group(name, () {
      late Network network;
      late Trajectories traj;

      setUpAll(() {
        network = NetworkSpec.fromJson(
          fixture['spec'] as Map<String, Object?>,
        ).build(couplings: testCouplings);
        final allFields = NodeField.values.toList();
        traj = network.inputData(
          flatten(fixture['inputs']),
          observed: fixture['observed'] == null
              ? null
              : flatten(fixture['observed']),
          timeSteps: fixture['time_steps'] == null
              ? null
              : Float64List.fromList(series(fixture['time_steps'])),
          fields: allFields,
        )!;
      });

      test('input nodes', () {
        expect(network.inputIdxs, fixture['input_idxs']);
      });

      test('update sequence matches pyhgf JAX', () {
        final (expected, actual) = jaxSequences(fixture, network);
        expect(actual, expected);
      });

      if (fixture['rshgf_update_sequence'] != null) {
        test('update sequence matches rshgf', () {
          final expected = [
            for (final s in fixture['rshgf_update_sequence'] as List<Object?>)
              '${(s as List)[0]}:${s[1]}',
          ];
          final seq = network.updateSequence;
          final actual = [
            for (final (i, step) in [...seq.predictions, ...seq.updates])
              '$i:${step.rshgfName}',
          ];
          expect(actual, expected);
        });
      }

      // Where pyhgf's own two backends disagree (ill-conditioned spikes), the
      // JAX tolerance is widened to that measured gap and the port is held to
      // the strict tolerance against rshgf, whose arithmetic it follows.
      final rsDiff = (fixture['rshgf_max_rel_diff'] as num?)?.toDouble() ?? 0;
      final jaxTol = rsDiff > relTol ? 2 * rsDiff : relTol;

      // The `*_mgf_correction` fixtures come from pyhgf JAX patched with the
      // proposed eHGF fix (Network.ehgfMgfCorrection), not released pyhgf.
      final reference = fixture['reference'] as String? ?? 'pyhgf JAX';

      test('trajectories match $reference (float64, rel tol '
          '${jaxTol.toStringAsExponential(1)})', () {
        expectTrajectories(
          traj,
          fixture['trajectories'] as Map<String, Object?>,
          jaxTol,
        );
        final vectors = vectorMismatches(fixture, traj, jaxTol, absTol);
        expect(vectors, isEmpty, reason: vectors.take(10).join('\n'));
      });

      test('surprise and response functions match $reference', () {
        final failures = surpriseMismatches(
          fixture,
          network,
          traj,
          // Surprise amplifies trajectory differences (π̂·δ², δ = x − μ̂
          // cancels), so where pyhgf's backends already disagree it gets 10×
          // the trajectory tolerance.
          rsDiff > relTol ? 10 * jaxTol : relTol,
          surpriseAbsTol,
        );
        expect(failures, isEmpty, reason: failures.take(10).join('\n'));
      });

      if (fixture['rshgf_trajectories'] != null) {
        test('trajectories match rshgf (rel tol $relTol)', () {
          expectTrajectories(
            traj,
            fixture['rshgf_trajectories'] as Map<String, Object?>,
            relTol,
          );
        });
      }
    });
  }
}
