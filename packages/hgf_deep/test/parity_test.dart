@TestOn('vm')
@Tags(['parity'])
library;

import 'dart:convert';
import 'dart:io';

import 'package:hgf_deep/hgf_deep.dart';
import 'package:test/test.dart';

import 'src/deep_check.dart';

void main() {
  final files =
      Directory('test/fixtures')
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.json'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

  test('fixtures exist', () {
    expect(files, isNotEmpty, reason: 'Run `melos run fixtures:deep` first');
  });

  // Fixtures whose reference is ill-conditioned, with a wider relative
  // tolerance (as hgf's parity suite widens it where pyhgf's backends
  // disagree).
  //
  // input_layer_feedforward_tonic: in op 2 (synaptic fit) layer 1 runs away
  // (docs/upstream P18/P19), its exponent roughly doubling per step (1e24,
  // 1e44, 1e83, 1e192, then NaN). A last-ulp difference from JAX/XLA
  // (1.8e-13 on a prediction) grows with it: ~1e-12 at 1e9, ~1e-11 at
  // 1e100, 4.4e-9 at 1e192. Ops 0 and 1 agree to ~1e-15; the NaN onset
  // must still match exactly.
  const relTolOverrides = {'input_layer_feedforward_tonic': 1e-8};

  for (final file in files) {
    final fixture = jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
    final rel = relTolOverrides[fixture['name']] ?? relTol;
    test('${fixture['name']} matches pyhgf (JAX f64)', () {
      final failures = checkDeepFixture(fixture, rel: rel);
      expect(failures, isEmpty, reason: failures.take(12).join('\n'));
    });
  }

  // DeepFixes.logSpaceVolatility rewrites the volatility terms with exact
  // identities and reads the stored known variance instead of a
  // reconstruction that cancels: wherever pyhgf is finite the values must
  // be pyhgf's (to rounding; the reconstruction loses up to
  // ε·Ω·π of relative accuracy).
  group('logSpaceVolatility leaves finite pyhgf values unchanged', () {
    for (final file in files) {
      final fixture =
          jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
      final base = fixesOf(fixture['settings'] as Map<String, Object?>);
      test('${fixture['name']}', () {
        final failures = checkDeepFixture(
          fixture,
          fixes: DeepFixes(
            curvatureFloor: base.curvatureFloor,
            logSpaceVolatility: true,
          ),
          finiteOnly: true,
          rel: 1e-7,
        );
        expect(failures, isEmpty, reason: failures.take(12).join('\n'));
      });
    }
  });
}
