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

  for (final file in files) {
    final fixture = jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
    test('${fixture['name']} matches pyhgf (JAX f64)', () {
      final failures = checkDeepFixture(fixture);
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
