@TestOn('vm')
@Tags(['parity'])
library;

import 'dart:convert';
import 'dart:io';

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
}
