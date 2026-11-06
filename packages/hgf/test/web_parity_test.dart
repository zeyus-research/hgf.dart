/// The parity suite on the JS / Wasm compilers: the fixtures are read on the
/// VM side through a hybrid isolate (browsers and Node have no file system
/// access from the test), then checked like `parity_test.dart`.
///
///     dart test -p chrome --tags parity          # dart2js in Chrome
///     dart test -p node --tags parity            # dart2js in Node
///     dart test -p chrome -c dart2wasm --tags parity
@TestOn('browser || node')
@Tags(['parity'])
library;

import 'dart:convert';

import 'package:test/test.dart';

import 'src/batch_check.dart';
import 'src/parity_check.dart';
import 'src/tangent_check.dart';

const _reader = '''
import 'dart:io';
import 'package:stream_channel/stream_channel.dart';

void hybridMain(StreamChannel<Object?> channel) {
  final files = Directory('test/fixtures')
      .listSync()
      .whereType<File>()
      .where((f) => f.path.endsWith('.json'))
      .toList()
    ..sort((a, b) => a.path.compareTo(b.path));
  for (final f in files) {
    channel.sink.add(f.readAsStringSync());
  }
  channel.sink.close();
}
''';

void main() {
  late List<Map<String, Object?>> fixtures;

  setUpAll(() async {
    final channel = spawnHybridCode(_reader);
    fixtures = [
      await for (final text in channel.stream)
        jsonDecode(text as String) as Map<String, Object?>,
    ];
  });

  test('every fixture matches pyhgf on this compiler', () {
    expect(fixtures, isNotEmpty);
    final failures = [for (final f in fixtures) ...fixtureMismatches(f)];
    expect(failures, isEmpty, reason: failures.take(10).join('\n'));
  });
  test('BatchNetwork matches Network bit for bit on this compiler', () {
    final failures = [for (final f in fixtures) ...batchMismatches(f)];
    expect(failures, isEmpty, reason: failures.take(10).join('\n'));
  });
  test('BatchNetwork tangents match finite differences on this compiler', () {
    final failures = [for (final f in fixtures) ...?checkTangents(f)?.$1];
    expect(failures, isEmpty, reason: failures.take(10).join('\n'));
  });
}
