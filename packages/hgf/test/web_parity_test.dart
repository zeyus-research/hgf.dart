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

  // Each check yields between fixtures: browser test suites share one
  // event loop, and a long synchronous run (the dp-state fixtures take
  // seconds each) keeps the other suites' pages from loading. Yielding
  // makes the default 30 s timeout apply (a synchronous body can't be
  // timed out), and each check takes minutes on dart2js.
  const long = Timeout(Duration(minutes: 10));
  Future<List<String>> each(
    Iterable<String> Function(Map<String, Object?>) check,
  ) async {
    final failures = <String>[];
    for (final f in fixtures) {
      failures.addAll(check(f));
      await Future<void>.delayed(Duration.zero);
    }
    return failures;
  }

  test('every fixture matches pyhgf on this compiler', () async {
    expect(fixtures, isNotEmpty);
    final failures = await each(fixtureMismatches);
    expect(failures, isEmpty, reason: failures.take(10).join('\n'));
  }, timeout: long);
  test('BatchNetwork matches Network bit for bit on this compiler', () async {
    final failures = await each(batchMismatches);
    expect(failures, isEmpty, reason: failures.take(10).join('\n'));
  }, timeout: long);
  test(
    'BatchNetwork tangents match finite differences on this compiler',
    () async {
      final failures = await each((f) => checkTangents(f)?.$1 ?? const []);
      expect(failures, isEmpty, reason: failures.take(10).join('\n'));
    },
    timeout: long,
  );
}
