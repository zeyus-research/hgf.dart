/// The GPU kernels against the f64 engine on every parity fixture whose node
/// kinds have a kernel: five perturbed parameter sets per fixture, total
/// surprise from `BatchNetwork` (f64, CPU) and from `GpuBatch` (f32, WebGPU).
///
/// Needs WebGPU in the test browser (`chrome-gpu` in `dart_test.yaml`).
/// `dart test` starts Chrome with `--disable-gpu`, so this runs on
/// SwiftShader (Chromium's software Vulkan), also on CI;
/// `native_parity_test.dart` runs the same check on the GPU.
///
///     dart test -p chrome-gpu -c dart2wasm --tags gpu
@TestOn('browser')
@Tags(['gpu'])
library;

import 'dart:convert';

import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/web.dart';
import 'package:test/test.dart';

import 'src/deep_gpu_check.dart' as deep;
import 'src/fit_check.dart';
import 'src/gpu_check.dart';

const _reader = '''
import 'dart:io';
import 'package:stream_channel/stream_channel.dart';

void hybridMain(StreamChannel<Object?> channel, Object? dir) {
  final files = Directory(dir! as String)
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
  late List<Map<String, Object?>> fixtures, deepFixtures;
  GpuBackend? backend;

  Future<List<Map<String, Object?>>> read(String dir) async => [
    await for (final text in spawnHybridCode(_reader, message: dir).stream)
      jsonDecode(text! as String) as Map<String, Object?>,
  ];

  setUpAll(() async {
    fixtures = await read('../hgf/test/fixtures');
    deepFixtures = await read('../hgf_deep/test/fixtures');
    backend = await WebGpuBackend.create();
  });

  test('every supported fixture: f32 GPU totals match f64 CPU', () async {
    if (backend == null) {
      markTestSkipped('No WebGPU adapter in this browser');
      return;
    }
    final failures = await checkAll(backend!, fixtures);
    expect(failures, isEmpty, reason: failures.join('\n'));
  });

  test('fit passes: f32 GPU values and gradients match f64 CPU', () async {
    if (backend == null) {
      markTestSkipped('No WebGPU adapter in this browser');
      return;
    }
    final fitter = GpuFitter(backend!);
    final failures = await checkFitCases(fitter, log: print);
    expect(fitter.cpuPasses, 0, reason: fitter.fallbackReason);
    expect(failures, isEmpty, reason: failures.take(20).join('\n'));
  });

  test('hgf_deep fixtures: batchUpdate and predict match f64 CPU', () async {
    if (backend == null) {
      markTestSkipped('No WebGPU adapter in this browser');
      return;
    }
    final bad = <String>[];
    for (final fixture in deepFixtures) {
      final name = fixture['name']! as String;
      final results = [
        for (final r in await deep.checkFixture(backend!, fixture))
          if (!r.$2.isNaN) r,
      ];
      bad.addAll([for (final f in deep.failures(name, results)) '$name: $f']);
    }
    expect(deepFixtures, isNotEmpty);
    expect(bad, isEmpty, reason: bad.join('\n'));
  });
}
