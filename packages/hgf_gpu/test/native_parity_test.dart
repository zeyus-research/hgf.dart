/// The check of `gpu_parity_test.dart` on wgpu-native (the machine's GPU),
/// through `dart:ffi`. Skipped if the library can't be loaded (install
/// wgpu-native, or run `tool/fetch_wgpu.sh` and set `WGPU_LIB`).
///
///     dart test --tags gpu test/native_parity_test.dart
@TestOn('vm')
@Tags(['gpu'])
library;

import 'dart:convert';
import 'dart:io';

import 'package:hgf_gpu/native.dart';
import 'package:test/test.dart';

import 'src/gpu_check.dart';

void main() {
  test('every supported fixture: f32 GPU totals match f64 CPU', () async {
    final WgpuBackend? backend;
    try {
      backend = WgpuBackend.create();
    } on ArgumentError catch (e) {
      markTestSkipped('wgpu-native not available: ${e.message}');
      return;
    }
    if (backend == null) {
      markTestSkipped('No wgpu adapter');
      return;
    }
    final files =
        Directory('../hgf/test/fixtures')
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.json'))
            .toList()
          ..sort((a, b) => a.path.compareTo(b.path));
    final fixtures = [
      for (final f in files)
        jsonDecode(f.readAsStringSync()) as Map<String, Object?>,
    ];
    final failures = await checkAll(backend, fixtures);
    expect(failures, isEmpty, reason: failures.join('\n'));
  });
}
