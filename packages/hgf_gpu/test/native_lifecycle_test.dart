/// Pipelines and buffers on wgpu-native release cleanly, in either order.
///
///     dart test --tags gpu test/native_lifecycle_test.dart
@TestOn('vm')
@Tags(['gpu'])
library;

import 'package:hgf_gpu/native.dart';
import 'package:test/test.dart';

import 'src/lifecycle_check.dart';

void main() {
  test('pipelines and buffers release in either order', () async {
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
    final failures = await checkLifecycle(backend, rounds: 50);
    expect(failures, isEmpty, reason: failures.join('\n'));
  });
}
