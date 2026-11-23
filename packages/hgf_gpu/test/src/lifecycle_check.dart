import 'dart:typed_data';

import 'package:hgf_gpu/hgf_gpu.dart';

const _double = '''
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&y)) { y[id.x] = 2.0 * x[id.x] + SHIFT; }
}
''';

/// Compiles, runs and releases pipelines and buffers on [backend] in both
/// orders (pipeline first, buffers first), [rounds] times, each round with a
/// freshly compiled kernel. Returns the mismatches (empty if all is well).
Future<List<String>> checkLifecycle(
  GpuBackend backend, {
  int rounds = 8,
}) async {
  const n = 100;
  final failures = <String>[];
  for (var r = 0; r < rounds; r++) {
    final pipeline = await backend.pipeline(
      _double.replaceAll('SHIFT', '$r.0'),
    );
    final x = backend.storage(n * 4), y = backend.storage(n * 4);
    final host = backend.hostFloats(n);
    for (var i = 0; i < n; i++) {
      host[i] = i.toDouble();
    }
    backend
      ..write(x, host)
      ..dispatch(pipeline, [x, y], (n + 63) ~/ 64);
    final out = await backend.read(y, bytes: n * 4);
    for (var i = 0; i < n; i++) {
      if (out[i] != 2.0 * i + r) {
        failures.add('round $r: y[$i] = ${out[i]}, expected ${2 * i + r}');
        break;
      }
    }
    if (r.isEven) {
      backend
        ..releasePipeline(pipeline)
        ..release(x)
        ..release(y);
    } else {
      backend
        ..release(x)
        ..release(y)
        ..releasePipeline(pipeline);
    }
  }
  // A buffer outliving the pipelines it was bound with still releases.
  final keep = backend.storage(16);
  final p = await backend.pipeline(_double.replaceAll('SHIFT', '0.0'));
  final other = backend.storage(16);
  backend
    ..write(keep, Float32List(4))
    ..dispatch(p, [keep, other], 1)
    ..releasePipeline(p)
    ..releasePipeline(p); // idempotent
  await backend.idle();
  backend
    ..release(other)
    ..release(keep);
  return failures;
}
