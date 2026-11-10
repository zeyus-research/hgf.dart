/// Reverse-mode spike: adjoint gradients (tape and checkpointed) against the
/// forward-mode tangents, on wgpu-native. Skipped if the library can't be
/// loaded.
///
///     dart test --tags gpu test/native_reverse_test.dart
@TestOn('vm')
@Tags(['gpu'])
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_gpu/native.dart';
import 'package:hgf_gpu/src/fit/grad_kernel.dart';
import 'package:hgf_gpu/src/trial.dart';
import 'package:test/test.dart';

import '../benchmark/src/reverse_cases.dart';

WgpuBackend? _backend() {
  try {
    return WgpuBackend.create();
  } on ArgumentError {
    return null;
  }
}

void main() {
  final backend = _backend();
  const batch = 256, nT = 200;

  for (final rc in reverseCases) {
    test('${rc.name}: reverse = forward', () async {
      final params = rc.params;
      final rng = math.Random(7);
      final kernels = {
        'forward': GradKernel(
          rc.network(),
          rc.response,
          params: params,
          directions: params.length,
        ),
        'tape': GradKernel(
          rc.network(),
          rc.response,
          params: params,
          mode: GradMode.reverseTape,
        ),
        'checkpoint': GradKernel(
          rc.network(),
          rc.response,
          params: params,
          mode: GradMode.reverseCheckpoint,
          checkpoint: 15, // 200 is not a multiple: a short last segment.
        ),
      };
      final fk = kernels['forward']!;
      final slots = rc.slots(fk, batch, rng);
      final data = rc.data(nT, rng);
      final resp = rc.responses(nT, batch, rng);
      final out = <String, Float32List>{};
      for (final MapEntry(:key, :value) in kernels.entries) {
        final job = await GradJob.create(
          backend!,
          value,
          batch: batch,
          nT: nT,
          slots: slots,
          data: data,
          responses: resp,
        );
        out[key] = await job.run();
        job.release();
      }
      final f = out['forward']!;
      var checked = 0;
      for (final mode in ['tape', 'checkpoint']) {
        final r = out[mode]!;
        for (var e = 0; e < batch; e++) {
          final nll = f[e];
          if (nll >= 3.3e38) continue;
          // The tangent kernel's primal compiles differently (contractions):
          // last-bit differences.
          expect(
            (r[e] - nll).abs() / math.max(1.0, nll.abs()),
            lessThan(1e-6),
            reason: '$mode likelihood, element $e',
          );
          for (var i = 0; i < params.length; i++) {
            final a = f[(1 + i) * batch + e], b = r[(1 + i) * batch + e];
            // f32, summed in a different order over 200 trials.
            final scale = math.max(1.0, a.abs());
            expect(
              (a - b).abs() / scale,
              lessThan(2e-4),
              reason: '$mode ∂/∂${params[i]}, element $e: $a vs $b',
            );
            checked++;
          }
        }
      }
      expect(checked, greaterThan(batch * params.length));
    }, skip: backend == null ? 'no wgpu-native' : false);
  }

  test('compileSsa: unbounded update has no SSA form', () {
    final net = Network()
      ..addNodes()
      ..addNodes(valueChildren: [0])
      ..addNodes(volatilityChildren: [1]);
    expect(
      () => GradKernel(
        net,
        GpuResponse.totalGaussianSurprise,
        params: const ['tv1'],
        mode: GradMode.reverseTape,
      ),
      throwsUnsupportedError,
    );
  });
}
