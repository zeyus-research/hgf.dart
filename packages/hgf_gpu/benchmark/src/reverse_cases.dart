// Models for the reverse-mode spike (test/native_reverse_test.dart and
// benchmark/bin/reverse_gpu.dart).
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_gpu/src/fit/grad_kernel.dart';
import 'package:hgf_gpu/src/trial.dart';

final class ReverseCase {
  const ReverseCase(
    this.name,
    this.network,
    this.response,
    this.params, {
    required this.nInputs,
    required this.binary,
  });

  final String name;
  final Network Function() network;
  final GpuResponse response;

  /// Slot variables differentiated (forward mode cycles through them).
  final List<String> params;
  final int nInputs;
  final bool binary;

  /// Per-element initial slots: the network's defaults, each parameter
  /// perturbed per element.
  Float32List slots(GradKernel k, int batch, math.Random rng) {
    final out = Float32List(k.nSlots * batch);
    for (var s = 0; s < k.nSlots; s++) {
      for (var e = 0; e < batch; e++) {
        out[s * batch + e] = k.initial[s];
      }
    }
    for (final v in params) {
      final s = k.slotOfVar[v]!;
      for (var e = 0; e < batch; e++) {
        final x0 = k.initial[s];
        final u = rng.nextDouble();
        out[s * batch + e] = switch (v) {
          _ when v.startsWith('tv') => x0 - 1.0 - 2.0 * u,
          _ when v.startsWith('p') => x0 * (0.5 + 1.5 * u),
          _ when v.startsWith('ac') => 0.95 + 0.05 * u,
          _ when v.startsWith('td') => 0.1 * (u - 0.5),
          _ when v.startsWith('rp') => 0.5 + 5.5 * u,
          _ when v.startsWith('k') || v.startsWith('a') => 0.5 + u,
          _ => x0 + (u - 0.5),
        };
      }
    }
    return out;
  }

  /// One shared series, `[t][input]`.
  Float32List data(int nT, math.Random rng) {
    final out = Float32List(nT * nInputs);
    for (var q = 0; q < nInputs; q++) {
      var p = 0.5, x = 0.0;
      for (var t = 0; t < nT; t++) {
        if (t % 40 == 0) p = 0.15 + 0.7 * rng.nextDouble();
        x += 0.1 * (rng.nextDouble() - 0.5) + (t % 60 == 0 ? 1.0 : 0.0);
        out[t * nInputs + q] = binary
            ? (rng.nextDouble() < p ? 1.0 : 0.0)
            : x + 0.2 * (rng.nextDouble() - 0.5);
      }
    }
    return out;
  }

  /// Random 0/1 responses `[t][element]` (for a response model that scores
  /// them), else null.
  Float32List? responses(int nT, int batch, math.Random rng) {
    if (!response.usesResponses) return null;
    return Float32List.fromList([
      for (var i = 0; i < nT * batch; i++) rng.nextBool() ? 1.0 : 0.0,
    ]);
  }
}

Network _binary2() => Network(volatilityUpdates: VolatilityUpdates.standard)
  ..addNodes(kind: NodeKind.binaryState)
  ..addNodes(valueChildren: [0]);

Network _continuous3(VolatilityUpdates u) => Network(volatilityUpdates: u)
  ..addNodes()
  ..addNodes(valueChildren: [0])
  ..addNodes(volatilityChildren: [1]);

/// 8 binary inputs, a value parent each, one shared volatility parent.
Network _wideBinary() {
  final net = Network(volatilityUpdates: VolatilityUpdates.standard);
  for (var i = 0; i < 8; i++) {
    net.addNodes(kind: NodeKind.binaryState);
  }
  for (var i = 0; i < 8; i++) {
    net.addNodes(valueChildren: [i]);
  }
  net.addNodes(volatilityChildren: [for (var i = 8; i < 16; i++) i]);
  return net;
}

const _c3Params = [
  'tv0', 'tv1', 'tv2', 'm1', 'p1', 'm2', 'p2', 'k2_1', 'td1', 'ac1', //
];

final reverseCases = <ReverseCase>[
  ReverseCase(
    'binary2-softmax',
    _binary2,
    const GpuBinarySoftmax(),
    const ['tv1', 'rp0', 'm1', 'p1', 'ac1', 'td1'],
    nInputs: 1,
    binary: true,
  ),
  ReverseCase(
    'continuous3-standard',
    () => _continuous3(VolatilityUpdates.standard),
    GpuResponse.totalGaussianSurprise,
    _c3Params,
    nInputs: 1,
    binary: false,
  ),
  ReverseCase(
    'continuous3-eHGF',
    () => _continuous3(VolatilityUpdates.eHGF),
    GpuResponse.totalGaussianSurprise,
    _c3Params,
    nInputs: 1,
    binary: false,
  ),
  ReverseCase(
    'wide-binary-33',
    _wideBinary,
    GpuResponse.firstLevelBinarySurprise,
    [
      for (var i = 8; i < 16; i++) 'tv$i',
      'tv16',
      for (var i = 8; i < 16; i++) 'm$i',
      for (var i = 8; i < 16; i++) 'p$i',
      for (var i = 8; i < 16; i++) 'k16_$i',
    ],
    nInputs: 8,
    binary: true,
  ),
];
