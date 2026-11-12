// Models for the gradient-cost benchmarks (reverse_cpu.dart here, and
// hgf_gpu's fit_reverse_gpu.dart).
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

typedef Case = ({
  String name,
  Network Function() network,
  ResponseModel response,
  List<Parameter> parameters,
  int nIn,
  bool binary,
});

Parameter node(String n, int i, NodeField f, double mean, double sd) =>
    Parameter.node(n, i, f, prior: Normal(mean, sd));

Network continuous3(VolatilityUpdates u) => Network(volatilityUpdates: u)
  ..addNodes()
  ..addNodes(valueChildren: [0])
  ..addNodes(volatilityChildren: [1]);

final c3Params = [
  node('w1', 1, NodeField.tonicVolatility, -4, 1),
  node('w2', 2, NodeField.tonicVolatility, -4, 1),
  node('w0', 0, NodeField.tonicVolatility, -2, 1),
  node('m1', 1, NodeField.mean, 0, 1),
  node('p1', 1, NodeField.precision, 1, 0.1),
  node('m2', 2, NodeField.mean, 0, 1),
  node('p2', 2, NodeField.precision, 1, 0.1),
  Parameter.volatilityCoupling('k', 2, 1, prior: const Normal(1, 0.1)),
  node('d1', 1, NodeField.tonicDrift, 0, 0.01),
  node('a1', 1, NodeField.autoconnectionStrength, 1, 0.01),
];

final cases = <Case>[
  (
    name: 'binary 2-level, softmax',
    network: () => Network(volatilityUpdates: VolatilityUpdates.standard)
      ..addNodes(kind: NodeKind.binaryState)
      ..addNodes(valueChildren: [0]),
    response: const BinarySoftmaxReadout(),
    parameters: [
      node('w1', 1, NodeField.tonicVolatility, -3, 1),
      Parameter.response('inverseTemperature', prior: const Normal(2, 0.5)),
      node('m1', 1, NodeField.mean, 0, 1),
      node('p1', 1, NodeField.precision, 1, 0.1),
      node('a1', 1, NodeField.autoconnectionStrength, 1, 0.01),
      node('d1', 1, NodeField.tonicDrift, 0, 0.01),
    ],
    nIn: 1,
    binary: true,
  ),
  for (final u in [
    VolatilityUpdates.standard,
    VolatilityUpdates.eHGF,
    VolatilityUpdates.unbounded,
  ])
    (
      name: 'continuous 3-level ${u.name}',
      network: () => continuous3(u),
      response: const ObservationSurprise(TotalGaussianSurprise()),
      parameters: c3Params,
      nIn: 1,
      binary: false,
    ),
  (
    name: '8 binary inputs',
    network: () {
      final net = Network(volatilityUpdates: VolatilityUpdates.standard);
      for (var i = 0; i < 8; i++) {
        net.addNodes(kind: NodeKind.binaryState);
      }
      for (var i = 0; i < 8; i++) {
        net.addNodes(valueChildren: [i]);
      }
      return net
        ..addNodes(volatilityChildren: [for (var i = 8; i < 16; i++) i]);
    },
    response: const ObservationSurprise(FirstLevelBinarySurprise()),
    parameters: [
      for (var i = 8; i < 16; i++)
        node('w$i', i, NodeField.tonicVolatility, -3, 1),
      node('w16', 16, NodeField.tonicVolatility, -4, 1),
      for (var i = 8; i < 16; i++) node('m$i', i, NodeField.mean, 0, 1),
      for (var i = 8; i < 16; i++) node('p$i', i, NodeField.precision, 1, 0.1),
      for (var i = 8; i < 16; i++)
        Parameter.volatilityCoupling('k$i', 16, i, prior: const Normal(1, 0.1)),
    ],
    nIn: 8,
    binary: true,
  ),
];

List<Subject> subjects(Case c, int n, int nT, math.Random rng) => [
  for (var s = 0; s < n; s++)
    () {
      final inputs = Float64List(nT * c.nIn);
      for (var q = 0; q < c.nIn; q++) {
        var x = 0.0, p = 0.5;
        for (var t = 0; t < nT; t++) {
          if (t % 40 == 0) p = 0.15 + 0.7 * rng.nextDouble();
          x += 0.1 * (rng.nextDouble() - 0.5) + (t % 60 == 0 ? 1.0 : 0.0);
          inputs[t * c.nIn + q] = c.binary
              ? (rng.nextDouble() < p ? 1.0 : 0.0)
              : x + 0.2 * (rng.nextDouble() - 0.5);
        }
      }
      return Subject(
        inputs: inputs,
        responses: Float64List.fromList([
          for (var t = 0; t < nT; t++) rng.nextBool() ? 1.0 : 0.0,
        ]),
      );
    }(),
];
