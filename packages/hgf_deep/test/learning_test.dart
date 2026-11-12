import 'dart:convert';
import 'dart:math' as math;

import 'package:hgf_deep/hgf_deep.dart';
import 'package:test/test.dart';

/// Random data with [rows] samples.
(Matrix, Matrix) data(int rows, int nx, int ny, int seed) {
  final rng = math.Random(seed);
  double g() => rng.nextDouble() * 2 - 1;
  return (
    Matrix.fromRows([
      for (var i = 0; i < rows; i++) [for (var j = 0; j < nx; j++) g()],
    ]),
    Matrix.fromRows([
      for (var i = 0; i < rows; i++) [for (var j = 0; j < ny; j++) g()],
    ]),
  );
}

DeepNetwork net({bool updateInputLayer = false}) =>
    DeepNetwork(coupling: CouplingFn.tanh, updateInputLayer: updateInputLayer)
      ..addLayer(2)
      ..addLayer(4)
      ..addLayer(3)
      ..weightInitialisation(const XavierInit(), seed: 3);

void expectClose(List<double> a, List<double> b, double tol) {
  expect(a.length, b.length);
  for (var i = 0; i < a.length; i++) {
    expect(
      (a[i] - b[i]).abs(),
      lessThanOrEqualTo(tol * math.max(1, a[i].abs())),
      reason: '[$i] ${a[i]} vs ${b[i]}',
    );
  }
}

void main() {
  final (x, y) = data(20, 3, 2, 1);
  final su = SynapticUncertainty(window: 10, priorVariance: 0.1);

  group('synapticUncertainty: arguments', () {
    test('needs its settings, and only with its kind', () {
      expect(
        () => net().fit(x, y, learningKind: LearningKind.synapticUncertainty),
        throwsArgumentError,
      );
      expect(
        () => net().fit(x, y, optimiser: Sgd(0.1), synapticUncertainty: su),
        throwsArgumentError,
      );
      expect(
        () => net().fit(
          x,
          y,
          optimiser: Sgd(0.1),
          learningKind: LearningKind.synapticUncertainty,
          synapticUncertainty: su,
        ),
        throwsArgumentError,
      );
    });

    test('is not available with mean-field updates', () {
      final n = DeepNetwork(meanFieldUpdates: true)
        ..addLayer(2)
        ..addLayer(3);
      expect(
        () => n.fit(
          x,
          y,
          learningKind: LearningKind.synapticUncertainty,
          synapticUncertainty: su,
        ),
        throwsArgumentError,
      );
    });

    test('validates the settings', () {
      expect(() => SynapticUncertainty(window: 0.5), throwsArgumentError);
      expect(
        () => SynapticUncertainty(window: 2, priorVariance: 0),
        throwsArgumentError,
      );
      expect(
        () => SynapticUncertainty(window: 2, learningRate: -1),
        throwsArgumentError,
      );
      expect(
        () => SynapticUncertainty(window: 2, incrementScale: 0),
        throwsArgumentError,
      );
    });
  });

  group('installWeightBelief', () {
    test('installs zeros where asked, and keeps an existing belief', () {
      final n = net();
      expect(n.weightPrecisionDeltas.every((d) => d == null), isTrue);
      n.installWeightBelief(layers: [2]);
      expect(n.weightPrecisionDeltas[1], isNull);
      expect(n.weightPrecisionDeltas[2]!.data.every((v) => v == 0), isTrue);
      n.weightPrecisionDeltas[2]!.data[0] = 5;
      n.installWeightBelief();
      expect(n.weightPrecisionDeltas[1], isNotNull);
      expect(n.weightPrecisionDeltas[2]!.data[0], 5);
      expect(() => n.installWeightBelief(layers: [0]), throwsArgumentError);
    });

    test('learning installs it and accumulates curvature', () {
      final n = net()
        ..fit(
          x,
          y,
          learningKind: LearningKind.synapticUncertainty,
          synapticUncertainty: su,
        );
      for (final d in n.weightPrecisionDeltas.skip(1)) {
        expect(d!.data.any((v) => v > 0), isTrue);
        expect(d.data.every((v) => v >= 0 && v.isFinite), isTrue);
      }
    });
  });

  test('without accumulated curvature the rule is gradient descent at '
      'rate learningRate·priorVariance', () {
    // Two layers: the only child is the clamped bottom layer, which has no
    // process noise, so the gradient is the precision-weighted one. A huge
    // window removes the reversion.
    DeepNetwork two() => DeepNetwork()
      ..addLayer(2)
      ..addLayer(3)
      ..weightInitialisation(const XavierInit(), seed: 5);
    final (x1, y1) = data(1, 3, 2, 9);
    final a = two()
      ..fit(
        x1,
        y1,
        learningKind: LearningKind.synapticUncertainty,
        synapticUncertainty: SynapticUncertainty(
          window: 1e15,
          priorVariance: 0.2,
          learningRate: 0.5,
        ),
      );
    final b = two()..fit(x1, y1, optimiser: const Sgd(0.1));
    expectClose(a.weights[1]!.data, b.weights[1]!.data, 1e-12);
  });

  group('sampleWeight', () {
    test('uniform weights give the plain mean', () {
      final a = net(), b = net();
      final ea = a.batchUpdate(x, y, optimiser: Adam(0.01));
      final eb = b.batchUpdate(
        x,
        y,
        optimiser: Adam(0.01),
        sampleWeight: List.filled(x.rows, 0.7),
      );
      expect(ea.data, eb.data);
      expectClose(a.weights[2]!.data, b.weights[2]!.data, 1e-12);
      expectClose(a.state(1).precision, b.state(1).precision, 1e-12);
      expectClose(a.state(1).meanVol!, b.state(1).meanVol!, 1e-12);
    });

    test('a weight of 2 counts a row twice', () {
      final (x3, y3) = data(3, 3, 2, 4);
      final dup = Matrix.fromRows([
        for (final r in [0, 0, 1, 2]) [for (var j = 0; j < 3; j++) x3.at(r, j)],
      ]);
      final dupY = Matrix.fromRows([
        for (final r in [0, 0, 1, 2]) [for (var j = 0; j < 2; j++) y3.at(r, j)],
      ]);
      final a = net()..batchUpdate(dup, dupY, optimiser: const Sgd(0.1));
      final b = net()
        ..batchUpdate(
          x3,
          y3,
          optimiser: const Sgd(0.1),
          sampleWeight: [2, 1, 1],
        );
      expectClose(a.weights[2]!.data, b.weights[2]!.data, 1e-12);
      expectClose(a.state(1).precision, b.state(1).precision, 1e-12);
    });

    test('masked rows change nothing', () {
      final (x3, y3) = data(3, 3, 2, 4);
      final first = Matrix.fromRows([
        [for (var j = 0; j < 3; j++) x3.at(0, j)],
      ]);
      final firstY = Matrix.fromRows([
        [for (var j = 0; j < 2; j++) y3.at(0, j)],
      ]);
      final a = net()..batchUpdate(first, firstY, optimiser: const Sgd(0.1));
      final b = net()
        ..batchUpdate(
          x3,
          y3,
          optimiser: const Sgd(0.1),
          sampleWeight: [1, 0, 0],
        );
      expectClose(a.weights[1]!.data, b.weights[1]!.data, 1e-12);
      expectClose(a.state(2).precision, b.state(2).precision, 1e-12);
    });

    test('must match the batch', () {
      expect(
        () => net().batchUpdate(x, y, sampleWeight: [1, 2]),
        throwsArgumentError,
      );
    });
  });

  group('updateInputLayer', () {
    test('moves the top precision, leaves its mean on the predictors', () {
      final a = net(updateInputLayer: true)..fit(x, y, optimiser: Adam(0.01));
      final top = a.state(2);
      expect(top.mean, [for (var j = 0; j < 3; j++) x.at(x.rows - 1, j)]);
      expect(top.precision.any((p) => p != 1.0), isTrue);
      expect(top.valuePredictionError.every((v) => v == 0), isTrue);
    });

    test('a binary top layer is left out (bit for bit)', () {
      DeepNetwork b(bool on) =>
          DeepNetwork(coupling: CouplingFn.tanh, updateInputLayer: on)
            ..addLayer(2)
            ..addLayer(4)
            ..addLayer(3, kind: LayerKind.binary)
            ..weightInitialisation(const XavierInit(), seed: 3);
      final (xb, yb) = data(10, 3, 2, 2);
      final on = b(true)..fit(xb, yb, optimiser: Adam(0.01));
      final off = b(false)..fit(xb, yb, optimiser: Adam(0.01));
      expect(on.weights[2]!.data, off.weights[2]!.data);
    });
  });

  test('toJson keeps the weight belief and updateInputLayer', () {
    final a = net(updateInputLayer: true)
      ..fit(
        x,
        y,
        learningKind: LearningKind.synapticUncertainty,
        synapticUncertainty: su,
      );
    final b = DeepNetwork.fromJson(
      jsonDecode(jsonEncode(a.toJson())) as Map<String, Object?>,
    );
    expect(b.settings.updateInputLayer, isTrue);
    expect(b.weightPrecisionDeltas[2]!.data, a.weightPrecisionDeltas[2]!.data);
    final (x2, y2) = data(8, 3, 2, 7);
    for (final n in [a, b]) {
      n.fit(
        x2,
        y2,
        learningKind: LearningKind.synapticUncertainty,
        synapticUncertainty: su,
      );
    }
    expect(b.weights[1]!.data, a.weights[1]!.data);
    expect(b.weightPrecisionDeltas[1]!.data, a.weightPrecisionDeltas[1]!.data);
  });

  test('snapshot and restore cover the weight belief', () {
    final a = net()..installWeightBelief();
    final snap = a.snapshot();
    a.fit(
      x,
      y,
      learningKind: LearningKind.synapticUncertainty,
      synapticUncertainty: su,
    );
    a.restore(snap);
    expect(a.weightPrecisionDeltas[1]!.data.every((v) => v == 0), isTrue);
  });

  test('checkGradientHealth leaves the state alone', () {
    final a = net()..fit(x, y, optimiser: Adam(0.01));
    final before = jsonEncode(a.toJson());
    final h = a.checkGradientHealth([0.1, 0.2, 0.3], [0.5, -0.5]);
    expect(jsonEncode(a.toJson()), before);
    expect(h.magnitudes.first, isNull);
    expect(h.magnitudes.skip(1).every((m) => m! > 0), isTrue);
    expect(h.dead, isEmpty);
    expect(h.partiallyDead, isFalse);
  });
}
