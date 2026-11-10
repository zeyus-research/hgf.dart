import 'dart:math' as math;

import 'package:hgf_deep/hgf_deep.dart';
import 'package:test/test.dart';

Matrix _randn(int rows, int cols, math.Random r) => Matrix.fromRows([
  for (var i = 0; i < rows; i++)
    [
      for (var j = 0; j < cols; j++)
        math.sqrt(-2 * math.log(1 - r.nextDouble())) *
            math.cos(2 * math.pi * r.nextDouble()),
    ],
]);

DeepNetwork _net({bool? volatilityParent}) => DeepNetwork()
  ..addLayer(3)
  ..addLayer(4, coupling: CouplingFn.tanh, volatilityParent: volatilityParent)
  ..addLayer(4, coupling: CouplingFn.tanh, volatilityParent: volatilityParent)
  ..addLayer(2)
  ..weightInitialisation(const XavierInit(), seed: 3);

void _expectSameState(DeepNetwork a, DeepNetwork b) {
  for (var l = 0; l < a.nLayers; l++) {
    for (final f in LayerField.values) {
      final x = a.state(l).field(f), y = b.state(l).field(f);
      expect(x == null, y == null, reason: 'layer $l $f presence');
      if (x == null) continue;
      for (var k = 0; k < x.length; k++) {
        // Bit for bit, NaN included.
        expect(
          x[k].isNaN ? 'NaN' : x[k],
          y![k].isNaN ? 'NaN' : y[k],
          reason: 'layer $l $f[$k]',
        );
      }
    }
  }
  for (var l = 1; l < a.nLayers; l++) {
    expect(a.weights[l]!.data, b.weights[l]!.data, reason: 'weights $l');
  }
}

void main() {
  final r = math.Random(5);
  final x = _randn(25, 2, r);
  final y = _randn(25, 3, r);

  test('tracing rows one by one is fit, bit for bit', () {
    final fitted = _net();
    final traced = _net();
    final pred = fitted.fit(x, y, optimiser: Adam(0.01));
    final adam = Adam(0.01);
    for (var t = 0; t < x.rows; t++) {
      final tr = traced.trace(x.row(t), y: y.row(t), optimiser: adam);
      expect(tr.prediction, pred.row(t).toList(), reason: 'prediction $t');
    }
    _expectSameState(fitted, traced);
  });

  test('without updatePrecisions too', () {
    final fitted = _net();
    final traced = _net();
    fitted.fit(x, y, optimiser: const Sgd(0.05), updatePrecisions: false);
    for (var t = 0; t < x.rows; t++) {
      traced.trace(
        x.row(t),
        y: y.row(t),
        optimiser: const Sgd(0.05),
        updatePrecisions: false,
      );
    }
    _expectSameState(fitted, traced);
  });

  test('phases run top-down then bottom-up', () {
    final net = _net();
    final tr = net.trace(x.row(0), y: y.row(0));
    expect(
      [for (final s in tr.steps) '${s.phase.name}:${s.layer}'],
      [
        'clampInput:3',
        'predict:2',
        'predict:1',
        'predict:0',
        'observe:0',
        'leafError:0',
        'posterior:1',
        'layerError:1',
        'posterior:2',
        'layerError:2',
        'weightUpdate:-1',
        'carry:-1',
      ],
    );
    final w = tr.steps.firstWhere((s) => s.phase == TracePhase.weightUpdate);
    expect(w.gradients, hasLength(4));
    expect(w.gradients![0], isNull);
    expect(w.gradients![1]!.rows, 3);
  });

  test('each phase changes only what it should', () {
    final net = _net();
    final tr = net.trace(x.row(0), y: y.row(0));
    final clamp = tr.steps[0].states;
    expect(clamp[3].mean, x.row(0));
    // Layer 1's prediction leaves layer 0 untouched.
    final p1 = tr.steps[2].states, p0 = tr.steps[3].states;
    expect(p1[0].expectedMean, isNot(p0[0].expectedMean));
    final obs = tr.steps[4].states;
    expect(obs[0].mean, y.row(0));
  });

  test(
    'a forward-only trace stops after the predictions and learns nothing',
    () {
      final net = _net();
      final before = net.weights[1]!.copy();
      final tr = net.trace(x.row(0));
      expect(tr.steps.last.phase, TracePhase.predict);
      expect(tr.steps.last.layer, 0);
      expect(tr.prediction, net.predict(Matrix.row(x.row(0))).row(0).toList());
      expect(net.weights[1]!.data, before.data);
    },
  );

  test('no optimiser: gradients recorded, weights unchanged', () {
    final net = _net();
    final before = [for (var l = 1; l < 4; l++) net.weights[l]!.copy()];
    final tr = net.trace(x.row(0), y: y.row(0));
    final g = tr.steps.firstWhere((s) => s.phase == TracePhase.weightUpdate);
    expect(g.gradients![1]!.data.any((v) => v != 0), isTrue);
    for (var l = 1; l < 4; l++) {
      expect(net.weights[l]!.data, before[l - 1].data);
    }
    expect(tr.weightsAt(tr.steps.length - 1)[1]!.data, before[0].data);
  });

  test('snapshot and restore round-trip', () {
    final net = _net();
    final snap = net.snapshot();
    final reference = _net();
    net.fit(x, y, optimiser: Adam(0.05));
    net.restore(snap);
    _expectSameState(net, reference);
  });

  test('Adam.copy continues independently, state included', () {
    final a = Adam(0.01);
    _net().fit(x, y, optimiser: a);
    final b = a.copy();
    final n1 = _net(), n2 = _net();
    n1.fit(x, y, optimiser: a);
    n2.fit(x, y, optimiser: b);
    expect(a.count, 50);
    expect(b.count, 50);
    // Same moments and step count: the same updates.
    expect(n1.weights[1]!.data, n2.weights[1]!.data);
    expect(n1.weights[3]!.data, n2.weights[3]!.data);
  });

  test('layers without a volatility parent have no volatility fields', () {
    final net = _net(volatilityParent: false);
    final tr = net.trace(x.row(0), y: y.row(0));
    expect(tr.steps.last.states[1].meanVol, isNull);
    expect(tr.steps.last.states[0].meanVol, isNotNull);
  });
}
