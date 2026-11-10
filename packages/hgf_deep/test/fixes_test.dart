/// The corrections of [DeepFixes] on the setting of upstream P18
/// (docs/upstream/README.md#p18): a 2-4-3 tanh network, N(0, 0.5²) weights
/// held fixed, N(0, 1) predictors and targets.
library;

import 'dart:math' as math;

import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';
import 'package:test/test.dart';

double _gauss(math.Random r) =>
    math.sqrt(-2 * math.log(1 - r.nextDouble())) *
    math.cos(2 * math.pi * r.nextDouble());

/// The first sample at which the hidden layer's mean is not finite, or −1.
int _firstNaN(DeepVolatilityUpdate update, DeepFixes fixes, int seed) {
  final r = math.Random(seed);
  final net = DeepNetwork(volatilityUpdates: update, fixes: fixes)
    ..addLayer(3)
    ..addLayer(4, coupling: CouplingFn.tanh)
    ..addLayer(2);
  Matrix randn(int rows, int cols, double sd) => Matrix.fromRows([
    for (var i = 0; i < rows; i++)
      [for (var j = 0; j < cols; j++) sd * _gauss(r)],
  ]);
  net.setWeights([randn(3, 5, 0.5), randn(4, 3, 0.5)]);
  const n = 500;
  net.fit(
    randn(n, 2, 1),
    randn(n, 3, 1),
    weightUpdate: false,
    record: {LayerField.mean},
  );
  final m = net.trajectories!.of(LayerField.mean, 1)!;
  for (var t = 0; t < n; t++) {
    for (var k = 0; k < 4; k++) {
      if (!m[t * 4 + k].isFinite) return t;
    }
  }
  return -1;
}

void main() {
  test('networks use the recommended fixes unless told otherwise', () {
    expect(DeepNetwork().settings.fixes, DeepFixes.recommended);
    expect(DeepNetwork(fixes: DeepFixes.none).settings.fixes.isNone, isTrue);
  });

  for (final update in DeepVolatilityUpdate.values) {
    test(
      'P18 (${update.name}): pyhgf runs away, the curvature floor does not',
      () {
        for (var seed = 0; seed < 5; seed++) {
          expect(
            _firstNaN(update, DeepFixes.none, seed),
            greaterThanOrEqualTo(0),
            reason: 'upstream behaviour, seed $seed',
          );
          expect(
            _firstNaN(update, const DeepFixes(curvatureFloor: true), seed),
            -1,
            reason: 'curvature floor, seed $seed',
          );
          expect(
            _firstNaN(update, DeepFixes.recommended, seed),
            -1,
            reason: 'all fixes, seed $seed',
          );
        }
      },
    );
  }

  test('logSpaceVolatility: an overflowing volatility level stays finite', () {
    // eHGF with a volatility level already far beyond exp's range: the
    // re-prediction is ∞ in plain arithmetic (∞·0 = NaN), its limit with
    // the log-space terms.
    DeepNetwork build(DeepFixes fixes) =>
        DeepNetwork(volatilityUpdates: DeepVolatilityUpdate.ehgf, fixes: fixes)
          ..addLayer(1)
          ..addLayer(
            2,
            coupling: CouplingFn.tanh,
            initial: {LayerField.meanVol: 800.0},
          )
          ..addLayer(1)
          ..weightInitialisation(const XavierInit(), seed: 1);
    final x = Matrix.column([0.3]), y = Matrix.column([1.5]);
    final plain = build(DeepFixes.none)..fit(x, y, weightUpdate: false);
    final fixed = build(DeepFixes.recommended)..fit(x, y, weightUpdate: false);
    expect(plain.state(1).effectivePrecision.any((v) => v.isNaN), isTrue);
    final s = fixed.state(1);
    expect(s.effectivePrecision, everyElement(1.0));
    expect(s.meanVol!.every((v) => v.isFinite), isTrue);
    expect(s.precisionVol!.every((v) => v.isFinite), isTrue);
  });

  test('a network learns moons (first accuracy check)', () {
    final (:train, :test) = split(moons(500, noise: 0.12, seed: 2), seed: 1);
    final net = DeepNetwork()
      ..addLayer(1, kind: LayerKind.binary)
      ..addLayer(12, coupling: CouplingFn.tanh, volatilityParent: false)
      ..addLayer(12, coupling: CouplingFn.tanh, volatilityParent: false)
      ..addLayer(2)
      ..weightInitialisation(const XavierInit(), seed: 1);
    final adam = Adam(0.01);
    final before = accuracy(net.predict(test.x), test.labels!);
    for (var e = 0; e < 15; e++) {
      final b = shuffled(train, seed: e);
      net.fit(b.x, b.y, optimiser: adam);
    }
    final after = accuracy(net.predict(test.x), test.labels!);
    expect(after, greaterThan(0.95), reason: 'from $before');
  });

  test('a network with volatility levels learns spirals', () {
    final (:train, :test) = split(spirals(600, seed: 3), seed: 1);
    final net = DeepNetwork()
      ..addLayer(3, kind: LayerKind.categorical)
      ..addLayer(24, coupling: CouplingFn.tanh)
      ..addLayer(24, coupling: CouplingFn.tanh)
      ..addLayer(2)
      ..weightInitialisation(const XavierInit(), seed: 1);
    final adam = Adam(0.01);
    for (var e = 0; e < 40; e++) {
      final b = shuffled(train, seed: e);
      net.fit(b.x, b.y, optimiser: adam);
    }
    expect(accuracy(net.predict(test.x), test.labels!), greaterThan(0.9));
  });
}
