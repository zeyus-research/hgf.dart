import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_fit/src/sample/sample.dart' show chainSeed;
import 'package:test/test.dart';

typedef _Target = (
  double Function(Float64List),
  Float64List Function(Float64List),
);

/// One chain of [Nuts.chain] (the generator), answered point by point as
/// `sampleFunction` answers them.
ChainOutput _reference(
  Nuts nuts,
  _Target target,
  Float64List y0,
  int chain, {
  required int warmup,
  required int draws,
  required int seed,
}) {
  final (u, g) = target;
  final rng = math.Random(chainSeed(seed, 0, chain));
  final out = ChainOutput();
  for (final req in nuts.chain(
    init: (_) => Float64List.fromList([
      for (final v in y0) v + (2 * rng.nextDouble() - 1),
    ]),
    warmup: warmup,
    draws: draws,
    random: rng,
    out: out,
  )) {
    final p = req.points.single;
    final v = u(p);
    req
      ..values = Float64List.fromList([v.isNaN ? double.infinity : v])
      ..gradients = [g(p)];
  }
  return out;
}

void _expectSame(ChainOutput a, ChainOutput b) {
  expect(a.error, b.error);
  expect(a.y, b.y);
  expect(a.logDensity, b.logDensity);
  expect(a.acceptStat, b.acceptStat);
  expect(a.energy, b.energy);
  expect(a.treeDepth, b.treeDepth);
  expect(a.leapfrogSteps, b.leapfrogSteps);
  expect(a.divergent, b.divergent);
  expect(a.warmupDivergences, b.warmupDivergences);
  expect(a.iterations, b.iterations);
  if (a.error == null) {
    expect(a.stepSize, b.stepSize);
    expect(a.inverseMetric, b.inverseMetric);
  }
}

void main() {
  // A correlated Gaussian with a wall beyond y₀ = 1.2 (divergences) and a
  // region of zero density below y₀ = −1.4 (failed starts at y0 = −1).
  final _Target target = (
    (y) {
      if (y[0] < -1.4) return double.infinity;
      final a = y[0], c = y[1] / 3;
      return (a * a - 1.6 * a * c + c * c) / 0.72 + (a > 1.2 ? 2000 : 0);
    },
    (y) {
      final a = y[0], c = y[1] / 3;
      return Float64List.fromList([
        (2 * a - 1.6 * c) / 0.72,
        (2 * c - 1.6 * a) / 0.72 / 3,
      ]);
    },
  );
  final cases = <String, (Nuts, int, int, Float64List)>{
    for (final m in MetricKind.values)
      '$m': (Nuts(metric: m), 300, 100, Float64List(2)),
    'shallow trees, dense': (
      const Nuts(metric: MetricKind.dense, maxTreeDepth: 2),
      150,
      50,
      Float64List(2),
    ),
    'no trees': (const Nuts(maxTreeDepth: 0), 10, 10, Float64List(2)),
    'short warmup': (const Nuts(), 12, 30, Float64List(2)),
    'no warmup, no draws': (const Nuts(), 0, 0, Float64List(2)),
    'failed starts': (
      const Nuts(maxInitAttempts: 2),
      40,
      20,
      Float64List.fromList([-1, 0]),
    ),
  };
  for (final MapEntry(key: name, value: (nuts, warmup, draws, y0))
      in cases.entries) {
    test('the ensemble reproduces Nuts.chain: $name', () {
      const chains = 6, seed = 11;
      final d = sampleFunction(
        target.$1,
        target.$2,
        y0,
        sampler: nuts,
        chains: chains,
        warmup: warmup,
        draws: draws,
        seed: seed,
      );
      final got = [...d.chains];
      expect(got.length + d.failedChains.length, chains);
      var ok = 0;
      for (var c = 0; c < chains; c++) {
        final ref = _reference(
          nuts,
          target,
          y0,
          c,
          warmup: warmup,
          draws: draws,
          seed: seed,
        );
        if (ref.error != null) continue;
        _expectSame(got[ok++], ref);
      }
      expect(ok, got.length);
      if (name == 'failed starts') {
        expect(d.failedChains, isNotEmpty);
        expect(d.chains, isNotEmpty);
      }
      if (warmup >= 300) {
        final divergent = d.chains.fold(
          0,
          (a, c) =>
              a + c.warmupDivergences + c.divergent.fold(0, (x, v) => x + v),
        );
        expect(divergent, greaterThan(0));
      }
      if (name == 'shallow trees, dense') {
        expect(d.maxDepthHits(2), greaterThan(0));
      }
    });
  }

  test('ensembles answer requests in flat arrays', () {
    final e = NutsEnsemble(const Nuts(), 2, warmup: 5, draws: 5);
    final outs = [ChainOutput(), ChainOutput(), ChainOutput()];
    for (var c = 0; c < 3; c++) {
      e.add(
        init: (_) => Float64List.fromList([c * 0.1, -c * 0.1]),
        random: math.Random(c),
        out: outs[c],
        subject: 2 - c,
      );
    }
    var rounds = 0;
    for (var n = e.request(); n > 0; n = e.request()) {
      expect(e.points.length, 2 * n);
      if (rounds++ == 0) {
        expect(e.pendingSubjects, [2, 1, 0]);
        expect(e.points, [0, 0, 0.1, -0.1, 0.2, -0.2]);
      }
      final v = Float64List(n);
      for (var j = 0; j < n; j++) {
        final a = e.points[2 * j], b = e.points[2 * j + 1];
        v[j] = (a * a + b * b) / 2;
      }
      e.answer(v, Float64List.fromList(e.points));
    }
    expect(rounds, greaterThan(10));
    for (final o in outs) {
      expect(o.iterations, 10);
      expect(o.y, hasLength(10));
    }
    expect(
      () => e.add(
        init: (_) => Float64List(2),
        random: math.Random(),
        out: ChainOutput(),
      ),
      throwsStateError,
    );
  });
}
