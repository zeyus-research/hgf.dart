// Reads data/binary_input.txt with dart:io.
@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

Float64List _binaryInput() => Float64List.fromList([
  for (final l in File('data/binary_input.txt').readAsLinesSync())
    if (l.trim().isNotEmpty) double.parse(l),
]);

HgfModel _binaryModel() => HgfModel(
  network: Network()
    ..addNodes(kind: NodeKind.binaryState)
    ..addNodes(valueChildren: [0]),
  response: const BinarySoftmaxReadout(),
  parameters: [
    Parameter.node(
      'omega2',
      1,
      NodeField.tonicVolatility,
      prior: TruncatedNormal(-3, 5, lower: -8, upper: 2),
    ),
    Parameter.response(
      'inverseTemperature',
      prior: const Uniform(0.2, 20),
      initial: 1.0,
    ),
  ],
);

/// A correlated Gaussian with sds [s1], [s2] and correlation [r].
(double Function(Float64List), Float64List Function(Float64List)) _gaussian(
  double s1,
  double s2,
  double r,
) {
  final det = 1 - r * r;
  return (
    (y) {
      final a = y[0] / s1, b = y[1] / s2;
      return (a * a - 2 * r * a * b + b * b) / (2 * det);
    },
    (y) {
      final a = y[0] / s1, b = y[1] / s2;
      return Float64List.fromList([
        (a - r * b) / det / s1,
        (b - r * a) / det / s2,
      ]);
    },
  );
}

double _correlation(PosteriorDraws d) {
  final a = d.parameter(0), b = d.parameter(1);
  final ma = d.mean(0), mb = d.mean(1);
  var s = 0.0, n = 0;
  for (var c = 0; c < a.length; c++) {
    for (var i = 0; i < a[c].length; i++) {
      s += (a[c][i] - ma) * (b[c][i] - mb);
      n++;
    }
  }
  return s / (n - 1) / d.sd(0) / d.sd(1);
}

void main() {
  group('diagnostics', () {
    // 3 chains x 101 draws of a deterministic, autocorrelated series; the
    // second variant is rounded to one decimal (ties in the ranks).
    List<Float64List> series({required bool rounded}) => [
      for (var c = 0; c < 3; c++)
        Float64List.fromList([
          for (var t = 0; t < 101; t++)
            () {
              final v =
                  math.sin(1.3 * t + c) +
                  0.5 * math.sin(0.07 * t * t + 2 * c) +
                  0.1 * c +
                  0.002 * t;
              return rounded ? (v * 10).roundToDouble() / 10 : v;
            }(),
        ]),
    ];
    // From arviz_stats 1.3 (rhat "rank", ess "bulk"/"tail"/"mean", mcse
    // "mean") on the same arrays.
    const expected = {
      false: [
        1.0032703580749973,
        209.11173973839115,
        349.2547968538651,
        208.21638708714735,
        0.0536646961695596,
      ],
      true: [
        1.0036474042870023,
        209.0072763770707,
        344.247288796699,
        208.12713105190605,
        0.05407727969594233,
      ],
    };
    for (final rounded in [false, true]) {
      test('match ArviZ${rounded ? ' with ties' : ''}', () {
        final x = series(rounded: rounded);
        final got = [rhat(x), essBulk(x), essTail(x), essMean(x), mcseMean(x)];
        for (var i = 0; i < got.length; i++) {
          expect(got[i], closeTo(expected[rounded]![i], 1e-9 * got[i]));
        }
      });
    }
    test('constant draws and too few chains', () {
      final flat = [Float64List(10)..fillRange(0, 10, 2), Float64List(10)];
      flat[1].fillRange(0, 10, 2);
      expect(essBulk(flat), 20);
      expect(
        rhat([Float64List.fromList(List.generate(10, (i) => i * 1.0))]),
        isNaN,
      );
    });
    test('quantileOf interpolates like NumPy', () {
      final s = Float64List.fromList([1, 2, 3, 4]);
      expect(quantileOf(s, 0.5), 2.5);
      expect(quantileOf(s, 0.05), closeTo(1.15, 1e-15));
      expect(quantileOf(s, 1), 4);
    });
  });

  group('sampleFunction', () {
    for (final metric in MetricKind.values) {
      test('correlated Gaussian, $metric metric', () {
        final (u, g) = _gaussian(1, 10, 0.9);
        final d = sampleFunction(
          u,
          g,
          Float64List(2),
          sampler: Nuts(metric: metric),
          seed: 1,
        );
        expect(d.nChains, 4);
        expect(d.nDraws, 1000);
        expect(d.divergences, 0);
        final s = d.summary();
        for (var i = 0; i < 2; i++) {
          expect(s[i].rhat, lessThan(1.01));
          expect(s[i].essBulk, greaterThan(400));
          expect(s[i].mean.abs(), lessThan(5 * s[i].mcseMean));
        }
        expect(s[0].sd, closeTo(1, 0.08));
        expect(s[1].sd, closeTo(10, 0.8));
        expect(_correlation(d), closeTo(0.9, 0.03));
        for (final e in d.eBfmi) {
          expect(e, greaterThan(0.3));
        }
        // Adaptation: the metric approaches the covariance.
        final inv = d.chains.first.inverseMetric;
        if (metric == MetricKind.diagonal) {
          expect(inv[1] / inv[0], greaterThan(30)); // truth 100
        } else if (metric == MetricKind.dense) {
          expect(inv[1] / math.sqrt(inv[0] * inv[3]), closeTo(0.9, 0.08));
        }
      });
    }

    test('the dense metric needs fewer steps on correlated targets', () {
      final (u, g) = _gaussian(1, 10, 0.95);
      int steps(MetricKind m) => sampleFunction(
        u,
        g,
        Float64List(2),
        sampler: Nuts(metric: m),
        draws: 300,
        seed: 2,
      ).leapfrogSteps;
      expect(steps(MetricKind.dense), lessThan(steps(MetricKind.diagonal)));
    });

    test('variance is unbiased over long runs', () {
      final d = sampleFunction(
        (y) => y[0] * y[0] / 2,
        (y) => Float64List.fromList([y[0]]),
        Float64List(1),
        draws: 10000,
        seed: 5,
      );
      final sq = [
        for (final c in d.parameter(0))
          Float64List.fromList([for (final v in c) v * v]),
      ];
      final m =
          sq.fold(0.0, (s, c) => s + c.fold(0.0, (a, v) => a + v)) / 40000;
      expect((m - 1).abs(), lessThan(4 * mcseMean(sq)));
    });

    test('a cliff in the density is reported as divergences', () {
      // A standard normal with a wall of +1e4 beyond y = 1.5.
      final d = sampleFunction(
        (y) => y[0] * y[0] / 2 + (y[0] > 1.5 ? 1e4 : 0),
        (y) => Float64List.fromList([y[0]]),
        Float64List(1),
        warmup: 200,
        draws: 200,
        seed: 3,
      );
      expect(d.divergences, greaterThan(0));
      for (final c in d.parameter(0)) {
        for (final v in c) {
          expect(v, lessThanOrEqualTo(1.5));
        }
      }
    });

    test('chains without a finite start fail alone', () {
      final d = sampleFunction(
        (y) => y[0] < 0 ? double.infinity : y[0] * y[0] / 2,
        (y) => Float64List.fromList([y[0]]),
        Float64List.fromList([-5]),
        sampler: const Nuts(maxInitAttempts: 3),
        warmup: 10,
        draws: 10,
      );
      expect(d.nChains, 0);
      expect(d.failedChains, hasLength(4));
      expect(d.failedChains.first, contains('no finite starting point'));
    });

    test('reproducible from the seed', () {
      final (u, g) = _gaussian(1, 2, 0.5);
      Float64List run(int seed) => sampleFunction(
        u,
        g,
        Float64List(2),
        warmup: 50,
        draws: 20,
        seed: seed,
      ).chains[1].y;
      expect(run(4), run(4));
      expect(run(4), isNot(run(5)));
    });
  });

  group('sampleNuts', () {
    final u = _binaryInput();
    final model = _binaryModel();
    Subject simulated(List<double> x, int seed) => Subject(
      inputs: u,
      responses: model.simulate(
        [Subject(inputs: u)],
        [x],
        math.Random(seed),
      )[0],
      id: 'sim$seed',
    );

    test('matches grid quadrature of the posterior', () {
      final s = simulated([-3.5, 3.0], 123);
      // Midpoint grid over the unconstrained plane (Jacobian included).
      const m = 301, lo = -8.0, hi = 8.0, h = (hi - lo) / (m - 1);
      final ys = Float64List(m * m * 2);
      for (var a = 0; a < m; a++) {
        for (var b = 0; b < m; b++) {
          ys[2 * (a * m + b)] = lo + a * h;
          ys[2 * (a * m + b) + 1] = lo + b * h;
        }
      }
      final f = Float64List(m * m);
      model.negLogPosterior([s], Int32List(m * m), ys, f, jacobian: true);
      final fMin = f.reduce(math.min);
      var z = 0.0;
      final s1 = [0.0, 0.0], s2 = [0.0, 0.0];
      for (var e = 0; e < m * m; e++) {
        final w = math.exp(fMin - f[e]);
        final x = model.toConstrained([ys[2 * e], ys[2 * e + 1]]);
        z += w;
        for (var i = 0; i < 2; i++) {
          s1[i] += w * x[i];
          s2[i] += w * x[i] * x[i];
        }
      }

      final fit = sampleNuts(
        model,
        [s],
        chains: 2,
        warmup: 500,
        draws: 500,
        seed: 1,
      );
      expect(fit.results, hasLength(1));
      final r = fit.results[0];
      expect(r.subject, same(s));
      expect(r.names, ['omega2', 'inverseTemperature']);
      expect(r.divergences, 0);
      for (final p in r.summary()) {
        final i = r.names.indexOf(p.name);
        final mean = s1[i] / z;
        final sd = math.sqrt(s2[i] / z - mean * mean);
        expect(p.mean, closeTo(mean, 4 * p.mcseMean), reason: p.name);
        expect(p.sd / sd, closeTo(1, 0.15), reason: p.name);
        expect(p.rhat, lessThan(1.02));
      }
      // Each pass is one leapfrog step of every running chain, with a
      // forward-difference gradient: k + 1 = 3 filters per chain.
      expect(fit.elements % 3, 0);
      expect(fit.elements, lessThanOrEqualTo(fit.passes * 2 * 3));
      final leapfrog = r.chains.fold(
        0,
        (a, c) => a + c.leapfrogSteps.fold(0, (b, v) => b + v),
      );
      expect(fit.elements, greaterThan(leapfrog * 3));
    });

    test('subjects sampled together give the same draws as alone', () {
      final a = simulated([-3.5, 3.0], 1), b = simulated([-2.0, 5.0], 2);
      PosteriorDraws run(List<Subject> s) => sampleNuts(
        model,
        s,
        chains: 2,
        warmup: 40,
        draws: 15,
        seed: 9,
      ).results[0];
      final alone = run([a]), together = run([a, b]);
      for (var c = 0; c < 2; c++) {
        expect(together.chains[c].y, alone.chains[c].y);
        expect(together.chains[c].stepSize, alone.chains[c].stepSize);
      }
    });

    test('exports for ArviZ and as CSV', () {
      final subjects = [
        simulated([-3.5, 3.0], 1),
        simulated([-2.0, 5.0], 2),
      ];
      final fit = sampleNuts(
        model,
        subjects,
        chains: 2,
        warmup: 30,
        draws: 7,
        start: [
          [-3.5, 3.0],
          [-2.0, 5.0],
        ],
      );
      final j = jsonDecode(fit.toJson()) as Map<String, dynamic>;
      final omega = (j['posterior'] as Map)['omega2'] as List;
      expect(omega, hasLength(2)); // chains
      expect(omega[0] as List, hasLength(7)); // draws
      expect((omega[0] as List)[3] as List, hasLength(2)); // subjects
      expect(((omega[1] as List)[3] as List)[1], fit.results[1].draw(1, 3)[0]);
      final stats = j['sample_stats'] as Map;
      expect(stats.keys, containsAll(['lp', 'diverging', 'tree_depth']));
      expect((j['dims'] as Map)['omega2'], ['subject']);
      expect((j['coords'] as Map)['subject'], ['sim1', 'sim2']);

      final csv = fit.results[0].toCsv().trim().split('\n');
      expect(csv, hasLength(1 + 2 * 7));
      expect(csv.first, startsWith('chain,draw,lp__,'));
      expect(csv.first, endsWith(',omega2,inverseTemperature'));
      final last = csv.last.split(',');
      expect(last[0], '1');
      expect(last[1], '6');
      expect(double.parse(last.last), fit.results[0].draw(1, 6)[1]);
    });
  });
}
