// Reads data/binary_input.txt with dart:io.
@TestOn('vm')
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_fit/src/linalg.dart';
import 'package:test/test.dart';

Float64List _binaryInput() => Float64List.fromList([
  for (final l in File('data/binary_input.txt').readAsLinesSync())
    if (l.trim().isNotEmpty) double.parse(l),
]);

Network _binary2({double? omega}) {
  final net = Network()
    ..addNodes(kind: NodeKind.binaryState)
    ..addNodes(valueChildren: [0]);
  if (omega != null) net.setInitial(1, NodeField.tonicVolatility, omega);
  return net;
}

final _omega = Parameter.node(
  'omega2',
  1,
  NodeField.tonicVolatility,
  prior: TruncatedNormal(-3, 5, lower: -8, upper: 2),
);
final _beta = Parameter.response(
  'inverseTemperature',
  prior: const Uniform(0.2, 20),
  initial: 1.0,
);

HgfModel _binaryModel() => HgfModel(
  network: _binary2(),
  response: const BinarySoftmaxReadout(),
  parameters: [_omega, _beta],
);

/// The exact log evidence and natural-scale posterior means and sds of a
/// two-parameter model, by a midpoint grid of [m]² points over `±w`
/// unconstrained sds around the Laplace mode (the density on the
/// unconstrained scale includes the Jacobian).
({double logZ, List<double> mean, List<double> sd}) _quadrature(
  HgfModel model,
  Subject subject,
  LaplaceResult r, {
  int m = 151,
  double w = 10,
}) {
  final sy = r.unconstrainedStandardErrors;
  final h0 = 2 * w * sy[0] / (m - 1), h1 = 2 * w * sy[1] / (m - 1);
  final ys = Float64List(m * m * 2);
  for (var a = 0; a < m; a++) {
    for (var b = 0; b < m; b++) {
      final e = a * m + b;
      ys[2 * e] = r.map.y[0] - w * sy[0] + a * h0;
      ys[2 * e + 1] = r.map.y[1] - w * sy[1] + b * h1;
    }
  }
  final f = Float64List(m * m);
  model.negLogPosterior([subject], Int32List(m * m), ys, f, jacobian: true);
  final fMin = f.reduce(math.min);
  var z = 0.0;
  final s1 = [0.0, 0.0], s2 = [0.0, 0.0];
  for (var e = 0; e < m * m; e++) {
    final wt = math.exp(fMin - f[e]);
    final x = model.toConstrained([ys[2 * e], ys[2 * e + 1]]);
    z += wt;
    for (var i = 0; i < 2; i++) {
      s1[i] += wt * x[i];
      s2[i] += wt * x[i] * x[i];
    }
  }
  final mean = [for (var i = 0; i < 2; i++) s1[i] / z];
  return (
    logZ: -fMin + math.log(z * h0 * h1),
    mean: mean,
    sd: [for (var i = 0; i < 2; i++) math.sqrt(s2[i] / z - mean[i] * mean[i])],
  );
}

void main() {
  final u = _binaryInput();
  final model = _binaryModel();

  Subject simulated(List<double> x, int seed, [HgfModel? m]) => Subject(
    inputs: u,
    responses: (m ?? model).simulate(
      [Subject(inputs: u)],
      [x],
      math.Random(seed),
    )[0],
  );

  group('linalg', () {
    test('Cholesky, log det and inverse', () {
      final a = Float64List.fromList([4, 2, 0.6, 2, 5, 1, 0.6, 1, 3]);
      final l = cholesky(a, 3)!;
      final inv = inverseFromCholesky(l, 3);
      for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
          var s = 0.0;
          for (var m = 0; m < 3; m++) {
            s += a[i * 3 + m] * inv[m * 3 + j];
          }
          expect(s, closeTo(i == j ? 1 : 0, 1e-14));
        }
      }
      // det = 4(15 − 1) − 2(6 − 0.6) + 0.6(2 − 3) = 44.6
      expect(logDetFromCholesky(l, 3), closeTo(math.log(44.6), 1e-14));
    });
    test('not positive definite', () {
      expect(cholesky(Float64List.fromList([1, 2, 2, 1]), 2), isNull);
      expect(cholesky(Float64List.fromList([1, 0, 0, 0]), 2), isNull);
      expect(cholesky(Float64List.fromList([1, 0, 0, double.nan]), 2), isNull);
    });
  });

  group('BatchedObjective.hessians', () {
    test('match central differences of the gradients, in one pass', () {
      final subjects = [
        simulated([-3.5, 3.0], 1),
      ];
      final obj = BatchedObjective(model, subjects);
      final p = model.toUnconstrained([-4.0, 2.0]);
      final (values, hs) = obj.hessians([0], [p]);
      expect(obj.passes, 1);
      expect(obj.elements, 9);
      expect(values[0], obj.evaluate([0], [p], [false]).$1[0]);
      final h = hs[0];
      expect(h[1], h[2]);
      // Each column by Richardson-extrapolated differences of gradients.
      Float64List grad(Float64List y) => obj.evaluate([0], [y], [true]).$2[0]!;
      for (var j = 0; j < 2; j++) {
        Float64List d(double s) {
          final a = Float64List.fromList(p)..[j] += s;
          final b = Float64List.fromList(p)..[j] -= s;
          final ga = grad(a), gb = grad(b);
          return Float64List.fromList([
            for (var i = 0; i < 2; i++) (ga[i] - gb[i]) / (2 * s),
          ]);
        }

        final d1 = d(1e-3), d2 = d(2e-3);
        for (var i = 0; i < 2; i++) {
          final rich = (4 * d1[i] - d2[i]) / 3;
          expect(h[i * 2 + j], closeTo(rich, 1e-4 * (1 + rich.abs())));
        }
      }
    });
  });

  group('laplace', () {
    test('agrees with quadrature where the posterior is well identified', () {
      for (final (truth, seed) in [
        ([-3.5, 3.0], 7),
        ([-2.0, 6.0], 7),
      ]) {
        final subject = simulated(truth, seed);
        final lap = laplace(model, fitMap(model, [subject]));
        expect(lap.passes, 1);
        expect(lap.elements, 9);
        final r = lap.results.single;
        expect(r.map.converged, isTrue);
        expect(r.positiveDefinite, isTrue);
        final q = _quadrature(model, subject, r);
        expect(r.logEvidence, closeTo(q.logZ, 0.02), reason: '$truth');
        final se = r.standardErrors;
        for (var i = 0; i < 2; i++) {
          expect(se[i], closeTo(q.sd[i], 0.05 * q.sd[i]), reason: '$truth');
          expect(r.map.x[i], closeTo(q.mean[i], 0.25 * q.sd[i]));
        }
        expect(lap.logEvidence, r.logEvidence);
      }
    });

    test('approximates a weakly identified posterior', () {
      final subject = simulated([-5.0, 1.0], 7);
      final r = laplace(model, fitMap(model, [subject])).results.single;
      final q = _quadrature(model, subject, r);
      expect(r.logEvidence, closeTo(q.logZ, 0.2));
      final se = r.standardErrors;
      for (var i = 0; i < 2; i++) {
        expect(se[i], closeTo(q.sd[i], 0.2 * q.sd[i]));
      }
    });

    test('derived quantities are consistent', () {
      final r = laplace(
        model,
        fitMap(model, [
          simulated([-3.5, 3.0], 2),
        ]),
      ).results.single;
      final c = r.covariance!, h = r.hessian;
      // C H = I
      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
          final s = c[i * 2] * h[j] + c[i * 2 + 1] * h[2 + j];
          expect(s, closeTo(i == j ? 1 : 0, 1e-12));
        }
      }
      final corr = r.correlation;
      expect(corr[0], closeTo(1, 1e-15));
      expect(corr[1], closeTo(c[1] / math.sqrt(c[0] * c[3]), 1e-15));
      // β has a logit transform onto (0.2, 20): the interval is asymmetric
      // around the estimate and inside the support; the delta-method sd is
      // about a quarter of the 95% width.
      final (lo, hi) = r.interval();
      final se = r.standardErrors;
      for (var i = 0; i < 2; i++) {
        expect(lo[i], lessThan(r.map.x[i]));
        expect(hi[i], greaterThan(r.map.x[i]));
        expect((hi[i] - lo[i]) / (2 * 1.959964), closeTo(se[i], 0.1 * se[i]));
      }
      expect(lo[1], greaterThan(0.2));
      final (lo50, hi50) = r.interval(0.5);
      expect(hi50[0] - lo50[0], lessThan(hi[0] - lo[0]));
      expect(() => r.interval(1), throwsArgumentError);
      expect(r.summary['omega2']!.standardError, se[0]);
    });

    test('95% intervals cover the true values about 95% of the time', () {
      const n = 40;
      final rng = math.Random(11);
      final truths = [
        for (var i = 0; i < n; i++)
          [-5.0 + 3.0 * rng.nextDouble(), 1.0 + 5.0 * rng.nextDouble()],
      ];
      final agents = [for (var i = 0; i < n; i++) Subject(inputs: u, id: i)];
      final responses = model.simulate(agents, truths, math.Random(5));
      final subjects = [
        for (var i = 0; i < n; i++)
          Subject(inputs: u, responses: responses[i], id: i),
      ];
      final lap = laplace(model, fitMap(model, subjects));
      expect(lap.passes, 1);
      expect(lap.elements, 9 * n);
      final covered = [0, 0];
      for (var s = 0; s < n; s++) {
        final (lo, hi) = lap.results[s].interval();
        for (var i = 0; i < 2; i++) {
          if (lo[i] <= truths[s][i] && truths[s][i] <= hi[i]) covered[i]++;
        }
      }
      for (var i = 0; i < 2; i++) {
        expect(covered[i] / n, inInclusiveRange(0.8, 1.0));
      }
    });

    test('log evidence prefers the right model and penalises a free '
        'parameter that buys nothing', () {
      // Data from ω = −3. Model B fixes ω at the truth, model C at a wrong
      // value; model A (ω free) fits at least as well as B but pays for
      // ω's broad prior.
      HgfModel fixed(double omega) => HgfModel(
        network: _binary2(omega: omega),
        response: const BinarySoftmaxReadout(),
        parameters: [_beta],
      );
      final subjects = [
        for (var s = 0; s < 10; s++) simulated([-3.0, 3.0], 100 + s),
      ];
      double evidence(HgfModel m) =>
          laplace(m, fitMap(m, subjects)).logEvidence;
      final a = evidence(model);
      final b = evidence(fixed(-3.0));
      final c = evidence(fixed(-7.5));
      expect(b, greaterThan(a));
      expect(a, greaterThan(c));
    });

    test('a non-positive-definite Hessian gives NaN, not garbage', () {
      // "dummy" targets ω₂ too, but omega2 is applied after it and
      // overwrites it: the likelihood doesn't depend on dummy, and its flat
      // prior adds no curvature, so the Hessian is singular.
      final m = HgfModel(
        network: _binary2(),
        response: const BinarySoftmaxReadout(),
        parameters: [
          Parameter.node(
            'dummy',
            1,
            NodeField.tonicVolatility,
            prior: const Flat(lower: -8, upper: 2),
          ),
          _omega,
          _beta,
        ],
      );
      final subject = simulated([-3.5, 3.0], 7);
      final r = laplace(m, fitMap(m, [subject])).results.single;
      expect(r.positiveDefinite, isFalse);
      expect(r.covariance, isNull);
      expect(r.logEvidence, isNaN);
      expect(r.standardErrors.every((v) => v.isNaN), isTrue);
      expect(r.interval().$1.every((v) => v.isNaN), isTrue);
      expect(r.correlation.every((v) => v.isNaN), isTrue);
    });
  });
}
