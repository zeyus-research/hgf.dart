import 'dart:math' as math;

import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

/// Trapezoid-rule integral of exp(f) over [a, b].
double _integrateExp(double Function(double) f, double a, double b) {
  const n = 200000;
  final h = (b - a) / n;
  var s = 0.5 * (math.exp(f(a)) + math.exp(f(b)));
  for (var i = 1; i < n; i++) {
    s += math.exp(f(a + i * h));
  }
  return s * h;
}

void main() {
  group('erf', () {
    // Reference values from mpmath (30 digits).
    const cases = <(double, double)>[
      (0.1, 0.112462916018284898404712251014),
      (1.0, 0.842700792949714869341220635083),
      (2.4, 0.99931148610335492111445002519),
    ];
    const tails = <(double, double)>[
      (2.5, 4.06952017444958939564215739975e-4),
      (3.0, 2.20904969985854413727761295823e-5),
      (6.0, 2.15197367124989131165933503992e-17),
      (10.0, 2.08848758376254475700078629496e-45),
    ];
    test('series region', () {
      for (final (x, v) in cases) {
        expect(erf(x), closeTo(v, 1e-15));
        expect(erf(-x), closeTo(-v, 1e-15));
      }
    });
    test('erfc tail keeps relative precision', () {
      for (final (x, v) in tails) {
        expect((erfc(x) - v).abs() / v, lessThan(1e-13), reason: 'x=$x');
        expect(erfc(-x), closeTo(2 - v, 1e-15));
      }
    });
    test('logNormalCdf', () {
      expect(logNormalCdf(0), closeTo(math.log(0.5), 1e-15));
      // log Φ(−10) and log Φ(−40) (mpmath).
      expect(logNormalCdf(-10), closeTo(-53.23128515051247, 1e-11));
      expect(logNormalCdf(-40), closeTo(-804.6084420137538, 1e-9));
      expect(logNormalCdf(8), closeTo(-6.220960574271785e-16, 1e-25));
    });
  });

  group('priors', () {
    final priors = <(Prior, double, double)>[
      (const Normal(1, 2), -30, 30),
      (TruncatedNormal(-3, 5, lower: -8, upper: 2), -8, 2),
      (TruncatedNormal(0, 1, lower: 3), 3, 20),
      (const Uniform(0.2, 20), 0.2, 20),
      (const HalfNormal(2), 0, 30),
      (const LogNormal(0, 0.5), 1e-12, 40),
      (const GammaPrior(3, 2), 1e-12, 40),
    ];
    for (final (p, a, b) in priors) {
      test('$p integrates to 1', () {
        expect(_integrateExp(p.logDensity, a, b), closeTo(1, 1e-6));
      });
      test('$p sample mean is in the support and matches', () {
        final rng = math.Random(1);
        var s = 0.0;
        const n = 20000;
        for (var i = 0; i < n; i++) {
          final x = p.sample(rng);
          expect(p.contains(x), isTrue);
          s += x;
        }
        // The density's mean, by quadrature.
        const m = 20000;
        final h = (b - a) / m;
        var mean = 0.0;
        for (var i = 0; i <= m; i++) {
          final x = a + i * h;
          mean += x * math.exp(p.logDensity(x)) * (i == 0 || i == m ? 0.5 : 1);
        }
        mean *= h;
        expect(s / n, closeTo(mean, 0.05 * (1 + mean.abs())));
      });
    }
    test('outside the support is -inf', () {
      expect(const Uniform(0, 1).logDensity(2), double.negativeInfinity);
      expect(const HalfNormal(1).logDensity(-1), double.negativeInfinity);
      expect(
        TruncatedNormal(0, 1, upper: 0).logDensity(0.1),
        double.negativeInfinity,
      );
    });
  });

  group('transforms', () {
    final transforms = <(Transform, List<double>)>[
      (const IdentityTransform(), [-3, 0, 5]),
      (const LowerBoundTransform(0.2), [0.3, 1, 50]),
      (const UpperBoundTransform(2), [-5, 0, 1.9]),
      (const IntervalTransform(-8, 2), [-7.9, -3, 1.5]),
    ];
    for (final (t, xs) in transforms) {
      test('${t.runtimeType} round trip and Jacobian', () {
        for (final x in xs) {
          final y = t.toUnconstrained(x);
          expect(t.toConstrained(y), closeTo(x, 1e-12 * (1 + x.abs())));
          const h = 1e-6;
          final dxdy =
              (t.toConstrained(y + h) - t.toConstrained(y - h)) / (2 * h);
          expect(t.logAbsJacobian(y), closeTo(math.log(dxdy.abs()), 1e-6));
        }
      });
    }
    test('forSupport picks by bounds', () {
      expect(
        Transform.forSupport(double.negativeInfinity, double.infinity),
        isA<IdentityTransform>(),
      );
      expect(
        Transform.forSupport(0, double.infinity),
        isA<LowerBoundTransform>(),
      );
      expect(
        Transform.forSupport(double.negativeInfinity, 0),
        isA<UpperBoundTransform>(),
      );
      expect(Transform.forSupport(0, 1), isA<IntervalTransform>());
    });
  });
}
