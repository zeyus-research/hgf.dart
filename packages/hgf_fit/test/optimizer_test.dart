import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

double _rosenbrock(Float64List x) {
  var s = 0.0;
  for (var i = 0; i < x.length - 1; i++) {
    final a = x[i + 1] - x[i] * x[i], b = 1 - x[i];
    s += 100 * a * a + b * b;
  }
  return s;
}

Float64List _rosenbrockGrad(Float64List x) {
  final g = Float64List(x.length);
  for (var i = 0; i < x.length - 1; i++) {
    final a = x[i + 1] - x[i] * x[i];
    g[i] += -400 * x[i] * a - 2 * (1 - x[i]);
    g[i + 1] += 200 * a;
  }
  return g;
}

void main() {
  group('L-BFGS', () {
    test('minimises Rosenbrock (2-D and 6-D)', () {
      for (final x0 in [
        [-1.2, 1.0],
        [-1.2, 1.0, -1.2, 1.0, -1.2, 1.0],
      ]) {
        final r = minimizeFunction(
          const Lbfgs(gradientTolerance: 1e-8, functionTolerance: 0),
          _rosenbrock,
          Float64List.fromList(x0),
          gradient: _rosenbrockGrad,
        );
        expect(r.converged, isTrue, reason: r.message);
        for (final v in r.x) {
          expect(v, closeTo(1, 1e-6));
        }
      }
    });

    test('a quadratic in few iterations', () {
      // f = Σ cᵢ (xᵢ − i)², badly scaled.
      const c = [1.0, 10.0, 100.0, 1000.0];
      double f(Float64List x) {
        var s = 0.0;
        for (var i = 0; i < 4; i++) {
          s += c[i] * (x[i] - i) * (x[i] - i);
        }
        return s;
      }

      final r = minimizeFunction(
        const Lbfgs(gradientTolerance: 1e-9, functionTolerance: 0),
        f,
        Float64List(4),
        gradient: (x) => Float64List.fromList([
          for (var i = 0; i < 4; i++) 2 * c[i] * (x[i] - i),
        ]),
      );
      expect(r.converged, isTrue);
      expect(r.iterations, lessThan(20));
      for (var i = 0; i < 4; i++) {
        expect(r.x[i], closeTo(i, 1e-9));
      }
    });

    test('backs off infinite regions', () {
      // −log x + x, minimum at 1; +∞ for x ≤ 0.
      final r = minimizeFunction(
        const Lbfgs(),
        (x) => x[0] > 0 ? -math.log(x[0]) + x[0] : double.infinity,
        Float64List.fromList([5.0]),
        gradient: (x) => Float64List.fromList([-1 / x[0] + 1]),
      );
      expect(r.converged, isTrue, reason: r.message);
      expect(r.x[0], closeTo(1, 1e-4));
    });

    test('reports a non-finite start', () {
      final r = minimizeFunction(
        const Lbfgs(),
        (x) => double.nan,
        Float64List(1),
        gradient: (x) => Float64List(1),
      );
      expect(r.converged, isFalse);
      expect(r.message, contains('not finite'));
    });
  });

  group('Nelder-Mead', () {
    for (final speculative in [true, false]) {
      test('minimises Rosenbrock (speculative: $speculative)', () {
        final r = minimizeFunction(
          NelderMead(speculative: speculative, xTolerance: 1e-8),
          _rosenbrock,
          Float64List.fromList([-1.2, 1.0]),
        );
        expect(r.converged, isTrue);
        expect(r.x[0], closeTo(1, 1e-5));
        expect(r.x[1], closeTo(1, 1e-5));
      });
    }
    test('one dimension', () {
      final r = minimizeFunction(
        const NelderMead(),
        (x) => (x[0] - 3) * (x[0] - 3),
        Float64List(1),
      );
      expect(r.converged, isTrue);
      expect(r.x[0], closeTo(3, 1e-5));
    });
  });
}
