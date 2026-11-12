// Reverse-mode gradients on the CPU (the trial's IR evaluated in f64,
// IrReverse) against the forward-mode tangents of BatchNetwork.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_fit/ir.dart';
import 'package:test/test.dart';

import 'src/fit_cases.dart';

HgfModel withMode(HgfModel m, GradientMode mode, {int above = 8}) => HgfModel(
  network: m.network,
  response: m.response,
  parameters: m.parameters,
  gradientMode: mode,
  reverseAbove: above,
);

void main() {
  for (final c in fitCases()) {
    test('${c.name}: reverse = forward (f64)', () {
      final fwd = withMode(c.model, GradientMode.forward);
      final rev = withMode(c.model, GradientMode.reverse);
      expect(rev.reverseUnsupported, isNull);
      expect(rev.usesReverseGradients, isTrue);
      final k = fwd.dimension;
      final n = c.subjects.length * c.points.length;
      final subjectOf = Int32List(n);
      final y = Float64List(n * k);
      var e = 0;
      for (final x in c.points) {
        final yy = fwd.toUnconstrained(x);
        for (var s = 0; s < c.subjects.length; s++) {
          subjectOf[e] = s;
          y.setRange(e * k, (e + 1) * k, yy);
          e++;
        }
      }
      for (final jacobian in [false, true]) {
        final outF = Float64List(n), outR = Float64List(n);
        final gF = Float64List(n * k), gR = Float64List(n * k);
        final nllF = Float64List(n), nllR = Float64List(n);
        fwd.negLogPosterior(
          c.subjects,
          subjectOf,
          y,
          outF,
          jacobian: jacobian,
          gradient: gF,
          negLogLikelihood: nllF,
        );
        rev.negLogPosterior(
          c.subjects,
          subjectOf,
          y,
          outR,
          jacobian: jacobian,
          gradient: gR,
          negLogLikelihood: nllR,
        );
        for (var j = 0; j < n; j++) {
          expect(
            (outR[j] - outF[j]).abs() / math.max(1, outF[j].abs()),
            lessThan(1e-10),
            reason: 'element $j value ${outR[j]} vs ${outF[j]}',
          );
          var scale = 1.0;
          for (var i = 0; i < k; i++) {
            scale = math.max(scale, gF[j * k + i].abs());
          }
          for (var i = 0; i < k; i++) {
            final a = gF[j * k + i], b = gR[j * k + i];
            expect(
              (a - b).abs() / scale,
              lessThan(1e-8),
              reason: 'element $j ∂/∂${fwd.names[i]}: forward $a, reverse $b',
            );
          }
        }
      }
    });
  }

  test('auto switches above reverseAbove parameters', () {
    final c = fitCases().first;
    final k = c.model.dimension;
    expect(
      withMode(c.model, GradientMode.auto, above: k).usesReverseGradients,
      isFalse,
    );
    expect(
      withMode(c.model, GradientMode.auto, above: k - 1).usesReverseGradients,
      isTrue,
    );
    expect(withMode(c.model, GradientMode.forward).usesReverseGradients, false);
  });

  test('models without an IR form fall back to forward mode', () {
    final sine = CustomCoupling(
      'sin',
      g: math.sin,
      dg: math.cos,
      d2g: (x) => -math.sin(x),
      d3g: (x) => -math.cos(x),
    );
    final m = HgfModel(
      network: Network()
        ..addNodes()
        ..addNodes(valueChildren: [0], coupling: sine),
      response: const ObservationSurprise(TotalGaussianSurprise()),
      parameters: [
        Parameter.node(
          'omega1',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-4, 1),
        ),
      ],
      gradientMode: GradientMode.reverse,
    );
    expect(m.reverseUnsupported, contains('Custom coupling'));
    expect(m.usesReverseGradients, isFalse);
  });

  test(
    'IrReverse: adjoints of every initial slot match finite differences',
    () {
      // A slot no parameter sets (the prior precision of node 2) as an
      // independent check of the slot-level adjoints.
      final c = fitCases()[1];
      final ir = IrReverse(c.model.network, irResponseOf(c.model.response));
      final nS = ir.nSlots;
      final subjects = c.subjects;
      final subjectOf = Int32List.fromList([0]);
      final slots = Float64List.fromList(ir.initialSlots);
      final adj = Float64List(nS), nll = Float64List(1);
      ir.evaluate(subjects, subjectOf, slots, nll, adjoints: adj);
      var checked = 0;
      for (final s in ir.carriedSlots) {
        final h = 1e-6 * math.max(1, slots[s].abs());
        double at(double v) {
          final p = Float64List.fromList(slots)..[s] = v;
          final o = Float64List(1);
          ir.evaluate(subjects, subjectOf, p, o);
          return o[0];
        }

        final fd = (at(slots[s] + h) - at(slots[s] - h)) / (2 * h);
        if (!fd.isFinite) continue;
        expect(
          (fd - adj[s]).abs(),
          lessThan(1e-5 * math.max(1, fd.abs())),
          reason: 'slot $s: adjoint ${adj[s]}, central difference $fd',
        );
        checked++;
      }
      expect(checked, greaterThan(5));
    },
  );

  group('exact Hessians (forward-over-reverse)', () {
    for (final c in fitCases()) {
      test('${c.name}: = central differences of the exact gradient', () {
        final m = withMode(c.model, GradientMode.reverse);
        final k = m.dimension;
        final x = c.points.first;
        final y0 = m.toUnconstrained(x);
        final subjectOf = Int32List.fromList([0]);
        for (final jacobian in [false, true]) {
          final h = m
              .exactHessians(c.subjects, subjectOf, y0, jacobian: jacobian)
              .single;
          Float64List grad(Float64List y) {
            final g = Float64List(k);
            m.negLogPosterior(
              c.subjects,
              subjectOf,
              y,
              Float64List(1),
              jacobian: jacobian,
              gradient: g,
            );
            return g;
          }

          var scale = 1.0;
          for (final v in h) {
            scale = math.max(scale, v.abs());
          }
          for (var j = 0; j < k; j++) {
            final step = 1e-5 * math.max(1, y0[j].abs());
            final up = Float64List.fromList(y0)..[j] += step;
            final dn = Float64List.fromList(y0)..[j] -= step;
            final gu = grad(up), gd = grad(dn);
            for (var i = 0; i < k; i++) {
              final fd = (gu[i] - gd[i]) / (2 * step);
              expect(
                (fd - h[i * k + j]).abs() / scale,
                lessThan(1e-6),
                reason:
                    'jacobian $jacobian, H[${m.names[i]}][${m.names[j]}]: '
                    'exact ${h[i * k + j]}, differences $fd',
              );
            }
          }
        }
      });
    }

    test('laplace: the same standard errors as with finite differences', () {
      final c = fitCases()[1];
      final m = withMode(c.model, GradientMode.reverse);
      final map = fitMap(m, c.subjects, random: math.Random(2));
      final fd = laplace(m, map), ex = laplace(m, map, exactHessian: true);
      for (var s = 0; s < map.results.length; s++) {
        final a = fd.results[s].standardErrors;
        final b = ex.results[s].standardErrors;
        // The second differences of values (step 1e-4) are good to about
        // 1e-4 relative (1.02e-4 on dart2wasm, whose exp differs).
        for (var i = 0; i < a.length; i++) {
          expect((a[i] - b[i]).abs() / b[i], lessThan(5e-4));
        }
        expect(
          (fd.results[s].logEvidence - ex.results[s].logEvidence).abs(),
          lessThan(1e-3),
        );
      }
    });
  });

  test('second derivatives of the built-in priors and transforms', () {
    double fd(double Function(double) f, double x) {
      final h = 1e-5 * math.max(1, x.abs());
      return (f(x + h) - f(x - h)) / (2 * h);
    }

    final priors = <Prior, List<double>>{
      const Normal(1, 2): [-3, 0.5, 4],
      const HalfNormal(1.5): [0.2, 1, 3],
      const LogNormal(0.3, 0.7): [0.4, 1.3, 5],
      const GammaPrior(3, 2): [0.5, 1.5, 4],
      const Uniform(-1, 2): [0, 1],
    };
    priors.forEach((p, xs) {
      for (final x in xs) {
        expect(
          p.d2LogDensity(x),
          closeTo(fd(p.dLogDensity, x), 1e-6),
          reason: '$p at $x',
        );
      }
    });
    final transforms = <Transform>[
      const IdentityTransform(),
      const LowerBoundTransform(0.5),
      const UpperBoundTransform(2),
      const IntervalTransform(-1, 3),
    ];
    for (final t in transforms) {
      for (final y in [-2.0, 0.3, 1.7]) {
        expect(
          t.d2ToConstrained(y),
          closeTo(fd(t.dToConstrained, y), 1e-6),
          reason: '$t at $y',
        );
        expect(
          t.d2LogAbsJacobian(y),
          closeTo(fd(t.dLogAbsJacobian, y), 1e-6),
          reason: '$t at $y',
        );
      }
    }
  });
}
