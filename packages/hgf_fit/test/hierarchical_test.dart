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

const _logSqrt2Pi = 0.9189385332046727;

Float64List _binaryInput() => Float64List.fromList([
  for (final l in File('data/binary_input.txt').readAsLinesSync())
    if (l.trim().isNotEmpty) double.parse(l),
]);

/// The binary 2-level HGF of pyhgf's multilevel notebook: ω₂ normal,
/// β positive (log transform), plus an independent initial mean μ₂.
HgfModel _model() => HgfModel(
  network: Network()
    ..addNodes(kind: NodeKind.binaryState)
    ..addNodes(valueChildren: [0]),
  response: const BinarySoftmaxReadout(),
  parameters: [
    Parameter.node(
      'omega2',
      1,
      NodeField.tonicVolatility,
      prior: const Normal(-4, 2),
    ),
    Parameter.response('inverseTemperature', prior: const LogNormal(0.5, 1)),
    Parameter.node('mu2', 1, NodeField.mean, prior: const Normal(0, 1)),
  ],
);

void main() {
  final u = _binaryInput().sublist(0, 120);
  final model = _model();
  final truth = [
    [-4.0, 2.0, 0.0],
    [-3.0, 4.0, 0.3],
    [-5.0, 1.5, -0.2],
    [-3.5, 3.0, 0.1],
  ];
  final responses = model.simulate(
    [for (final _ in truth) Subject(inputs: u)],
    truth,
    math.Random(7),
  );
  final subjects = [
    for (var i = 0; i < truth.length; i++)
      Subject(inputs: u, responses: responses[i], id: 'a$i'),
  ];

  HierarchicalModel hier({
    bool centered = false,
    Set<String> shared = const {},
    Set<String> pooled = const {'omega2', 'inverseTemperature'},
  }) => HierarchicalModel(
    model,
    pooled: {
      if (pooled.contains('omega2'))
        'omega2': const Population(mean: Normal(-5, 5), sd: HalfNormal(2)),
      if (pooled.contains('inverseTemperature'))
        'inverseTemperature': const Population(
          mean: Normal(0, 2),
          sd: HalfNormal(1),
        ),
    },
    shared: shared,
    centered: centered,
  );

  group('HierarchicalModel', () {
    test('layout, names and indices', () {
      final h = hier(shared: {'mu2'}, pooled: {'omega2'});
      expect(h.dimension(4), 2 + 1 + 4 * 2);
      expect(h.names(subjects.sublist(0, 2)), [
        'omega2_mu',
        'omega2_sigma',
        'mu2',
        'omega2[a0]',
        'inverseTemperature[a0]',
        'omega2[a1]',
        'inverseTemperature[a1]',
      ]);
      expect(h.indexOf('omega2', group: 'mu'), 0);
      expect(h.indexOf('omega2', group: 'sigma'), 1);
      expect(h.indexOf('mu2'), 2);
      expect(h.indexOf('omega2', subject: 1), 5);
      expect(h.indexOf('inverseTemperature', subject: 1), 6);
      expect(() => h.indexOf('mu2', group: 'mu'), throwsArgumentError);
      expect(() => h.indexOf('omega2'), throwsArgumentError);
    });

    test('rejects bad declarations', () {
      expect(
        () => HierarchicalModel(model, pooled: {'nope': const Population()}),
        throwsArgumentError,
      );
      expect(
        () => HierarchicalModel(
          model,
          pooled: {'omega2': const Population()},
          shared: {'omega2'},
        ),
        throwsArgumentError,
      );
      expect(
        () => HierarchicalModel(
          model,
          pooled: {'omega2': const Population(sd: Normal(0, 1))},
        ),
        throwsArgumentError,
      );
    });

    for (final centered in [false, true]) {
      final name = centered ? 'centred' : 'non-centred';

      test('$name objective equals the joint density by hand', () {
        final h = hier(centered: centered, shared: {'mu2'}, pooled: {'omega2'});
        final rng = math.Random(3);
        final theta = Float64List.fromList([
          for (var i = 0; i < h.dimension(4); i++) 0.4 * rng.nextDouble() - 0.2,
        ]);
        theta[0] = -4.2; // μ
        theta[1] = math.log(0.8); // log σ
        final (v, _) = h.potential(subjects, [theta]);

        final x = h.toConstrained(theta);
        final mu = x[0], sd = x[1], mu2 = x[2];
        var expected = 0.0;
        // Hyperpriors with the log-σ Jacobian.
        expected -= const Normal(-5, 5).logDensity(mu);
        expected -= const HalfNormal(2).logDensity(sd) + math.log(sd);
        // The shared μ₂ with its prior (identity transform).
        expected -= const Normal(0, 1).logDensity(mu2);
        for (var s = 0; s < 4; s++) {
          final omega = x[h.indexOf('omega2', subject: s)];
          final beta = x[h.indexOf('inverseTemperature', subject: s)];
          final nll = Float64List(1), out = Float64List(1);
          model.negLogPosterior(
            [subjects[s]],
            Int32List(1),
            model.toUnconstrained([omega, beta, mu2]),
            out,
            negLogLikelihood: nll,
          );
          expected += nll[0];
          // β independent: its prior and log Jacobian.
          expected -= const LogNormal(0.5, 1).logDensity(beta) + math.log(beta);
          // ω₂ pooled.
          if (centered) {
            final r = (omega - mu) / sd;
            expected += 0.5 * r * r + math.log(sd) + _logSqrt2Pi;
          } else {
            final z = theta[h.indexOf('omega2', subject: s)];
            expect(omega, closeTo(mu + sd * z, 1e-12));
            expected += 0.5 * z * z + _logSqrt2Pi;
          }
        }
        expect(v[0], closeTo(expected, 1e-9 * expected.abs()));
      });

      test('$name gradient matches finite differences', () {
        for (final h in [
          hier(centered: centered),
          hier(centered: centered, shared: {'mu2'}, pooled: {'omega2'}),
          hier(centered: centered, shared: {'omega2'}, pooled: {}),
        ]) {
          final rng = math.Random(11);
          final theta = h.initialPoint(
            subjects.length,
            start: [for (final t in truth) t],
          );
          for (var i = 0; i < theta.length; i++) {
            theta[i] += 0.3 * (2 * rng.nextDouble() - 1);
          }
          final (v, g) = h.potential(subjects, [theta]);
          expect(v[0].isFinite, isTrue);
          final pts = <Float64List>[];
          const step = 1e-5;
          for (var i = 0; i < theta.length; i++) {
            for (final sgn in [1.0, -1.0]) {
              pts.add(Float64List.fromList(theta)..[i] += sgn * step);
            }
          }
          final (fv, _) = h.potential(subjects, pts);
          for (var i = 0; i < theta.length; i++) {
            final fd = (fv[2 * i] - fv[2 * i + 1]) / (2 * step);
            expect(
              g[0][i],
              closeTo(fd, 1e-5 * math.max(1, fd.abs())),
              reason: '$h entry $i',
            );
          }
        }
      });
    }

    test('initial point from estimates', () {
      final h = hier();
      final theta = h.initialPoint(4, start: truth);
      final x = h.toConstrained(theta);
      final omegas = [for (final t in truth) t[0]];
      final m = omegas.reduce((a, b) => a + b) / 4;
      expect(x[0], closeTo(m, 1e-12));
      for (var s = 0; s < 4; s++) {
        for (final (i, name) in [
          (0, 'omega2'),
          (1, 'inverseTemperature'),
          (2, 'mu2'),
        ]) {
          expect(x[h.indexOf(name, subject: s)], closeTo(truth[s][i], 1e-12));
        }
      }
    });
  });

  group('sampleHierarchical', () {
    test('with a flat likelihood the draws follow the priors', () {
      // A response model that ignores everything: the posterior is the
      // prior, μ ~ N(1, 0.5), σ ~ HalfNormal(1), y_s ~ N(μ, σ).
      final flat = HgfModel(
        network: Network()
          ..addNodes(kind: NodeKind.binaryState)
          ..addNodes(valueChildren: [0]),
        response: ResponseModel.fn(
          surprise: (_, _, _, _, _, _) => 0.0,
          surprisePartials: (_, _, _, _, _, _, _) => 0.0,
          usesResponses: false,
        ),
        parameters: [
          Parameter.node(
            'omega2',
            1,
            NodeField.tonicVolatility,
            prior: const Normal(0, 1),
          ),
        ],
      );
      expect(flat.hasExactGradients, isTrue);
      final h = HierarchicalModel(
        flat,
        pooled: {
          'omega2': const Population(mean: Normal(1, 0.5), sd: HalfNormal(1)),
        },
      );
      final subs = [
        for (var s = 0; s < 3; s++) Subject(inputs: u.sublist(0, 5)),
      ];
      final fit = sampleHierarchical(
        h,
        subs,
        chains: 4,
        warmup: 500,
        draws: 2000,
        seed: 2,
      );
      expect(fit.divergences, 0);
      final mu = fit.summary('omega2', group: 'mu');
      final sd = fit.summary('omega2', group: 'sigma');
      final y = fit.summary('omega2', subject: 1);
      expect(mu.mean, closeTo(1, 4 * mu.mcseMean));
      expect(mu.sd, closeTo(0.5, 0.05));
      expect(sd.mean, closeTo(math.sqrt(2 / math.pi), 4 * sd.mcseMean));
      expect(y.mean, closeTo(1, 4 * y.mcseMean));
      // Var y = Var μ + E σ² = 0.25 + 1.
      expect(y.sd, closeTo(math.sqrt(1.25), 0.1));
    });

    test('centred and non-centred give the same posterior', () {
      HierarchicalFit run(bool centered) => sampleHierarchical(
        hier(centered: centered),
        subjects,
        chains: 2,
        warmup: 300,
        draws: 400,
        start: truth,
        seed: 5,
      );
      final a = run(false), b = run(true);
      expect(a.passes, greaterThan(0));
      // Every pass evaluates each running chain's point for all subjects.
      expect(a.elements, lessThanOrEqualTo(a.passes * 2 * subjects.length));
      for (final (name, group, subject) in [
        ('omega2', 'mu', null),
        ('inverseTemperature', 'mu', null),
        ('omega2', null, 0),
        ('inverseTemperature', null, 2),
        ('mu2', null, 1),
      ]) {
        final sa = a.summary(name, group: group, subject: subject);
        final sb = b.summary(name, group: group, subject: subject);
        final se = math.sqrt(
          sa.mcseMean * sa.mcseMean + sb.mcseMean * sb.mcseMean,
        );
        expect(
          sa.mean,
          closeTo(sb.mean, 4.5 * se),
          reason: '$name $group $subject',
        );
      }
    });

    test('exports for ArviZ', () {
      final h = hier(shared: {'mu2'}, pooled: {'omega2'});
      final fit = sampleHierarchical(
        h,
        subjects,
        chains: 2,
        warmup: 20,
        draws: 7,
        seed: 1,
      );
      final a = jsonDecode(fit.toJson()) as Map<String, dynamic>;
      final post = a['posterior'] as Map<String, dynamic>;
      expect(post.keys, {
        'omega2_mu',
        'omega2_sigma',
        'mu2',
        'omega2',
        'inverseTemperature',
      });
      expect((post['omega2_mu'] as List), hasLength(2));
      expect((post['omega2_mu'] as List)[0], hasLength(7));
      final w = post['omega2'] as List;
      expect(w, hasLength(2));
      expect(w[1], hasLength(7));
      expect(w[1][3], hasLength(4));
      expect(w[1][3][2], fit.draws.draw(1, 3)[h.indexOf('omega2', subject: 2)]);
      expect(a['dims'], {
        'omega2': ['subject'],
        'inverseTemperature': ['subject'],
      });
      expect(a['coords'], {
        'subject': ['a0', 'a1', 'a2', 'a3'],
      });
      expect((a['sample_stats'] as Map)['lp'], hasLength(2));
    });
  });

  group('fitEmpiricalBayes', () {
    final full = _binaryInput();
    final m2 = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0]),
      response: const BinarySoftmaxReadout(),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-4, 2),
        ),
        Parameter.response(
          'inverseTemperature',
          prior: const LogNormal(0.5, 1),
        ),
      ],
    );
    final h = HierarchicalModel(
      m2,
      pooled: {
        'omega2': const Population(),
        'inverseTemperature': const Population(),
      },
    );

    test('subject objective is the population density on y', () {
      // One EM objective evaluated by hand: −log N(y | μ, σ) per pooled
      // parameter plus the likelihood.
      final eb = fitEmpiricalBayes(h, subjects, maxIterations: 1);
      expect(eb.iterations, 1);
      final s0 = eb.subjects[0];
      final nll = Float64List(1), out = Float64List(1);
      m2.negLogPosterior(
        [subjects[0]],
        Int32List(1),
        s0.y,
        out,
        negLogLikelihood: nll,
      );
      // The starting populations: N(T(start), 1) for each parameter.
      final mu = m2.toUnconstrained(m2.start());
      var expected = nll[0];
      for (var i = 0; i < 2; i++) {
        final z = s0.y[i] - mu[i];
        expected += 0.5 * z * z + _logSqrt2Pi;
      }
      expect(s0.negLogPosterior, closeTo(expected, 1e-9));
    });

    test('recovers the population of simulated agents', () {
      final rng = math.Random(42);
      const muW = -3.5, sdW = 0.6, muB = 1.0, sdB = 0.3;
      final truth = [
        for (var i = 0; i < 60; i++)
          [
            muW + sdW * const Normal(0, 1).sample(rng),
            math.exp(muB + sdB * const Normal(0, 1).sample(rng)),
          ],
      ];
      final r = m2.simulate(
        [for (final _ in truth) Subject(inputs: full)],
        truth,
        rng,
      );
      final subs = [
        for (var i = 0; i < truth.length; i++)
          Subject(inputs: full, responses: r[i]),
      ];
      final eb = fitEmpiricalBayes(h, subs);
      expect(eb.converged, isTrue);
      // EM never lowers the (Laplace) evidence, up to optimiser noise.
      for (var i = 1; i < eb.history.length; i++) {
        expect(eb.history[i], greaterThan(eb.history[i - 1] - 1e-3));
      }
      // Against the moments of the agents actually drawn (60 agents leave
      // the population's own sampling error of 0.08 / 0.04 otherwise).
      double mean(List<double> v) => v.reduce((a, c) => a + c) / v.length;
      double sd(List<double> v) {
        final m = mean(v);
        return math.sqrt(
          v.fold(0.0, (a, c) => a + (c - m) * (c - m)) / v.length,
        );
      }

      final tw = [for (final t in truth) t[0]];
      final tb = [for (final t in truth) math.log(t[1])];
      final w = eb.populations['omega2']!;
      final b = eb.populations['inverseTemperature']!;
      expect(w.mean, closeTo(mean(tw), 0.1));
      expect(w.sd, closeTo(sd(tw), 0.1));
      expect(b.mean, closeTo(mean(tb), 0.05));
      expect(b.sd, closeTo(sd(tb), 0.05));
      // Shrinkage: the estimates are closer to the group mean than the
      // independent MAP estimates under a flat-ish prior.
      final map = fitMap(m2, subs);
      double spread(List<double> v) {
        final m = v.reduce((a, c) => a + c) / v.length;
        return v.fold(0.0, (a, c) => a + (c - m) * (c - m));
      }

      expect(
        spread([for (final s in eb.subjects) s.y[0]]),
        lessThan(spread([for (final f in map.results) f.y[0]])),
      );
      for (final s in eb.subjects) {
        expect(s.covariance, isNotNull);
        expect(s.logEvidence.isFinite, isTrue);
      }
    });

    test('shared parameters are not supported', () {
      expect(
        () => fitEmpiricalBayes(
          HierarchicalModel(
            m2,
            pooled: {'omega2': const Population()},
            shared: {'inverseTemperature'},
          ),
          subjects,
        ),
        throwsUnsupportedError,
      );
    });
  });
}
