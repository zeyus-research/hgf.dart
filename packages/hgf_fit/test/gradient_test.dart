// Reads data/binary_input.txt with dart:io.
@TestOn('vm')
library;

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

double _normal(math.Random rng) =>
    math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
    math.cos(2 * math.pi * rng.nextDouble());

/// A noisy random walk with calm and volatile phases.
Float64List _walk(int n, math.Random rng) {
  final out = Float64List(n);
  var x = 0.0;
  for (var t = 0; t < n; t++) {
    x += ((t ~/ 50).isEven ? 0.05 : 0.4) * _normal(rng);
    out[t] = x + 0.5 * _normal(rng);
  }
  return out;
}

/// Richardson-extrapolated central differences of [f] at [y] along each
/// coordinate.
Float64List _richardson(double Function(Float64List) f, Float64List y) {
  final g = Float64List(y.length);
  for (var i = 0; i < y.length; i++) {
    double d(double h) {
      final a = Float64List.fromList(y)..[i] += h;
      final b = Float64List.fromList(y)..[i] -= h;
      return (f(a) - f(b)) / (2 * h);
    }

    g[i] = (4 * d(1e-4) - d(2e-4)) / 3;
  }
  return g;
}

/// Checks [model]'s exact gradients against finite differences at each of
/// [points] (natural scale), for every subject, with and without the
/// Jacobian terms.
void _checkGradients(
  HgfModel model,
  List<Subject> subjects,
  List<List<double>> points, {
  double tol = 1e-6,
}) {
  expect(model.hasExactGradients, isTrue);
  final k = model.dimension;
  for (final jacobian in [false, true]) {
    for (final x in points) {
      final y = model.toUnconstrained(x);
      final n = subjects.length;
      final subjectOf = Int32List.fromList([for (var s = 0; s < n; s++) s]);
      final ys = Float64List(n * k);
      for (var s = 0; s < n; s++) {
        ys.setRange(s * k, (s + 1) * k, y);
      }
      final out = Float64List(n), plain = Float64List(n);
      final grad = Float64List(n * k);
      model.negLogPosterior(
        subjects,
        subjectOf,
        ys,
        out,
        jacobian: jacobian,
        gradient: grad,
      );
      // The value is the same with and without tangents, bit for bit.
      model.negLogPosterior(subjects, subjectOf, ys, plain, jacobian: jacobian);
      expect(out, plain);
      for (var s = 0; s < n; s++) {
        double f(Float64List yy) {
          final o = Float64List(1);
          model.negLogPosterior(
            [subjects[s]],
            Int32List(1),
            yy,
            o,
            jacobian: jacobian,
          );
          return o[0];
        }

        final fd = _richardson(f, y);
        for (var i = 0; i < k; i++) {
          expect(
            grad[s * k + i],
            closeTo(fd[i], tol * (1 + fd[i].abs())),
            reason: '${model.names[i]} subject $s at $x (jacobian: $jacobian)',
          );
        }
      }
    }
  }
}

void main() {
  final u = _binaryInput();

  group('derivatives of priors and transforms', () {
    final priors = <(Prior, List<double>)>[
      (const Normal(1, 2), [-3, 0.5, 4]),
      (TruncatedNormal(-3, 5, lower: -8, upper: 2), [-7, -3, 1.5]),
      (const Uniform(0.2, 20), [1, 10]),
      (const HalfNormal(2), [0.1, 1, 5]),
      (const LogNormal(0.5, 0.7), [0.2, 1, 6]),
      (const GammaPrior(2.5, 1.5), [0.3, 1, 4]),
      (const Flat(), [-1, 3]),
    ];
    for (final (prior, xs) in priors) {
      test('$prior', () {
        for (final x in xs) {
          const h = 1e-5;
          final fd =
              (prior.logDensity(x + h) - prior.logDensity(x - h)) / (2 * h);
          expect(prior.dLogDensity(x), closeTo(fd, 1e-7 * (1 + fd.abs())));
          final t = prior.defaultTransform;
          final y = t.toUnconstrained(x);
          final dx =
              (t.toConstrained(y + h) - t.toConstrained(y - h)) / (2 * h);
          expect(t.dToConstrained(y), closeTo(dx, 1e-7 * (1 + dx.abs())));
          final dj =
              (t.logAbsJacobian(y + h) - t.logAbsJacobian(y - h)) / (2 * h);
          expect(t.dLogAbsJacobian(y), closeTo(dj, 1e-7 * (1 + dj.abs())));
        }
      });
    }
  });

  group('exact gradients match finite differences', () {
    test('binary 2-level, softmax decisions (ω₂, β)', () {
      final model = HgfModel(
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
          ),
        ],
      );
      final r = model.simulate(
        [Subject(inputs: u), Subject(inputs: u)],
        [
          [-3.5, 3.0],
          [-1.5, 1.0],
        ],
        math.Random(1),
      );
      // A missed response and a short series too.
      final r1 = Float64List.fromList(r[1])..[10] = double.nan;
      final subjects = [
        Subject(inputs: u, responses: r[0]),
        Subject(inputs: u, responses: r1),
        Subject(
          inputs: Float64List.sublistView(u, 0, 150),
          responses: Float64List.sublistView(r[0], 0, 150),
        ),
      ];
      _checkGradients(model, subjects, [
        [-4.0, 2.0],
        [-2.5, 6.0],
      ]);
    });

    test('continuous 3-level, Gaussian readout, every update mode', () {
      final rng = math.Random(3);
      final data = _walk(300, rng);
      for (final mode in VolatilityUpdates.values) {
        final net = Network(volatilityUpdates: mode)
          ..addNodes(precision: 4)
          ..addNodes(valueChildren: [0], tonicVolatility: -3)
          ..addNodes(volatilityChildren: [1], tonicVolatility: -4);
        final model = HgfModel(
          network: net,
          response: const GaussianReadout(),
          parameters: [
            Parameter.node(
              'omega2',
              1,
              NodeField.tonicVolatility,
              prior: const Normal(-3, 2),
            ),
            Parameter.node(
              'omega3',
              2,
              NodeField.tonicVolatility,
              prior: const Normal(-4, 2),
            ),
            Parameter.volatilityCoupling(
              'kappa',
              2,
              1,
              prior: const HalfNormal(2),
            ),
            Parameter.node(
              'inputPrecision',
              0,
              NodeField.precision,
              prior: const LogNormal(1, 1),
            ),
            Parameter.response('precision', prior: const GammaPrior(2, 1)),
          ],
        );
        final sim = model.simulate(
          [Subject(inputs: data)],
          [
            [-3.0, -4.0, 1.0, 4.0, 2.0],
          ],
          math.Random(4),
        )[0];
        // Missing observations, irregular time steps.
        final observed = Float64List(data.length)..fillRange(0, data.length, 1);
        for (var t = 40; t < 50; t++) {
          observed[t] = 0;
        }
        final steps = Float64List.fromList([
          for (var t = 0; t < data.length; t++) t.isEven ? 1.0 : 0.5,
        ]);
        final subjects = [
          Subject(inputs: data, responses: sim),
          Subject(
            inputs: data,
            responses: sim,
            observed: observed,
            timeSteps: steps,
          ),
        ];
        _checkGradients(model, subjects, [
          [-3.0, -4.0, 1.0, 4.0, 2.0],
          [-2.0, -5.0, 0.5, 2.0, 0.5],
        ], tol: 2e-6);
      }
    });

    test('perceptual fit: observation surprise, value coupling, prior '
        'mean', () {
      final data = _walk(200, math.Random(5));
      final model = HgfModel(
        network: Network(volatilityUpdates: VolatilityUpdates.eHGF)
          ..addNodes(precision: 4)
          ..addNodes(valueChildren: [0], tonicVolatility: -3)
          ..addNodes(volatilityChildren: [1], tonicVolatility: -4),
        response: const ObservationSurprise(TotalGaussianSurprise()),
        parameters: [
          Parameter.node(
            'omega2',
            1,
            NodeField.tonicVolatility,
            prior: const Normal(-3, 2),
          ),
          Parameter.valueCoupling('alpha', 1, 0, prior: const Normal(1, 0.3)),
          Parameter.node('mu2', 1, NodeField.mean, prior: const Normal(0, 1)),
        ],
      );
      _checkGradients(
        model,
        [Subject(inputs: data)],
        [
          [-3.0, 1.0, 0.0],
          [-2.0, 0.8, 0.5],
        ],
      );
    });

    test('categorical (3 categories): choices and observation surprise', () {
      // Nodes: 0 categorical, 1-3 binary, 4-6 their logits, 7 the shared
      // volatility parent.
      Network net() => Network()
        ..addCategoricalState(nCategories: 3, volatilityTonicVolatility: -6);
      // Drifting category probabilities; one-hot rows, a few missing.
      final rng = math.Random(9);
      const nTime = 160;
      final picks = [
        for (var t = 0; t < nTime; t++)
          () {
            final likely = (t ~/ 40) % 3;
            final r = rng.nextDouble();
            return r < 0.7 ? likely : (likely + 1 + (r < 0.85 ? 0 : 1)) % 3;
          }(),
      ];
      final u = Float64List.fromList([
        for (final k in picks)
          for (var c = 0; c < 3; c++) c == k ? 1.0 : 0.0,
      ]);
      final observed = Float64List.fromList([
        for (var t = 0; t < nTime; t++) t % 37 == 5 ? 0.0 : 1.0,
      ]);
      final logits = Parameter(
        'omega_logits',
        targets: [
          for (final i in [4, 5, 6]) NodeTarget(i, NodeField.tonicVolatility),
        ],
        prior: const Normal(-2, 2),
      );
      final volatility = Parameter.node(
        'omega_vol',
        7,
        NodeField.tonicVolatility,
        prior: const Normal(-6, 2),
      );
      final choices = HgfModel(
        network: net(),
        response: const CategoricalSoftmaxReadout(),
        parameters: [
          logits,
          volatility,
          Parameter.response(
            'inverseTemperature',
            prior: const LogNormal(0.7, 1),
          ),
        ],
      );
      expect(choices.hasExactGradients, isTrue);
      final r = choices
          .simulate(
            [Subject(inputs: u)],
            [
              [-2.0, -6.0, 3.0],
            ],
            math.Random(2),
          )
          .single;
      // Choices are category indices; a sharp agent mostly picks the most
      // expected one.
      expect(r.every((v) => v == 0 || v == 1 || v == 2), isTrue);
      final rMissed = Float64List.fromList(r)..[7] = double.nan;
      _checkGradients(
        choices,
        [
          Subject(inputs: u, responses: r),
          Subject(inputs: u, responses: rMissed, observed: observed),
        ],
        // Not ω ≈ −1: the gradient there is ~3e3 and so curved that the
        // h = 1e-4 differences are off by 1.4e-6 (relative; at h = 1e-5
        // they match the exact gradient to 1e-10).
        [
          [-2.0, -6.0, 3.0],
          [-1.5, -5.0, 0.7],
        ],
      );
      final perceptual = HgfModel(
        network: net(),
        response: const ObservationSurprise(CategoricalSurprise()),
        parameters: [logits, volatility],
      );
      _checkGradients(
        perceptual,
        [Subject(inputs: u)],
        [
          [-2.0, -6.0],
          [-3.0, -4.5],
        ],
      );
    });

    test('a parameter whose target a later one overwrites gets only its '
        'prior gradient', () {
      final model = HgfModel(
        network: Network()
          ..addNodes(kind: NodeKind.binaryState)
          ..addNodes(valueChildren: [0]),
        response: const BinarySoftmaxReadout(),
        parameters: [
          Parameter.node(
            'shadowed',
            1,
            NodeField.tonicVolatility,
            prior: const Normal(0, 1),
          ),
          Parameter.node(
            'omega2',
            1,
            NodeField.tonicVolatility,
            prior: const Normal(-3, 2),
          ),
        ],
      );
      final r = model.simulate(
        [Subject(inputs: u)],
        [
          [0.0, -3.0],
        ],
        math.Random(2),
      )[0];
      final s = [Subject(inputs: u, responses: r)];
      _checkGradients(model, s, [
        [0.5, -3.0],
      ]);
      final out = Float64List(1), g = Float64List(2);
      model.negLogPosterior(
        s,
        Int32List(1),
        model.toUnconstrained([0.5, -3.0]),
        out,
        gradient: g,
      );
      expect(g[0], closeTo(0.5, 1e-12)); // −d log N(0.5; 0, 1)
    });

    test('ResponseModel.fn with a tangent form', () {
      // Decisions with a lapse rate: p = ε / 2 + (1 − ε) μ̂.
      final lapse = ResponseModel.fn(
        parameters: const {'lapse': 0.1},
        surprise: (net, e, t, r, params, o) {
          final p =
              0.5 * params[o] +
              (1 - params[o]) * net.get(e, 0, NodeField.expectedMean);
          return binarySurprise(r, p);
        },
        surprisePartials: (net, e, t, r, params, o, partials) {
          final eps = params[o], m = net.get(e, 0, NodeField.expectedMean);
          final p = 0.5 * eps + (1 - eps) * m;
          final ds = binarySurpriseDerivative(r, p);
          partials
            ..add(0, NodeField.expectedMean, ds * (1 - eps))
            ..addParameter(0, ds * (0.5 - m));
          return binarySurprise(r, p);
        },
      );
      final model = HgfModel(
        network: Network()
          ..addNodes(kind: NodeKind.binaryState)
          ..addNodes(valueChildren: [0]),
        response: lapse,
        parameters: [
          Parameter.node(
            'omega2',
            1,
            NodeField.tonicVolatility,
            prior: const Normal(-3, 2),
          ),
          Parameter.response('lapse', prior: const Uniform(0, 0.5)),
        ],
      );
      final responses = Float64List.fromList([
        for (var t = 0; t < u.length; t++) t % 7 == 0 ? 1 - u[t] : u[t],
      ]);
      _checkGradients(
        model,
        [Subject(inputs: u, responses: responses)],
        [
          [-3.0, 0.1],
          [-2.0, 0.3],
        ],
      );
    });
  });

  group('without exact gradients', () {
    final noTangents = ResponseModel.fn(
      surprise: (net, e, t, r, params, o) =>
          binarySurprise(r, net.get(e, 0, NodeField.expectedMean)),
    );
    final model = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0]),
      response: noTangents,
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-3, 2),
        ),
      ],
    );

    test('fall back to finite differences', () {
      expect(model.hasExactGradients, isFalse);
      final s = [Subject(inputs: u, responses: u)];
      expect(BatchedObjective(model, s).gradients, Gradients.central);
      expect(
        () => BatchedObjective(model, s, gradients: Gradients.exact),
        throwsArgumentError,
      );
      expect(
        () => model.negLogPosterior(
          s,
          Int32List(1),
          Float64List(1),
          Float64List(1),
          gradient: Float64List(1),
        ),
        throwsUnsupportedError,
      );
    });

    test('a node field without tangents', () {
      final m = HgfModel(
        network: Network()
          ..addNodes(kind: NodeKind.binaryState)
          ..addNodes(valueChildren: [0]),
        response: const BinarySoftmaxReadout(),
        parameters: [
          Parameter.node(
            'observed',
            1,
            NodeField.observed,
            prior: const Uniform(0, 1),
          ),
        ],
      );
      expect(m.hasExactGradients, isFalse);
    });
  });

  test('fitMap with exact and finite-difference gradients agree', () {
    final model = HgfModel(
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
        Parameter.response('inverseTemperature', prior: const Uniform(0.2, 20)),
      ],
    );
    final truth = [
      for (var i = 0; i < 6; i++) [-5.0 + 0.6 * i, 1.0 + 0.8 * i],
    ];
    final responses = model.simulate(
      [for (final _ in truth) Subject(inputs: u)],
      truth,
      math.Random(7),
    );
    final subjects = [
      for (final r in responses) Subject(inputs: u, responses: r),
    ];
    final exact = fitMap(model, subjects);
    final central = fitMap(model, subjects, gradients: Gradients.central);
    expect(exact.elements, lessThan(central.elements));
    for (var s = 0; s < subjects.length; s++) {
      final a = exact.results[s], b = central.results[s];
      expect(a.converged, isTrue);
      expect(a.negLogPosterior, lessThanOrEqualTo(b.negLogPosterior + 1e-6));
      for (var i = 0; i < 2; i++) {
        expect(a.x[i], closeTo(b.x[i], 1e-3 * (1 + b.x[i].abs())));
      }
    }
  });
}
