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

Network _binary2() => Network()
  ..addNodes(kind: NodeKind.binaryState)
  ..addNodes(valueChildren: [0]);

Network _continuous2() => Network()
  ..addNodes(precision: 4)
  ..addNodes(valueChildren: [0])
  ..addNodes(volatilityChildren: [1]);

HgfModel _binaryModel() => HgfModel(
  network: _binary2(),
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

/// A random walk whose step size switches between calm and volatile phases,
/// observed with noise (sd 0.5, matching the input precision of 4 in
/// [_continuous2]). With nearly noiseless inputs the beliefs follow the
/// data whatever ω is, and ω can't be recovered.
Float64List _walk(int n, math.Random rng) {
  final out = Float64List(n);
  var x = 0.0;
  for (var t = 0; t < n; t++) {
    final sd = (t ~/ 100).isEven ? 0.05 : 0.4;
    x += sd * _normal(rng);
    out[t] = x + 0.5 * _normal(rng);
  }
  return out;
}

double _normal(math.Random rng) =>
    math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
    math.cos(2 * math.pi * rng.nextDouble());

double _corr(List<double> a, List<double> b) {
  final n = a.length;
  final ma = a.reduce((x, y) => x + y) / n, mb = b.reduce((x, y) => x + y) / n;
  var sab = 0.0, saa = 0.0, sbb = 0.0;
  for (var i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) * (a[i] - ma);
    sbb += (b[i] - mb) * (b[i] - mb);
  }
  return sab / math.sqrt(saa * sbb);
}

void main() {
  final u = _binaryInput();

  group('negLogPosterior', () {
    test('matches Network.surprise plus the priors, bit for bit', () {
      final model = _binaryModel();
      final rng = math.Random(3);
      final responses = Float64List.fromList([
        for (var t = 0; t < u.length; t++) rng.nextBool() ? 1.0 : 0.0,
      ]);
      final subject = Subject(inputs: u, responses: responses);
      for (final (omega, beta) in [(-4.0, 1.0), (-2.5, 3.2), (-6.0, 0.4)]) {
        final net = _binary2()..setInitial(1, NodeField.tonicVolatility, omega);
        final nll = net.surprise(
          u,
          BinarySoftmaxInverseTemperature(responses, inverseTemperature: beta),
        );
        final expected =
            nll -
            model.parameters[0].prior.logDensity(omega) -
            model.parameters[1].prior.logDensity(beta);
        final y = model.toUnconstrained([omega, beta]);
        final x = model.toConstrained(y);
        final out = Float64List(1), parts = Float64List(1);
        model.negLogPosterior(
          [subject],
          Int32List(1),
          y,
          out,
          negLogLikelihood: parts,
        );
        // The transform round trip may move x by an ulp; compare at x.
        final net2 = _binary2()..setInitial(1, NodeField.tonicVolatility, x[0]);
        expect(
          parts[0],
          net2.surprise(
            u,
            BinarySoftmaxInverseTemperature(
              responses,
              inverseTemperature: x[1],
            ),
          ),
        );
        expect(out[0], closeTo(expected, 1e-9 * expected.abs()));
        // negLogPosteriorAt maps x back to y, another round trip.
        expect(
          model.negLogPosteriorAt(subject, x),
          closeTo(out[0], 1e-12 * out[0].abs()),
        );
      }
    });

    test('Gaussian readout matches GaussianResponse', () {
      final data = _walk(300, math.Random(5));
      final responses = Float64List.fromList([for (final v in data) v + 0.3])
        ..[17] = double.nan;
      final model = HgfModel(
        network: _continuous2(),
        response: const GaussianReadout(node: 1),
        parameters: [
          Parameter.response('precision', prior: const GammaPrior(2, 1)),
        ],
      );
      final out = Float64List(1), nll = Float64List(1);
      final y = model.toUnconstrained([2.5]);
      model.negLogPosterior(
        [Subject(inputs: data, responses: responses)],
        Int32List(1),
        y,
        out,
        negLogLikelihood: nll,
      );
      final p = model.toConstrained(y)[0];
      expect(
        nll[0],
        _continuous2().surprise(
          data,
          GaussianResponse(responses, precision: p, node: 1),
        ),
      );
    });

    test('per-element data, groups and coupling targets agree with '
        'one-at-a-time evaluation', () {
      final rng = math.Random(9);
      final model = HgfModel(
        network: _continuous2(),
        response: const ObservationSurprise(TotalGaussianSurprise()),
        parameters: [
          Parameter.node(
            'omega2',
            2,
            NodeField.tonicVolatility,
            prior: const Normal(-4, 2),
          ),
          Parameter.volatilityCoupling(
            'kappa',
            2,
            1,
            prior: const LogNormal(0, 0.5),
          ),
          Parameter.node(
            'pi_u',
            0,
            NodeField.precision,
            prior: const LogNormal(4, 1),
          ),
        ],
      );
      final steps = Float64List.fromList([
        for (var t = 0; t < 150; t++) 0.5 + rng.nextDouble(),
      ]);
      final subjects = [
        Subject(inputs: _walk(200, rng)),
        Subject(inputs: _walk(200, rng)),
        Subject(inputs: _walk(150, rng), timeSteps: steps),
        Subject(
          inputs: _walk(200, rng),
          observed: Float64List.fromList([
            for (var t = 0; t < 200; t++) t % 7 == 3 ? 0.0 : 1.0,
          ]),
        ),
      ];
      final subjectOf = Int32List.fromList([0, 1, 2, 3, 1, 3, 2, 0]);
      final y = Float64List(subjectOf.length * 3);
      for (var i = 0; i < y.length; i++) {
        y[i] = 0.3 * _normal(rng);
      }
      final out = Float64List(subjectOf.length);
      model.negLogPosterior(subjects, subjectOf, y, out);
      for (var e = 0; e < subjectOf.length; e++) {
        final x = model.toConstrained(y.sublist(e * 3, e * 3 + 3));
        final s = subjects[subjectOf[e]];
        final net = _continuous2()
          ..setInitial(2, NodeField.tonicVolatility, x[0])
          ..setVolatilityCoupling(2, 1, x[1])
          ..setInitial(0, NodeField.precision, x[2]);
        final nll = net.surprise(
          s.inputs,
          const TotalGaussianSurprise(),
          observed: s.observed,
          timeSteps: s.timeSteps,
        );
        expect(out[e], closeTo(nll - model.logPrior(x), 1e-9), reason: '$e');
      }
    });

    test('jacobian adds the transforms\' log-Jacobian', () {
      final model = _binaryModel();
      final s = Subject(inputs: u, responses: u);
      final y = model.toUnconstrained([-3.0, 2.0]);
      final a = Float64List(1), b = Float64List(1);
      model.negLogPosterior([s], Int32List(1), y, a);
      model.negLogPosterior([s], Int32List(1), y, b, jacobian: true);
      final jac =
          model.parameters[0].transform.logAbsJacobian(y[0]) +
          model.parameters[1].transform.logAbsJacobian(y[1]);
      expect(b[0], closeTo(a[0] - jac, 1e-9));
    });
  });

  group('BatchedObjective', () {
    test('finite-difference gradients match a Richardson estimate', () {
      final model = _binaryModel();
      final responses = model.simulate(
        [Subject(inputs: u)],
        [
          [-3.5, 3.0],
        ],
        math.Random(1),
      )[0];
      final subjects = [Subject(inputs: u, responses: responses)];
      final obj = BatchedObjective(model, subjects);
      final p = model.toUnconstrained([-4.0, 2.0]);
      final (_, grads) = obj.evaluate([0], [p], [true]);
      expect(obj.passes, 1);
      expect(obj.elements, 5);
      double f(Float64List y) => obj.evaluate([0], [y], [false]).$1[0];
      for (var i = 0; i < 2; i++) {
        double d(double h) {
          final a = Float64List.fromList(p)..[i] += h;
          final b = Float64List.fromList(p)..[i] -= h;
          return (f(a) - f(b)) / (2 * h);
        }

        final rich = (4 * d(1e-3) - d(2e-3)) / 3;
        expect(grads[0]![i], closeTo(rich, 1e-5 * (1 + rich.abs())));
      }
    });
  });

  group('validation', () {
    test('unknown response parameter', () {
      expect(
        () => HgfModel(
          network: _binary2(),
          response: const BinarySoftmaxReadout(),
          parameters: [Parameter.response('beta', prior: const Uniform(0, 1))],
        ),
        throwsArgumentError,
      );
    });
    test('duplicate names and bad couplings', () {
      final omega = Parameter.node(
        'w',
        1,
        NodeField.tonicVolatility,
        prior: const Normal(0, 1),
      );
      expect(
        () => HgfModel(
          network: _binary2(),
          response: const BinarySoftmaxReadout(),
          parameters: [omega, omega],
        ),
        throwsArgumentError,
      );
      expect(
        () => HgfModel(
          network: _binary2(),
          response: const BinarySoftmaxReadout(),
          parameters: [
            Parameter.volatilityCoupling('k', 1, 0, prior: const Normal(1, 1)),
          ],
        ),
        throwsArgumentError,
      );
    });
    test('missing responses', () {
      final model = _binaryModel();
      expect(
        () => model.negLogPosteriorAt(Subject(inputs: u), [-3, 1]),
        throwsArgumentError,
      );
      expect(
        () => model.negLogPosteriorAt(
          Subject(inputs: u, responses: Float64List(10)),
          [-3, 1],
        ),
        throwsArgumentError,
      );
    });
    test('initial outside the prior', () {
      expect(
        () =>
            Parameter.response('b', prior: const Uniform(0.2, 20), initial: 0),
        throwsArgumentError,
      );
    });
  });

  group('recovery', () {
    test('binary HGF + softmax (pyhgf notebook 4), L-BFGS and Nelder-Mead '
        'agree', () {
      final model = _binaryModel();
      const n = 20;
      final omegas = [for (var i = 0; i < n; i++) -6.0 + 5.0 * i / (n - 1)];
      final betas = [for (var i = 0; i < n; i++) 0.5 + 5.5 * i / (n - 1)];
      final responses = model.simulate(
        [for (var i = 0; i < n; i++) Subject(inputs: u)],
        [
          for (var i = 0; i < n; i++) [omegas[i], betas[i]],
        ],
        math.Random(123),
      );
      final subjects = [
        for (var i = 0; i < n; i++)
          Subject(inputs: u, responses: responses[i], id: i),
      ];
      final lb = fitMap(model, subjects);
      final nm = fitMap(model, subjects, optimizer: const NelderMead());
      expect(lb.results.every((r) => r.converged), isTrue);
      expect(nm.results.every((r) => r.converged), isTrue);
      // Every subject's optimisers are advanced in lock step: a few dozen
      // passes for all 20, not hundreds each.
      expect(lb.passes, lessThan(40));
      for (var i = 0; i < n; i++) {
        expect(
          lb.results[i].negLogPosterior,
          closeTo(nm.results[i].negLogPosterior, 1e-4),
        );
        expect(lb.results[i].x[0], closeTo(nm.results[i].x[0], 0.02));
      }
      expect(
        _corr(omegas, [for (final r in lb.results) r.x[0]]),
        greaterThan(0.85),
      );
      expect(
        _corr(betas, [for (final r in lb.results) r.x[1]]),
        greaterThan(0.85),
      );
      final r0 = lb.results.first;
      expect(r0.logPrior, closeTo(model.logPrior(r0.x), 1e-9));
    });

    test('continuous HGF + Gaussian readout', () {
      final rng = math.Random(11);
      final model = HgfModel(
        network: _continuous2(),
        response: const GaussianReadout(node: 1),
        parameters: [
          Parameter.node(
            'omega2',
            2,
            NodeField.tonicVolatility,
            prior: const Normal(-4, 3),
          ),
          Parameter.response('precision', prior: const LogNormal(2, 2)),
        ],
      );
      const n = 12;
      final omegas = [for (var i = 0; i < n; i++) -7.0 + 5.0 * i / (n - 1)];
      final precisions = [
        for (var i = 0; i < n; i++) math.exp(1 + 3.0 * ((i * 5) % n) / n),
      ];
      final inputs = [for (var i = 0; i < n; i++) _walk(400, rng)];
      final responses = model.simulate(
        [for (final x in inputs) Subject(inputs: x)],
        [
          for (var i = 0; i < n; i++) [omegas[i], precisions[i]],
        ],
        rng,
      );
      final fit = fitMap(model, [
        for (var i = 0; i < n; i++)
          Subject(inputs: inputs[i], responses: responses[i]),
      ]);
      expect(fit.results.every((r) => r.converged), isTrue);
      expect(
        _corr(omegas, [for (final r in fit.results) r.x[0]]),
        greaterThan(0.8),
      );
      expect(
        _corr(
          [for (final p in precisions) math.log(p)],
          [for (final r in fit.results) math.log(r.x[1])],
        ),
        greaterThan(0.95),
      );
    });

    test('restarts never do worse', () {
      final model = _binaryModel();
      final responses = model.simulate(
        [Subject(inputs: u)],
        [
          [-2.0, 4.0],
        ],
        math.Random(4),
      );
      final s = [Subject(inputs: u, responses: responses[0])];
      final one = fitMap(
        model,
        s,
        start: [
          [-7.5, 15.0],
        ],
      );
      final many = fitMap(
        model,
        s,
        start: [
          [-7.5, 15.0],
        ],
        restarts: 4,
        random: math.Random(2),
      );
      expect(
        many.results[0].negLogPosterior,
        lessThanOrEqualTo(one.results[0].negLogPosterior + 1e-9),
      );
    });

    test('simulate is reproducible', () {
      final model = _binaryModel();
      final a = model.simulate(
        [Subject(inputs: u)],
        [
          [-3.0, 2.0],
        ],
        math.Random(7),
      );
      final b = model.simulate(
        [Subject(inputs: u)],
        [
          [-3.0, 2.0],
        ],
        math.Random(7),
      );
      expect(a[0], b[0]);
      expect(a[0].every((v) => v == 0 || v == 1), isTrue);
    });
  });
}
