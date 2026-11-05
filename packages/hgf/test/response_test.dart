import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

Network binary2() => Network()
  ..addNodes(kind: NodeKind.binaryState)
  ..addNodes(valueChildren: [0], tonicVolatility: -3.0);

Network continuous2() => Network()
  ..addNodes(precision: 1e4)
  ..addNodes(valueChildren: [0], mean: 1.0, tonicVolatility: -8.0)
  ..addNodes(volatilityChildren: [1], tonicVolatility: -2.0);

final u = Float64List.fromList([
  for (var t = 0; t < 120; t++) ((t * 7) % 11 < 6) ? 1.0 : 0.0,
]);
final y = Float64List.fromList([
  for (var t = 0; t < 120; t++) 1.0 + 0.01 * (t % 17) - 0.004 * (t % 5),
]);

void main() {
  group('surprise math (pyhgf docstring values)', () {
    test('binarySurprise', () {
      expect(binarySurprise(1.0, 0.7), closeTo(0.35667497, 1e-7));
      expect(binarySurprise(0.0, 0.7), closeTo(-math.log(0.3), 1e-12));
      // Clipped to [1e-6, 1 - 1e-6] unless disabled.
      expect(binarySurprise(1.0, 0.0), closeTo(-math.log(1e-6), 1e-9));
      expect(binarySurprise(1.0, 0.0, clipping: false), double.infinity);
    });

    test('gaussianSurprise', () {
      expect(gaussianSurprise(2.0, 0.0, 1.0), closeTo(2.9189386, 1e-7));
      // It is the negative log of the normal density.
      const x = 0.3, m = -0.2, p = 4.0;
      final logPdf =
          0.5 * math.log(p / (2 * math.pi)) - 0.5 * p * (x - m) * (x - m);
      expect(gaussianSurprise(x, m, p), closeTo(-logPdf, 1e-12));
    });

    test('sigmoidInverseTemperature', () {
      expect(sigmoidInverseTemperature(0.7, 1.0), closeTo(0.7, 1e-15));
      expect(sigmoidInverseTemperature(0.5, 7.0), closeTo(0.5, 1e-15));
      expect(sigmoidInverseTemperature(0.7, 3.0), greaterThan(0.7));
      expect(sigmoidInverseTemperature(0.7, 0.5), lessThan(0.7));
    });

    test('binarySurpriseFinitePrecision', () {
      // With very precise observations it reduces to the binary surprise
      // (up to the log of pyhgf's density normalisation, π/√(2π)).
      const p = 1e3;
      final norm = math.log(p / math.sqrt(2 * math.pi));
      expect(
        binarySurpriseFinitePrecision(1.0, 0.7, p) + norm,
        closeTo(binarySurprise(1.0, 0.7), 1e-9),
      );
    });
  });

  group('Network.surprise', () {
    test('total equals the sum of per-trial values and of inputData', () {
      final net = continuous2();
      final perTrial = Float64List(y.length);
      final total = net.surprise(
        y,
        const FirstLevelGaussianSurprise(),
        perTrial: perTrial,
      );
      expect(total, closeTo(perTrial.reduce((a, b) => a + b), 1e-9));
      final traj = net.inputData(
        y,
        response: const FirstLevelGaussianSurprise(),
      )!;
      expect(traj.responseSurprise, perTrial);
    });

    test('matches step-by-step use of the response function', () {
      final net = binary2();
      final responses = Float64List.fromList([
        for (var t = 0; t < u.length; t++) u[(t + 1) % u.length],
      ]);
      final rf = BinarySoftmax(responses);
      final total = net.surprise(u, rf);

      net.reset();
      rf.prepare(net, u.length);
      var stepped = 0.0;
      for (var t = 0; t < u.length; t++) {
        net.step(Float64List.fromList([u[t]]));
        stepped += rf.surprise(net, t);
      }
      expect(stepped, total);
    });

    test('ResponseFunction.fn sees the trial index and the state', () {
      final net = binary2();
      final seen = <int>[];
      final rf = ResponseFunction.fn((n, t) {
        seen.add(t);
        return binarySurprise(u[t], n.get(0, NodeField.expectedMean));
      });
      final custom = net.surprise(u, rf);
      final builtin = net.surprise(u, const FirstLevelBinarySurprise());
      expect(seen, List.generate(u.length, (t) => t));
      expect(custom, closeTo(builtin, 1e-12));
    });

    test('BinarySoftmaxInverseTemperature with β = 1 is BinarySoftmax', () {
      final net = binary2();
      final r = Float64List.fromList([for (final x in u) 1 - x]);
      expect(
        net.surprise(
          u,
          BinarySoftmaxInverseTemperature(r, inverseTemperature: 1),
        ),
        closeTo(net.surprise(u, BinarySoftmax(r)), 1e-9),
      );
    });

    test('NaN responses are missed trials', () {
      final net = binary2();
      final r = Float64List.fromList(u);
      final full = Float64List(u.length);
      net.surprise(u, BinarySoftmax(r), perTrial: full);
      r[3] = double.nan;
      r[10] = double.nan;
      final missed = Float64List(u.length);
      final total = net.surprise(u, BinarySoftmax(r), perTrial: missed);
      expect(missed[3], 0.0);
      expect(missed[10], 0.0);
      expect(
        total,
        closeTo(full.reduce((a, b) => a + b) - full[3] - full[10], 1e-9),
      );
    });
  });

  group('GaussianResponse', () {
    test('is the Gaussian surprise of responses around the belief', () {
      final net = continuous2();
      final traj = net.inputData(y)!;
      final belief = traj.series(0, NodeField.expectedMean);
      final responses = Float64List.fromList([
        for (var t = 0; t < y.length; t++) belief[t] + 0.05 * math.sin(t * 1.0),
      ]);
      const zeta = 50.0;
      var expected = 0.0;
      for (var t = 0; t < y.length; t++) {
        expected += gaussianSurprise(responses[t], belief[t], zeta);
      }
      expect(
        net.surprise(y, GaussianResponse(responses, precision: zeta)),
        closeTo(expected, 1e-9),
      );
    });

    test('can read a posterior of another node', () {
      final net = continuous2();
      final traj = net.inputData(y)!;
      final mu1 = traj.series(1, NodeField.mean);
      final rf = GaussianResponse(mu1, node: 1, field: NodeField.mean);
      // Responses equal to the belief: only the normalising term is left.
      expect(
        net.surprise(y, rf),
        closeTo(y.length * 0.5 * math.log(2 * math.pi), 1e-9),
      );
    });

    test('is minimised near the true response precision', () {
      final net = continuous2();
      final belief = net.inputData(y)!.series(0, NodeField.expectedMean);
      final rng = math.Random(7);
      double gauss() =>
          math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
          math.cos(2 * math.pi * rng.nextDouble());
      const sd = 0.1;
      final responses = Float64List.fromList([
        for (var t = 0; t < y.length; t++) belief[t] + sd * gauss(),
      ]);
      double nll(double p) =>
          net.surprise(y, GaussianResponse(responses, precision: p));
      final truth = 1 / (sd * sd);
      expect(nll(truth), lessThan(nll(truth / 4)));
      expect(nll(truth), lessThan(nll(truth * 4)));
    });

    test('rejects a non-positive precision', () {
      expect(
        () => GaussianResponse(Float64List(1), precision: 0),
        throwsArgumentError,
      );
    });
  });

  group('prepare() checks', () {
    test('too few responses', () {
      expect(
        () => binary2().surprise(u, BinarySoftmax(Float64List(3))),
        throwsArgumentError,
      );
    });

    test('wrong node kinds', () {
      expect(
        () => binary2().surprise(u, const FirstLevelGaussianSurprise()),
        throwsArgumentError,
      );
      expect(
        () => continuous2().surprise(y, const FirstLevelBinarySurprise()),
        throwsArgumentError,
      );
    });

    test('missing node', () {
      expect(
        () => binary2().surprise(u, BinarySoftmax(u, node: 9)),
        throwsArgumentError,
      );
    });

    test('perTrial length', () {
      expect(
        () => binary2().surprise(
          u,
          const FirstLevelBinarySurprise(),
          perTrial: Float64List(2),
        ),
        throwsArgumentError,
      );
    });
  });

  group('Trajectories surprise', () {
    test('nodeSurprise and totalSurprise', () {
      final traj = binary2().inputData(u)!;
      final s0 = traj.nodeSurprise(0);
      final s1 = traj.nodeSurprise(1);
      for (var t = 0; t < u.length; t++) {
        expect(
          s0[t],
          binarySurprise(u[t], traj.at(0, NodeField.expectedMean, t)),
        );
      }
      final total = traj.totalSurprise();
      for (var t = 0; t < u.length; t++) {
        expect(total[t], closeTo(s0[t] + s1[t], 1e-12));
      }
    });

    test('needs the fields it reads', () {
      final traj = binary2().inputData(u, fields: const [NodeField.mean])!;
      expect(() => traj.nodeSurprise(0), throwsArgumentError);
    });
  });
}
