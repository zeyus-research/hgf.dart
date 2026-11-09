import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_deep/hgf_deep.dart';
import 'package:test/test.dart';

/// Elementwise equality up to a few ulps (width 1 and wider batches sum in
/// different orders).
Matcher _ulps(List<double> expected) => pairwiseCompare<double, double>(
  expected,
  (e, a) => (e - a).abs() <= 1e-14 * (1 + e.abs()),
  'within 1e-14',
);

Matrix _randn(int rows, int cols, math.Random rng) => Matrix(
  rows,
  cols,
  Float64List.fromList([
    for (var k = 0; k < rows * cols; k++) rng.nextDouble() * 2 - 1,
  ]),
);

DeepNetwork _net({Coupling coupling = CouplingFn.tanh}) =>
    DeepNetwork(coupling: coupling)
      ..addLayer(2)
      ..addLayer(4)
      ..addLayer(3)
      ..weightInitialisation(const XavierInit(), seed: 3);

void main() {
  group('building', () {
    test('weights are child × (size + bias), unit by default', () {
      final net = DeepNetwork()
        ..addLayer(2)
        ..addLayer(3)
        ..addLayer(4, addConstantInput: false);
      expect(net.layerSizes, [2, 3, 4]);
      expect(net.weights[0], isNull);
      expect((net.weights[1]!.rows, net.weights[1]!.cols), (2, 4));
      expect((net.weights[2]!.rows, net.weights[2]!.cols), (3, 4));
      expect(net.weights[1]!.data.every((w) => w == 1.0), isTrue);
    });

    test('one-to-one layers are the identity and need matching sizes', () {
      final net = DeepNetwork()
        ..addLayer(3)
        ..addLayer(3, fullyConnected: false, addConstantInput: false);
      expect(net.weights[1]!.data, Matrix.identity(3).data);
      expect(
        () => DeepNetwork()
          ..addLayer(3)
          ..addLayer(2, fullyConnected: false, addConstantInput: false),
        throwsArgumentError,
      );
      expect(
        () => DeepNetwork()
          ..addLayer(3)
          ..addLayer(3, fullyConnected: false),
        throwsArgumentError,
      );
    });

    test('categorical only at the bottom', () {
      expect(
        () => DeepNetwork()
          ..addLayer(2)
          ..addLayer(3, kind: LayerKind.categorical),
        throwsArgumentError,
      );
      expect(
        () => DeepNetwork()..addLayer(3, kind: LayerKind.categorical),
        returnsNormally,
      );
    });

    test('volatility parents follow carryPrecision', () {
      final carried = DeepNetwork()..addLayer(2);
      expect(carried.state(0).hasVolatility, isTrue);
      final fresh = DeepNetwork(carryPrecision: false)
        ..addLayer(2)
        ..addLayer(2, kind: LayerKind.binary);
      expect(fresh.state(0).hasVolatility, isFalse);
      expect(fresh.state(1).hasVolatility, isTrue);
      expect(
        () =>
            DeepNetwork(carryPrecision: false)
              ..addLayer(2, volatilityParent: true),
        throwsArgumentError,
      );
    });

    test('initial overrides are applied (volatility ones dropped without a '
        'volatility level)', () {
      final net = DeepNetwork()
        ..addLayer(
          2,
          volatilityParent: false,
          initial: {LayerField.precision: 5.0, LayerField.meanVol: 3.0},
        );
      expect(net.state(0).precision, [5.0, 5.0]);
      expect(net.state(0).meanVol, isNull);
    });

    test('tonic volatility needs the network flag', () {
      expect(
        () => DeepNetwork()..addLayer(2, tonicVolatility: -2.0),
        throwsArgumentError,
      );
      expect(
        () =>
            DeepNetwork(tonicVolatility: true)
              ..addLayer(2, tonicVolatility: -2),
        returnsNormally,
      );
    });

    test('mean-field and feedforward uncertainty exclude each other', () {
      expect(
        () => DeepNetwork(meanFieldUpdates: true, feedforwardUncertainty: true),
        throwsArgumentError,
      );
    });

    test('setWeights checks shapes and copies', () {
      final net = _net();
      final w = [Matrix(2, 5), Matrix(4, 4)];
      net.setWeights(w);
      w[0].data[0] = 9;
      expect(net.weights[1]!.data[0], 0.0);
      expect(
        () => net.setWeights([Matrix(2, 4), Matrix(4, 4)]),
        throwsArgumentError,
      );
    });
  });

  group('weight initialisation', () {
    test('bias column zero, deterministic per seed, layers independent', () {
      final a = DeepNetwork()
        ..addLayer(3)
        ..addLayer(3)
        ..addLayer(3)
        ..weightInitialisation(const HeInit(), seed: 7);
      final b = DeepNetwork()
        ..addLayer(3)
        ..addLayer(3)
        ..addLayer(3)
        ..weightInitialisation(const HeInit(), seed: 7);
      for (var l = 1; l < 3; l++) {
        expect(a.weights[l]!.data, b.weights[l]!.data);
        for (var i = 0; i < 3; i++) {
          expect(a.weights[l]!.at(i, 3), 0.0);
        }
      }
      expect(a.weights[1]!.data, isNot(a.weights[2]!.data));
    });

    test('orthogonal columns (tall) or rows (wide)', () {
      final rng = math.Random(1);
      for (final (p, c) in [(3, 5), (5, 3)]) {
        final w = const OrthogonalInit().draw(p, c, rng);
        final tall = c >= p;
        final n = tall ? p : c;
        for (var a = 0; a < n; a++) {
          for (var b = 0; b < n; b++) {
            var dot = 0.0;
            for (var k = 0; k < (tall ? c : p); k++) {
              dot += tall
                  ? w[k * p + a] * w[k * p + b]
                  : w[a * p + k] * w[b * p + k];
            }
            expect(dot, closeTo(a == b ? 1.0 : 0.0, 1e-12));
          }
        }
      }
    });

    test('sparse keeps the requested fraction', () {
      final w = const SparseInit(sparsity: 0.75).draw(8, 10, math.Random(2));
      expect(w.where((v) => v != 0).length, 20);
    });

    test('xavier stays within its limit', () {
      final w = const XavierInit().draw(4, 6, math.Random(3));
      final limit = math.sqrt(6 / 10);
      expect(w.every((v) => v.abs() <= limit), isTrue);
    });
  });

  group('optimisers', () {
    test('SGD steps against the gradient', () {
      final w = [null, Matrix.filled(1, 2, 1.0)];
      const Sgd(0.1).step(w, [
        null,
        Float64List.fromList([2.0, -1.0]),
      ]);
      expect(w[1]!.data, [closeTo(0.8, 1e-15), closeTo(1.1, 1e-15)]);
    });

    test('the first Adam step has size lr', () {
      final w = [Matrix.filled(1, 1, 1.0)];
      final adam = Adam(0.01);
      adam.step(w, [
        Float64List.fromList([3.0]),
      ]);
      expect(w[0].data[0], closeTo(0.99, 1e-9));
      expect(adam.count, 1);
    });

    test('an Adam instance continues across fits; a new one restarts', () {
      final rng = math.Random(5);
      final x = _randn(20, 3, rng);
      final y = _randn(20, 2, rng);
      final whole = _net();
      whole.fit(x, y, optimiser: Adam(0.01));
      final split = _net();
      final adam = Adam(0.01);
      split.fit(
        Matrix(10, 3, x.data.sublist(0, 30)),
        Matrix(10, 2, y.data.sublist(0, 20)),
        optimiser: adam,
      );
      split.fit(
        Matrix(10, 3, x.data.sublist(30)),
        Matrix(10, 2, y.data.sublist(20)),
        optimiser: adam,
      );
      expect(split.weights[1]!.data, whole.weights[1]!.data);
      final restarted = _net();
      restarted.fit(
        Matrix(10, 3, x.data.sublist(0, 30)),
        Matrix(10, 2, y.data.sublist(0, 20)),
        optimiser: Adam(0.01),
      );
      restarted.fit(
        Matrix(10, 3, x.data.sublist(30)),
        Matrix(10, 2, y.data.sublist(20)),
        optimiser: Adam(0.01),
      );
      expect(restarted.weights[1]!.data, isNot(whole.weights[1]!.data));
    });
  });

  group('learning', () {
    test('fit reduces the error on a learnable target', () {
      final rng = math.Random(11);
      final x = _randn(400, 3, rng);
      final y = Matrix(400, 2);
      for (var s = 0; s < 400; s++) {
        final a = x.at(s, 0), b = x.at(s, 1), c = x.at(s, 2);
        y.set(s, 0, 0.5 * a - 0.3 * b + 0.2);
        y.set(s, 1, -0.4 * c + 0.1 * a);
      }
      final net = DeepNetwork(volatilityUpdates: DeepVolatilityUpdate.standard)
        ..addLayer(2)
        ..addLayer(3, addConstantInput: false)
        ..weightInitialisation(const XavierInit(), seed: 1);
      double mse(Matrix p, int from, int to) {
        var sum = 0.0;
        for (var s = from; s < to; s++) {
          for (var j = 0; j < 2; j++) {
            final d = p.at(s, j) - y.at(s, j);
            sum += d * d;
          }
        }
        return sum / (to - from);
      }

      final pred = net.fit(x, y, optimiser: Adam(0.02));
      expect(mse(pred, 350, 400), lessThan(mse(pred, 0, 50) * 0.5));
    });

    test('predict matches the fit-time prediction of the same network', () {
      final rng = math.Random(4);
      final x = _randn(5, 3, rng);
      final net = _net();
      final frozen = net.fit(x, _randn(5, 2, rng), weightUpdate: false);
      expect(net.predict(x).data, _ulps(frozen.data));
    });

    test('a one-sample batch makes the same weight step as one fit step', () {
      final rng = math.Random(6);
      final x = _randn(1, 3, rng);
      final y = _randn(1, 2, rng);
      final seq = _net();
      final bat = _net();
      seq.fit(x, y, optimiser: const Sgd(0.05));
      bat.batchUpdate(x, y, optimiser: const Sgd(0.05));
      for (var l = 1; l < 3; l++) {
        expect(bat.weights[l]!.data, _ulps(seq.weights[l]!.data));
      }
      expect(bat.state(1).precision, _ulps(seq.state(1).precision));
    });

    test('batch input errors equal one width-1 sweep per sample', () {
      final rng = math.Random(8);
      final x = _randn(6, 3, rng);
      final y = _randn(6, 2, rng);
      final batch = _net().batchUpdate(x, y);
      for (var s = 0; s < 6; s++) {
        final single = _net().batchUpdate(
          Matrix(1, 3, x.row(s)),
          Matrix(1, 2, y.row(s)),
        );
        for (var j = 0; j < 3; j++) {
          expect(batch.at(s, j), closeTo(single.at(0, j), 1e-14));
        }
      }
    });

    test('updatePrecisions: false keeps the carried beliefs', () {
      final rng = math.Random(9);
      final net = _net();
      final before = net.state(1).precision.toList();
      net.fit(
        _randn(10, 3, rng),
        _randn(10, 2, rng),
        optimiser: Adam(0.01),
        updatePrecisions: false,
      );
      expect(net.state(1).precision, before);
    });

    test('recorded trajectories hold every step', () {
      final rng = math.Random(10);
      final net = _net();
      final pred = net.fit(
        _randn(7, 3, rng),
        _randn(7, 2, rng),
        optimiser: Adam(0.01),
        record: {LayerField.expectedMean, LayerField.meanVol},
      );
      final traj = net.trajectories!;
      expect(traj.steps, 7);
      expect(traj.of(LayerField.expectedMean, 0), pred.data);
      expect(traj.of(LayerField.meanVol, 1)!.length, 7 * 4);
    });

    test('binary and categorical outputs are probabilities', () {
      final rng = math.Random(12);
      final binary = DeepNetwork()
        ..addLayer(3, kind: LayerKind.binary)
        ..addLayer(2)
        ..weightInitialisation(const HeInit(), seed: 2);
      final p = binary.predict(_randn(10, 2, rng));
      expect(p.data.every((v) => v > 0 && v < 1), isTrue);
      final cat = DeepNetwork()
        ..addLayer(4, kind: LayerKind.categorical)
        ..addLayer(2)
        ..weightInitialisation(const HeInit(), seed: 2);
      final q = cat.predict(_randn(10, 2, rng));
      for (var s = 0; s < 10; s++) {
        var sum = 0.0;
        for (var j = 0; j < 4; j++) {
          sum += q.at(s, j);
        }
        expect(sum, closeTo(1.0, 1e-12));
      }
    });

    test('fit validates its inputs', () {
      final net = _net();
      expect(() => net.fit(Matrix(2, 3), Matrix(2, 2)), throwsArgumentError);
      expect(
        () => net.fit(Matrix(2, 2), Matrix(2, 2), optimiser: Adam(0.1)),
        throwsArgumentError,
      );
      expect(
        () => net.fit(Matrix(2, 3), Matrix(3, 2), optimiser: Adam(0.1)),
        throwsArgumentError,
      );
    });
  });
}
