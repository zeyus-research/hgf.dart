import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

/// `nTime` one-hot rows over [n] categories, mostly category 0 at first and
/// mostly the last one after the midpoint.
Float64List oneHot(int nTime, int n, {int seed = 3}) {
  final rng = math.Random(seed);
  final out = Float64List(nTime * n);
  for (var t = 0; t < nTime; t++) {
    final favoured = t < nTime ~/ 2 ? 0 : n - 1;
    final k = rng.nextDouble() < 0.6 ? favoured : rng.nextInt(n);
    out[t * n + k] = 1.0;
  }
  return out;
}

void expectSameSeries(
  Trajectories a,
  int nodeA,
  Trajectories b,
  int nodeB,
  String what,
) {
  for (final f in a.fields) {
    expect(
      a.series(nodeA, f),
      b.series(nodeB, f),
      reason: '$what: $f node $nodeA vs $nodeB',
    );
  }
  for (final f in a.vectorFields) {
    if (a.vectorLength(nodeA, f) == 0) continue;
    expect(
      a.vectorSeries(nodeA, f),
      b.vectorSeries(nodeB, f),
      reason: '$what: $f node $nodeA vs $nodeB',
    );
  }
}

const allFields = NodeField.values;

void main() {
  group('construction', () {
    test("2n + 2 nodes in pyhgf's order", () {
      final net = Network()..addCategoricalState(nCategories: 3);
      expect(net.nNodes, 8);
      expect(
        [for (var i = 0; i < 8; i++) net.kindOf(i)],
        [
          NodeKind.categoricalState,
          ...List.filled(3, NodeKind.binaryState),
          ...List.filled(4, NodeKind.continuousState),
        ],
      );
      expect(net.inputIdxs, [0]);
      expect(net.nInputColumns, 3);
      expect(net.vectorLength(0, VectorField.mean), 3);
      expect(net.vectorLength(0, VectorField.alpha), 3);
      expect(net.vectorLength(1, VectorField.alpha), 0);
      expect(net.vectorLength(0, VectorField.xis), 0);
    });

    test("pyhgf's initial beliefs", () {
      final net = Network()..addCategoricalState();
      expect(net.nNodes, 10); // 4 categories by default
      expect(net.getVector(0, VectorField.alpha), [1.0, 1.0, 1.0, 1.0]);
      expect(net.getVector(0, VectorField.mean), List.filled(4, 0.25));
      for (var i = 1; i <= 4; i++) {
        expect(net.get(i, NodeField.mean), 0.25);
        expect(net.get(i, NodeField.precision), 1.0);
      }
      for (var i = 5; i <= 8; i++) {
        expect(net.get(i, NodeField.mean), -math.log(3));
        expect(net.get(i, NodeField.tonicVolatility), -4.0);
      }
      expect(net.get(9, NodeField.mean), 0.0);
      expect(net.get(9, NodeField.tonicVolatility), -4.0);
    });

    test('the implied nodes take parameters', () {
      final net = Network()
        ..addCategoricalState(
          nCategories: 2,
          binaryMean: 0.4,
          binaryPrecision: 2.0,
          valueMean: 0.3,
          valuePrecision: 3.0,
          valueTonicVolatility: -2.0,
          volatilityMean: 0.1,
          volatilityPrecision: 4.0,
          volatilityTonicVolatility: -5.0,
        );
      expect(net.get(1, NodeField.mean), 0.4);
      expect(net.get(2, NodeField.precision), 2.0);
      expect(net.get(3, NodeField.mean), 0.3);
      expect(net.get(4, NodeField.precision), 3.0);
      expect(net.get(4, NodeField.tonicVolatility), -2.0);
      expect(net.get(5, NodeField.mean), 0.1);
      expect(net.get(5, NodeField.precision), 4.0);
      expect(net.get(5, NodeField.tonicVolatility), -5.0);
    });

    test('rejects bad declarations', () {
      expect(
        () => Network().addCategoricalState(nCategories: 1),
        throwsArgumentError,
      );
      expect(
        () => Network().addNodes(kind: NodeKind.categoricalState),
        throwsArgumentError,
      );
      // Edges to a categorical node are set by addCategoricalState only.
      final net = Network()..addCategoricalState(nCategories: 2);
      expect(() => net.addNodes(valueChildren: [0]), throwsArgumentError);
      expect(() => net.addNodes(valueParents: [0]), throwsArgumentError);
      expect(() => net.addNodes(volatilityChildren: [0]), throwsArgumentError);
    });

    test('the categorical posterior runs last, binary nodes get none', () {
      // A continuous HGF first: pyhgf gives the binary nodes the continuous
      // posterior update here and crashes (upstream P7).
      final net = Network()
        ..addNodes(precision: 1e4)
        ..addNodes(valueChildren: [0])
        ..addCategoricalState(nCategories: 3);
      final updates = net.updateSequence.updates;
      expect(updates.last, (2, UpdateStep.posteriorCategorical));
      for (final (node, step) in updates) {
        if (net.kindOf(node) == NodeKind.binaryState) {
          expect(step, UpdateStep.predictionErrorBinary);
        }
      }
      expect(
        updates.where((s) => s.$1 == 2).map((s) => s.$2),
        unorderedEquals([
          UpdateStep.predictionErrorCategorical,
          UpdateStep.posteriorCategorical,
        ]),
      );
    });

    test('NetworkSpec reads categorical-state nodes', () {
      final net = NetworkSpec.fromJson({
        'nodes': [
          {
            'kind': 'categorical-state',
            'params': {
              'n_categories': 3,
              'binary_parameters': {'tonic_volatility_2': -2.0, 'mean_3': 0.5},
            },
          },
        ],
      }).build();
      expect(net.nNodes, 8);
      expect(net.get(4, NodeField.tonicVolatility), -2.0);
      expect(net.get(7, NodeField.mean), 0.5);
      expect(
        NetworkSpec.fromJson({
          'nodes': [
            {'kind': 'categorical-state'},
          ],
        }).build().nNodes,
        10,
      );
      for (final bad in [
        {'kind': 'categorical-state', 'n_nodes': 2},
        {'kind': 'categorical-state', 'value_children': 0},
        {
          'kind': 'categorical-state',
          'params': {'dimension': 3},
        },
        {
          'kind': 'categorical-state',
          'params': {
            'binary_parameters': {'mean_4': 0.0},
          },
        },
      ]) {
        expect(
          () => NetworkSpec.fromJson({
            'nodes': [bad],
          }).build(),
          throwsFormatException,
          reason: '$bad',
        );
      }
    });
  });

  group('filtering', () {
    test('networks around a categorical node leave it unchanged', () {
      final u = oneHot(200, 3);
      final alone = (Network()..addCategoricalState(nCategories: 3)).inputData(
        u,
        fields: allFields,
      )!;

      // A continuous HGF before it (the order that crashes pyhgf), and a
      // second categorical node after it: inputs [y, cat 3, cat 2].
      final net = Network()
        ..addNodes(precision: 1e4)
        ..addNodes(valueChildren: [0])
        ..addCategoricalState(nCategories: 3)
        ..addCategoricalState(nCategories: 2);
      expect(net.inputIdxs, [0, 2, 10]);
      expect(net.nInputColumns, 6);
      final u2 = oneHot(200, 2, seed: 9);
      final data = Float64List(200 * 6);
      for (var t = 0; t < 200; t++) {
        data[t * 6] = 1.0 + 0.01 * math.sin(t / 7);
        data.setRange(t * 6 + 1, t * 6 + 4, u, t * 3);
        data.setRange(t * 6 + 4, t * 6 + 6, u2, t * 2);
      }
      final both = net.inputData(data, fields: allFields)!;
      for (var i = 0; i < 8; i++) {
        expectSameSeries(both, 2 + i, alone, i, 'after a continuous HGF');
      }

      final second = (Network()..addCategoricalState(nCategories: 2)).inputData(
        u2,
        fields: allFields,
      )!;
      for (var i = 0; i < 6; i++) {
        expectSameSeries(both, 10 + i, second, i, 'second categorical node');
      }
    });

    test('alpha follows the observed frequencies', () {
      final net = Network()..addCategoricalState(nCategories: 3);
      // Category 0 on 80% of trials.
      final u = Float64List(300 * 3);
      for (var t = 0; t < 300; t++) {
        u[t * 3 + (t % 5 == 0 ? 1 + (t ~/ 5) % 2 : 0)] = 1.0;
      }
      final traj = net.inputData(u, fields: allFields)!;
      final alpha = traj.vectorAt(0, VectorField.alpha, 299);
      expect(alpha[0], greaterThan(alpha[1]));
      expect(alpha[0], greaterThan(alpha[2]));
      final p = traj.vectorAt(0, VectorField.mean, 299);
      expect(p[0], greaterThan(0.6));
      expect(net.get(0, NodeField.mean), p[0]);
      final kl = traj.series(0, NodeField.klDivergence);
      expect(kl.every((v) => v.isFinite && v >= -1e-12), isTrue);
      final s = traj.series(0, NodeField.surprise);
      expect(s.every((v) => v.isFinite && v > 0), isTrue);
    });

    test("a missing observation gives alpha = ±∞ (pyhgf's behaviour)", () {
      final u = oneHot(10, 3);
      final observed = Float64List(10)..fillRange(0, 10, 1.0);
      observed[4] = 0.0;
      final net = Network()..addCategoricalState(nCategories: 3);
      final traj = net.inputData(u, observed: observed, fields: allFields)!;
      final alpha = traj.vectorAt(0, VectorField.alpha, 4);
      expect(alpha.every((a) => a.isInfinite), isTrue);
      final kl = traj.series(0, NodeField.klDivergence);
      expect(kl[4].isNaN && kl[5].isNaN, isTrue);
      expect(kl[6].isFinite, isTrue);
      // The binary HGFs see no prediction error on that trial.
      expect(traj.series(1, NodeField.valuePredictionError)[4], 0.0);
    });

    test('step() reproduces inputData()', () {
      final u = oneHot(60, 4);
      final net = Network()..addCategoricalState();
      final batch = net.inputData(u, fields: allFields)!;
      net.reset();
      for (var t = 0; t < 60; t++) {
        net.step(Float64List.sublistView(u, t * 4, t * 4 + 4));
        expect(
          net.getVector(0, VectorField.alpha),
          batch.vectorAt(0, VectorField.alpha, t),
        );
        expect(
          net.get(0, NodeField.klDivergence),
          batch.series(0, NodeField.klDivergence)[t],
        );
        expect(net.get(5, NodeField.mean), batch.series(5, NodeField.mean)[t]);
      }
    });

    test('saveState / restoreState carry alpha', () {
      final u = oneHot(40, 3);
      final net = Network()..addCategoricalState(nCategories: 3);
      net.inputData(Float64List.sublistView(u, 0, 60), record: false);
      final state = net.saveState();
      final a = net.inputData(
        Float64List.sublistView(u, 60),
        append: true,
        fields: allFields,
      )!;
      net
        ..reset()
        ..restoreState(state);
      final b = net.inputData(
        Float64List.sublistView(u, 60),
        append: true,
        fields: allFields,
      )!;
      for (var i = 0; i < net.nNodes; i++) {
        expectSameSeries(a, i, b, i, 'restored');
      }
    });

    test('setInitialVector sets the prior alpha', () {
      final net = Network()..addCategoricalState(nCategories: 3);
      net.setInitialVector(0, VectorField.alpha, [5.0, 1.0, 1.0]);
      final traj = net.inputData(oneHot(1, 3), fields: allFields)!;
      final fresh = (Network()..addCategoricalState(nCategories: 3)).inputData(
        oneHot(1, 3),
        fields: allFields,
      )!;
      // The prior only enters the KL divergence of the first trial.
      expect(
        traj.vectorAt(0, VectorField.alpha, 0),
        fresh.vectorAt(0, VectorField.alpha, 0),
      );
      expect(
        traj.series(0, NodeField.klDivergence)[0],
        isNot(fresh.series(0, NodeField.klDivergence)[0]),
      );
      expect(net.getVector(0, VectorField.alpha), isNot([5.0, 1.0, 1.0]));
      net.reset();
      expect(net.getVector(0, VectorField.alpha), [5.0, 1.0, 1.0]);
    });

    test('toJson writes alpha and the vector mean per trial', () {
      final net = Network()..addCategoricalState(nCategories: 2);
      final json = net.inputData(oneHot(3, 2))!.toJson();
      expect((json['0']!['alpha']! as List).length, 3);
      expect((json['0']!['alpha']![0] as List).length, 2);
      expect((json['0']!['mean']![0] as List).length, 2);
      expect(json['1']!.containsKey('alpha'), isFalse);
    });

    test('node surprise is undefined for the categorical node', () {
      final traj = (Network()..addCategoricalState(nCategories: 2)).inputData(
        oneHot(5, 2),
      )!;
      expect(traj.nodeSurprise(0).every((v) => v.isNaN), isTrue);
      expect(traj.nodeSurprise(1).every((v) => v.isFinite), isTrue);
    });
  });

  group('math', () {
    // References: Python's math.lgamma, JAX's digamma (float64).
    test('logGamma and gammaSign', () {
      for (final (x, lg) in [
        (0.5, 0.5723649429247004),
        (1.0, 0.0),
        (1.5, -0.12078223763524543),
        (2.5, 0.2846828704729196),
        (10.0, 12.801827480081467),
        (123.456, 469.60554712992956),
        (1e6, 12815504.569147611),
        (-0.5, 1.265512123484645),
        (-1.5, 0.8600470153764812),
        (-2.7, -0.07140708531564521),
        (1e-8, 18.42068073818021),
      ]) {
        expect(logGamma(x), closeTo(lg, 1e-13 * math.max(1.0, lg.abs())));
      }
      expect(logGamma(0.0), double.infinity);
      expect(logGamma(-3.0), double.infinity);
      expect(logGamma(double.infinity), double.infinity);
      expect(gammaSign(2.0), 1.0);
      expect(gammaSign(-0.5), -1.0);
      expect(gammaSign(-1.5), 1.0);
      expect(gammaSign(-2.7), -1.0);
      expect(gammaSign(-2.0).isNaN, isTrue);
    });

    test('digamma', () {
      for (final (x, psi) in [
        (0.5, -1.9635100260214255),
        (1.0, -0.5772156649015334),
        (1.5, 0.03648997397857712),
        (2.5, 0.7031566406452437),
        (10.0, 2.2517525890667214),
        (123.456, 4.811829323828985),
        (1e6, 13.81551005796419),
        (-0.5, 0.03648997397857731),
        (-1.5, 0.703156640645244),
        (-2.7, -1.115347129140689),
        (1e-8, -100000000.57721567),
      ]) {
        expect(digamma(x), closeTo(psi, 1e-13 * math.max(1.0, psi.abs())));
      }
      expect(digamma(0.0).isNaN, isTrue);
      expect(digamma(-2.0).isNaN, isTrue);
      expect(digamma(double.infinity), double.infinity);
    });

    test('dirichletKullbackLeibler', () {
      // References: pyhgf.math.dirichlet_kullback_leibler.
      expect(
        dirichletKullbackLeibler([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]),
        closeTo(1.5734509276026944, 1e-13),
      );
      expect(
        dirichletKullbackLeibler([2.5, 0.7, 1.3, 4.0], [1.1, 2.2, 0.9, 3.0]),
        closeTo(2.4686607239954457, 1e-13),
      );
      expect(dirichletKullbackLeibler([3.0, 4.0], [3.0, 4.0]), 0.0);
      // pyhgf: NaN (Γ(180) overflows); in log space it stays finite.
      expect(
        dirichletKullbackLeibler([60.0, 70.0, 50.0], [61.0, 69.0, 52.0]),
        closeTo(0.044353766040615916, 1e-12),
      );
      expect(
        () => dirichletKullbackLeibler([1.0], [1.0, 2.0]),
        throwsArgumentError,
      );
    });
  });

  group('not supported yet', () {
    test('BatchNetwork', () {
      final net = Network()..addCategoricalState(nCategories: 2);
      expect(() => BatchNetwork(net, 4), throwsUnsupportedError);
    });

    test('sample()', () {
      final net = Network()..addCategoricalState(nCategories: 2);
      expect(() => net.sample(5), throwsUnsupportedError);
    });
  });
}
