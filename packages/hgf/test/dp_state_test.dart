@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf/src/dirichlet_process.dart';
import 'package:hgf/src/jax_random.dart';
import 'package:test/test.dart';

import 'src/fixtures.dart';

/// Draws from three normal clusters (centres −6, 0.5, 7), in random order.
Float64List mixture(int nTime, {int seed = 5}) {
  final rng = math.Random(seed);
  const centres = [-6.0, 0.5, 7.0];
  return Float64List.fromList([
    for (var t = 0; t < nTime; t++)
      centres[rng.nextInt(3)] +
          math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
              math.cos(2 * math.pi * rng.nextDouble()),
  ]);
}

/// A dp-state node (at [base]) with [k] clusters.
/// A dp-state node with [k] clusters, appended to [net].
void addDp(Network net, {int k = 3, double alpha = 0.1}) {
  final dp = net.nNodes;
  net
    ..addDpState(batchSize: k, alpha: alpha)
    ..addEfState(nNodes: k, valueChildren: [dp], nus: 15.0);
}

Network dpNetwork({int k = 3, double alpha = 0.1}) {
  final net = Network();
  addDp(net, k: k, alpha: alpha);
  return net;
}

void expectSameNode(
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

/// Distance in units in the last place.
int ulps(double a, double b) {
  final bits = ByteData(16)
    ..setFloat64(0, a)
    ..setFloat64(8, b);
  return (bits.getInt64(0) - bits.getInt64(8)).abs();
}

void main() {
  group('JAX random', () {
    test('threefry split of key(42) matches JAX', () {
      final (a, b) = jaxSplit(jaxKey(42));
      expect(
        [a.$1, a.$2, b.$1, b.$2],
        [1832780943, 270669613, 64467757, 2916123636],
      );
    });

    test('erfInv inverts erf at known points', () {
      expect(erfInv(0.0), 0.0);
      expect(erfInv(1.0), double.infinity);
      expect(erfInv(-1.0), double.negativeInfinity);
      expect(erfInv(1.5).isNaN, isTrue);
      // erf(0.5) and erf(2.5) (both branches of the polynomial).
      expect(erfInv(0.5204998778130465), closeTo(0.5, 1e-15));
      expect(erfInv(0.9995930479825550), closeTo(2.5, 1e-12));
      expect(erfInv(-0.5204998778130465), closeTo(-0.5, 1e-15));
    });
  });

  group("pyhgf's cluster proposals", () {
    final ref =
        jsonDecode(
              File('test/fixtures/dp/dp_reference.json').readAsStringSync(),
            )
            as Map<String, Object?>;

    test('match JAX to a few ulp', () {
      final p = DpProposals.instance;
      final stride = ref['stride'] as int;
      final mu = series(ref['mu']), sigma = series(ref['sigma']);
      var worst = 0;
      for (var k = 0; k < mu.length; k++) {
        worst = math.max(worst, ulps(p.mu[k * stride], mu[k]));
        worst = math.max(worst, ulps(p.sigma[k * stride], sigma[k]));
      }
      expect(worst, lessThanOrEqualTo(4));
      final muSum = p.mu.fold(0.0, (a, b) => a + b);
      final sigmaSum = p.sigma.fold(0.0, (a, b) => a + b);
      expect(muSum, closeTo(decode(ref['mu_sum']), 1e-9));
      expect(sigmaSum, closeTo(decode(ref['sigma_sum']), 1e-9));
    });

    test('candidates match get_candidate', () {
      final scratch = DpScratch(10);
      for (final raw in ref['candidates'] as List<Object?>) {
        final c = raw as Map<String, Object?>;
        final means = Float64List.fromList(series(c['expected_means']));
        final sigmas = Float64List.fromList(series(c['expected_sigmas']));
        final (mean, sigma) = dpCandidate(
          decode(c['value']),
          decode(c['sensory_precision']),
          means,
          sigmas,
          0,
          means.length,
          scratch,
        );
        final expMean = decode(c['mean']), expSigma = decode(c['sigma']);
        expect((mean - expMean).abs() / expMean.abs(), lessThan(1e-12));
        expect((sigma - expSigma).abs() / expSigma, lessThan(1e-12));
      }
    });
    // pyhgf's own test_get_candidate expects other values (5.026636,
    // 1.2752448): float32 draws, with its conftest's legacy (non-partitionable)
    // threefry. The first case above is the same call in float64 with JAX's
    // default PRNG, which is what pyhgf users get (P15).
  });

  group('construction', () {
    test("pyhgf's layout and defaults", () {
      final net = dpNetwork();
      expect(net.nNodes, 4);
      expect(net.kindOf(0), NodeKind.dpState);
      expect(net.inputIdxs, [0]);
      expect(net.nInputColumns, 1);
      expect(net.get(0, NodeField.alpha), 0.1);
      expect(net.get(0, NodeField.sensoryPrecision), 1.0);
      expect(net.getVector(0, VectorField.n), [0.0, 0.0, 0.0]);
      expect(net.getVector(0, VectorField.expectedSigmas), [1.0, 1.0, 1.0]);
      expect(net.getVector(0, VectorField.activated), [0.0, 0.0, 0.0]);
      expect(net.getVector(1, VectorField.xis), [0.0, 1.0]);
    });

    test('the dp step runs before its clusters', () {
      final seq = dpNetwork().updateSequence;
      expect(seq.predictions, [(0, UpdateStep.predictionDp)]);
      expect(seq.updates, [
        (0, UpdateStep.predictionErrorDp),
        for (var i = 1; i <= 3; i++) (i, UpdateStep.predictionErrorEfFixed),
      ]);
    });

    test('rejects bad declarations', () {
      expect(
        () => Network().addNodes(kind: NodeKind.dpState),
        throwsArgumentError,
      );
      // Clusters need an existing dp-state child...
      expect(
        () => Network()
          ..addNodes()
          ..addEfState(valueChildren: [0]),
        throwsArgumentError,
      );
      // ... and are univariate with a fixed learning rate.
      expect(
        () => Network()
          ..addDpState(batchSize: 1)
          ..addEfState(valueChildren: [0], dimension: 2),
        throwsArgumentError,
      );
      expect(
        () => Network()
          ..addDpState(batchSize: 1)
          ..addEfState(valueChildren: [0], learning: EfLearning.hgf1),
        throwsArgumentError,
      );
      // No other edges to a dp-state node.
      expect(
        () => Network()
          ..addDpState(batchSize: 1)
          ..addNodes(valueChildren: [0]),
        throwsArgumentError,
      );
      // One cluster per batch element (pyhgf fails on the array shapes).
      expect(
        () =>
            (Network()
                  ..addDpState(batchSize: 4)
                  ..addEfState(nNodes: 3, valueChildren: [0]))
                .build(),
        throwsStateError,
      );
    });

    test('NetworkSpec reads dp-state nodes and their clusters', () {
      final net = NetworkSpec.fromJson({
        'nodes': [
          {
            'kind': 'dp-state',
            'params': {'batch_size': 2, 'alpha': 0.5, 'sensory_precision': 2.0},
          },
          {
            'kind': 'ef-state',
            'n_nodes': 2,
            'value_children': [0],
            'params': {'nus': 9.0},
          },
        ],
      }).build();
      expect(net.nNodes, 3);
      expect(net.get(0, NodeField.alpha), 0.5);
      expect(net.get(0, NodeField.sensoryPrecision), 2.0);
      expect(net.get(2, NodeField.nus), 9.0);
      expect(net.inputIdxs, [0]);
      expect(
        () => NetworkSpec.fromJson({
          'nodes': [
            {
              'kind': 'dp-state',
              'params': {'concentration': 1.0},
            },
          ],
        }).build(),
        throwsFormatException,
      );
    });
  });

  group('filtering', () {
    final x = mixture(120);

    test('finds the three clusters', () {
      final net = dpNetwork();
      net.inputData(x, record: false);
      final means = net.getVector(0, VectorField.expectedMeans)..sort();
      expect(means[0], closeTo(-6.0, 1.0));
      expect(means[1], closeTo(0.5, 1.0));
      expect(means[2], closeTo(7.0, 1.0));
      final n = net.getVector(0, VectorField.n);
      expect(n.reduce((a, b) => a + b), 120);
      expect(net.get(0, NodeField.nTotal), 120);
    });

    test('networks around a dp node leave it unchanged', () {
      final alone = dpNetwork().inputData(x, fields: NodeField.values)!;

      // A continuous HGF first: pyhgf crashes on this order (P7).
      final after = Network()
        ..addNodes(precision: 1e4)
        ..addNodes(valueChildren: [0]);
      addDp(after);
      // A second dp node after it.
      addDp(after, k: 2, alpha: 1.0);
      final data = Float64List(x.length * 3);
      for (var t = 0; t < x.length; t++) {
        data[3 * t] = x[t];
        data[3 * t + 1] = x[t];
        data[3 * t + 2] = -x[t];
      }
      expect(after.inputIdxs, [0, 2, 6]);
      final traj = after.inputData(data, fields: NodeField.values)!;
      for (var i = 0; i < 4; i++) {
        expectSameNode(alone, i, traj, i + 2, 'dp after a continuous HGF');
      }
    });

    test('a missing observation is skipped (diverges from pyhgf, P14)', () {
      final observed = Float64List(x.length)..fillRange(0, x.length, 1.0);
      observed[50] = 0.0;
      final data = Float64List.fromList(x)..[50] = double.nan;
      final net = dpNetwork();
      final traj = net.inputData(
        data,
        observed: observed,
        fields: NodeField.values,
      )!;
      final nTotal = traj.series(0, NodeField.nTotal);
      expect(nTotal[49], 50);
      expect(nTotal[50], 50);
      expect(nTotal[51], 51);
      expect(
        traj.vectorAt(0, VectorField.n, 50),
        traj.vectorAt(0, VectorField.n, 49),
      );
      for (var c = 1; c <= 3; c++) {
        expect(
          traj.vectorAt(c, VectorField.xis, 50),
          traj.vectorAt(c, VectorField.xis, 49),
        );
        expect(traj.series(c, NodeField.mean)[50].isNaN, isFalse);
      }
      expect(
        net.getVector(0, VectorField.expectedMeans).any((m) => m.isNaN),
        isFalse,
      );
    });

    test('more openings than clusters are counted and dropped', () {
      final net = dpNetwork(k: 2, alpha: 50.0);
      final traj = net.inputData(x.sublist(0, 20), fields: NodeField.values)!;
      expect(traj.series(0, NodeField.nActiveCluster).last, 20);
      expect(net.getVector(0, VectorField.activated), [1.0, 1.0]);
      expect(net.getVector(0, VectorField.n), [1.0, 1.0]);
    });

    test('step() reproduces inputData()', () {
      final a = dpNetwork()..inputData(x, record: false);
      final b = dpNetwork();
      for (final v in x) {
        b.step(Float64List.fromList([v]));
      }
      for (final f in [VectorField.n, VectorField.expectedMeans]) {
        expect(b.getVector(0, f), a.getVector(0, f));
      }
      expect(b.getVector(2, VectorField.xis), a.getVector(2, VectorField.xis));
    });

    test('saveState / restoreState and reset carry the clusters', () {
      final net = dpNetwork()..inputData(x.sublist(0, 60), record: false);
      final state = net.saveState();
      final first = net.inputData(x.sublist(60), append: true)!;
      net.restoreState(state);
      final second = net.inputData(x.sublist(60), append: true)!;
      expectSameNode(first, 0, second, 0, 'restored');
      net.reset();
      expect(net.getVector(0, VectorField.n), [0.0, 0.0, 0.0]);
      expect(net.get(0, NodeField.nTotal), 0);
      expect(net.getVector(3, VectorField.xis), [0.0, 1.0]);
    });

    test('node surprise is undefined for the dp node', () {
      final traj = dpNetwork().inputData(x)!;
      expect(traj.nodeSurprise(0).every((s) => s.isNaN), isTrue);
    });
  });

  group('surprise (hgf.dart extension)', () {
    final x = mixture(120);

    /// The base measure's predictive density at [v], by brute force.
    double baseDensity(double v) {
      final p = DpProposals.instance;
      var sum = 0.0;
      for (var j = 0; j < p.mu.length; j++) {
        final z = (v - p.mu[j]) / p.sigma[j];
        sum += math.exp(-0.5 * z * z) / (p.sigma[j] * math.sqrt(2 * math.pi));
      }
      return sum / p.mu.length;
    }

    test('first trial: the base measure alone', () {
      final net = dpNetwork()..inputData(x.sublist(0, 1), record: false);
      expect(
        net.get(0, NodeField.surprise),
        closeTo(-math.log(baseDensity(x[0])), 1e-12),
      );
    });

    test('matches the mixture computed from the state before each trial', () {
      final net = dpNetwork();
      for (var t = 0; t < 60; t++) {
        final alpha = net.get(0, NodeField.alpha);
        final nTotal = net.get(0, NodeField.nTotal);
        final counts = net.getVector(0, VectorField.n);
        final active = net.getVector(0, VectorField.activated);
        double pdf(double v, double m, double sd) =>
            math.exp(-0.5 * ((v - m) / sd) * ((v - m) / sd)) /
            (sd * math.sqrt(2 * math.pi));
        var mix = alpha / (alpha + nTotal) * baseDensity(x[t]);
        for (var i = 0; i < 3; i++) {
          // The cluster's prediction, as the dp step forms it.
          final xi = net.getVector(1 + i, VectorField.xis);
          final sd = math.sqrt(xi[1] - xi[0] * xi[0]);
          mix +=
              counts[i] * active[i] / (alpha + nTotal) * pdf(x[t], xi[0], sd);
        }
        net.step(Float64List.fromList([x[t]]));
        expect(
          net.get(0, NodeField.surprise),
          closeTo(-math.log(mix), 1e-9),
          reason: 'trial $t',
        );
      }
    });

    test('stays finite far from every cluster', () {
      final net = dpNetwork()..inputData(x, record: false);
      net.step(Float64List.fromList([1e3]));
      final s = net.get(0, NodeField.surprise);
      expect(s.isFinite, isTrue);
      expect(s, greaterThan(100));
    });

    test('DpSurprise sums it; missing observations add nothing', () {
      final observed = Float64List(x.length)..fillRange(0, x.length, 1);
      for (var t = 5; t < x.length; t += 7) {
        observed[t] = 0;
      }
      final traj = dpNetwork().inputData(
        x,
        observed: observed,
        fields: const [NodeField.surprise],
      )!;
      final perTrial = Float64List(x.length);
      final total = dpNetwork().surprise(
        x,
        const DpSurprise(),
        observed: observed,
        perTrial: perTrial,
      );
      final recorded = traj.series(0, NodeField.surprise);
      expect(perTrial, recorded);
      for (var t = 5; t < x.length; t += 7) {
        expect(perTrial[t], 0.0);
      }
      expect(total, closeTo(recorded.reduce((a, b) => a + b), 1e-9));
      expect(const DpSurprise().supportsGradients, isFalse);
      expect(
        () => Network().addNodes().surprise(Float64List(3), const DpSurprise()),
        throwsArgumentError,
      );
    });

    test('BatchNetwork matches Network bit for bit, per alpha', () {
      final alphas = Float64List.fromList([0.01, 0.3, 5.0]);
      final batch = BatchNetwork(dpNetwork(), alphas.length)
        ..setInitial(0, NodeField.alpha, alphas);
      final totals = batch.surprise(x, const DpSurprise());
      for (var e = 0; e < alphas.length; e++) {
        final net = dpNetwork()..setInitial(0, NodeField.alpha, alphas[e]);
        expect(totals[e], net.surprise(x, const DpSurprise()), reason: '$e');
      }
    });

    test('a sensible alpha explains three clusters best', () {
      final mix = mixture(300);
      double nll(double alpha) =>
          (dpNetwork(k: 6)..setInitial(0, NodeField.alpha, alpha)).surprise(
            mix,
            const DpSurprise(),
          );
      final small = nll(0.1), huge = nll(1e4);
      expect(small, lessThan(huge));
    });
  });

  group('BatchNetwork', () {
    // Bit-for-bit agreement with Network on the fixtures is in batch_test.
    final x = mixture(60);

    test('each element clusters with its own alpha', () {
      final alphas = Float64List.fromList([0.01, 0.1, 1.0, 50.0]);
      final batch = BatchNetwork(dpNetwork(), alphas.length)
        ..setInitial(0, NodeField.alpha, alphas)
        ..inputData(x);
      for (var e = 0; e < alphas.length; e++) {
        final net = dpNetwork()
          ..setInitial(0, NodeField.alpha, alphas[e])
          ..inputData(x, record: false);
        for (final f in [VectorField.n, VectorField.expectedMeans]) {
          expect(batch.getVector(e, 0, f), net.getVector(0, f));
        }
        expect(
          batch.get(e, 0, NodeField.nActiveCluster),
          net.get(0, NodeField.nActiveCluster),
        );
      }
      expect(batch.get(0, 0, NodeField.nActiveCluster), lessThan(5));
      expect(batch.get(3, 0, NodeField.nActiveCluster), greaterThan(20));
    });

    test('blocks are capped by the proposal caches', () {
      // 64 MB of 320 KB rows: 69 elements with 3 clusters.
      expect(BatchNetwork(dpNetwork(), 500).blockSize, 69);
      expect(BatchNetwork(dpNetwork(), 10).blockSize, 10);
      expect(BatchNetwork(dpNetwork(), 500, blockSize: 8).blockSize, 8);
    });

    test('reset restores the initial clusters', () {
      final batch = BatchNetwork(dpNetwork(), 2)..inputData(x);
      expect(batch.get(1, 0, NodeField.nTotal), 60);
      batch.reset();
      expect(batch.getVector(1, 0, VectorField.n), [0.0, 0.0, 0.0]);
      expect(batch.getVector(1, 3, VectorField.xis), [0.0, 1.0]);
      expect(batch.get(1, 0, NodeField.nTotal), 0);
    });
  });

  group('not supported yet', () {
    test('sample()', () {
      expect(() => dpNetwork().sample(5), throwsUnsupportedError);
    });
  });
}
