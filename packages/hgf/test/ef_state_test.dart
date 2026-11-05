import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

/// A continuous 2-level HGF (input 0), a bivariate ef node (input 2, fixed
/// rate) and a univariate one with hgf-2 learning (input 3, nodes 4..9).
Network mixed() => Network()
  ..addNodes(precision: 1e2)
  ..addNodes(valueChildren: [0], tonicVolatility: -4.0)
  ..addEfState(dimension: 2)
  ..addEfState(learning: EfLearning.hgf2);

/// `nTime` rows of [mixed]'s 4 input columns.
Float64List mixedData(int nTime) {
  final rng = math.Random(7);
  return Float64List.fromList([
    for (var t = 0; t < nTime; t++) ...[
      1.0 + 0.1 * math.sin(t / 9),
      rng.nextDouble(),
      2.0 * rng.nextDouble() - 1.0,
      0.5 + 0.2 * math.cos(t / 5) + 0.05 * rng.nextDouble(),
    ],
  ]);
}

void expectSameRun(Trajectories a, Trajectories b) {
  for (final f in a.fields) {
    for (var i = 0; i < a.nNodes; i++) {
      expect(a.series(i, f), b.series(i, f), reason: '$f node $i');
    }
  }
  expect(a.vectorFields, b.vectorFields);
  for (final f in a.vectorFields) {
    for (var i = 0; i < a.nNodes; i++) {
      expect(a.vectorSeries(i, f), b.vectorSeries(i, f), reason: '$f $i');
    }
  }
}

void main() {
  group('construction', () {
    test('hgf learning adds one chain per sufficient statistic', () {
      for (final (learning, perChain) in [
        (EfLearning.fixed, 0),
        (EfLearning.hgf1, 2),
        (EfLearning.hgf2, 3),
        (EfLearning.hgf3, 4),
      ]) {
        final uni = Network()..addEfState(learning: learning);
        expect(uni.nNodes, 1 + 2 * perChain, reason: '$learning');
        final tri = Network()..addEfState(dimension: 3, learning: learning);
        expect(tri.nNodes, 1 + 9 * perChain, reason: '$learning');
      }
    });

    test('inputs, columns and vector lengths', () {
      final net = mixed();
      expect(net.inputIdxs, [0, 2, 3]);
      expect(net.nInputColumns, 4);
      expect(net.vectorLength(2, VectorField.mean), 2);
      expect(net.vectorLength(2, VectorField.xis), 5);
      expect(net.vectorLength(3, VectorField.xis), 2);
      expect(net.vectorLength(0, VectorField.xis), 0);
      expect(net.getVector(2, VectorField.xis), [0, 0, 1, 0, 1]);
      expect(net.get(3, NodeField.nus), 3.0);
    });

    test('rejects bad declarations', () {
      expect(() => Network().addEfState(dimension: 0), throwsArgumentError);
      expect(
        () => Network().addEfState(xis: [0.0, 1.0, 2.0]),
        throwsArgumentError,
      );
      expect(
        () => Network().addNodes(kind: NodeKind.efState),
        throwsArgumentError,
      );
      // An ef node only takes value parents.
      final net = Network()..addEfState();
      expect(() => net.addNodes(valueParents: [0]), throwsArgumentError);
      expect(() => net.addNodes(volatilityChildren: [0]), throwsArgumentError);
      // With parents, it needs one per sufficient statistic.
      final one = Network()
        ..addEfState()
        ..addNodes(valueChildren: [0]);
      one.addNodes(valueChildren: [1]);
      expect(one.build, throwsStateError);
    });

    test('NetworkSpec reads pyhgf ef-state parameters', () {
      final net = NetworkSpec.fromJson({
        'nodes': [
          {
            'kind': 'ef-state',
            'n_nodes': 2,
            'params': {
              'dimension': 2,
              'distribution': 'multivariate-normal',
              'learning': 'hgf-1',
              'nus': 5,
            },
          },
        ],
      }).build();
      expect(net.nNodes, 2 + 2 * 5 * 2);
      expect(net.get(1, NodeField.nus), 5.0);
      expect(
        () => NetworkSpec.fromJson({
          'nodes': [
            {
              'kind': 'ef-state',
              'params': {'dimension': 2},
            },
          ],
        }).build(),
        throwsFormatException,
      );
    });
  });

  group('filtering', () {
    test('fixed rate: xis is a running exponential average of t(x)', () {
      final net = Network()..addEfState(nus: 1.0); // rate 1/2
      final traj = net.inputSeries([2.0, 4.0]);
      expect(traj.vectorAt(0, VectorField.xis, 0), [1.0, 2.5]);
      expect(traj.vectorAt(0, VectorField.xis, 1), [2.5, 9.25]);
      expect(traj.vectorAt(0, VectorField.observationSs, 1), [4.0, 16.0]);
      expect(traj.series(0, NodeField.mean), [2.0, 4.0]);
    });

    test('a missing observation leaves xis alone', () {
      final net = Network()..addEfState(nus: 1.0);
      final traj = net.inputSeries([2.0, 4.0], observed: [1.0, 0.0]);
      expect(traj.vectorAt(0, VectorField.xis, 1), [1.0, 2.5]);
    });

    test('step() reproduces inputData() with vector inputs', () {
      final data = mixedData(200);
      final observed = Float64List.fromList([
        for (var t = 0; t < 200; t++) ...[1.0, t % 7 == 0 ? 0.0 : 1.0, 1.0],
      ]);
      final net = mixed();
      final whole = net.inputData(
        data,
        observed: observed,
        fields: [...defaultRecordedFields, NodeField.nus],
      )!;
      net.reset();
      for (var t = 0; t < 200; t++) {
        net.step(
          Float64List.sublistView(data, 4 * t, 4 * t + 4),
          observed: Float64List.sublistView(observed, 3 * t, 3 * t + 3),
        );
        for (final f in whole.vectorFields) {
          for (final i in [2, 3]) {
            expect(
              net.getVector(i, f),
              whole.vectorAt(i, f, t),
              reason: '$f node $i t $t',
            );
          }
        }
      }
      expect(net.get(3, NodeField.nus), whole.at(3, NodeField.nus, 199));
    });

    test('checks the data and observed shapes', () {
      final net = mixed();
      expect(() => net.inputData(Float64List(9)), throwsArgumentError);
      expect(
        () => net.inputData(Float64List(8), observed: Float64List(8)),
        throwsArgumentError,
      );
      expect(
        net.inputData(Float64List(8), observed: Float64List(6)),
        isNotNull,
      );
      expect(() => net.step(Float64List(3)), throwsArgumentError);
    });

    test('setInitialVector sets the prior xis', () {
      final net = Network()..addEfState(nus: 1.0);
      net.setInitialVector(0, VectorField.xis, [1.0, 3.0]);
      expect(net.getInitialVector(0, VectorField.xis), [1.0, 3.0]);
      expect(net.getVector(0, VectorField.xis), [0.0, 1.0]); // until reset
      final traj = net.inputSeries([3.0]);
      expect(traj.vectorAt(0, VectorField.xis, 0), [2.0, 6.0]);
      expect(
        () => net.setInitialVector(0, VectorField.xis, [1.0]),
        throwsArgumentError,
      );
      final viaArg = Network()..addEfState(nus: 1.0, xis: [1.0, 3.0]);
      expect(viaArg.inputSeries([3.0]).vectorAt(0, VectorField.xis, 0), [
        2.0,
        6.0,
      ]);
    });

    test('saveState / restoreState carry the vectors', () {
      final data = mixedData(120);
      final net = mixed()
        ..inputData(Float64List.sublistView(data, 0, 4 * 60), record: false);
      final state = net.saveState();
      final rest = Float64List.sublistView(data, 4 * 60);
      final a = net.inputData(rest, append: true)!;
      net.restoreState(state);
      final b = net.inputData(rest, append: true)!;
      expectSameRun(a, b);
    });

    test('inputData(into:) reuses vector columns', () {
      final data = mixedData(80);
      final net = mixed();
      final first = net.inputData(data)!;
      final fresh = mixed().inputData(data)!;
      final again = net.inputData(data, into: first)!;
      expect(identical(again, first), isTrue);
      expectSameRun(again, fresh);
      final other = Network()..addEfState(dimension: 3);
      expect(
        () => other.inputData(Float64List(80 * 3), into: first),
        throwsArgumentError,
      );
    });

    test('toJson writes vector fields per trial', () {
      final net = Network()..addEfState(dimension: 2);
      final json = net
          .inputData(Float64List.fromList([1.0, 2.0, 3.0, 4.0]))!
          .toJson();
      expect(json['0']!['mean'], [
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      expect((json['0']!['xis']! as List).length, 2);
    });

    test('node surprise is undefined for ef nodes', () {
      final net = Network()..addEfState(learning: EfLearning.hgf1);
      final traj = net.inputSeries([1.0, 1.1, 0.9]);
      expect(traj.nodeSurprise(0).every((v) => v.isNaN), isTrue);
      expect(traj.totalSurprise().every((v) => v.isFinite), isTrue);
    });
  });

  group('not supported yet', () {
    test('BatchNetwork', () {
      expect(() => BatchNetwork(mixed(), 4), throwsUnsupportedError);
    });

    test('sample()', () {
      expect(() => mixed().sample(10), throwsUnsupportedError);
    });
  });
}
