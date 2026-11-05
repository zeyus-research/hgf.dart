import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

Network binary3() => Network()
  ..addNodes(kind: NodeKind.binaryState)
  ..addNodes(valueChildren: [0], tonicVolatility: -3.0)
  ..addNodes(volatilityChildren: [1], tonicVolatility: -2.0);

Network continuous2() => Network()
  ..addNodes(precision: 1e2)
  ..addNodes(valueChildren: [0], tonicVolatility: -4.0)
  ..addNodes(volatilityChildren: [1], tonicVolatility: -2.0);

/// Two inputs of different kinds sharing a parent.
Network mixed() => Network()
  ..addNodes(kind: NodeKind.binaryState)
  ..addNodes(precision: 10.0)
  ..addNodes(valueChildren: [0, 1], tonicVolatility: -3.0);

void expectSameTrajectories(Trajectories a, Trajectories b) {
  for (final f in a.fields) {
    for (var i = 0; i < a.nNodes; i++) {
      expect(a.series(i, f), b.series(i, f), reason: '$f node $i');
    }
  }
}

void main() {
  for (final (name, make) in [
    ('binary', binary3),
    ('continuous', continuous2),
    ('mixed inputs', mixed),
  ]) {
    test('$name: replaying the draws with inputData gives the same run', () {
      final net = make();
      final nIn = net.inputIdxs.length;
      final obs = Float64List(300 * nIn);
      final sampled = net.sample(
        300,
        random: math.Random(1),
        observations: obs,
      )!;
      final replayed = net.inputData(obs)!;
      expectSameTrajectories(sampled, replayed);
      // The draws are the input nodes' recorded means.
      for (var k = 0; k < nIn; k++) {
        final m = sampled.series(net.inputIdxs[k], NodeField.mean);
        for (var t = 0; t < 300; t++) {
          expect(m[t], obs[t * nIn + k]);
        }
      }
    });
  }

  test('time steps are honoured', () {
    final net = continuous2();
    final dt = Float64List.fromList([for (var t = 0; t < 50; t++) 0.5 + t % 3]);
    final obs = Float64List(50);
    final sampled = net.sample(
      50,
      random: math.Random(2),
      timeSteps: dt,
      observations: obs,
    )!;
    expectSameTrajectories(sampled, net.inputData(obs, timeSteps: dt)!);
  });

  test('a seed makes it reproducible', () {
    final net = binary3();
    final a = net.sample(100, random: math.Random(7))!;
    final b = net.sample(100, random: math.Random(7))!;
    expectSameTrajectories(a, b);
    final c = net.sample(100, random: math.Random(8))!;
    expect(c.series(0, NodeField.mean), isNot(a.series(0, NodeField.mean)));
  });

  test('binary draws follow the predicted probability', () {
    final net = binary3();
    final traj = net.sample(20000, random: math.Random(3))!;
    final u = traj.series(0, NodeField.mean);
    final p = traj.series(0, NodeField.expectedMean);
    var sumU = 0.0, sumP = 0.0;
    for (var t = 0; t < u.length; t++) {
      expect(u[t], anyOf(0.0, 1.0));
      sumU += u[t];
      sumP += p[t];
    }
    // Σu − Σp is a sum of 20000 zero-mean terms with variance ≤ 1/4.
    expect((sumU - sumP).abs(), lessThan(4 * math.sqrt(20000 / 4)));
  });

  test('continuous draws are N(expectedMean, 1/expectedPrecision)', () {
    final net = continuous2();
    final traj = net.sample(20000, random: math.Random(4))!;
    final x = traj.series(0, NodeField.mean);
    final m = traj.series(0, NodeField.expectedMean);
    final p = traj.series(0, NodeField.expectedPrecision);
    var s1 = 0.0, s2 = 0.0;
    for (var t = 0; t < x.length; t++) {
      final z = (x[t] - m[t]) * math.sqrt(p[t]);
      s1 += z;
      s2 += z * z;
    }
    final n = x.length;
    expect(s1 / n, closeTo(0.0, 4 / math.sqrt(n)));
    expect(s2 / n, closeTo(1.0, 4 * math.sqrt(2 / n)));
  });

  group('saveState / restoreState', () {
    test('branches continue identically', () {
      final net = binary3();
      net.sample(40, random: math.Random(5));
      final state = net.saveState();
      final a = net.sample(30, random: math.Random(6), append: true)!;
      net.restoreState(state);
      final b = net.sample(30, random: math.Random(6), append: true)!;
      expectSameTrajectories(a, b);
    });

    test('append continues from the current beliefs', () {
      final net = continuous2();
      final obs = Float64List(60);
      net.sample(60, random: math.Random(9), observations: obs);
      final whole = net.inputData(obs)!;
      net.inputData(Float64List.sublistView(obs, 0, 25), record: false);
      final tail = net.inputData(
        Float64List.sublistView(obs, 25),
        append: true,
      )!;
      for (var t = 0; t < 35; t++) {
        expect(
          tail.at(1, NodeField.mean, t),
          whole.at(1, NodeField.mean, t + 25),
        );
      }
    });

    test('rejects a state from another network', () {
      final a = binary3()..build();
      final b = binary3()..build();
      expect(() => b.restoreState(a.saveState()), throwsArgumentError);
    });

    test('restore clears a pending prediction', () {
      final net = binary3();
      final state = net.saveState();
      net.predict();
      expect(net.saveState, throwsStateError);
      net.restoreState(state);
      expect(net.saveState, returnsNormally);
    });
  });

  test('argument checks', () {
    final net = binary3();
    expect(() => net.sample(-1), throwsArgumentError);
    expect(() => net.sample(3, timeSteps: Float64List(2)), throwsArgumentError);
    expect(
      () => net.sample(3, observations: Float64List(2)),
      throwsArgumentError,
    );
  });
}
