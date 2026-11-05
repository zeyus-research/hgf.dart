// Reads a fixture file (dart:io) in the ehgfMgfCorrection tests.
@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

Network twoLevel({VolatilityUpdates v = VolatilityUpdates.unbounded}) =>
    Network(volatilityUpdates: v)
      ..addNodes(precision: 1e4)
      ..addNodes(valueChildren: [0], mean: 1.0, tonicVolatility: -8.0)
      ..addNodes(volatilityChildren: [1], tonicVolatility: -2.0);

final series = Float64List.fromList([
  for (var t = 0; t < 200; t++) 1.0 + 0.01 * (t % 17) - 0.004 * (t % 5),
]);

void main() {
  group('pyhgf unit values', () {
    // Values from pyhgf's tests/test_continuous.py (JAX backend).
    test('one-node HGF, one observation', () {
      final net = Network()
        ..addNodes()
        ..addNodes(valueChildren: [0]);
      final traj = net.inputSeries([0.2]);
      expect(traj.at(1, NodeField.precision, 0), closeTo(1.9820137, 1e-6));
      expect(
        traj.at(1, NodeField.expectedPrecision, 0),
        closeTo(0.98201376, 1e-6),
      );
      expect(traj.at(1, NodeField.mean, 0), closeTo(0.10090748, 1e-6));
    });

    test('two-node HGF with a volatility parent on the input', () {
      final net = Network()
        ..addNodes()
        ..addNodes(valueChildren: [0])
        ..addNodes(volatilityChildren: [0]);
      final traj = net.inputSeries([0.2]);
      expect(
        traj.at(0, NodeField.expectedPrecision, 0),
        closeTo(0.2715764, 1e-6),
      );
      expect(traj.at(1, NodeField.precision, 0), closeTo(1.3574047, 1e-6));
      expect(traj.at(1, NodeField.mean, 0), closeTo(0.0553101, 1e-6));
      expect(traj.at(2, NodeField.precision, 0), closeTo(1.0834352, 1e-6));
      expect(traj.at(2, NodeField.mean, 0), closeTo(-0.10491979, 1e-6));
    });
  });

  group('structure', () {
    test('inputs are the nodes without children', () {
      expect(twoLevel().inputIdxs, [0]);
    });

    test('inputs get no autoconnection and no tonic volatility', () {
      final net = Network()
        ..addNodes(tonicVolatility: -3.0, autoconnectionStrength: 0.5)
        ..addNodes(valueChildren: [0]);
      expect(net.getInitial(0, NodeField.tonicVolatility), 0.0);
      expect(net.getInitial(0, NodeField.autoconnectionStrength), 0.0);
      expect(net.getInitial(1, NodeField.tonicVolatility), -4.0);
    });

    test('update sequence: roots predict first, leaves update first', () {
      final seq = twoLevel().updateSequence;
      expect(seq.predictions.map((s) => s.$1), [2, 1, 0]);
      expect(seq.updates, [
        (0, UpdateStep.predictionErrorContinuous),
        (1, UpdateStep.posteriorContinuous),
        (1, UpdateStep.predictionErrorContinuous),
        (2, UpdateStep.posteriorContinuousUnbounded),
      ]);
    });

    test('edges to nodes that do not exist yet are rejected', () {
      expect(() => Network().addNodes(valueChildren: [0]), throwsArgumentError);
    });

    test('strength lists must match the index lists', () {
      expect(
        () => Network()
          ..addNodes()
          ..addNodes(valueChildren: [0], valueChildrenStrengths: [1.0, 2.0]),
        throwsArgumentError,
      );
    });

    test('binary nodes take no volatility edges', () {
      expect(
        () => Network()
          ..addNodes()
          ..addNodes(kind: NodeKind.binaryState, volatilityChildren: [0]),
        throwsArgumentError,
      );
    });
  });

  group('filtering', () {
    test('step() reproduces inputData() exactly', () {
      final whole = twoLevel().inputData(series)!;
      final live = twoLevel()..build();
      final obs = Float64List(1);
      for (var t = 0; t < series.length; t++) {
        obs[0] = series[t];
        live.step(obs);
        for (var node = 0; node < 3; node++) {
          expect(
            live.get(node, NodeField.mean),
            whole.at(node, NodeField.mean, t),
          );
        }
      }
    });

    test('predict() + step() equals step(), and exposes the prediction', () {
      final a = twoLevel()..build();
      final b = twoLevel()..build();
      final obs = Float64List(1);
      for (var t = 0; t < 50; t++) {
        obs[0] = series[t];
        b.predict();
        final upcoming = b.get(1, NodeField.expectedMean);
        b
          ..predict() // idempotent
          ..step(obs);
        a.step(obs);
        expect(b.get(1, NodeField.expectedMean), upcoming);
        for (var node = 0; node < 3; node++) {
          expect(b.get(node, NodeField.mean), a.get(node, NodeField.mean));
        }
      }
    });

    test('a pending prediction must match the step and blocks inputData', () {
      final net = twoLevel()..predict(timeStep: 2.0);
      expect(() => net.step(Float64List.fromList([1.0])), throwsStateError);
      expect(() => net.inputData(series, append: true), throwsStateError);
      net.reset(); // clears it
      net.inputData(series, append: true);
    });

    test('inputData restarts from the initial state; append continues', () {
      final net = twoLevel();
      final a = net.inputData(series)!;
      final b = net.inputData(series)!;
      expect(b.series(1, NodeField.mean), a.series(1, NodeField.mean));

      final half = series.length ~/ 2;
      final first = Float64List.sublistView(series, 0, half);
      final second = Float64List.sublistView(series, half);
      net.inputData(first, record: false);
      final cont = net.inputData(second, append: true)!;
      expect(
        cont.at(1, NodeField.mean, cont.nTime - 1),
        a.at(1, NodeField.mean, a.nTime - 1),
      );
    });

    test('setInitial changes the next run', () {
      // ω1 and μ2 only enter through ω1 + κ·μ2, so after enough trials μ2
      // absorbs a shift in ω1 (and μ1 barely moves); check μ2.
      final net = twoLevel();
      final base = net.inputData(series)!.at(2, NodeField.mean, 199);
      net.setInitial(1, NodeField.tonicVolatility, -2.0);
      final changed = net.inputData(series)!.at(2, NodeField.mean, 199);
      expect(changed, closeTo(base - 6.0, 1e-6));
      net.setInitial(1, NodeField.tonicVolatility, -8.0);
      expect(net.inputData(series)!.at(2, NodeField.mean, 199), base);
    });

    test('a missing observation leaves the parent at its prediction', () {
      final net = twoLevel(v: VolatilityUpdates.standard);
      final observed = Float64List(series.length)
        ..fillRange(0, series.length, 1);
      observed[50] = 0;
      final traj = net.inputData(series, observed: observed)!;
      expect(
        traj.at(1, NodeField.mean, 50),
        traj.at(1, NodeField.expectedMean, 50),
      );
      expect(
        traj.at(1, NodeField.precision, 50),
        traj.at(1, NodeField.expectedPrecision, 50),
      );
    });

    test('input shapes are validated', () {
      final net = Network()
        ..addNodes(nNodes: 2)
        ..addNodes(valueChildren: [0, 1]);
      expect(
        () => net.inputData(Float64List.fromList([1, 2, 3])),
        throwsArgumentError,
      );
      expect(() => net.step(Float64List.fromList([1])), throwsArgumentError);
      expect(
        () => net.step(
          Float64List.fromList([1, 2]),
          observed: Float64List.fromList([1]),
        ),
        throwsArgumentError,
      );
    });
  });

  group('export', () {
    test('CSV has a header and one row per node and time step', () {
      final traj = twoLevel().inputData(series.sublist(0, 3))!;
      final lines = const LineSplitter().convert(traj.toCsv());
      expect(lines.first, startsWith('time,node,mean,expected_mean'));
      expect(lines, hasLength(1 + 3 * 3));
    });

    test('JSON mirrors node_trajectories', () {
      final traj = twoLevel().inputData(series.sublist(0, 3))!;
      final json = jsonDecode(traj.toJsonString()) as Map<String, Object?>;
      expect(json.keys, ['0', '1', '2']);
      expect((json['1'] as Map)['mean'], hasLength(3));
    });
  });

  group('coupling functions', () {
    test('derivatives match finite differences', () {
      for (final fn in CouplingFn.values) {
        for (final x in [-2.3, -0.4, 0.3, 1.7]) {
          const h = 1e-5;
          final d1 = (fn.f(x + h) - fn.f(x - h)) / (2 * h);
          final d2 = (fn.df(x + h) - fn.df(x - h)) / (2 * h);
          expect(fn.df(x), closeTo(d1, 1e-6), reason: '${fn.name} df($x)');
          if (fn != CouplingFn.gelu) {
            // rshgf's GELU uses an erfc approximation, so its analytic d2f is
            // only close to the derivative of its df.
            expect(fn.d2f(x), closeTo(d2, 1e-6), reason: '${fn.name} d2f($x)');
          }
        }
      }
    });

    test('names round-trip through pyhgf names', () {
      for (final fn in CouplingFn.values) {
        expect(CouplingFn.parse(fn.pyhgfName), fn);
      }
      expect(CouplingFn.parse('identity'), CouplingFn.linear);
      expect(() => CouplingFn.parse('sin'), throwsArgumentError);
    });
  });

  group('CustomCoupling', () {
    final tanhCustom = CustomCoupling(
      'tanh',
      g: CouplingFn.tanh.f,
      dg: CouplingFn.tanh.df,
      d2g: CouplingFn.tanh.d2f,
    );

    Network net(Coupling c, {bool meanField = false}) =>
        Network(meanFieldUpdates: meanField)
          ..addNodes(precision: 100)
          ..addNodes(valueChildren: [0], coupling: c)
          ..addNodes(volatilityChildren: [1]);

    test('behaves exactly like the named function it wraps', () {
      for (final mf in [false, true]) {
        final a = net(tanhCustom, meanField: mf).inputData(series)!;
        final b = net(CouplingFn.tanh, meanField: mf).inputData(series)!;
        for (final f in defaultRecordedFields) {
          for (var i = 0; i < 3; i++) {
            expect(a.series(i, f), b.series(i, f), reason: '$f node $i');
          }
        }
      }
    });

    test('check() catches a wrong derivative', () {
      expect(tanhCustom.check, returnsNormally);
      final wrong = CustomCoupling(
        'sin',
        g: math.sin,
        dg: math.cos,
        d2g: math.sin, // sign error
      );
      expect(wrong.check, throwsStateError);
    });

    test('NetworkSpec resolves custom names, then named ones', () {
      final spec = NetworkSpec.fromJson({
        'nodes': [
          {'kind': 'continuous-state'},
          {
            'kind': 'continuous-state',
            'value_children': [0],
            'coupling_fn': 'mine',
          },
        ],
      });
      expect(spec.build, throwsArgumentError);
      final a = spec.build(couplings: {'mine': tanhCustom}).inputData(series)!;
      final b = Network()
        ..addNodes()
        ..addNodes(valueChildren: [0], coupling: CouplingFn.tanh);
      expect(
        a.series(1, NodeField.mean),
        b.inputData(series)!.series(1, NodeField.mean),
      );
    });
  });

  group('constant-state', () {
    Network biased(double alpha) => Network()
      ..addNodes(precision: 1e4)
      ..addNodes(valueChildren: [0], tonicVolatility: -4.0)
      ..addNodes(
        kind: NodeKind.constantState,
        valueChildren: [1],
        valueChildrenStrengths: [alpha],
      );

    test('adds α to the child drift and no variance', () {
      final plain = Network()
        ..addNodes(precision: 1e4)
        ..addNodes(valueChildren: [0], tonicVolatility: -4.0, tonicDrift: 0.3);
      final a = biased(0.3).inputSeries([0.1, 0.4, 0.2]);
      final b = plain.inputSeries([0.1, 0.4, 0.2]);
      for (final f in [NodeField.expectedMean, NodeField.expectedPrecision]) {
        for (var t = 0; t < 3; t++) {
          expect(a.at(1, f, t), closeTo(b.at(1, f, t), 1e-12), reason: '$f');
        }
      }
    });

    test('is never updated and is not an input', () {
      final net = biased(1.0);
      expect(net.inputIdxs, [0]);
      final seq = net.updateSequence;
      expect(
        [...seq.predictions, ...seq.updates].map((s) => s.$1),
        isNot(contains(2)),
      );
      net.inputSeries([5.0, -3.0]);
      expect(net.get(2, NodeField.mean), 1.0);
      expect(net.get(2, NodeField.expectedPrecision), double.infinity);
    });

    test('rejects parents, volatility edges and parameters', () {
      expect(
        () => Network()
          ..addNodes()
          ..addNodes(kind: NodeKind.constantState, volatilityChildren: [0]),
        throwsArgumentError,
      );
      expect(
        () => Network()
          ..addNodes()
          ..addNodes(kind: NodeKind.constantState, valueParents: [0]),
        throwsArgumentError,
      );
      expect(
        () => Network()
          ..addNodes()
          ..addNodes(
            kind: NodeKind.constantState,
            valueChildren: [0],
            mean: 2.0,
          ),
        throwsArgumentError,
      );
      // Pointing at a constant node as anything but a value parent.
      expect(
        () => Network()
          ..addNodes()
          ..addNodes(kind: NodeKind.constantState, valueChildren: [0])
          ..addNodes(volatilityChildren: [1]),
        throwsArgumentError,
      );
    });

    test('needs a value child', () {
      final net = Network()
        ..addNodes()
        ..addNodes(kind: NodeKind.constantState, valueChildren: [0]);
      expect(net.build, returnsNormally);
      // A child declared later through valueParents also counts.
      final late = Network()
        ..addNodes(kind: NodeKind.constantState)
        ..addNodes(valueParents: [0]);
      expect(late.inputIdxs, [1]);
      final orphan = Network()
        ..addNodes()
        ..addNodes(kind: NodeKind.constantState);
      expect(orphan.build, throwsStateError);
    });

    test('coupling is forced to linear', () {
      final a = Network()
        ..addNodes(precision: 1e4)
        ..addNodes(
          kind: NodeKind.constantState,
          valueChildren: [0],
          coupling: CouplingFn.tanh,
        );
      final b = Network()
        ..addNodes(precision: 1e4)
        ..addNodes(kind: NodeKind.constantState, valueChildren: [0]);
      expect(
        a.inputSeries([0.5]).at(0, NodeField.expectedMean, 0),
        b.inputSeries([0.5]).at(0, NodeField.expectedMean, 0),
      );
    });
  });

  test('lambertW0 solves w·e^w = z', () {
    for (final z in [0.0, 0.5, 1.0, 10.0, 1e6]) {
      final w = lambertW0(z);
      expect(w * math.exp(w), closeTo(z, 1e-9 * (1 + z)));
    }
  });

  group('inputData(into:)', () {
    test('reuses the columns and matches a fresh run', () {
      final net = twoLevel();
      final fresh = net.inputData(series)!;
      final other = Float64List.fromList([for (final v in series) v * 2]);
      final reused = net.inputData(other)!;
      final again = net.inputData(series, into: reused)!;
      expect(identical(again, reused), isTrue);
      for (final f in fresh.fields) {
        expect(again.raw(f), fresh.raw(f), reason: f.name);
      }
    });

    test('fills responseSurprise, or NaN without a response', () {
      final net = twoLevel();
      const rf = TotalGaussianSurprise();
      final fresh = net.inputData(series, response: rf)!;
      final reused = net.inputData(series, response: rf, into: fresh)!;
      final expected = Float64List.fromList(fresh.responseSurprise!);
      expect(reused.responseSurprise, expected);
      net.inputData(series, into: reused);
      expect(reused.responseSurprise!.every((v) => v.isNaN), isTrue);
    });

    test('rejects a different shape, fields or a missing column', () {
      final net = twoLevel();
      final traj = net.inputData(series)!;
      final shorter = Float64List.sublistView(series, 0, 100);
      expect(() => net.inputData(shorter, into: traj), throwsArgumentError);
      final four = Network()
        ..addNodes()
        ..addNodes(valueChildren: [0])
        ..addNodes(valueChildren: [1])
        ..addNodes(volatilityChildren: [2]);
      expect(() => four.inputData(series, into: traj), throwsArgumentError);
      final binary = Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0])
        ..addNodes(volatilityChildren: [1]);
      final u = Float64List(series.length);
      expect(() => binary.inputData(u, into: traj), throwsArgumentError);
      expect(
        () => net.inputData(series, fields: [NodeField.mean], into: traj),
        throwsArgumentError,
      );
      expect(
        () => net.inputData(
          series,
          response: const TotalGaussianSurprise(),
          into: traj,
        ),
        throwsArgumentError,
      );
    });
  });

  group('ehgfMgfCorrection (diverges from pyhgf)', () {
    // The 3-level HGF on USD/CHF, from the parity fixture.
    final fixture =
        jsonDecode(
              File(
                'test/fixtures/continuous_3level_eHGF.json',
              ).readAsStringSync(),
            )
            as Map<String, Object?>;
    final usdchf = Float64List.fromList([
      for (final row in fixture['inputs'] as List<Object?>)
        ((row as List<Object?>)[0] as num).toDouble(),
    ]);

    Network threeLevel(
      VolatilityUpdates v, {
      bool meanField = false,
      bool correction = false,
    }) => NetworkSpec.fromJson({
      ...fixture['spec'] as Map<String, Object?>,
      'network': {
        'volatility_updates': v.pyhgfName,
        'mean_field_updates': meanField,
        'ehgf_mgf_correction': correction,
      },
    }).build();

    int? firstNonFinite(Trajectories traj) {
      for (var t = 0; t < traj.nTime; t++) {
        for (var i = 0; i < traj.nNodes; i++) {
          if (!traj.at(i, NodeField.mean, t).isFinite) return t;
        }
      }
      return null;
    }

    test('off by default, and read from the spec', () {
      expect(Network().ehgfMgfCorrection, isFalse);
      expect(
        threeLevel(VolatilityUpdates.eHGF, correction: true).ehgfMgfCorrection,
        isTrue,
      );
    });

    test('keeps the relaxed 3-level eHGF finite where pyhgf diverges', () {
      final stock = threeLevel(VolatilityUpdates.eHGF).inputData(usdchf)!;
      expect(firstNonFinite(stock), 44); // as in pyhgf (JAX and rshgf)
      final fixed = threeLevel(
        VolatilityUpdates.eHGF,
        correction: true,
      ).inputData(usdchf)!;
      expect(firstNonFinite(fixed), isNull);
      // The volatility parent now gains precision from its child.
      var gains = 0;
      for (var t = 0; t < fixed.nTime; t++) {
        if (fixed.at(2, NodeField.precision, t) >
            fixed.at(2, NodeField.expectedPrecision, t)) {
          gains++;
        }
      }
      expect(gains, greaterThan(0.9 * fixed.nTime));
    });

    test('is a no-op for mean-field eHGF, standard and unbounded', () {
      for (final (v, mf) in [
        (VolatilityUpdates.eHGF, true),
        (VolatilityUpdates.standard, false),
        (VolatilityUpdates.unbounded, false),
        (VolatilityUpdates.unbounded, true),
      ]) {
        final a = threeLevel(v, meanField: mf).inputData(usdchf)!;
        final b = threeLevel(
          v,
          meanField: mf,
          correction: true,
        ).inputData(usdchf)!;
        for (final f in a.fields) {
          expect(b.raw(f), a.raw(f), reason: '$v meanField=$mf ${f.name}');
        }
      }
    });
  });
}
