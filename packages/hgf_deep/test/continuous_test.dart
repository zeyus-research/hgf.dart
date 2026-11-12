// Continuous layers against hgf's nodalised Network, as pyhgf's
// tests/test_vectorised_continuous.py checks its vectorised layers against
// its nodalised backend: the same topology built both ways must filter the
// same observations to the same beliefs. Parity with pyhgf itself is in
// parity_test (the continuous_* fixtures).
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart'
    show Network, NetworkFixes, NodeField, Trajectories, VolatilityUpdates;
import 'package:hgf_deep/hgf_deep.dart';
import 'package:test/test.dart';

final _u = [0.2, 0.5, -0.3, 1.0, 0.1, -0.7, 0.4];

const _fields = {
  LayerField.mean: NodeField.mean,
  LayerField.precision: NodeField.precision,
  LayerField.expectedMean: NodeField.expectedMean,
  LayerField.expectedPrecision: NodeField.expectedPrecision,
};

VolatilityUpdates _nodal(DeepVolatilityUpdate v) => switch (v) {
  DeepVolatilityUpdate.standard => VolatilityUpdates.standard,
  DeepVolatilityUpdate.ehgf => VolatilityUpdates.eHGF,
  DeepVolatilityUpdate.unbounded => VolatilityUpdates.unbounded,
};

/// Filter [rows] through both networks and compare every node in
/// [nodeToLayer] (`node → (layer, position)`).
void _parity(
  Network nod,
  DeepNetwork deep,
  List<List<double>> rows,
  Map<int, (int, int)> nodeToLayer, {
  List<double>? timeSteps,
  double tol = 1e-9,
}) {
  final flat = Float64List.fromList([for (final r in rows) ...r]);
  final Trajectories traj = nod.inputData(
    flat,
    timeSteps: timeSteps == null ? null : Float64List.fromList(timeSteps),
  )!;
  deep.inputData(
    Matrix.fromRows(rows),
    timeSteps: timeSteps,
    record: _fields.keys.toSet(),
  );
  final n = rows.length;
  nodeToLayer.forEach((node, at) {
    final (layer, pos) = at;
    _fields.forEach((lf, nf) {
      final a = traj.series(node, nf);
      final width = deep.layerSizes[layer];
      final b = deep.trajectories!.of(lf, layer)!;
      for (var t = 0; t < n; t++) {
        final x = a[t], y = b[t * width + pos];
        expect(
          (x - y).abs() <= tol * math.max(1.0, x.abs()),
          isTrue,
          reason:
              'node $node / layer $layer[$pos] ${lf.pyhgfName} t=$t: '
              '$x vs $y',
        );
      }
    });
  });
}

List<List<double>> _col(List<double> u) => [
  for (final v in u) [v],
];
List<List<double>> _pair(List<double> u) => [
  for (final v in u) [v, v * 0.5 + 0.1],
];

Network _net(DeepVolatilityUpdate v, {bool meanField = false}) => Network(
  volatilityUpdates: _nodal(v),
  meanFieldUpdates: meanField,
  fixes: NetworkFixes.none,
);

DeepNetwork _deep(DeepVolatilityUpdate v, {bool meanField = false}) =>
    DeepNetwork(volatilityUpdates: v, meanFieldUpdates: meanField);

void main() {
  const all = DeepVolatilityUpdate.values;
  const dense = [DeepVolatilityUpdate.standard, DeepVolatilityUpdate.ehgf];

  test('two-level chain', () {
    final nod = _net(DeepVolatilityUpdate.standard)
      ..addNodes()
      ..addNodes(valueChildren: [0]);
    final deep = _deep(DeepVolatilityUpdate.standard)
      ..addLayer(1, kind: LayerKind.continuous)
      ..addLayer(1, kind: LayerKind.continuous);
    _parity(nod, deep, _col(_u), {0: (0, 0), 1: (1, 0)});
  });

  for (final v in all) {
    final oneToOne = v != DeepVolatilityUpdate.unbounded;
    test('${v.pyhgfName}: noise parent of the observation layer', () {
      final nod = _net(v)
        ..addNodes()
        ..addNodes(valueChildren: [0])
        ..addNodes(volatilityChildren: [0]);
      final deep = _deep(v)
        ..addLayer(1, kind: LayerKind.continuous)
        ..addLayer(1, kind: LayerKind.continuous)
        ..addLayer(
          1,
          kind: LayerKind.continuous,
          volatilityChildren: 0,
          volatilityFullyConnected: oneToOne,
        );
      _parity(nod, deep, _col(_u), {0: (0, 0), 1: (1, 0), 2: (2, 0)});
    });

    for (final mf in [false, true]) {
      test('${v.pyhgfName}${mf ? ' mean-field' : ''}: three levels', () {
        final nod = _net(v, meanField: mf)
          ..addNodes()
          ..addNodes(valueChildren: [0])
          ..addNodes(volatilityChildren: [1]);
        final deep = _deep(v, meanField: mf)
          ..addLayer(1, kind: LayerKind.continuous)
          ..addLayer(1, kind: LayerKind.continuous)
          ..addLayer(
            1,
            kind: LayerKind.continuous,
            volatilityChildren: 1,
            volatilityFullyConnected: oneToOne,
          );
        _parity(nod, deep, _col(_u), {0: (0, 0), 1: (1, 0), 2: (2, 0)});
      });
    }

    test('${v.pyhgfName}: noise parent below the value parent', () {
      final nod = _net(v)
        ..addNodes()
        ..addNodes(volatilityChildren: [0])
        ..addNodes(valueChildren: [0]);
      final deep = _deep(v)
        ..addLayer(1, kind: LayerKind.continuous)
        ..addLayer(
          1,
          kind: LayerKind.continuous,
          volatilityChildren: 0,
          volatilityFullyConnected: oneToOne,
        )
        ..addLayer(1, kind: LayerKind.continuous, valueChildren: 0);
      _parity(nod, deep, _col(_u), {0: (0, 0), 1: (1, 0), 2: (2, 0)});
    });
  }

  for (final v in dense) {
    test('${v.pyhgfName}: dense layers', () {
      final nod = _net(v)
        ..addNodes(nNodes: 2)
        ..addNodes(
          nNodes: 2,
          valueChildren: [0, 1],
          valueChildrenStrengths: [0.5, 0.5],
        )
        ..addNodes(volatilityChildren: [2, 3]);
      final deep = _deep(v)
        ..addLayer(2, kind: LayerKind.continuous)
        ..addLayer(2, kind: LayerKind.continuous)
        ..addLayer(1, kind: LayerKind.continuous, volatilityChildren: 1);
      _parity(nod, deep, _pair(_u), {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
        4: (2, 0),
      });
    });

    for (final mf in [false, true]) {
      test('${v.pyhgfName}${mf ? ' mean-field' : ''}: two volatility '
          'parents', () {
        final nod = _net(v, meanField: mf)
          ..addNodes()
          ..addNodes(valueChildren: [0])
          ..addNodes(
            volatilityChildren: [1],
            volatilityChildrenStrengths: [0.5],
          )
          ..addNodes(
            volatilityChildren: [1],
            volatilityChildrenStrengths: [0.5],
          );
        final deep = _deep(v, meanField: mf)
          ..addLayer(1, kind: LayerKind.continuous)
          ..addLayer(1, kind: LayerKind.continuous)
          ..addLayer(2, kind: LayerKind.continuous, volatilityChildren: 1);
        _parity(nod, deep, _col(_u), {
          0: (0, 0),
          1: (1, 0),
          2: (2, 0),
          3: (2, 1),
        });
      });
    }

    test('${v.pyhgfName}: value and volatility parent at once', () {
      final nod = _net(v)
        ..addNodes()
        ..addNodes(valueChildren: [0])
        ..addNodes(valueChildren: [1], volatilityChildren: [0]);
      final deep = _deep(v)
        ..addLayer(1, kind: LayerKind.continuous)
        ..addLayer(1, kind: LayerKind.continuous)
        ..addLayer(
          1,
          kind: LayerKind.continuous,
          valueChildren: 1,
          volatilityChildren: 0,
        );
      _parity(nod, deep, _col(_u), {0: (0, 0), 1: (1, 0), 2: (2, 0)});
    });

    test('${v.pyhgfName}: tanh value coupling', () {
      final nod = _net(v)
        ..addNodes()
        ..addNodes(valueChildren: [0], coupling: CouplingFn.tanh)
        ..addNodes(volatilityChildren: [0]);
      final deep = _deep(v)
        ..addLayer(1, kind: LayerKind.continuous)
        ..addLayer(1, kind: LayerKind.continuous, coupling: CouplingFn.tanh)
        ..addLayer(1, kind: LayerKind.continuous, volatilityChildren: 0);
      _parity(nod, deep, _col(_u), {0: (0, 0), 1: (1, 0), 2: (2, 0)});
    });
  }

  test('unbounded: one-to-one volatility layer', () {
    const v = DeepVolatilityUpdate.unbounded;
    final nod = _net(v)
      ..addNodes(nNodes: 2)
      ..addNodes(
        nNodes: 2,
        valueChildren: [0, 1],
        valueChildrenStrengths: [0.5, 0.5],
      )
      ..addNodes(volatilityChildren: [2])
      ..addNodes(volatilityChildren: [3]);
    final deep = _deep(v)
      ..addLayer(2, kind: LayerKind.continuous)
      ..addLayer(2, kind: LayerKind.continuous)
      ..addLayer(
        2,
        kind: LayerKind.continuous,
        volatilityChildren: 1,
        volatilityFullyConnected: false,
      );
    _parity(nod, deep, _pair(_u), {
      0: (0, 0),
      1: (0, 1),
      2: (1, 0),
      3: (1, 1),
      4: (2, 0),
      5: (2, 1),
    });
  });

  test('irregular time steps', () {
    const v = DeepVolatilityUpdate.standard;
    final nod = _net(v)
      ..addNodes()
      ..addNodes(valueChildren: [0])
      ..addNodes(volatilityChildren: [1]);
    final deep = _deep(v)
      ..addLayer(1, kind: LayerKind.continuous)
      ..addLayer(1, kind: LayerKind.continuous)
      ..addLayer(1, kind: LayerKind.continuous, volatilityChildren: 1);
    _parity(
      nod,
      deep,
      _col(_u),
      {0: (0, 0), 1: (1, 0), 2: (2, 0)},
      timeSteps: [1.0, 0.5, 2.0, 1.5, 1.0, 3.0, 0.1],
    );
  });

  test('parameter overrides', () {
    const v = DeepVolatilityUpdate.standard;
    final nod = _net(v)
      ..addNodes()
      ..addNodes(
        valueChildren: [0],
        tonicVolatility: -2.0,
        tonicDrift: 0.1,
        autoconnectionStrength: 0.9,
      );
    final deep = _deep(v)
      ..addLayer(1, kind: LayerKind.continuous)
      ..addLayer(
        1,
        kind: LayerKind.continuous,
        tonicVolatility: -2.0,
        tonicDrift: 0.1,
        autoconnectionStrength: 0.9,
      );
    _parity(nod, deep, _col(_u), {0: (0, 0), 1: (1, 0)});
  });

  group('builder', () {
    DeepNetwork two() =>
        DeepNetwork(volatilityUpdates: DeepVolatilityUpdate.standard)
          ..addLayer(1, kind: LayerKind.continuous)
          ..addLayer(1, kind: LayerKind.continuous);

    test('fan-in couplings', () {
      final net = DeepNetwork(volatilityUpdates: DeepVolatilityUpdate.ehgf)
        ..addLayer(3, kind: LayerKind.continuous)
        ..addLayer(4, kind: LayerKind.continuous, valueCoupling: 2.0)
        ..addLayer(
          2,
          kind: LayerKind.continuous,
          volatilityChildren: 1,
          volatilityCoupling: 0.5,
        );
      expect(net.weights[1]!.data, everyElement(0.5));
      expect(net.weights[1]!.rows, 3);
      expect(net.volatilityWeights[2]!.data, everyElement(0.25));
      expect(net.weights[2], isNull);
      expect(net.volatilityWeights[1], isNull);
      // The observation layer's defaults: λ = 0, ω = 0.
      expect(net.configs[0].autoconnectionStrength, 0.0);
      expect(net.configs[0].tonicVolatility, 0.0);
      expect(net.configs[1].tonicVolatility, -4.0);
      // Only layers with a volatility parent carry a volatility error.
      expect(net.state(1).volatilityPredictionError, isNotNull);
      expect(net.state(0).volatilityPredictionError, isNull);
      expect(net.state(1).meanVol, isNull);
    });

    test('rejects invalid layers and leaves the network as it was', () {
      final net = two();
      expect(
        () => net.addLayer(1),
        throwsArgumentError,
        reason: 'mixing kinds',
      );
      expect(
        () => net.addLayer(1, kind: LayerKind.continuous, valueChildren: 0),
        throwsArgumentError,
        reason: 'layer 0 already has a value parent',
      );
      expect(
        () => net.addLayer(1, kind: LayerKind.continuous, valueChildren: 5),
        throwsRangeError,
      );
      expect(
        () =>
            net.addLayer(1, kind: LayerKind.continuous, addConstantInput: true),
        throwsArgumentError,
      );
      expect(
        () =>
            net.addLayer(1, kind: LayerKind.continuous, tonicVolatilityVol: -2),
        throwsArgumentError,
      );
      expect(
        () => DeepNetwork().addLayer(2, tonicDrift: 0.1),
        throwsArgumentError,
      );
      expect(
        () => DeepNetwork()
          ..addLayer(1, kind: LayerKind.continuous)
          ..addLayer(2, kind: LayerKind.continuous, volatilityChildren: 0),
        throwsArgumentError,
        reason: 'unbounded needs one-to-one κ',
      );
      expect(
        () => DeepNetwork()
          ..addLayer(1, kind: LayerKind.continuous)
          ..addLayer(
            1,
            kind: LayerKind.continuous,
            valueChildren: 0,
            volatilityChildren: 0,
            volatilityFullyConnected: false,
          ),
        throwsArgumentError,
        reason: 'unbounded parents cannot also be value parents',
      );
      // A noise parent that is also the volatility child of the input's
      // value parent closes a cycle (pyhgf raises in the sweep).
      final cyc = DeepNetwork(volatilityUpdates: DeepVolatilityUpdate.standard)
        ..addLayer(1, kind: LayerKind.continuous)
        ..addLayer(1, kind: LayerKind.continuous, volatilityChildren: 0);
      expect(
        () => cyc.addLayer(
          1,
          kind: LayerKind.continuous,
          valueChildren: 0,
          volatilityChildren: 1,
        ),
        throwsArgumentError,
      );
      expect(cyc.nLayers, 2);
      expect(net.nLayers, 2);
    });

    test('deep entry points are rejected, and inputData elsewhere', () {
      final net = two();
      final x = Matrix.row([0.1]);
      expect(() => net.fit(x, x, optimiser: Sgd(0.1)), throwsStateError);
      expect(() => net.predict(x), throwsStateError);
      expect(() => net.batchUpdate(x, x), throwsStateError);
      expect(
        () => net.weightInitialisation(const XavierInit()),
        throwsStateError,
      );
      expect(
        () =>
            (DeepNetwork()
                  ..addLayer(1)
                  ..addLayer(1))
                .inputData(x),
        throwsStateError,
      );
      expect(
        () => net.inputData(x, record: {LayerField.meanVol}),
        throwsArgumentError,
      );
    });

    test('the state carries over between calls', () {
      final a = two()..inputData(Matrix.fromRows(_col(_u)));
      final b = two()
        ..inputData(Matrix.fromRows(_col(_u.sublist(0, 3))))
        ..inputData(Matrix.fromRows(_col(_u.sublist(3))));
      for (var l = 0; l < 2; l++) {
        expect(b.state(l).mean, a.state(l).mean);
        expect(b.state(l).precision, a.state(l).precision);
      }
    });

    test('toJson / fromJson round trip', () {
      final net = DeepNetwork(volatilityUpdates: DeepVolatilityUpdate.ehgf)
        ..addLayer(2, kind: LayerKind.continuous)
        ..addLayer(
          2,
          kind: LayerKind.continuous,
          coupling: CouplingFn.tanh,
          fullyConnected: false,
          tonicDrift: 0.2,
        )
        ..addLayer(
          1,
          kind: LayerKind.continuous,
          volatilityChildren: 0,
          volatilityCoupling: 0.7,
          initial: {LayerField.precision: 3.0},
        );
      net.weights[1]!.data[0] = 0.9;
      net.volatilityWeights[2]!.data[1] = -0.3;
      final rows = _pair(_u);
      net.inputData(Matrix.fromRows(rows.sublist(0, 4)));
      final again = DeepNetwork.fromJson(net.toJson());
      final a = net.inputData(Matrix.fromRows(rows.sublist(4)));
      final b = again.inputData(Matrix.fromRows(rows.sublist(4)));
      expect(b.data, a.data);
      for (var l = 0; l < 3; l++) {
        expect(again.state(l).mean, net.state(l).mean);
        expect(again.state(l).precision, net.state(l).precision);
      }
    });
  });
}
