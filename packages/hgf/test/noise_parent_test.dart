// Noise parents of continuous input nodes (pyhgf #477) and the P21 fix.
// Their parity with pyhgf is in parity_test (the noise_parent_* fixtures).
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

/// An input (0) with a tanh value parent (1) and a noise parent (2), and
/// either a bias parent of strength [bias] (3) or, with [asDrift], the same
/// bias as the input's tonic drift.
Network _network(
  VolatilityUpdates mode,
  double bias, {
  required bool asDrift,
  NetworkFixes? fixes,
}) {
  final net = Network(volatilityUpdates: mode, fixes: fixes)
    ..addNodes(tonicDrift: asDrift ? bias : 0.0)
    ..addNodes(
      valueChildren: [0],
      coupling: CouplingFn.tanh,
      tonicVolatility: -3,
    )
    ..addNodes(volatilityChildren: [0], tonicVolatility: -4);
  if (!asDrift) {
    net.addNodes(
      kind: NodeKind.constantState,
      valueChildren: [0],
      valueChildrenStrengths: [bias],
    );
  }
  return net;
}

Float64List _inputs(int n) {
  final rng = math.Random(3);
  return Float64List.fromList([
    for (var t = 0; t < n; t++)
      0.4 + 0.3 * math.sin(t / 9) + 0.2 * (rng.nextDouble() - 0.5),
  ]);
}

void main() {
  final u = _inputs(200);
  final steps = Float64List.fromList([
    for (var t = 0; t < u.length; t++) t.isEven ? 0.5 : 1.5,
  ]);

  for (final mode in VolatilityUpdates.values) {
    test('${mode.name}: with the P21 fix a bias parent is a tonic drift', () {
      final withBias = _network(mode, 0.4, asDrift: false);
      expect(withBias.fixes, NetworkFixes.recommended);
      final a = withBias.inputSeries(u, timeSteps: steps);
      final b = _network(
        mode,
        0.4,
        asDrift: true,
      ).inputSeries(u, timeSteps: steps);
      for (var i = 0; i < 3; i++) {
        for (final f in [
          NodeField.mean,
          NodeField.precision,
          NodeField.expectedMean,
          NodeField.expectedPrecision,
          NodeField.volatilityPredictionError,
        ]) {
          final x = a.series(i, f), y = b.series(i, f);
          for (var t = 0; t < x.length; t++) {
            expect(
              (x[t] - y[t]).abs() / math.max(1.0, y[t].abs()),
              lessThan(1e-12),
              reason: 'node $i ${f.name} t=$t: ${x[t]} vs ${y[t]}',
            );
          }
        }
      }
    });
  }

  test('without it (pyhgf) the bias parent adds (t α)² of noise variance', () {
    const alpha = 0.4;
    final net = _network(
      VolatilityUpdates.standard,
      alpha,
      asDrift: false,
      fixes: NetworkFixes.none,
    ).inputSeries(u.sublist(0, 1));
    final fixed = _network(
      VolatilityUpdates.standard,
      alpha,
      asDrift: false,
    ).inputSeries(u.sublist(0, 1));
    // ε = π̂_u (V + d²) − 1, so the extra variance is Δε / π̂_u.
    final eps = net.series(0, NodeField.volatilityPredictionError);
    final epsFixed = fixed.series(0, NodeField.volatilityPredictionError);
    final pihat = net.series(0, NodeField.expectedPrecision)[0];
    expect((eps[0] - epsFixed[0]) / pihat, closeTo(alpha * alpha, 1e-12));
  });

  test('fixes survive toJson / fromJson and default to defaultFixes', () {
    final saved = Network.defaultFixes;
    addTearDown(() => Network.defaultFixes = saved);
    final off = _network(
      VolatilityUpdates.eHGF,
      0.4,
      asDrift: false,
      fixes: NetworkFixes.none,
    );
    expect(Network.fromJson(off.toJson()).fixes, NetworkFixes.none);
    Network.defaultFixes = NetworkFixes.none;
    expect(Network().fixes, NetworkFixes.none);
    final json = off.toJson()..remove('fixes');
    expect(Network.fromJson(json).fixes, NetworkFixes.none);
    Network.defaultFixes = NetworkFixes.recommended;
    expect(Network.fromJson(json).fixes, NetworkFixes.recommended);
  });

  test('the noise prediction error runs after the value parents and before '
      'the noise parents', () {
    final net = _network(VolatilityUpdates.unbounded, 0.4, asDrift: false)
      ..build();
    final steps = net.updateSequence.updates;
    int at(int node, UpdateStep s) => steps.indexOf((node, s));
    final noise = at(0, UpdateStep.noisePredictionErrorContinuous);
    expect(noise, greaterThan(at(1, UpdateStep.posteriorContinuousUnbounded)));
    // The noise parent's only volatility child is a noise child: the
    // standard update in every scheme.
    expect(at(2, UpdateStep.posteriorContinuous), greaterThan(noise));
  });
}
