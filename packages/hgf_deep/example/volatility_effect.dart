// What a hidden layer's volatility level does to learning. Every volatile
// node of a deep HGF can carry an internal volatility level: its belief
// about how fast the node's value drifts between samples (Ω = e^μ_vol). A
// high volatility makes the node's prior vague, so each sample moves it
// further; the level learns from the size of those moves.
//
// The same 2 → 16 → 16 → 1 network learns the moons and circles tasks with:
//   * no volatility level in the hidden layers,
//   * a volatility level under each update scheme (pyhgf's default network),
//   * a volatility level plus a value-level tonic volatility ω = −4, which
//     keeps the drift small unless the level learns otherwise,
// each with hgf_deep's corrections (DeepFixes.recommended) and as pyhgf
// computes it (DeepFixes.none). Printed: test accuracy, whether the beliefs
// went NaN, and the hidden layers' mean μ_vol and predicted precision π̂ at
// the end.
//
//   dart run example/volatility_effect.dart [epochs]
import 'dart:math' as math;

import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';

import 'src/train.dart';

typedef Variant = ({
  String name,
  bool volatility,
  DeepVolatilityUpdate update,
  bool tonic,
});

const variants = <Variant>[
  (
    name: 'no volatility level',
    volatility: false,
    update: DeepVolatilityUpdate.unbounded,
    tonic: false,
  ),
  (
    name: 'volatility, unbounded',
    volatility: true,
    update: DeepVolatilityUpdate.unbounded,
    tonic: false,
  ),
  (
    name: 'volatility, eHGF',
    volatility: true,
    update: DeepVolatilityUpdate.ehgf,
    tonic: false,
  ),
  (
    name: 'volatility, standard',
    volatility: true,
    update: DeepVolatilityUpdate.standard,
    tonic: false,
  ),
  (
    name: 'volatility + ω = −4, standard',
    volatility: true,
    update: DeepVolatilityUpdate.standard,
    tonic: true,
  ),
];

String _run(Dataset ds, Variant v, DeepFixes fixes, int epochs) {
  final (:train, :test) = split(ds, seed: 1);
  final net =
      DeepNetwork(
          volatilityUpdates: v.update,
          tonicVolatility: v.tonic,
          fixes: fixes,
        )
        ..addLayer(1, kind: LayerKind.binary)
        ..addLayer(
          16,
          coupling: CouplingFn.tanh,
          volatilityParent: v.volatility,
        )
        ..addLayer(
          16,
          coupling: CouplingFn.tanh,
          volatilityParent: v.volatility,
        )
        ..addLayer(2)
        ..weightInitialisation(const XavierInit(), seed: 1);
  final adam = Adam(0.01);
  int? nanAt;
  for (var e = 1; e <= epochs; e++) {
    final b = shuffled(train, seed: e);
    net.fit(b.x, b.y, optimiser: adam);
    if (nanAt == null && firstNonFiniteLayer(net) != null) nanAt = e;
  }
  final acc = accuracy(net.predict(test.x), test.labels!);
  final stats = StringBuffer();
  if (v.volatility) {
    var mv = 0.0, lpi = 0.0, n = 0;
    for (var l = 1; l <= 2; l++) {
      final s = net.state(l);
      for (var k = 0; k < s.nodes; k++) {
        if (!s.meanVol![k].isFinite) continue;
        mv += s.meanVol![k];
        lpi += math.log(s.expectedPrecision[k]) / math.ln10;
        n++;
      }
    }
    stats.write(
      n == 0
          ? '  (all NaN)'
          : '  μ_vol ${(mv / n).toStringAsFixed(1).padLeft(6)}'
                '  π̂ 1e${(lpi / n).toStringAsFixed(0)}',
    );
  }
  return '${acc.toStringAsFixed(3)}'
      '${nanAt == null ? '         ' : ' NaN@${'$nanAt'.padRight(4)}'}'
      '$stats';
}

void main(List<String> args) {
  final epochs = args.isEmpty ? 20 : int.parse(args[0]);
  for (final ds in [moons(600, noise: 0.12), circles(600, noise: 0.08)]) {
    print('\n== ${ds.name} ($epochs epochs; test accuracy)');
    for (final v in variants) {
      print(v.name);
      for (final (label, fixes) in [
        ('  hgf_deep', DeepFixes.recommended),
        ('  pyhgf   ', DeepFixes.none),
      ]) {
        print('$label  ${_run(ds, v, fixes, epochs)}');
      }
    }
  }
}
