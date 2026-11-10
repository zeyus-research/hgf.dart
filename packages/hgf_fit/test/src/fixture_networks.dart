import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/codegen.dart';

double _sigmoid(double x) => 1 / (1 + math.exp(-x));

/// The custom couplings of hgf's parity fixtures.
final fixtureCouplings = <String, Coupling>{
  'sin': CustomCoupling(
    'sin',
    g: math.sin,
    dg: math.cos,
    d2g: (x) => -math.sin(x),
    d3g: (x) => -math.cos(x),
  ),
  'softplus': CustomCoupling(
    'softplus',
    g: (x) => math.max(x, 0.0) + math.log(1.0 + math.exp(-x.abs())),
    dg: _sigmoid,
    d2g: (x) => _sigmoid(x) * (1.0 - _sigmoid(x)),
  ),
};

double _decode(Object? v) => switch (v) {
  num() => v.toDouble(),
  'NaN' => double.nan,
  'Infinity' => double.infinity,
  '-Infinity' => double.negativeInfinity,
  _ => throw FormatException('$v'),
};

Float64List _flat(Object? rows) => Float64List.fromList([
  for (final row in rows as List<Object?>)
    if (row is List) ...row.map(_decode) else _decode(row),
]);

/// Changes parameters after building, as the playground's sliders do
/// (not on input nodes unless [inputs]).
void perturb(Network network, {bool inputs = true}) {
  for (var i = 0; i < network.nNodes; i++) {
    if (network.kindOf(i) != NodeKind.continuousState) continue;
    if (!inputs && network.inputIdxs.contains(i)) continue;
    network.setInitial(
      i,
      NodeField.tonicVolatility,
      network.getInitial(i, NodeField.tonicVolatility) - 0.25,
    );
    network.setInitial(i, NodeField.mean, 0.1 * i);
    final vp = network.valueParentsOf(i);
    if (vp.isNotEmpty && network.kindOf(vp.first) == NodeKind.continuousState) {
      network.setValueCoupling(vp.first, i, 0.75);
    }
  }
}

/// One of hgf's parity fixtures, as built (and perturbed, if [perturbed]).
final class FixtureNetwork {
  FixtureNetwork(
    this.name,
    this.spec,
    this.network,
    this.data,
    this.perturbed, {
    this.rshgfMaxRelDiff = 0,
  });

  final String name;

  /// The fixture's pyhgf-style spec.
  final Map<String, Object?> spec;
  final Network network;
  final ExportData data;
  final bool perturbed;

  /// Where pyhgf's two backends disagree (the parity suite then widens the
  /// JAX tolerance to twice this).
  final double rshgfMaxRelDiff;
}

/// Every fixture of `packages/hgf/test/fixtures`, plain and perturbed
/// (see [perturb]).
List<FixtureNetwork> fixtureNetworks({bool perturbInputs = true}) {
  final files =
      Directory('../hgf/test/fixtures')
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.json'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));
  return [
    for (final file in files)
      for (final perturbed in [false, true])
        if (jsonDecode(file.readAsStringSync())
            case final Map<String, Object?> fx)
          () {
            final net = NetworkSpec.fromJson(
              fx['spec'] as Map<String, Object?>,
            ).build(couplings: fixtureCouplings);
            if (perturbed) perturb(net, inputs: perturbInputs);
            return FixtureNetwork(
              '${fx['name']}${perturbed ? ' (perturbed)' : ''}',
              fx['spec'] as Map<String, Object?>,
              net,
              ExportData(
                inputs: _flat(fx['inputs']),
                observed: fx['observed'] == null ? null : _flat(fx['observed']),
                timeSteps: fx['time_steps'] == null
                    ? null
                    : _flat(fx['time_steps']),
              ),
              perturbed,
              rshgfMaxRelDiff:
                  (fx['rshgf_max_rel_diff'] as num?)?.toDouble() ?? 0,
            );
          }(),
  ];
}
