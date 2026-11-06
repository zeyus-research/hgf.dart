@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

import 'src/fixtures.dart';

/// Filters [fixture]'s inputs, recording every field.
Trajectories _run(Network network, Map<String, Object?> fixture) =>
    network.inputData(
      flatten(fixture['inputs']),
      observed: fixture['observed'] == null
          ? null
          : flatten(fixture['observed']),
      timeSteps: fixture['time_steps'] == null
          ? null
          : Float64List.fromList(series(fixture['time_steps'])),
      fields: NodeField.values,
    )!;

/// Bit-level equality (NaNs with the same payload compare equal).
void _sameBits(List<double> a, List<double> b, String what) {
  expect(a.length, b.length, reason: what);
  final x = Float64List.fromList(a).buffer.asUint64List();
  final y = Float64List.fromList(b).buffer.asUint64List();
  for (var i = 0; i < x.length; i++) {
    if (x[i] != y[i]) fail('$what[$i]: ${a[i]} != ${b[i]}');
  }
}

/// Changes parameters after build, which the snapshot must carry.
void _perturb(Network network) {
  for (var i = 0; i < network.nNodes; i++) {
    if (network.kindOf(i) != NodeKind.continuousState) continue;
    network.setInitial(
      i,
      NodeField.tonicVolatility,
      network.getInitial(i, NodeField.tonicVolatility) - 0.25,
    );
    final vp = network.valueParentsOf(i);
    if (vp.isNotEmpty && network.kindOf(vp.first) == NodeKind.continuousState) {
      network.setValueCoupling(vp.first, i, 0.75);
    }
  }
}

void main() {
  final files =
      Directory('test/fixtures')
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.json'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

  group('Network.toJson round trip filters bit for bit', () {
    for (final file in files) {
      final fixture =
          jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
      for (final perturbed in [false, true]) {
        test('${fixture['name']}${perturbed ? ' (perturbed)' : ''}', () {
          Network build() => NetworkSpec.fromJson(
            fixture['spec'] as Map<String, Object?>,
          ).build(couplings: testCouplings);
          final original = build();
          if (perturbed) _perturb(original);
          final text = jsonEncode(original.toJson());
          final copy = Network.fromJson(
            jsonDecode(text) as Map<String, Object?>,
            couplings: testCouplings,
          );
          expect(copy.nNodes, original.nNodes);
          expect(copy.inputIdxs, original.inputIdxs);
          expect(
            copy.updateSequence.toString(),
            original.updateSequence.toString(),
          );
          // Serialising the copy gives the same text.
          expect(jsonEncode(copy.toJson()), text);
          final a = _run(original, fixture), b = _run(copy, fixture);
          for (final f in NodeField.values) {
            _sameBits(b.raw(f), a.raw(f), f.name);
          }
          for (final f in VectorField.values) {
            final Float64List va;
            try {
              va = a.rawVector(f);
            } on ArgumentError {
              continue; // not recorded: the network has no such vectors
            }
            _sameBits(b.rawVector(f), va, f.name);
          }
        });
      }
    }
  });

  test('non-finite values and vector initials survive', () {
    final net = Network(maxPosteriorPrecision: double.infinity)
      ..addEfState(dimension: 2, learning: EfLearning.fixed)
      ..addNodes(precision: 1e4)
      ..addNodes(valueChildren: [1], tonicVolatility: -3);
    net.setInitial(2, NodeField.mean, double.negativeInfinity);
    final xis = net.getInitialVector(0, VectorField.xis);
    net.setInitialVector(0, VectorField.xis, [for (final x in xis) x + 0.5]);
    final copy = Network.fromJson(
      jsonDecode(jsonEncode(net.toJson())) as Map<String, Object?>,
    );
    expect(copy.maxPosteriorPrecision, double.infinity);
    expect(copy.getInitial(2, NodeField.mean), double.negativeInfinity);
    expect(copy.getInitialVector(0, VectorField.xis), [
      for (final x in xis) x + 0.5,
    ]);
    expect(copy.get(2, NodeField.mean), double.negativeInfinity);
  });

  test('custom couplings must be supplied', () {
    final c = CustomCoupling(
      'cube',
      g: (x) => x * x * x,
      dg: (x) => 3 * x * x,
      d2g: (x) => 6 * x,
    );
    final net = Network()
      ..addNodes()
      ..addNodes(valueChildren: [0], coupling: c);
    final json = net.toJson();
    expect(() => Network.fromJson(json), throwsArgumentError);
    expect(Network.fromJson(json, couplings: {'cube': c}).couplingOf(1), c);
  });

  test('rejects the pyhgf spec format', () {
    expect(
      () => Network.fromJson({
        'nodes': [
          {'kind': 'continuous-state'},
        ],
      }),
      throwsFormatException,
    );
  });

  test('a copy can grow like the original', () {
    final net = Network()
      ..addNodes(precision: 1e4)
      ..addNodes(valueChildren: [0], tonicVolatility: -3);
    final copy = Network.fromJson(net.toJson());
    for (final n in [net, copy]) {
      n.addNodes(volatilityChildren: [1], tonicVolatility: -5);
    }
    final u = [for (var t = 0; t < 50; t++) 0.1 * (t % 7)];
    _sameBits(
      copy.inputSeries(u).series(2, NodeField.mean),
      net.inputSeries(u).series(2, NodeField.mean),
      'mean',
    );
  });
}
