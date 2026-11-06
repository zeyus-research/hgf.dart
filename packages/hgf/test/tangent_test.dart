@TestOn('vm')
@Tags(['parity'])
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'package:test/test.dart';

import 'src/tangent_check.dart';

void main() {
  final files =
      Directory('test/fixtures')
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.json'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

  group('BatchNetwork tangents match finite differences', () {
    for (final file in files) {
      final fixture =
          jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
      test(fixture['name'] as String, () {
        final result = checkTangents(fixture);
        if (result == null) {
          markTestSkipped('node kinds without batched kernels');
          return;
        }
        final (failures, compared) = result;
        expect(compared, greaterThan(100), reason: 'too few comparisons');
        expect(
          failures,
          isEmpty,
          reason:
              '${failures.length} failures:\n${failures.take(12).join('\n')}',
        );
      });
    }
  });

  group('response functions: gradient of the total surprise', () {
    final rng = math.Random(4);
    var x = 0.0;
    final cont = Float64List.fromList([
      for (var t = 0; t < 150; t++) x += 0.2 * (rng.nextDouble() - 0.5),
    ]);
    final bin = Float64List.fromList([
      for (var t = 0; t < 150; t++) rng.nextDouble() < 0.7 ? 1.0 : 0.0,
    ]);
    final decisions = Float64List.fromList([
      for (var t = 0; t < 150; t++) t % 5 == 0 ? double.nan : 1 - bin[t],
    ]);
    final reports = Float64List.fromList([
      for (var t = 0; t < 150; t++) cont[t] + 0.1 * (rng.nextDouble() - 0.5),
    ]);
    Network continuous() => Network()
      ..addNodes(precision: 1e2)
      ..addNodes(valueChildren: [0], tonicVolatility: -3)
      ..addNodes(volatilityChildren: [1], tonicVolatility: -4);
    Network binary() => Network()
      ..addNodes(kind: NodeKind.binaryState)
      ..addNodes(valueChildren: [0], tonicVolatility: -2);
    final cases = <(String, Network Function(), Float64List, ResponseFunction)>[
      (
        'FirstLevelGaussianSurprise',
        continuous,
        cont,
        const FirstLevelGaussianSurprise(),
      ),
      (
        'TotalGaussianSurprise',
        continuous,
        cont,
        const TotalGaussianSurprise(),
      ),
      (
        'GaussianResponse',
        continuous,
        cont,
        GaussianResponse(reports, precision: 50),
      ),
      (
        'GaussianResponse (mean of node 1)',
        continuous,
        cont,
        GaussianResponse(
          reports,
          precision: 50,
          node: 1,
          field: NodeField.mean,
        ),
      ),
      (
        'FirstLevelBinarySurprise',
        binary,
        bin,
        const FirstLevelBinarySurprise(),
      ),
      ('BinarySoftmax', binary, bin, BinarySoftmax(decisions)),
      (
        'BinarySoftmaxInverseTemperature',
        binary,
        bin,
        BinarySoftmaxInverseTemperature(decisions, inverseTemperature: 3),
      ),
    ];
    for (final (name, make, data, response) in cases) {
      test(name, () {
        expect(response.supportsGradients, isTrue);
        double total(double dOmega) {
          final net = make();
          final b = BatchNetwork(net, 1)
            ..setInitialAt(
              0,
              1,
              NodeField.tonicVolatility,
              net.getInitial(1, NodeField.tonicVolatility) + dOmega,
            );
          return b.surprise(data, response)[0];
        }

        final net = make();
        final b = BatchNetwork(net, 1, tangents: 2)
          ..setInitialTangent(
            1,
            NodeField.tonicVolatility,
            0,
            Float64List.fromList([1]),
          );
        final grad = Float64List(2);
        final t = b.surprise(data, response, gradient: grad);
        expect(t[0], total(0));
        double d(double h) => (total(h) - total(-h)) / (2 * h);
        final fd = (4 * d(1e-4) - d(2e-4)) / 3;
        expect(grad[0], closeTo(fd, 1e-6 * (1 + fd.abs())));
        expect(grad[1], 0.0, reason: 'direction 1 is seeded nowhere');
      });
    }
  });
}
