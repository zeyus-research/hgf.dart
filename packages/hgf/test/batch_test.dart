@TestOn('vm')
@Tags(['parity'])
library;

import 'dart:convert';
import 'dart:io';

import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

import 'src/batch_check.dart';

void main() {
  final files =
      Directory('test/fixtures')
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.json'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

  group('BatchNetwork matches Network bit for bit', () {
    for (final file in files) {
      final fixture =
          jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
      test(fixture['name'] as String, () {
        final failures = batchMismatches(fixture);
        expect(failures, isEmpty, reason: failures.take(10).join('\n'));
      });
    }
  });
  group('BatchNetwork API', () {
    Network hgf({double kappa = 1.0}) => Network()
      ..addNodes(precision: 1e4)
      ..addNodes(valueChildren: [0], mean: 1.0, tonicVolatility: -4.0)
      ..addNodes(
        volatilityChildren: [1],
        volatilityChildrenStrengths: [kappa],
        tonicVolatility: -3.0,
      );
    final u = Float64List.fromList([
      for (var t = 0; t < 200; t++) 1.0 + 0.01 * (t % 17) - 0.002 * t,
    ]);

    test('element state, initial values and read()', () {
      final batch = BatchNetwork(hgf(), 3)
        ..setInitial(
          1,
          NodeField.tonicVolatility,
          Float64List.fromList([-5, -4, -3]),
        )
        ..setInitialAt(2, 2, NodeField.mean, 0.5);
      expect(batch.batchSize, 3);
      expect(batch.nNodes, 3);
      expect(batch.inputIdxs, [0]);
      expect(batch.getInitial(0, 1, NodeField.tonicVolatility), -5);
      expect(batch.getInitial(2, 2, NodeField.mean), 0.5);
      expect(batch.get(2, 2, NodeField.mean), 0.0, reason: 'until reset');
      batch.reset();
      expect(batch.get(2, 2, NodeField.mean), 0.5);
      batch.inputData(u);
      final mu1 = batch.read(1, NodeField.mean);
      for (var e = 0; e < 3; e++) {
        expect(mu1[e], batch.get(e, 1, NodeField.mean));
      }
      expect(mu1[0], isNot(mu1[2]));
    });

    test('one element per parameter set equals a Network loop', () {
      final omegas = Float64List.fromList([-6, -5, -4, -3]);
      final batch = BatchNetwork(hgf(), omegas.length)
        ..setInitial(1, NodeField.tonicVolatility, omegas);
      final nll = batch.surprise(u, const TotalGaussianSurprise());
      for (var e = 0; e < omegas.length; e++) {
        final net = hgf()..setInitial(1, NodeField.tonicVolatility, omegas[e]);
        expect(nll[e], net.surprise(u, const TotalGaussianSurprise()));
      }
    });

    test('per-element coupling strengths', () {
      final kappas = Float64List.fromList([0.5, 1.0, 2.0]);
      final batch = BatchNetwork(hgf(), 3)
        ..setVolatilityCoupling(2, 1, kappas)
        ..inputData(u);
      for (var e = 0; e < 3; e++) {
        final net = hgf(kappa: kappas[e])..inputData(u, record: false);
        expect(batch.get(e, 2, NodeField.mean), net.get(2, NodeField.mean));
      }
      expect(
        () => batch.setVolatilityCoupling(1, 0, kappas),
        throwsArgumentError,
      );
      expect(
        () => batch.setValueCoupling(1, 0, Float64List(2)),
        throwsArgumentError,
      );
    });

    test('ResponseFunction.fn needs a batch form', () {
      final batch = BatchNetwork(hgf(), 2);
      final scalarOnly = ResponseFunction.fn(
        (net, t) => net.get(0, NodeField.mean),
      );
      expect(() => batch.surprise(u, scalarOnly), throwsUnsupportedError);
      final both = ResponseFunction.fn(
        (net, t) => net.get(1, NodeField.expectedMean),
        batch: (net, t, e) => net.get(e, 1, NodeField.expectedMean),
      );
      final totals = batch.surprise(u, both);
      expect(totals[0], hgf().surprise(u, both));
      expect(totals[1], totals[0]);
    });

    test('rejects bad arguments', () {
      expect(() => BatchNetwork(hgf(), 0), throwsArgumentError);
      expect(() => BatchNetwork(hgf(), 2, blockSize: 0), throwsArgumentError);
      final batch = BatchNetwork(hgf(), 2);
      expect(() => batch.get(2, 0, NodeField.mean), throwsRangeError);
      expect(() => batch.get(0, 3, NodeField.mean), throwsRangeError);
      expect(
        () => batch.setInitial(1, NodeField.mean, Float64List(3)),
        throwsArgumentError,
      );
      expect(
        () => batch.inputData(Float64List(5), perElement: true),
        throwsArgumentError,
      );
      expect(
        () => batch.inputData(u, observed: Float64List(3)),
        throwsArgumentError,
      );
      expect(() => batch.step(Float64List(2)), throwsArgumentError);
    });

    test('is unusable once the network gains nodes', () {
      final net = hgf();
      final batch = BatchNetwork(net, 2);
      net.addNodes(valueChildren: [2]);
      expect(() => batch.inputData(u), throwsStateError);
      net.build();
      expect(() => batch.inputData(u), throwsStateError);
    });
  });

  group('Network coupling setters', () {
    test('match building with that strength, before and after build', () {
      Network build(double kappa, double alpha) => Network()
        ..addNodes(precision: 1e4)
        ..addNodes(
          valueChildren: [0],
          valueChildrenStrengths: [alpha],
          mean: 1.0,
        )
        ..addNodes(
          volatilityChildren: [1],
          volatilityChildrenStrengths: [kappa],
        );
      final u = Float64List.fromList([for (var t = 0; t < 100; t++) t * 0.01]);
      final expected = build(1.7, 0.8)..inputData(u, record: false);

      final before = build(1.0, 1.0)
        ..setVolatilityCoupling(2, 1, 1.7)
        ..setValueCoupling(1, 0, 0.8)
        ..inputData(u, record: false);
      final after = build(1.0, 1.0)..build();
      after
        ..setVolatilityCoupling(2, 1, 1.7)
        ..setValueCoupling(1, 0, 0.8)
        ..inputData(u, record: false);
      for (final net in [before, after]) {
        for (var i = 0; i < 3; i++) {
          for (final f in NodeField.values) {
            expect(net.get(i, f), expected.get(i, f), reason: '$i $f');
          }
        }
      }
      expect(
        () => before.setVolatilityCoupling(1, 0, 1.0),
        throwsArgumentError,
      );
      expect(() => before.setValueCoupling(2, 1, 1.0), throwsArgumentError);
      expect(() => before.setValueCoupling(5, 1, 1.0), throwsRangeError);
    });
  });
}
