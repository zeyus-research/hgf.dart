import 'dart:convert';
import 'dart:math' as math;

import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';
import 'package:test/test.dart';

void main() {
  group('Dataset.fromCsv', () {
    const iris = '''
sepal_length,sepal_width,petal_length,species
5.1,3.5,1.4,setosa
7.0,3.2,4.7,versicolor
6.3,3.3,6.0,virginica
4.9,,1.4,setosa
"5.8",2.7,5.1,"virginica"
''';

    test('text classes become sorted labels; missing rows are dropped', () {
      final ds = Dataset.fromCsv(iris, target: ['species'], name: 'iris');
      expect(ds.length, 4);
      expect(ds.featureNames, ['sepal_length', 'sepal_width', 'petal_length']);
      expect(ds.classNames, ['setosa', 'versicolor', 'virginica']);
      expect(ds.labels, [0, 1, 2, 2]);
      expect(ds.y.cols, 3); // one-hot
      expect(ds.x.at(3, 0), 5.8);
    });

    test('features chosen by name; two classes give one 0/1 column', () {
      final ds = Dataset.fromCsv(
        'a;b;label\n1,5;2;yes\n-0,5;3;no\n',
        target: ['label'],
        features: ['b', 'a'],
      );
      expect(ds.featureNames, ['b', 'a']);
      expect(ds.x.data, [2, 1.5, 3, -0.5]);
      expect(ds.classNames, ['no', 'yes']);
      expect(ds.y.cols, 1);
      expect(ds.y.data, [1, 0]);
    });

    test('regression: real-valued or explicit, several targets', () {
      final ds = Dataset.fromCsv(
        'x\ty\tz\n0\t0.5\t1\n1\t1.5\t2\n2\t2.5\t3\n',
        target: ['y', 'z'],
      );
      expect(ds.isClassification, isFalse);
      expect(ds.y.cols, 2);
      expect(ds.y.data, [0.5, 1, 1.5, 2, 2.5, 3]);
      final counts = Dataset.fromCsv(
        'x,n\n0,1\n1,2\n2,3\n',
        target: ['n'],
        classification: false,
      );
      expect(counts.isClassification, isFalse);
      // Whole numbers with few values: classes by default.
      expect(Dataset.fromCsv('x,n\n0,1\n1,2\n2,1\n', target: ['n']).classes, 2);
    });

    test('errors name the column', () {
      expect(
        () => Dataset.fromCsv(iris, target: ['colour']),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('sepal_length'),
          ),
        ),
      );
      expect(
        () => Dataset.fromCsv(iris, target: ['sepal_length']),
        throwsFormatException,
      );
    });
  });

  test('minibatches cover every row once, per epoch', () {
    final ds = moons(103, seed: 1);
    final batches = minibatches(ds, 25, seed: 3).toList();
    expect([for (final b in batches) b.length], [25, 25, 25, 25, 3]);
    final rows = <String>{
      for (final b in batches)
        for (var i = 0; i < b.length; i++) '${b.x.at(i, 0)},${b.x.at(i, 1)}',
    };
    expect(rows.length, 103);
    final ordered = minibatches(ds, 50, shuffle: false).first;
    expect(ordered.x.at(0, 0), ds.x.at(0, 0));
  });

  group('DeepNetwork.toJson / fromJson', () {
    DeepNetwork trained() {
      final net =
          DeepNetwork(
              coupling: CouplingFn.tanh,
              tonicVolatility: true,
              volatilityUpdates: DeepVolatilityUpdate.ehgf,
            )
            ..addLayer(3, kind: LayerKind.categorical)
            ..addLayer(8, initial: {LayerField.precision: 2.0})
            ..addLayer(6, coupling: CouplingFn.relu, volatilityParent: false)
            ..addLayer(2)
            ..weightInitialisation(const XavierInit(), seed: 4);
      final ds = spirals(60, seed: 2);
      net.fit(ds.x, ds.y, optimiser: Adam(1e-2));
      return net;
    }

    test('round-trips through jsonEncode: same predictions and learning', () {
      final a = trained();
      a.debugLayers[1].tonicVolatility![2] = -3.25;
      final b = DeepNetwork.fromJson(
        jsonDecode(jsonEncode(a.toJson())) as Map<String, Object?>,
      );
      expect(b.layerSizes, a.layerSizes);
      expect(b.settings.volatilityUpdates, DeepVolatilityUpdate.ehgf);
      expect(b.debugLayers[1].tonicVolatility![2], -3.25);
      final ds = spirals(30, seed: 9);
      expect(b.predict(ds.x).data, a.predict(ds.x).data);
      for (var l = 0; l < a.nLayers; l++) {
        for (final f in LayerField.values) {
          expect(b.state(l).field(f), a.state(l).field(f), reason: '$l $f');
        }
      }
      // Learning on continues identically (with equal fresh optimisers).
      final pa = a.fit(ds.x, ds.y, optimiser: Adam(1e-2));
      final pb = b.fit(ds.x, ds.y, optimiser: Adam(1e-2));
      expect(pb.data, pa.data);
      expect(b.weights[2]!.data, a.weights[2]!.data);
    });

    test('non-finite beliefs and custom couplings', () {
      final sine = CustomCoupling(
        'sine',
        g: math.sin,
        dg: math.cos,
        d2g: (x) => -math.sin(x),
      );
      final net = DeepNetwork()
        ..addLayer(1)
        ..addLayer(2, coupling: sine);
      net.state(1).mean[0] = double.nan;
      net.state(1).precision[1] = double.infinity;
      final json = jsonDecode(jsonEncode(net.toJson())) as Map<String, Object?>;
      expect(() => DeepNetwork.fromJson(json), throwsArgumentError);
      final back = DeepNetwork.fromJson(json, couplings: {'sine': sine});
      expect(back.state(1).mean[0].isNaN, isTrue);
      expect(back.state(1).precision[1], double.infinity);
      expect(back.configs[1].coupling, same(sine));
      expect(
        () => DeepNetwork.fromJson({'format': 'x'}),
        throwsFormatException,
      );
    });
  });
}
