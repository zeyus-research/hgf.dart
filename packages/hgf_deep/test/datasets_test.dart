import 'dart:math' as math;

import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';
import 'package:test/test.dart';

void main() {
  group('generators', () {
    for (final (name, ds, classes) in [
      ('xor', xor(100, seed: 1), 2),
      ('moons', moons(100, seed: 1), 2),
      ('circles', circles(100, seed: 1), 2),
      ('spirals', spirals(99, seed: 1), 3),
    ]) {
      test('$name: shapes, balance and encoding', () {
        expect(ds.x.rows, ds.length);
        expect(ds.x.cols, 2);
        expect(ds.classes, classes);
        final counts = List.filled(classes, 0);
        for (final l in ds.labels!) {
          counts[l]++;
        }
        for (final c in counts) {
          expect(c, greaterThan(ds.length ~/ classes - 15));
        }
        if (classes == 2) {
          expect(ds.y.cols, 1);
          for (var i = 0; i < ds.length; i++) {
            expect(ds.y.at(i, 0), ds.labels![i].toDouble());
          }
        } else {
          expect(ds.y.cols, classes);
          for (var i = 0; i < ds.length; i++) {
            expect(ds.y.at(i, ds.labels![i]), 1.0);
          }
        }
      });
    }

    test('xor labels follow the quadrant signs', () {
      final ds = xor(200, noise: 0.0, seed: 2);
      for (var i = 0; i < ds.length; i++) {
        final differ = ds.x.at(i, 0) * ds.x.at(i, 1) < 0;
        expect(ds.labels![i], differ ? 1 : 0);
      }
    });

    test('deterministic per seed', () {
      expect(moons(50, seed: 7).x.data, moons(50, seed: 7).x.data);
      expect(moons(50, seed: 7).x.data, isNot(moons(50, seed: 8).x.data));
    });

    test('sine is regression', () {
      final ds = sine(30, noise: 0, seed: 3);
      expect(ds.isClassification, isFalse);
      expect(ds.classes, 0);
    });
  });

  test('split partitions the rows', () {
    final ds = moons(100, seed: 1);
    final (:train, :test) = split(ds, testFraction: 0.2, seed: 4);
    expect(train.length, 80);
    expect(test.length, 20);
    final all = {
      for (var i = 0; i < train.length; i++) '${train.x.row(i)}',
      for (var i = 0; i < test.length; i++) '${test.x.row(i)}',
    };
    expect(all.length, 100);
  });

  test('standardiser: zero mean, unit sd on the fitted set', () {
    final x = Matrix.fromRows([
      [1, 10, 5],
      [2, 20, 5],
      [3, 30, 5],
    ]);
    final z = Standardiser.fit(x).apply(x);
    for (var j = 0; j < 2; j++) {
      var m = 0.0, v = 0.0;
      for (var i = 0; i < 3; i++) {
        m += z.at(i, j);
      }
      for (var i = 0; i < 3; i++) {
        v += (z.at(i, j) - m / 3) * (z.at(i, j) - m / 3);
      }
      expect(m / 3, closeTo(0, 1e-12));
      expect(v / 3, closeTo(1, 1e-12));
    }
    expect([for (var i = 0; i < 3; i++) z.at(i, 2)], [0, 0, 0]);
  });

  group('metrics', () {
    test('accuracy and confusion, binary', () {
      final p = Matrix.column([0.9, 0.2, 0.6, 0.4]);
      expect(accuracy(p, [1, 0, 0, 0]), 0.75);
      expect(confusionMatrix(p, [1, 0, 0, 0], 2), [
        [2, 1],
        [0, 1],
      ]);
    });

    test('accuracy, categorical; NaN rows never count', () {
      final p = Matrix.fromRows([
        [0.1, 0.7, 0.2],
        [0.5, 0.3, 0.2],
        [double.nan, 0.5, 0.5],
      ]);
      expect(predictedClasses(p), [1, 0, -1]);
      expect(accuracy(p, [1, 0, 1]), closeTo(2 / 3, 1e-15));
    });

    test('mse, r2 and cross-entropy by hand', () {
      final y = Matrix.column([1, 2, 3]);
      final p = Matrix.column([1, 2, 4]);
      expect(mse(p, y), closeTo(1 / 3, 1e-15));
      // ss_tot = 2, ss_res = 1.
      expect(r2(p, y), closeTo(0.5, 1e-15));
      final pb = Matrix.column([0.8, 0.25]);
      final yb = Matrix.column([1, 0]);
      expect(
        crossEntropy(pb, yb),
        closeTo(-(math.log(0.8) + math.log(0.75)) / 2, 1e-12),
      );
      final pc = Matrix.fromRows([
        [0.7, 0.2, 0.1],
      ]);
      expect(crossEntropy(pc, oneHot([1], 3)), closeTo(-math.log(0.2), 1e-12));
      expect(
        crossEntropy(Matrix.column([double.nan]), Matrix.column([1])),
        isNaN,
      );
    });
  });
}
