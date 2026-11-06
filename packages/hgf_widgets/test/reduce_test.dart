import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

void main() {
  group('SeriesReducer.line', () {
    test('keeps every sample when sparse, split at gaps', () {
      final r = SeriesReducer()
        ..line(Float64List.fromList([1, 2, double.nan, 4, 5]), 0, 4, 100);
      expect(r.length, 4);
      expect(r.runEnds, [2, 4]);
      expect(r.x.sublist(0, 4), [0, 1, 3, 4]);
    });

    test('keeps each column\'s min and max in order when dense', () {
      final y = Float64List(10000);
      for (var t = 0; t < y.length; t++) {
        y[t] = (t % 7).toDouble();
      }
      y[5003] = 100; // a spike survives
      final r = SeriesReducer()..line(y, 0, y.length - 1, 100);
      expect(r.length, lessThanOrEqualTo(200));
      expect(r.y.sublist(0, r.length), contains(100.0));
      expect(r.runEnds, [r.length]);
    });

    test('all-NaN columns break the line', () {
      final y = Float64List(1000)..fillRange(0, 1000, 1);
      y.fillRange(400, 600, double.nan);
      final r = SeriesReducer()..line(y, 0, 999, 100);
      expect(r.runEnds.length, 2);
    });

    test('respects the visible window', () {
      final y = Float64List.fromList(List.generate(100, (i) => i * 1.0));
      final r = SeriesReducer()..line(y, 10, 19, 50);
      expect(r.length, 10);
      expect(r.y[0], 10);
    });
  });

  test('envelope takes the outermost bounds per column', () {
    final r = SeriesReducer()
      ..envelope(0, 999, 10, (t) => -t.toDouble(), (t) => t.toDouble());
    expect(r.length, 10);
    expect(r.y[9], -999);
    expect(r.y2[9], 999);
  });

  test('niceTicks', () {
    expect(niceTicks(0, 1), [0, 0.2, 0.4, 0.6000000000000001, 0.8, 1]);
    expect(niceTicks(0, 1, target: 2), [0, 0.5, 1]);
    expect(niceTicks(-3.3, 7.1, target: 3), [-2, 0, 2, 4, 6]);
    expect(niceTicks(1, 1), isEmpty);
    expect(niceTicks(0, double.infinity), isEmpty);
  });

  test('large tick labels stay distinct', () {
    final ticks = niceTicks(0, 130000);
    final step = ticks[1] - ticks[0];
    final labels = [for (final t in ticks) formatTick(t, step)];
    expect(labels.toSet().length, labels.length, reason: '$labels');
  });
}
