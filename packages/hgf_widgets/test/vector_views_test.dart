import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/gestures.dart';
import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

Widget host(Widget child) => MaterialApp(home: Scaffold(body: child));

List<TrajectoryPlot> plots(WidgetTester tester) =>
    tester.widgetList<TrajectoryPlot>(find.byType(TrajectoryPlot)).toList();

Future<void> hover(WidgetTester tester) async {
  final gesture = await tester.createGesture(kind: PointerDeviceKind.mouse);
  for (final p in find.byType(TrajectoryPlot).evaluate()) {
    await gesture.moveTo(tester.getCenter(find.byWidget(p.widget)));
    await tester.pump();
  }
  await gesture.removePointer();
}

void main() {
  testWidgets('categorical: binary parents folded, logits share a panel', (
    tester,
  ) async {
    final net = Network()..addCategoricalState(nCategories: 3);
    final cats = [0, 1, 2, 2, 1, 0, 0, 2];
    final data = Float64List(cats.length * 3);
    for (final (t, c) in cats.indexed) {
      data[t * 3 + c] = 1;
    }
    final traj = net.inputData(data)!;
    await tester.pumpWidget(
      host(HgfTrajectoryView(network: net, trajectories: traj)),
    );
    // Volatility parent, logits, categorical input, surprise.
    expect(HgfTrajectoryView.panelCount(net), 4);
    final p = plots(tester);
    expect(p.map((p) => p.title), [
      'x₇',
      'x₄–x₆ (category logits)',
      'x₀ (categorical input)',
      'surprise',
    ]);
    // One probability line and one observation row per category; the
    // observed category's row has a dot at each trial.
    final layers = p[2].layers;
    expect(layers.whereType<LineLayer>(), hasLength(3));
    final rows = layers.whereType<PointLayer>().toList();
    expect(rows, hasLength(3));
    for (final (t, c) in cats.indexed) {
      for (var k = 0; k < 3; k++) {
        expect(rows[k].values[t].isFinite, k == c, reason: 't=$t k=$k');
      }
    }
    await hover(tester);
    expect(tester.takeException(), isNull);
  });

  testWidgets('dp: clusters folded, observations split by cluster', (
    tester,
  ) async {
    final net = Network()
      ..addDpState(batchSize: 3, alpha: 0.1)
      ..addEfState(nNodes: 3, valueChildren: [0], nus: 3.0);
    final u = Float64List.fromList([
      for (var t = 0; t < 30; t++) (t ~/ 10) * 4.0 - 4 + 0.01 * (t % 3),
    ]);
    final traj = net.inputData(u)!;
    await tester.pumpWidget(
      host(HgfTrajectoryView(network: net, trajectories: traj)),
    );
    // No continuous or binary node: no surprise panel.
    expect(HgfTrajectoryView.panelCount(net), 1);
    final layers = plots(tester).single.layers;
    final points = layers.whereType<PointLayer>().toList();
    final counts = traj.vectorAt(0, VectorField.n, 29);
    // One point layer per cluster that got observations; every observation
    // is in exactly one, at its value.
    expect(points, hasLength(counts.where((c) => c > 0).length));
    for (var t = 0; t < 30; t++) {
      final hits = points.where((p) => p.values[t].isFinite).toList();
      expect(hits, hasLength(1), reason: 't=$t');
      expect(hits.single.values[t], u[t]);
    }
    // A predicted-mean line per opened cluster, starting the trial after
    // it opened.
    final lines = layers.whereType<LineLayer>().toList();
    expect(lines, hasLength(points.length));
    expect(lines.first.values[0].isNaN, isTrue);
    expect(lines.first.values[1].isFinite, isTrue);
    await hover(tester);
    expect(tester.takeException(), isNull);
  });

  testWidgets('ef-state: mean ± sd from the previous trial\'s ξ', (
    tester,
  ) async {
    final net = Network()..addEfState(nus: 4);
    final u = Float64List.fromList([1, 2, 0, 3, 2]);
    final traj = net.inputData(u)!;
    await tester.pumpWidget(
      host(HgfTrajectoryView(network: net, trajectories: traj)),
    );
    final layers = plots(tester).single.layers;
    final band = layers.whereType<BandLayer>().single;
    final mean = layers.whereType<LineLayer>().single;
    final xi = traj.vectorSeries(0, VectorField.xis);
    expect(mean.values[0].isNaN, isTrue);
    for (var t = 1; t < 5; t++) {
      final m = xi[(t - 1) * 2], sd2 = xi[(t - 1) * 2 + 1] - m * m;
      expect(mean.values[t], m);
      expect(band.valueAt(t), closeTo(math.sqrt(math.max(0, sd2)), 1e-12));
    }
  });

  test('categories past the palette share the overflow colour', () {
    const c = HgfColors.light;
    expect(c.category(0), c.belief);
    expect(c.category(7), isNot(c.overflow));
    expect(c.category(8), c.overflow);
  });
}
