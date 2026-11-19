import 'dart:typed_data';

import 'package:flutter/gestures.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf_widgets/hgf_widgets.dart';
import 'package:material_ui/material_ui.dart';

Float64List ramp(int n) =>
    Float64List.fromList([for (var t = 0; t < n; t++) t.toDouble()]);

Widget host(Widget child) => MaterialApp(home: Scaffold(body: child));

void main() {
  group('PlotViewport', () {
    test('null shows everything; resolve keeps windows inside', () {
      expect(PlotViewport.resolve(null, 100), (0.0, 99.0));
      expect(PlotViewport.resolve((-10, 10), 100), (0.0, 20.0));
      expect(PlotViewport.resolve((95, 120), 100), (74.0, 99.0));
      expect(PlotViewport.resolve((-5, 500), 100), (0.0, 99.0));
      expect(PlotViewport.resolve(null, 1), (0.0, 1.0));
    });

    test('zoom keeps the anchor in place, and zooming out resets', () {
      final v = PlotViewport();
      v.zoom(0.5, 25, 101); // anchor at 1/4 of (0, 100)
      expect(v.value, (12.5, 62.5));
      v.zoom(4, 25, 101);
      expect(v.value, isNull);
    });

    test('min span, pan clamps, follow', () {
      final v = PlotViewport();
      v.show(50, 50.5, 101);
      expect(v.value, (48.25, 52.25));
      v.pan(1000, 101);
      expect(v.value, (96.0, 100.0));
      v.show(10, 30, 101);
      v.follow(35, 101);
      expect(v.value!.$2, greaterThanOrEqualTo(35));
      expect(v.value!.$2 - v.value!.$1, closeTo(20, 1e-9));
      v.follow(2, 101);
      expect(v.value!.$1, lessThanOrEqualTo(2));
      v.reset();
      v.pan(5, 101); // not zoomed: nothing to pan
      expect(v.value, isNull);
    });
  });

  group('TrajectoryPlot zoom and pan', () {
    Future<(PlotViewport, ValueNotifier<int?>)> pumpPlot(
      WidgetTester tester, {
      bool inList = false,
    }) async {
      final vp = PlotViewport();
      final hover = ValueNotifier<int?>(null);
      addTearDown(vp.dispose);
      addTearDown(hover.dispose);
      final plot = TrajectoryPlot(
        layers: [LineLayer(ramp(1001))],
        viewport: vp,
        hover: hover,
      );
      await tester.pumpWidget(
        host(
          inList
              ? ListView(
                  children: [
                    SizedBox(height: 300, child: plot),
                    const SizedBox(height: 2000),
                  ],
                )
              : plot,
        ),
      );
      return (vp, hover);
    }

    Future<void> scroll(WidgetTester tester, Offset at, Offset delta) async {
      final p = TestPointer(1, PointerDeviceKind.mouse);
      await tester.sendEventToBinding(p.hover(at));
      await tester.sendEventToBinding(p.scroll(delta));
      await tester.pump();
    }

    testWidgets('wheel zooms about the pointer; double-click resets', (
      tester,
    ) async {
      final (vp, _) = await pumpPlot(tester);
      final c = tester.getCenter(find.byType(TrajectoryPlot));
      await scroll(tester, c, const Offset(0, -300));
      final (a, b) = vp.value!;
      expect(b - a, lessThan(1000));
      // The trial under the pointer stays (about) where it was.
      final r = PlotGeometry(tester.getSize(find.byType(TrajectoryPlot))).plot;
      final topLeft = tester.getTopLeft(find.byType(TrajectoryPlot));
      final before = (c.dx - topLeft.dx - r.left) / r.width * 1000;
      final after = a + (c.dx - topLeft.dx - r.left) / r.width * (b - a);
      expect(after, closeTo(before, 1e-6));

      await tester.tapAt(c);
      await tester.pump(const Duration(milliseconds: 50));
      await tester.tapAt(c);
      await tester.pumpAndSettle();
      expect(vp.value, isNull);
    });

    testWidgets('in a list, plain scroll scrolls and Ctrl + scroll zooms', (
      tester,
    ) async {
      final (vp, _) = await pumpPlot(tester, inList: true);
      final c = tester.getCenter(find.byType(TrajectoryPlot));
      await scroll(tester, c, const Offset(0, 40));
      expect(vp.value, isNull);
      expect(
        tester.state<ScrollableState>(find.byType(Scrollable)).position.pixels,
        greaterThan(0),
      );

      final c2 = tester.getCenter(find.byType(TrajectoryPlot));
      await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
      await scroll(tester, c2, const Offset(0, -300));
      await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
      expect(vp.value, isNotNull);
    });

    testWidgets('mouse drag pans a zoomed plot; sideways scroll pans', (
      tester,
    ) async {
      final (vp, _) = await pumpPlot(tester);
      vp.show(400, 500, 1001);
      await tester.pump();
      final c = tester.getCenter(find.byType(TrajectoryPlot));
      final g = await tester.startGesture(c, kind: PointerDeviceKind.mouse);
      for (var i = 0; i < 10; i++) {
        await g.moveBy(const Offset(-20, 0));
      }
      await g.up();
      await tester.pump();
      final (a, b) = vp.value!;
      expect(a, greaterThan(400));
      expect(b - a, closeTo(100, 1e-9));

      await scroll(tester, c, const Offset(-150, 0));
      expect(vp.value!.$1, lessThan(a));
      await tester.pump(const Duration(seconds: 1));
    });

    testWidgets('touch drag scrubs the hover; two fingers pinch', (
      tester,
    ) async {
      final (vp, hover) = await pumpPlot(tester);
      final c = tester.getCenter(find.byType(TrajectoryPlot));
      final one = await tester.startGesture(c, kind: PointerDeviceKind.touch);
      await one.moveBy(const Offset(60, 0));
      await one.moveBy(const Offset(10, 0));
      await tester.pump();
      expect(hover.value, isNotNull);
      expect(vp.value, isNull);
      await one.up();

      final f1 = await tester.startGesture(
        c - const Offset(50, 0),
        kind: PointerDeviceKind.touch,
        pointer: 7,
      );
      final f2 = await tester.startGesture(
        c + const Offset(50, 0),
        kind: PointerDeviceKind.touch,
        pointer: 8,
      );
      await f1.moveBy(const Offset(-50, 0));
      await f2.moveBy(const Offset(50, 0));
      await tester.pump();
      final (a, b) = vp.value!;
      // Fingers twice as far apart: half the trials.
      expect(b - a, closeTo(500, 1));
      await f1.up();
      await f2.up();
      await tester.pump(const Duration(seconds: 1));
    });

    testWidgets('a fixed xRange does not zoom', (tester) async {
      final vp = PlotViewport();
      addTearDown(vp.dispose);
      await tester.pumpWidget(
        host(
          TrajectoryPlot(
            layers: [LineLayer(ramp(100))],
            xRange: (10, 20),
            viewport: vp,
          ),
        ),
      );
      final c = tester.getCenter(find.byType(TrajectoryPlot));
      await scroll(tester, c, const Offset(0, -300));
      expect(vp.value, isNull);
    });
  });

  testWidgets('HgfTrajectoryView panels share the viewport', (tester) async {
    // Two plots under one viewport: zooming one moves the other.
    final vp = PlotViewport();
    addTearDown(vp.dispose);
    await tester.pumpWidget(
      host(
        Column(
          children: [
            PlotViewportBar(viewport: vp, length: 1001),
            Expanded(
              child: TrajectoryPlot(
                layers: [LineLayer(ramp(1001))],
                viewport: vp,
              ),
            ),
            Expanded(
              child: TrajectoryPlot(
                layers: [LineLayer(ramp(1001))],
                viewport: vp,
              ),
            ),
          ],
        ),
      ),
    );
    expect(find.text('all 1001 trials'), findsOneWidget);
    await tester.tap(find.byTooltip('Zoom in'));
    await tester.pump();
    expect(vp.value, (250.0, 750.0));
    expect(find.text('trials 250–750 of 1001'), findsOneWidget);

    // Dragging the bar's thumb pans.
    final track = tester.getCenter(find.byType(PlotViewportBar));
    await tester.dragFrom(track, const Offset(-40, 0));
    await tester.pump();
    expect(vp.value!.$1, lessThan(250));

    await tester.tap(find.byTooltip('Show everything'));
    await tester.pump();
    expect(vp.value, isNull);
    expect(tester.takeException(), isNull);
  });
}
