import 'package:flutter/gestures.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_playground/src/builder/builder_model.dart';
import 'package:hgf_playground/src/builder/ui/canvas_viewport.dart';
import 'package:hgf_playground/src/model/presets.dart';
import 'package:material_ui/material_ui.dart';

/// A canvas with a fixed content box and one draggable thing (a square at
/// the canvas origin, 0..40), recording what the viewport asks of it.
class _Probe extends StatefulWidget {
  const _Probe(this.content, this.fit);
  final Rect content;
  final ViewFit fit;

  @override
  State<_Probe> createState() => _ProbeState();
}

class _ProbeState extends State<_Probe> with CanvasViewport<_Probe> {
  final log = <String>[];
  int opened = 0;

  void reopen() => setState(() => opened++);

  @override
  Rect? get contentBounds => widget.content;
  @override
  ViewFit get viewFit => widget.fit;
  @override
  int get openedVersion => opened;

  bool _onThing(Offset screen) =>
      (Offset.zero & const Size(40, 40)).contains(toCanvas(screen));

  @override
  bool startDrag(Offset at) {
    if (!_onThing(at)) return false;
    log.add('start');
    return true;
  }

  @override
  void updateDrag(Offset at) => log.add('update');
  @override
  void endDrag() => log.add('end');
  @override
  void onTap(Offset at) => log.add(doubleTap(at) ? 'double' : 'tap');
  @override
  void openMenu(Offset at) => log.add('menu');

  /// The zoom the last painter was built with.
  double? painted;

  @override
  Widget build(BuildContext context) => viewport(
    painter: () => _Blank(painted = scale),
    onKey: (_, _) => KeyEventResult.ignored,
  );
}

class _Blank extends CustomPainter {
  _Blank(this.scale);
  final double scale;

  @override
  void paint(Canvas canvas, Size size) {}
  @override
  bool shouldRepaint(_Blank old) => false;
}

void main() {
  Future<_ProbeState> pump(
    WidgetTester tester,
    Size view,
    Rect content,
    ViewFit fit,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Align(
          alignment: Alignment.topLeft,
          child: SizedBox.fromSize(size: view, child: _Probe(content, fit)),
        ),
      ),
    );
    return tester.state<_ProbeState>(find.byType(_Probe));
  }

  const pipeline = ViewFit(
    margin: 40,
    maxScale: 1.2,
    narrowMinScale: 0.65,
    showStart: true,
  );

  group('fit to view', () {
    testWidgets('centres content that fits', (tester) async {
      final v = await pump(
        tester,
        const Size(800, 600),
        const Rect.fromLTWH(0, 0, 400, 200),
        pipeline,
      );
      // (400 + 80) wide at most 1.2×: 576 of 800.
      expect(v.scale, 1.2);
      // The first frame already paints the fitted view.
      expect(v.painted, 1.2);
      final box = const Rect.fromLTWH(0, 0, 400, 200).inflate(40);
      expect(v.toScreen(box.center), const Offset(400, 300));
    });

    testWidgets('on a phone, stops at a readable zoom and shows the start', (
      tester,
    ) async {
      // The pipeline of a preset, data to fit: 1316 wide.
      const content = Rect.fromLTWH(0, 0, 1316, 120);
      final v = await pump(tester, const Size(340, 380), content, pipeline);
      expect(v.scale, 0.65);
      // The left edge (with its margin) at the canvas's left; the height
      // fits, so it is centred.
      expect(v.toScreen(const Offset(-40, 0)).dx, closeTo(0, 1e-9));
      final midY = v.toScreen(content.center).dy;
      expect(midY, closeTo(190, 1e-9));
    });

    testWidgets('on a wide canvas, fits all of it as before', (tester) async {
      const content = Rect.fromLTWH(0, 0, 1800, 120);
      final v = await pump(tester, const Size(700, 500), content, pipeline);
      expect(v.scale, closeTo(700 / 1880, 1e-9));
      expect(v.toScreen(content.center).dx, closeTo(350, 1e-9));
    });

    testWidgets('without showStart, overflowing content stays centred', (
      tester,
    ) async {
      const content = Rect.fromLTWH(0, 0, 3000, 100);
      final v = await pump(
        tester,
        const Size(700, 500),
        content,
        const ViewFit(margin: 70, maxScale: 1.4),
      );
      expect(v.scale, 0.3);
      expect(v.toScreen(content.center).dx, closeTo(350, 1e-9));
    });

    testWidgets('again when a new document opens', (tester) async {
      final v = await pump(
        tester,
        const Size(800, 600),
        const Rect.fromLTWH(0, 0, 400, 200),
        pipeline,
      );
      v.zoomBy(2, Offset.zero);
      await tester.pump();
      expect(v.scale, 2.4);
      v.reopen();
      await tester.pump();
      expect(v.scale, 1.2);
    });
  });

  group('gestures', () {
    testWidgets('a drag on the background pans, on a thing drags it', (
      tester,
    ) async {
      final v = await pump(
        tester,
        const Size(800, 600),
        const Rect.fromLTWH(0, 0, 400, 200),
        pipeline,
      );
      final thing = v.toScreen(const Offset(20, 20));
      await tester.dragFrom(thing, const Offset(60, 0));
      expect(v.log.first, 'start');
      expect(v.log.last, 'end');
      expect(v.log.where((e) => e == 'update'), isNotEmpty);
      expect(v.toScreen(const Offset(20, 20)), thing); // not panned

      v.log.clear();
      final empty = v.toScreen(const Offset(300, 150));
      await tester.dragFrom(empty, const Offset(50, 30));
      expect(v.log, isEmpty);
      expect(v.toScreen(const Offset(20, 20)), thing + const Offset(50, 30));
    });

    testWidgets('taps, double taps, scroll zoom and right clicks', (
      tester,
    ) async {
      final v = await pump(
        tester,
        const Size(800, 600),
        const Rect.fromLTWH(0, 0, 400, 200),
        pipeline,
      );
      const at = Offset(500, 300);
      await tester.tapAt(at);
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapAt(at);
      await tester.pump(const Duration(milliseconds: 500));
      await tester.tapAt(at);
      expect(v.log, ['tap', 'double', 'tap']);

      // Scrolling zooms about the pointer.
      final under = v.toCanvas(at);
      tester.binding.handlePointerEvent(
        const PointerScrollEvent(position: at, scrollDelta: Offset(0, -200)),
      );
      await tester.pump();
      expect(v.scale, greaterThan(1.2));
      expect(v.toScreen(under).dx, closeTo(at.dx, 1e-9));

      // A right click whose mouse jitters opens the menu, and does not pan.
      v.log.clear();
      final before = v.pan;
      final g = await tester.startGesture(
        at,
        kind: PointerDeviceKind.mouse,
        buttons: kSecondaryMouseButton,
      );
      await g.moveBy(const Offset(3, 2));
      await g.up();
      await tester.pump();
      expect(v.log, ['menu']);
      expect(v.pan, before);
    });
  });

  group('the pipeline layout', () {
    test('has no empty columns', () {
      final m = BuilderModel()..openPreset(presetById('binary3'));
      expect(m.document.transforms, isEmpty);
      expect(m.document.response, isNull);
      final pos = m.pipelinePositions;
      expect(pos[DocPart.data], Offset.zero);
      expect(pos[DocPart.network], const Offset(280, 0));
      expect(pos[DocPart.fit], const Offset(560, 0));
    });

    test('keeps cards where they were when a column appears', () {
      final m = BuilderModel()..openPreset(presetById('binary3'));
      m.addTransform(at: const Offset(300, 300));
      final pos = m.pipelinePositions;
      expect(pos[DocPart.network], const Offset(280, 0));
      expect(pos[DocPart.fit], const Offset(560, 0));
      // Laid out again, the transforms get their column.
      m.autoLayoutPipeline();
      final again = m.pipelinePositions;
      expect(again[DocPart.network], const Offset(560, 0));
      expect(again[DocPart.fit], const Offset(840, 0));
      // Undo goes back to the pinned layout, and again to the automatic one.
      m.undo();
      expect(m.pipelinePositions[DocPart.network], const Offset(280, 0));
      m.undo();
      expect(m.document.transforms, isEmpty);
      expect(m.pipelinePositions[DocPart.network], const Offset(280, 0));
    });
  });
}
