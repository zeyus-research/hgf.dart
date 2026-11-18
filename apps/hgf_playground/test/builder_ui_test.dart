import 'package:flutter/gestures.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_playground/src/app.dart';
import 'package:hgf_playground/src/builder/builder_model.dart';
import 'package:hgf_playground/src/builder/ui/builder_canvas.dart';
import 'package:hgf_playground/src/builder/ui/builder_page.dart';
import 'package:hgf_playground/src/builder/ui/pipeline_canvas.dart';
import 'package:hgf_playground/src/builder/ui/pipeline_inspector.dart'
    show Pick;
import 'package:hgf_playground/src/builder/ui/search_picker.dart';
import 'package:hgf_playground/src/model/presets.dart';
import 'package:material_ui/material_ui.dart';

/// The builder's interaction details: drags that leave the rest of the page
/// alone, context menus, searchable pickers and the phone layout.
void main() {
  Future<void> pumpPage(
    WidgetTester tester,
    BuilderModel m, {
    Size size = const Size(1300, 900),
  }) async {
    tester.view
      ..physicalSize = size
      ..devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(
      MaterialApp(
        theme: buildTheme(Brightness.light),
        home: BuilderPage(model: m),
      ),
    );
    await tester.pumpAndSettle();
  }

  group('moves', () {
    test('notify the layout only', () {
      final m = BuilderModel()..openPreset(presetById('continuous2'));
      var model = 0, layout = 0;
      m
        ..addListener(() => model++)
        ..layoutChanges.addListener(() => layout++);
      final id = m.doc.nodes.first.id;
      m.moveNode(id, const Offset(10, 20));
      m.movePart(DocPart.network, const Offset(600, 40));
      expect((model, layout), (0, 2));
      expect(m.positions[id], const Offset(10, 20));
      expect(m.pipelinePositions[DocPart.network], const Offset(600, 40));
    });

    testWidgets('a node drag repaints the canvas, not the page', (
      tester,
    ) async {
      final m = BuilderModel(doc: NetworkDocument(nodes: const []));
      m.addNode(NodeKind.continuousState, at: Offset.zero);
      m
        ..showView(BuilderView.network)
        ..select(null);
      await pumpPage(tester, m);
      var pageBuilds = 0;
      m.addListener(() => pageBuilds++);
      final box = tester.getRect(find.byType(BuilderCanvas));
      // The fitted view puts the only node in the middle.
      final g = await tester.startGesture(box.center);
      await tester.pump(const Duration(milliseconds: 20));
      for (var k = 1; k <= 10; k++) {
        await g.moveBy(const Offset(6, 0));
        await tester.pump(const Duration(milliseconds: 16));
      }
      await g.up();
      await tester.pumpAndSettle();
      // The drag's start: one undo step and the selection.
      expect(pageBuilds, 2);
      expect(m.positions['x0']!.dx, greaterThan(20));
    });
  });

  group('context menus', () {
    for (final view in [BuilderView.pipeline, BuilderView.network]) {
      testWidgets('right-click on the ${view.name} background', (tester) async {
        final m = BuilderModel()..openPreset(presetById('continuous2'));
        m.showView(view);
        await pumpPage(tester, m);
        final canvas = view == BuilderView.network
            ? find.byType(BuilderCanvas)
            : find.byType(PipelineCanvas);
        final at = tester.getRect(canvas).topLeft + const Offset(30, 30);
        // A real mouse moves a little between press and release.
        final g = await tester.startGesture(
          at,
          kind: PointerDeviceKind.mouse,
          buttons: kSecondaryButton,
        );
        await g.moveBy(const Offset(3, 1));
        await g.up();
        await tester.pumpAndSettle();
        expect(
          find.textContaining(
            view == BuilderView.network
                ? 'Add a continuous'
                : 'Add a transform',
          ),
          findsOneWidget,
        );
      });
    }
  });

  group('search picker', () {
    test('matches subscripts and every word', () {
      expect(searchMatches('ω x₁₂ · tonic volatility', 'x12 tonic'), isTrue);
      expect(searchMatches('ω x₁₂ · tonic volatility', 'X1 VOL'), isTrue);
      expect(searchMatches('ω x₁₂ · tonic volatility', 'x13'), isFalse);
      expect(searchMatches('anything', '  '), isTrue);
    });

    testWidgets('a long Pick searches; a short one drops down', (tester) async {
      String? picked;
      final long = {for (var i = 0; i < 20; i++) 'c$i': 'column $i'};
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: Column(
              children: [
                Pick<String>(
                  label: 'Long',
                  value: 'c3',
                  items: long,
                  onChanged: (v) => picked = v,
                ),
                Pick<String>(
                  label: 'Short',
                  value: 'a',
                  items: const {'a': 'A', 'b': 'B'},
                  onChanged: (v) => picked = v,
                ),
              ],
            ),
          ),
        ),
      );
      expect(find.byType(DropdownButton<String>), findsOneWidget);
      await tester.tap(find.text('column 3'));
      await tester.pumpAndSettle();
      expect(find.byType(TextField), findsOneWidget);
      await tester.enterText(find.byType(TextField), '17');
      await tester.pumpAndSettle();
      expect(find.byType(ListTile), findsOneWidget);
      await tester.tap(find.text('column 17'));
      await tester.pumpAndSettle();
      expect(picked, 'c17');
      expect(find.byType(TextField), findsNothing);
    });
  });

  group('phones', () {
    double sheetHeight(WidgetTester tester) => tester
        .getSize(
          find
              .descendant(
                of: find.byType(DraggableScrollableSheet),
                matching: find.byType(Material),
              )
              .first,
        )
        .height;

    testWidgets('selecting opens the properties sheet below the canvas', (
      tester,
    ) async {
      final m = BuilderModel()..openPreset(presetById('continuous2'));
      await pumpPage(tester, m, size: const Size(400, 860));
      final peek = sheetHeight(tester);
      expect(peek, lessThan(120));

      m.select(const PartSelection(DocPart.network));
      await tester.pumpAndSettle();
      final open = sheetHeight(tester);
      expect(open, greaterThan(peek + 150));
      // Its top stays below the canvas.
      final canvas = tester.getRect(find.byType(PipelineCanvas));
      final sheet = tester.getRect(
        find
            .descendant(
              of: find.byType(DraggableScrollableSheet),
              matching: find.byType(Material),
            )
            .first,
      );
      expect(sheet.top, greaterThanOrEqualTo(canvas.bottom));
      expect(find.text('Edit the network'), findsOneWidget);

      // The handle closes it again.
      await tester.tap(find.bySemanticsLabel('Properties: open or close'));
      await tester.pumpAndSettle();
      expect(sheetHeight(tester), closeTo(peek, 1));
    });
  });
}
