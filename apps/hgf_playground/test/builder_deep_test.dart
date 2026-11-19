import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_deep/hgf_deep.dart' show LayerKind;
import 'package:hgf_fit/document.dart';
import 'package:hgf_playground/src/app.dart';
import 'package:hgf_playground/src/builder/builder_model.dart';
import 'package:hgf_playground/src/builder/ui/builder_page.dart';
import 'package:hgf_playground/src/builder/ui/deep_canvas.dart';
import 'package:hgf_playground/src/builder/ui/deep_inspector.dart';
import 'package:hgf_playground/src/builder/ui/pipeline_canvas.dart';
import 'package:hgf_playground/src/model/presets.dart';
import 'package:material_ui/material_ui.dart';

const _csv =
    'x1,x2,press\n'
    '0.1,0.5,left\n'
    '0.4,-0.2,right\n'
    '-0.3,0.1,none\n'
    '0.2,0.2,left\n';

BuilderModel _deepOnly() {
  final m = BuilderModel(doc: NetworkDocument(nodes: const []));
  m.removePart(DocPart.network);
  m.addDeep();
  return m;
}

BuilderModel _continuousOnly() {
  final m = _deepOnly();
  final d = m.document.deepById('deep')!;
  m.setDeep(
    'deep',
    d.copyWith(
      layers: const [
        DocLayer(size: 1, kind: LayerKind.continuous),
        DocLayer(size: 1, kind: LayerKind.continuous, tonicVolatility: -2),
      ],
      features: const [],
      targets: const [],
      categories: () => null,
    ),
  );
  return m;
}

void main() {
  group('BuilderModel with deep networks', () {
    test('a deep network alone: no HGF cards, only its own checks', () {
      final m = _deepOnly();
      expect(m.document.hasNetwork, isFalse);
      expect(m.pipelineParts, [DocPart.data, DocPart.deep('deep')]);
      expect(m.issues.where((i) => i.isError), isEmpty, reason: '${m.issues}');
      // Without columns yet: warnings that name the network's card.
      expect(
        m.issues.every((i) => i.parts.contains(DocPart.deep('deep'))),
        isTrue,
      );
      // Without data: nothing to train on yet.
      expect(m.previewError, contains('Load data to train'));
    });

    test('a preset plus a deep network; deleting the HGF network', () {
      final m = BuilderModel()..openPreset(presetById('continuous2'));
      final id = m.addDeep();
      expect(m.pipelineParts, contains(DocPart.network));
      expect(m.pipelineParts, contains(DocPart.deep(id)));
      expect(m.trajectories, isNotNull);
      m.removePart(DocPart.network);
      expect(m.document.hasNetwork, isFalse);
      expect(m.document.inputs, isEmpty);
      expect(m.pipelineParts, isNot(contains(DocPart.fit)));
      m.undo();
      expect(m.document.hasNetwork, isTrue);
      expect(m.trajectories, isNotNull);
    });

    test('columns size the layers; text targets become labels', () {
      final m = _deepOnly()..loadData(CsvData.parse(_csv), name: 'presses.csv');
      m
        ..bindDeepColumn('deep', 'x1', target: false)
        ..bindDeepColumn('deep', 'x2', target: false)
        ..bindDeepColumn('deep', 'press', target: true);
      final d = m.document.deepById('deep')!;
      expect(d.features, ['x1', 'x2']);
      expect(d.featureLayer!.size, 2);
      expect(d.targets, ['press']);
      expect(d.categories, ['left', 'right', 'none']);
      expect(d.targetLayer!.kind, LayerKind.categorical);
      expect(d.targetLayer!.size, 3);
      expect(m.issues, isEmpty, reason: '${m.issues}');
      m.unbindDeepColumn('deep', 'x2', target: false);
      expect(m.document.deepById('deep')!.featureLayer!.size, 1);
    });

    test('insert, edit and remove layers; undo', () {
      final m = _deepOnly()..showDeep('deep');
      expect(m.view, BuilderView.deep);
      m.insertLayer(1); // above the hidden layer
      var d = m.deep!;
      expect(d.layers, hasLength(4));
      expect(m.selectedLayer, 2);
      m.setLayer(2, d.layers[2].copyWith(size: 3, coupling: () => 'relu'));
      d = m.deep!;
      expect(d.layers[2].size, 3);
      expect(d.build().configs[2].coupling, CouplingFn.relu);
      m.deleteSelection();
      expect(m.deep!.layers, hasLength(3));
      // The feature and target layers stay.
      m.removeLayer(0);
      m.removeLayer(2);
      expect(m.deep!.layers, hasLength(3));
      m.undo();
      expect(m.deep!.layers, hasLength(4));
    });

    test('continuous levels: insert, rewire and remove keep the chain', () {
      final m = _continuousOnly()..showDeep('deep');
      // observations ← level 1; a level on top predicts level 1's
      // volatility.
      m.insertLayer(1);
      var d = m.deep!;
      expect(d.layers, hasLength(3));
      expect(m.selectedLayer, 2);
      m.setLayer(2, d.withChildren(2, value: null, volatility: 1));
      d = m.deep!;
      List<(int?, int?)> wiring() => [
        for (final c in m.deep!.build().configs)
          (c.valueChild, c.volatilityChild),
      ];
      expect(wiring(), [(null, null), (0, null), (null, 1)]);
      // A level between the observations and level 1 slots into the chain;
      // the volatility parent follows level 1 up.
      m.insertLayer(0);
      expect(wiring(), [(null, null), (0, null), (1, null), (null, 2)]);
      expect(m.deep!.layers[1].size, m.deep!.layers[0].size);
      // Any level above the observations can go while two layers remain.
      expect(m.canRemoveLayer(3), isTrue);
      expect(m.canRemoveLayer(0), isFalse);
      m.removeLayer(1);
      expect(wiring(), [(null, null), (0, null), (null, 1)]);
      m.removeLayer(2);
      m.removeLayer(1);
      expect(m.deep!.layers, hasLength(2), reason: 'two layers stay');
      expect(m.issues.where((i) => i.isError), isEmpty, reason: '${m.issues}');
    });

    test('a renamed deep network keeps its card; the JSON round-trips', () {
      final m = _deepOnly();
      m.setDeep('deep', m.document.deepById('deep')!.copyWith(name: 'press'));
      final json = m.toJsonText();
      final back = ModelDocument.fromJson(
        jsonDecode(json) as Map<String, Object?>,
      );
      expect(back.deep.single.label, 'press');
      expect(back.hasNetwork, isFalse);
    });
  });

  group('BuilderPage with deep networks', () {
    for (final width in [1300.0, 480.0]) {
      testWidgets('renders the pipeline, the layers and the panels '
          '${width.toInt()} wide', (tester) async {
        tester.view
          ..physicalSize = Size(width, 900)
          ..devicePixelRatio = 1;
        addTearDown(tester.view.reset);
        final m = BuilderModel()..openPreset(presetById('continuous2'));
        m.addDeep();
        await tester.pumpWidget(
          MaterialApp(
            theme: buildTheme(Brightness.light),
            home: BuilderPage(model: m),
          ),
        );
        await tester.pumpAndSettle();
        // Narrow, the inspector is below the fold (built lazily).
        final wide = width >= 900;
        if (wide) expect(find.byType(DeepPanel), findsOneWidget);
        m.showDeep('deep');
        await tester.pumpAndSettle();
        expect(find.byType(DeepCanvas), findsOneWidget);
        m.select(const LayerSelection(1));
        await tester.pumpAndSettle();
        if (wide) expect(find.byType(DeepLayerPanel), findsOneWidget);
        m.removePart(DocPart.network);
        m.showView(BuilderView.pipeline);
        await tester.pumpAndSettle();
        expect(tester.takeException(), isNull);
      });
    }

    testWidgets('drag a column onto the deep card; open it; insert a layer', (
      tester,
    ) async {
      tester.view
        ..physicalSize = const Size(1300, 900)
        ..devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      final m = _deepOnly()..loadData(CsvData.parse(_csv), name: 'p.csv');
      await tester.pumpWidget(
        MaterialApp(
          theme: buildTheme(Brightness.light),
          home: BuilderPage(model: m),
        ),
      );
      await tester.pumpAndSettle();

      final box = tester.getRect(find.byType(PipelineCanvas));
      final cards = pipelineGeometry(m).cards;
      var r = cards.first.rect;
      for (final c in cards.skip(1)) {
        r = r.expandToInclude(c.rect);
      }
      final bounds = r.inflate(40);
      final scale = [
        box.width / bounds.width,
        box.height / bounds.height,
      ].reduce((a, b) => a < b ? a : b).clamp(0.3, 1.2);
      final pan = box.size.center(Offset.zero) - bounds.center * scale;
      Offset screen(Offset p) => box.topLeft + pan + p * scale;

      final data = cards.first;
      final deep = cards.firstWhere((c) => c.part == DocPart.deep('deep'));
      Future<void> drag(String column, int row) async {
        final from = screen(
          data.rowPort(data.rows.indexWhere((x) => x.label == column)),
        );
        final to = screen(deep.rowCentre(row) + const Offset(60, 0));
        final g = await tester.startGesture(from);
        await tester.pump(const Duration(milliseconds: 20));
        for (var k = 1; k <= 10; k++) {
          await g.moveTo(Offset.lerp(from, to, k / 10)!);
          await tester.pump(const Duration(milliseconds: 16));
        }
        await g.up();
        await tester.pumpAndSettle();
      }

      // Under the status row: "drop feature columns here", the targets.
      int port({required bool target}) => deep.rows.indexWhere(
        (x) => x.port is DeepIn && (x.port! as DeepIn).target == target,
      );
      expect(deep.rows.first.port, isNull);
      await drag('x1', port(target: false));
      expect(m.document.deepById('deep')!.features, ['x1']);
      await drag('press', port(target: true));
      expect(m.document.deepById('deep')!.categories, [
        'left',
        'right',
        'none',
      ]);

      // Double-click opens the layers.
      final head = screen(
        pipelineGeometry(
              m,
            ).cards.firstWhere((c) => c.part == DocPart.deep('deep')).at +
            const Offset(cardWidth / 2, 12),
      );
      await tester.tapAt(head);
      await tester.pump(const Duration(milliseconds: 50));
      await tester.tapAt(head);
      await tester.pumpAndSettle();
      expect(m.view, BuilderView.deep);

      final canvas = tester.getRect(find.byType(DeepCanvas));
      final g = DeepGeometry(m.deep!, canvas.size);
      await tester.tapAt(canvas.topLeft + g.plus[1]);
      await tester.pumpAndSettle();
      expect(m.deep!.layers, hasLength(4));
      expect(find.byType(DeepLayerPanel), findsOneWidget);
      final g2 = DeepGeometry(m.deep!, canvas.size);
      await tester.tapAt(canvas.topLeft + g2.bars[0].center);
      await tester.pumpAndSettle();
      expect(m.selectedLayer, 0);
      expect(find.text('target'), findsWidgets);
    });

    for (final width in [1300.0, 480.0]) {
      testWidgets('a continuous level\'s wiring in the inspector '
          '${width.toInt()} wide', (tester) async {
        tester.view
          ..physicalSize = Size(width, 900)
          ..devicePixelRatio = 1;
        addTearDown(tester.view.reset);
        final m = _continuousOnly()..showDeep('deep');
        m.insertLayer(1);
        final d = m.deep!;
        m.setLayer(2, d.withChildren(2, value: null, volatility: 1));
        await tester.pumpWidget(
          MaterialApp(
            theme: buildTheme(Brightness.light),
            home: BuilderPage(model: m),
          ),
        );
        await tester.pumpAndSettle();
        expect(find.byType(DeepCanvas), findsOneWidget);
        if (width >= 900) {
          expect(find.byType(DeepLayerPanel), findsOneWidget);
          expect(
            find.text('Volatility parent of (predicts its volatility)'),
            findsOneWidget,
          );
          expect(find.text('Volatility coupling κ (empty: 1)'), findsOneWidget);
          await tester.tap(find.text('Add a level above'));
          await tester.pumpAndSettle();
          expect(m.deep!.layers, hasLength(4));
          expect(m.deep!.valueChildOf(3), 2);
          // The observation layer: its noise, no wiring.
          m.select(const LayerSelection(0));
          await tester.pumpAndSettle();
          expect(find.text('Observation noise ω (empty: 0)'), findsOneWidget);
          expect(find.textContaining('Value parent of'), findsNothing);
        }
        expect(tester.takeException(), isNull);
      });
    }
  });
}
