import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_playground/src/app.dart';
import 'package:hgf_playground/src/builder/builder_model.dart';
import 'package:hgf_playground/src/builder/doc_layout.dart';
import 'package:hgf_playground/src/builder/ui/builder_canvas.dart';
import 'package:hgf_playground/src/builder/ui/inspector.dart' show withId;
import 'package:hgf_playground/src/builder/ui/builder_page.dart';
import 'package:hgf_playground/src/builder/ui/pipeline_canvas.dart';
import 'package:hgf_playground/src/model/presets.dart';
import 'package:material_ui/material_ui.dart';

void main() {
  group('BuilderModel', () {
    for (final p in presets) {
      test('${p.id}: opens without issues and builds the same network', () {
        final m = BuilderModel(doc: NetworkDocument(nodes: const []))
          ..openPreset(p);
        expect(m.issues, isEmpty, reason: '${m.issues}');
        // The same graph and initial values (a node's `params` may come
        // back as overrides).
        expect(
          _comparable(m.network!.toJson()),
          _comparable((p.build()..build()).toJson()),
        );
        expect(m.trajectories, isNotNull, reason: m.previewError);
        expect(m.previewError, isNull);
        // Parents above their children.
        for (final e in m.doc.edges) {
          final pp = m.positions[ownerIdOf(e.parent)]!;
          final cp = m.positions[ownerIdOf(e.child)]!;
          expect(pp.dy, lessThan(cp.dy), reason: '${e.parent} → ${e.child}');
        }
      });
    }

    test('builds a binary 3-level HGF from scratch, with undo and redo', () {
      final m = BuilderModel(doc: NetworkDocument(nodes: const []));
      expect(m.issues.single.message, contains('no nodes'));
      final u = m.addNode(NodeKind.binaryState);
      expect(m.issues.single.severity, IssueSeverity.warning);
      final x1 = m.addLinked(u, EdgeType.value);
      final x2 = m.addLinked(x1, EdgeType.volatility);
      expect(m.issues, isEmpty);
      final net = m.network!;
      expect(net.nNodes, 3);
      expect(net.inputIdxs, [0]);
      expect(net.valueParentsOf(0), [1]);
      expect(net.volatilityParentsOf(1), [2]);
      expect(m.trajectories!.nTime, m.trials);
      expect(m.positions[x2]!.dy, lessThan(m.positions[x1]!.dy));

      m.undo();
      expect(m.doc.nodes, hasLength(2));
      m.undo();
      m.undo();
      expect(m.doc.nodes, isEmpty);
      expect(m.canUndo, isFalse);
      m
        ..redo()
        ..redo()
        ..redo();
      expect(m.doc.nodes.map((n) => n.id), [u, x1, x2]);
      expect(m.issues, isEmpty);
    });

    test('checkConnection rejects what the engine cannot build', () {
      final m = BuilderModel()..openPreset(presetById('binary3'));
      // x0 is binary: no volatility parents.
      final c = m.checkConnection('x2', 'x0', EdgeType.volatility);
      expect(c.problem, contains('value edges only'));
      expect(m.checkConnection('x0', 'x0', EdgeType.value).problem, isNotNull);
      // A cycle.
      expect(
        m.checkConnection('x0', 'x2', EdgeType.value).problem,
        contains('cycle'),
      );
      // Twice.
      expect(
        m.checkConnection('x1', 'x0', EdgeType.value).problem,
        contains('twice'),
      );
      expect(m.checkConnection('x2', 'x0', EdgeType.value).problem, isNull);
      expect(m.doc.edges, hasLength(2), reason: 'checks change nothing');
    });

    test('a slider drag is one undo step and re-filters each move', () {
      final m = BuilderModel()..openPreset(presetById('continuous3'));
      final r0 = m.revision;
      m.checkpoint();
      for (final w in [-3.0, -2.5, -2.0]) {
        m.setParam('x1', NodeField.tonicVolatility, w, record: false);
      }
      expect(m.revision, r0 + 3);
      expect(m.initial('x1', NodeField.tonicVolatility), -2);
      m.undo();
      expect(m.initial('x1', NodeField.tonicVolatility), -4);
      expect(m.canUndo, isFalse);
    });

    test('renaming keeps positions, edges and the selection', () {
      final m = BuilderModel()..openPreset(presetById('binary3'));
      final at = m.positions['x1'];
      m.select(const NodeSelection('x1'));
      final n = m.doc.node('x1')! as ScalarNode;
      m.replaceNode('x1', ScalarNode('belief', kind: n.kind, params: n.params));
      expect(m.positions['belief'], at);
      expect(m.positions.containsKey('x1'), isFalse);
      expect(m.selectedNode, 'belief');
      expect(m.doc.edges.map((e) => '${e.parent}>${e.child}'), [
        'belief>x0',
        'x2>belief',
      ]);
      expect(m.issues, isEmpty);
    });

    test('compound nodes: a categorical node and an edge to its parts', () {
      final m = BuilderModel(doc: NetworkDocument(nodes: const []));
      final cat = m.addNode(NodeKind.categoricalState);
      expect(m.issues, isEmpty);
      final v = m.addNode(NodeKind.continuousState);
      expect(
        m.checkConnection(v, cat, EdgeType.volatility).problem,
        contains('its own'),
      );
      final c = m.checkConnection(v, '$cat.7', EdgeType.volatility);
      expect(c.problem, isNull);
      m.addEdge(c.edge);
      expect(m.issues, isEmpty);
      expect(m.network!.volatilityParentsOf(7), contains(8));
    });

    test('dp-state: clusters checked, preview capped', () {
      final m = BuilderModel(doc: NetworkDocument(nodes: const []));
      final dp = m.addNode(NodeKind.dpState);
      expect(m.hasErrors, isTrue);
      final ef = m.addNode(NodeKind.efState);
      m.replaceNode(ef, EfStateNode(ef, nNodes: 4));
      m.addEdge(m.checkConnection(ef, dp, EdgeType.value).edge);
      expect(m.issues, isEmpty);
      expect(m.trials, 150);
      expect(m.trajectories!.nTime, 150);
    });

    test('saves and opens again: document, name and positions', () {
      final m = BuilderModel()..openPreset(presetById('igt'));
      m.moveNode('x3', const Offset(12.5, -40));
      final text = m.toJsonText();
      final back = BuilderModel(doc: NetworkDocument(nodes: const []))
        ..openJson(text);
      expect(back.name, m.name);
      expect(back.positions['x3'], const Offset(12.5, -40));
      expect(jsonEncode(back.doc.toJson()), jsonEncode(m.doc.toJson()));
      expect(back.issues, isEmpty);
      // A network's JSON opens too.
      final net = presetById('binary3').build()..build();
      back.openJson(jsonEncode(net.toJson()), name: 'net');
      expect(back.doc.nodes, hasLength(3));
    });

    test('a broken coupling expression is an error on its node', () {
      final m = BuilderModel()..openPreset(presetById('continuous2'));
      m.replaceNode(
        'x1',
        ScalarNode('x1', kind: NodeKind.continuousState, coupling: 'tanh(x'),
      );
      expect(m.nodeIssues['x1'], IssueSeverity.error);
      expect(m.network, isNull);
      expect(m.previewError, isNotNull);
      m.undo();
      expect(m.network, isNotNull);
    });
  });

  group('Pipeline', () {
    test('a choice task opens with its response, simulated', () {
      final m = BuilderModel()..openPreset(presetById('igt'));
      final r = m.document.response!;
      expect(r.options, hasLength(4));
      expect(r.reveals, hasLength(4));
      expect(m.issues.where((i) => i.isError), isEmpty);
      expect(m.compiled, isNotNull);
      final choices = m.subject!.responses!;
      expect(choices.every((c) => c >= 0 && c < 4), isTrue);
      // A choice reveals only its deck's outcome.
      expect(m.subject!.observed, isNotNull);
    });

    test('adds a response and outputs; undo takes them back', () {
      final m = BuilderModel()..openPreset(presetById('binary3'));
      expect(m.document.inputs.single.name, 'observation');
      m.addResponse();
      expect(m.document.response!.readout['type'], 'BinarySoftmaxReadout');
      expect(m.selectedPart, DocPart.response);
      final r = m.subject!.responses!;
      expect(r.every((v) => v == 0 || v == 1), isTrue);
      final name = m.addOutput(node: 'x1', field: 'value_prediction_error');
      expect(m.outputs[name], hasLength(m.trials));
      expect(m.pipelineParts, contains(DocPart.output(name)));
      m.deleteSelection();
      expect(m.document.outputs, isEmpty);
      m.undo();
      expect(m.document.outputs, hasLength(1));
      m
        ..undo()
        ..undo();
      expect(m.document.response, isNull);
    });

    test('previews loaded data per participant, through transforms', () {
      final m = BuilderModel()..openPreset(presetById('continuous2'));
      final rows = [
        for (final p in ['a', 'b'])
          for (var t = 0; t < 40; t++) '$p,$t,${p == 'a' ? t * 0.1 : -t * 0.1}',
      ];
      m.loadData(
        CsvData.parse('who,time,raw\n${rows.join('\n')}\n'),
        name: 'two.csv',
      );
      // The input reads "observation", which the data lacks.
      expect(m.partIssues[DocPart.input('observation')], IssueSeverity.error);
      expect(m.dataColumns, ['who', 'time', 'raw']);

      final t = m.addTransform(column: 'raw');
      expect(m.document.transform(t)!.expression.toString(), 'raw');
      m.bindColumn(t, 'observation');
      m.update(
        m.document.copyWith(
          data: const DocData(participantColumn: 'who', timeColumn: 'time'),
        ),
      );
      expect(m.issues.where((i) => i.isError), isEmpty);
      expect(m.participants, ['a', 'b']);
      final a = m.lastMean('x0')!;
      m.showParticipant(1);
      expect(m.lastMean('x0'), lessThan(a));

      // Outputs line up with the data's rows.
      m.addOutput(node: 'x0', field: 'mean');
      final out = CsvData.parse(m.outputsCsv()!);
      expect(out.rows, hasLength(80));
      expect(out.header.last, 'x0_mean');

      // Renaming the transform renames what reads it.
      m.update(m.document.replaceTransform(t, DocTransform('scaled', 'raw')));
      expect(m.document.inputs.single.columns, ['scaled']);
      expect(m.issues.where((i) => i.isError), isEmpty);
    });

    test('renaming a node keeps the outputs and the response on it', () {
      final m = BuilderModel()..openPreset(presetById('binary3'));
      m
        ..addResponse()
        ..addOutput(node: 'x1', field: 'mean');
      m.replaceNode('x1', withId(m.doc.node('x1')!, 'belief'));
      expect(m.document.outputs.single.node, 'belief');
      expect(m.issues.where((i) => i.isError), isEmpty);
      // The binary input: still read by the response.
      expect(m.document.response!.readout['node'], 'x0');
    });

    test('saves and opens the whole model, with both layouts', () {
      final m = BuilderModel()..openPreset(presetById('continuous2'));
      m
        ..addTransform(column: 'observation')
        ..addOutput(node: 'x1', field: 'mean')
        ..movePart(DocPart.network, const Offset(600, 40));
      final back = BuilderModel(doc: NetworkDocument(nodes: const []))
        ..openJson(m.toJsonText());
      expect(back.document.transforms, hasLength(1));
      expect(back.document.outputs, hasLength(1));
      expect(back.pipelinePositions[DocPart.network], const Offset(600, 40));
      expect(back.positions, m.positions);
    });
  });

  group('BuilderPage', () {
    for (final width in [1300.0, 480.0]) {
      testWidgets('renders every preset ${width.toInt()} wide', (tester) async {
        tester.view
          ..physicalSize = Size(width, 900)
          ..devicePixelRatio = 1;
        addTearDown(tester.view.reset);
        final m = BuilderModel();
        await tester.pumpWidget(
          MaterialApp(
            theme: buildTheme(Brightness.light),
            home: BuilderPage(model: m),
          ),
        );
        for (final p in presets) {
          m.openPreset(p);
          await tester.pumpAndSettle();
          expect(tester.takeException(), isNull, reason: p.id);
        }
      });
    }

    testWidgets('drag a column onto the network\'s input to read it', (
      tester,
    ) async {
      tester.view
        ..physicalSize = const Size(1300, 900)
        ..devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      final m = BuilderModel()..openPreset(presetById('continuous2'));
      m.loadData(CsvData.parse('t,y\n0,0.1\n1,0.4\n2,0.2\n'), name: 'y.csv');
      await tester.pumpWidget(
        MaterialApp(
          theme: buildTheme(Brightness.light),
          home: BuilderPage(model: m),
        ),
      );
      await tester.pumpAndSettle();
      expect(m.view, BuilderView.pipeline);

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
      final yRow = data.rows.indexWhere((row) => row.label == 'y');
      final net = cards.firstWhere((c) => c.part == DocPart.network);
      final from = screen(data.rowPort(yRow));
      final to = screen(net.rowCentre(0) + const Offset(60, 0));
      final g = await tester.startGesture(from);
      await tester.pump(const Duration(milliseconds: 20));
      for (var k = 1; k <= 10; k++) {
        await g.moveTo(Offset.lerp(from, to, k / 10)!);
        await tester.pump(const Duration(milliseconds: 16));
      }
      await g.up();
      await tester.pumpAndSettle();
      expect(m.document.inputs.single.columns, ['y']);
      expect(m.selectedPart, DocPart.input('observation'));
      expect(m.issues.where((i) => i.isError), isEmpty);
      expect(m.trajectories!.nTime, 3);

      // Double-clicking the network opens it.
      final centre = screen(net.at + const Offset(cardWidth / 2, 12));
      await tester.tapAt(centre);
      await tester.pump(const Duration(milliseconds: 50));
      await tester.tapAt(centre);
      await tester.pumpAndSettle();
      expect(m.view, BuilderView.network);
      expect(find.byType(BuilderCanvas), findsOneWidget);
    });

    testWidgets('drag from a port to a node connects them; Del deletes', (
      tester,
    ) async {
      tester.view
        ..physicalSize = const Size(1300, 900)
        ..devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      final m = BuilderModel(doc: NetworkDocument(nodes: const []));
      m.addNode(NodeKind.continuousState, at: Offset.zero); // x0
      m.addNode(NodeKind.continuousState, at: const Offset(0, -120)); // x1
      m
        ..showView(BuilderView.network)
        ..opened += 1; // fit the view to both
      await tester.pumpWidget(
        MaterialApp(
          theme: buildTheme(Brightness.light),
          home: BuilderPage(model: m),
        ),
      );
      await tester.pumpAndSettle();

      // The canvas fits the bounds (inflated by 70) into its box.
      final box = tester.getRect(find.byType(BuilderCanvas));
      final bounds = boundsOf(m.positions.values)!.inflate(70);
      final scale = [
        box.width / bounds.width,
        box.height / bounds.height,
      ].reduce((a, b) => a < b ? a : b).clamp(0.3, 1.4);
      final pan = box.size.center(Offset.zero) - bounds.center * scale;
      Offset screen(Offset p) => box.topLeft + pan + p * scale;

      // x1's volatility child port (bottom right) to x0.
      final x1 = m.doc.node('x1')!;
      final from = portPosition(x1, m.positions['x1']!, Port.bottomVolatility);
      final g = await tester.startGesture(screen(from));
      await tester.pump(const Duration(milliseconds: 20));
      for (var k = 1; k <= 10; k++) {
        await g.moveTo(Offset.lerp(screen(from), screen(Offset.zero), k / 10)!);
        await tester.pump(const Duration(milliseconds: 16));
      }
      await g.up();
      await tester.pumpAndSettle();
      expect(m.doc.edges, hasLength(1));
      final e = m.doc.edges.single;
      expect((e.type, e.parent, e.child), (EdgeType.volatility, 'x1', 'x0'));
      expect(m.selectedEdge, 0);
      expect(find.text('x₁ → x₀'), findsWidgets); // the inspector

      // Delete it from the keyboard.
      await tester.sendKeyEvent(LogicalKeyboardKey.delete);
      await tester.pump();
      expect(m.doc.edges, isEmpty);

      // Dragging a node moves it (one undo step).
      final before = m.positions['x0']!;
      await tester.dragFrom(screen(before), const Offset(60, 0));
      await tester.pumpAndSettle();
      expect(m.positions['x0']!.dx, greaterThan(before.dx + 20));
      m.undo();
      expect(m.positions['x0'], before);
    });
  });
}

String _comparable(Map<String, Object?> json) => jsonEncode({
  ...json,
  'nodes': [
    for (final n in json['nodes']! as List<Object?>)
      {...(n! as Map<String, Object?>)}..remove('params'),
  ],
});
