import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/rendering.dart' show RenderRepaintBoundary;
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_playground/src/app.dart';
import 'package:hgf_playground/src/builder/builder_model.dart';
import 'package:hgf_playground/src/builder/ui/builder_page.dart';
import 'package:hgf_playground/src/builder/ui/deep_preview.dart';
import 'package:hgf_playground/src/builder/ui/pipeline_canvas.dart';
import 'package:hgf_playground/src/model/presets.dart';
import 'package:material_ui/material_ui.dart';

/// Two participants; the press is "left" when x1 + x2 > 0, else "right".
CsvData _data(int n) {
  final r = math.Random(4);
  return CsvData(
    ['participant', 'x1', 'x2', 'press', 'u'],
    [
      for (final p in ['a', 'b'])
        for (var t = 0; t < n; t++)
          () {
            final x1 = 2 * r.nextDouble() - 1, x2 = 2 * r.nextDouble() - 1;
            return [
              p,
              '$x1',
              '$x2',
              x1 + x2 > 0 ? 'left' : 'right',
              '${r.nextDouble()}',
            ];
          }(),
    ],
  );
}

/// A classifier of the press from x1 and x2, trained on the split's train
/// rows.
BuilderModel _classifier(BuilderModel m, CsvData data, {int budget = 5000}) {
  // The test VM runs unoptimised code: give the small networks time.
  m.deepBudgetMs = budget;
  m.loadData(data, name: 'presses.csv');
  m
    ..update(
      m.document.copyWith(
        data: m.document.data.copyWith(participantColumn: () => 'participant'),
      ),
    )
    ..addStep('split');
  final id = m.addDeep();
  m
    ..bindDeepColumn(id, 'x1', target: false)
    ..bindDeepColumn(id, 'x2', target: false)
    ..bindDeepColumn(id, 'press', target: true);
  final d = m.document.deepById(id)!;
  m.setDeep(
    id,
    d.copyWith(
      training: d.training.copyWith(
        epochs: 3,
        trainOn: () => Expr.parse("split == 'train'"),
      ),
    ),
  );
  return m;
}

void main() {
  test('a small network trains at once, scored on held-out rows', () {
    final m = _classifier(
      BuilderModel(doc: NetworkDocument(nodes: const [])),
      _data(40),
    );
    final r = m.deepResult('deep');
    expect(r, isNotNull, reason: m.deepStatus('deep'));
    expect(r!.testScore, isNotNull);
    expect(r.trainScore!.name, 'accuracy');
    expect(m.deepStatus('deep'), contains('held out accuracy'));
    // It gives what the exported scripts give: DeepTraining on the
    // pipeline's table.
    final doc = m.document;
    final l = doc.deepPipeline(doc.deepById('deep')!);
    final table = l.transform(m.csv!);
    final want = DeepTraining.run(doc.deepById('deep')!, table, scope: l.scope);
    expect(r.predictions.data, want.predictions.data);
    // Two participants for the preview bar.
    expect(m.participants, ['a', 'b']);
    expect(m.deepColumns, ['deep_p_left', 'deep_p_right']);
  });

  test('the HGF network reads a deep network\'s prediction', () {
    final m = _classifier(
      BuilderModel()..openPreset(presetById('continuous2')),
      _data(40),
    );
    final input = m.document.inputs.first.name;
    m.bindColumn('deep_p_left', input);
    expect(
      m.issues.where((i) => i.isError),
      isEmpty,
      reason: '${m.issues.map((i) => i.message)}',
    );
    expect(m.compiled!.layout.deep.single.id, 'deep');
    expect(m.deepPending, isEmpty);
    // The input is the prediction, row by row.
    final s = m.subject!;
    final r = m.deepResult('deep')!;
    expect(s.inputs.first, r.predictions.at(0, 0));
    // A transform cannot read it.
    m.update(
      m.document.withTransform(DocTransform('twice', 'deep_p_left * 2')),
    );
    expect(
      m.issues.map((i) => i.message),
      contains(contains('the deep networks run after the transforms')),
    );
  });

  testWidgets('a big network trains in the background', (tester) async {
    final m = _classifier(
      BuilderModel(doc: NetworkDocument(nodes: const [])),
      _data(3000),
      budget: 15,
    );
    final d = m.document.deepById('deep')!;
    m.setDeep('deep', d.copyWith(training: d.training.copyWith(epochs: 40)));
    expect(m.deepResult('deep'), isNull);
    expect(m.deepRun('deep'), isNotNull);
    expect(m.deepStatus('deep'), startsWith('training…'));
    // Slices between frames (the test's clock drives the timers).
    var progress = 0.0;
    for (var k = 0; k < 20000 && m.deepResult('deep') == null; k++) {
      await tester.pump(const Duration(milliseconds: 10));
      if (m.deepRun('deep') case final r?) {
        expect(r.progress, greaterThanOrEqualTo(progress));
        progress = r.progress;
      }
    }
    expect(m.deepResult('deep'), isNotNull);
    expect(m.deepRun('deep'), isNull);
    m.dispose();
  });

  testWidgets('the preview plots the predictions; the card lists them', (
    tester,
  ) async {
    tester.view
      ..physicalSize = const Size(1300, 900)
      ..devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    final m = _classifier(
      BuilderModel(doc: NetworkDocument(nodes: const [])),
      _data(40),
    );
    await tester.pumpWidget(
      MaterialApp(
        theme: buildTheme(Brightness.light),
        home: BuilderPage(model: m),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.byType(DeepPreview), findsOneWidget);
    final card = pipelineGeometry(
      m,
    ).cards.firstWhere((c) => c.part == DocPart.deep('deep'));
    expect(
      [
        for (final r in card.rows)
          if (r.port case ColumnOut(:final column)) column,
      ],
      ['deep_p_left', 'deep_p_right'],
    );
    expect(card.rows.first.label, contains('held out'));
    m.showDeep('deep');
    await tester.pumpAndSettle();
    expect(find.byType(DeepPreview), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  // A screenshot for a look by eye: DEEP_SHOT=path.png (the pipeline with
  // the HGF input reading the prediction, the deep card selected).
  testWidgets('screenshot', (tester) async {
    final shot = Platform.environment['DEEP_SHOT'];
    if (shot == null) return;
    tester.view
      ..physicalSize = const Size(1400, 900)
      ..devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    final m = _classifier(
      BuilderModel()..openPreset(presetById('continuous2')),
      _data(40),
    );
    m.bindColumn('deep_p_left', m.document.inputs.first.name);
    m.select(PartSelection(DocPart.deep('deep')));
    final key = GlobalKey();
    await tester.pumpWidget(
      RepaintBoundary(
        key: key,
        child: MaterialApp(
          theme: buildTheme(Brightness.light),
          home: BuilderPage(model: m),
        ),
      ),
    );
    await tester.pumpAndSettle();
    await tester.runAsync(() async {
      final boundary =
          key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
      final image = await boundary.toImage();
      final png = await image.toByteData(format: ui.ImageByteFormat.png);
      File(shot).writeAsBytesSync(png!.buffer.asUint8List());
    });
  });
}
