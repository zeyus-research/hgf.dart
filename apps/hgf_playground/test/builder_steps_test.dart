import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_playground/src/app.dart';
import 'package:hgf_playground/src/builder/builder_model.dart';
import 'package:hgf_playground/src/builder/ui/builder_page.dart';
import 'package:hgf_playground/src/builder/ui/pipeline_canvas.dart';
import 'package:hgf_playground/src/builder/ui/step_inspector.dart';
import 'package:material_ui/material_ui.dart';

const _csv = '''
participant,t,x,trial
a,0,1,1
a,1,2,1
a,2,3,2
a,3,4,2
a,4,5,3
b,0,5,1
b,1,6,2
b,2,7,2
''';

BuilderModel _model() {
  final m = BuilderModel()
    ..open(
      ModelDocument(
        name: 'steps',
        network: NetworkDocument(
          nodes: [
            ScalarNode('x', kind: NodeKind.continuousState),
            ScalarNode('x_v', kind: NodeKind.continuousState),
          ],
          edges: const [DocEdge(EdgeType.value, 'x_v', 'x')],
        ),
        inputs: [
          DocInput('x', node: 'x', columns: ['x']),
        ],
        data: const DocData(participantColumn: 'participant', timeColumn: 't'),
      ),
    )
    ..loadData(CsvData.parse(_csv), name: 'steps.csv');
  for (final k in BuilderModel.stepKinds.keys) {
    m.addStep(k);
  }
  return m;
}

void main() {
  for (final width in [1400.0, 480.0]) {
    testWidgets('renders the step cards and panels ${width.toInt()} wide', (
      tester,
    ) async {
      tester.view
        ..physicalSize = Size(width, 900)
        ..devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      final m = _model();
      expect(m.issues.where((i) => i.isError), isEmpty, reason: '${m.issues}');
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
      final cards = pipelineGeometry(m).cards;
      expect(cards.where((c) => c.part.startsWith('step:')).length, 5);
      for (var i = 0; i < 5; i++) {
        m.select(PartSelection(DocPart.step(i)));
        await tester.pumpAndSettle();
        if (width >= 900) expect(find.byType(StepPanel), findsOneWidget);
      }
      // A screenshot for a look by eye: STEPS_SHOT=path.png.
      final shot = Platform.environment['STEPS_SHOT'];
      if (shot != null && width >= 900) {
        m.select(PartSelection(DocPart.step(0)));
        await tester.pumpAndSettle();
        await tester.runAsync(() async {
          final boundary =
              key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
          final image = await boundary.toImage();
          final png = await image.toByteData(format: ui.ImageByteFormat.png);
          File(shot).writeAsBytesSync(png!.buffer.asUint8List());
        });
      }
      m.removePart(DocPart.step(0));
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull);
    });
  }
}
