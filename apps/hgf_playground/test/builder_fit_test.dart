@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_fit/hgf_fit.dart' show GroupRow;
import 'package:hgf_playground/src/builder/builder_fit.dart';
import 'package:hgf_playground/src/builder/builder_model.dart';
import 'package:hgf_playground/src/model/fit_config.dart';
import 'package:hgf_playground/src/model/fit_engine.dart';
import 'package:hgf_playground/src/model/fit_jobs.dart';
import 'package:hgf_playground/src/model/presets.dart';
import 'package:hgf_playground/src/app.dart' show buildTheme;
import 'package:hgf_playground/src/builder/ui/builder_page.dart';
import 'package:hgf_playground/src/builder/ui/fit_panel.dart';
import 'package:material_ui/material_ui.dart';

import 'wire_engine.dart';

/// A 2-level continuous HGF answering with Gaussian guesses of the next
/// observation, its volatility and the response precision free.
BuilderModel _builder({int participants = 4}) {
  final m = BuilderModel()..openPreset(presetById('continuous2'));
  m
    ..setTrials(100)
    ..addResponse();
  m.setFree(
    nodeTarget('x1', NodeField.tonicVolatility),
    true,
    current: m.initial('x1', NodeField.tonicVolatility),
  );
  m.setFree(responseTarget('precision'), true, current: 1);
  m.setSimParticipants(participants);
  return m;
}

void main() {
  const cpu = FitConfig(
    device: FitDevice.cpu,
    workers: 2,
    restarts: 1,
    warmup: 60,
    draws: 40,
  );

  test('simulated participants differ and come with their own values', () {
    final m = _builder();
    expect(m.issues.where((i) => i.isError), isEmpty, reason: '${m.issues}');
    expect(m.document.parameterNames, ['omega_x1', 'response_precision']);
    // Several participants name a participant column.
    expect(m.document.data.participantColumn, 'participant');
    expect(m.participants, ['1', '2', '3', '4']);
    final truth = m.simulatedTruth;
    expect(truth.keys, ['1', '2', '3', '4']);
    expect(truth['1'], isNot(truth['2']));
    final subs = m.allSubjects();
    expect(subs.map((s) => s.id), ['1', '2', '3', '4']);
    expect(subs[0].inputs, isNot(subs[1].inputs));
    // The preview shows the selected one, at its own values.
    final a = m.lastMean('x0');
    m.showParticipant(2);
    expect(m.lastMean('x0'), isNot(a));
    // Saved, the simulated data reads back as four participants.
    final csv = CsvData.parse(m.dataCsv()!);
    expect(m.compiled!.subjects(csv), hasLength(4));
  });

  test('the free chips follow the document', () {
    final m = _builder(participants: 1);
    final t = nodeTarget('x1', NodeField.tonicVolatility);
    expect(m.isFree(t), isTrue);
    m.setFree(t, false, current: 0);
    expect(m.isFree(t), isFalse);
    expect(m.document.parameterNames, ['response_precision']);
    m.undo();
    expect(m.isFree(t), isTrue);
    expect(m.pipelineParts.last, DocPart.fit);
  });

  for (final (method, pooled) in [
    ('map', false),
    ('map', true),
    ('nuts', false),
    ('nuts', true),
  ]) {
    test(
      'fits $method${pooled ? ' with a pooled ω' : ''} through the wire',
      () async {
        final m = _builder();
        var doc = m.document.copyWith(fit: DocFit(method: method, chains: 2));
        if (pooled) doc = doc.withRole('omega_x1', ParameterRole.pooled);
        m.update(doc);
        final engine = WireEngine(LocalFitEngine());
        addTearDown(engine.close);
        final f = BuilderFitting(m, engine: engine, config: () => cpu);
        expect(f.unavailable, isNull);
        // x1: the input's own mean is just the last observation.
        final before = m.lastMean('x1');
        await f.fit();
        expect(f.error, isNull);
        final r = f.result!;
        expect(r.participants, ['1', '2', '3', '4']);
        expect(r.names, ['omega_x1', 'response_precision']);
        for (final x in r.estimate) {
          expect(x.every((v) => v.isFinite), isTrue);
        }
        expect(r.method, switch ((method, pooled)) {
          ('map', false) => 'MAP per participant',
          ('map', true) => 'empirical Bayes',
          ('nuts', false) => 'NUTS per participant',
          _ => 'hierarchical NUTS',
        });
        if (pooled) {
          expect(r.group.map((g) => g.kind), [
            GroupRow.typical,
            GroupRow.spread,
          ]);
        } else {
          expect(r.group, isEmpty);
        }
        expect(f.stale, isFalse);
        // The preview moved to the estimates, and back.
        expect(m.lastMean('x1'), isNot(before));
        f.setPreviewAtEstimates(false);
        expect(m.lastMean('x1'), before);

        final table = f.estimatesCsv()!.split('\n\n');
        final rows = CsvData.parse(table.first);
        expect(rows.header.take(5), [
          'participant',
          'omega_x1',
          'omega_x1_lower',
          'omega_x1_upper',
          'omega_x1_true',
        ]);
        expect(rows.rows, hasLength(4));
        expect(table, hasLength(pooled ? 2 : 1));

        // Editing the model makes the result stale.
        m.setParam('x1', NodeField.mean, 0.5);
        expect(f.stale, isTrue);
      },
    );
  }

  for (final width in [1300.0, 480.0]) {
    testWidgets('the fit panel renders, with results, ${width.toInt()} wide', (
      tester,
    ) async {
      tester.view
        ..physicalSize = Size(width, 1400)
        ..devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      final m = _builder();
      m.update(m.document.withRole('omega_x1', ParameterRole.pooled));
      final f = BuilderFitting(m, engine: _IdleEngine(), config: () => cpu);
      Float64List row(double a, double b) => Float64List.fromList([a, b]);
      f.result = DocumentFitOutcome(
        method: 'empirical Bayes',
        names: m.document.parameterNames,
        participants: m.participants,
        estimate: [for (var k = 0; k < 4; k++) row(-4.0 - k * 0.3, 1.0)],
        lower: [for (var k = 0; k < 4; k++) row(-5.0 - k * 0.3, 0.5)],
        upper: [for (var k = 0; k < 4; k++) row(-3.0 - k * 0.3, 2.0)],
        group: const [
          (
            name: 'omega_x1',
            kind: GroupRow.typical,
            value: -4.4,
            lower: double.nan,
            upper: double.nan,
          ),
          (
            name: 'omega_x1',
            kind: GroupRow.spread,
            value: 0.2,
            lower: double.nan,
            upper: double.nan,
          ),
        ],
        converged: true,
        millis: 1234,
        where: '2 isolates',
        logEvidence: -100,
      );
      m
        ..refreshPreview()
        ..select(const PartSelection(DocPart.fit));
      await tester.pumpWidget(
        MaterialApp(
          theme: buildTheme(Brightness.light),
          home: BuilderPage(model: m),
        ),
      );
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull);
      expect(find.text('Free parameters'), findsOneWidget);
      expect(find.byType(ForestPlot), findsNWidgets(2));
      expect(find.textContaining('typical'), findsOneWidget);
      expect(find.text('Fit 4 participants'), findsOneWidget);

      // A tap on a participant's row previews it.
      await tester.ensureVisible(find.byType(ForestPlot).first);
      await tester.pumpAndSettle();
      final plot = tester.getRect(find.byType(ForestPlot).first);
      await tester.tapAt(plot.topLeft + const Offset(80, 4 + 2.5 * 12));
      await tester.pumpAndSettle();
      expect(m.participant, 2);
    });
  }

  test('the group level needs participants; sharing needs NUTS', () {
    final m = _builder(participants: 1);
    final f = BuilderFitting(m, engine: LocalFitEngine(), config: () => cpu);
    m.update(m.document.withRole('omega_x1', ParameterRole.pooled));
    expect(f.unavailable, contains('several participants'));
    m.update(m.document.withRole('response_precision', ParameterRole.shared));
    expect(
      m.issues.where((i) => i.isError).map((i) => i.message).join(),
      contains('needs NUTS'),
    );
    expect(m.partIssues[DocPart.fit], IssueSeverity.error);
  });
}

/// A [FitEngine] that never runs: the widget test only renders.
final class _IdleEngine implements FitEngine {
  @override
  Future<GpuInfo> probeGpu() async => (adapter: null, reason: 'test');

  @override
  Future<R> run<R>(FitJobSpec<R> job, void Function(JobEvent) onEvent) =>
      throw UnimplementedError();

  @override
  void cancel() {}

  @override
  Future<void> close() async {}
}
