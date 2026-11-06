import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_playground/src/model/fitting.dart';
import 'package:hgf_playground/src/model/playground.dart';
import 'package:hgf_playground/src/model/presets.dart';
import 'package:hgf_playground/src/ui/fit_card.dart';

Future<FitWorkers> _inline() => FitWorkers.spawn(count: 0);

void main() {
  test('candidates per preset; vector node kinds are not fitted', () {
    for (final p in presets) {
      final f = Fitting(
        Playground(presetId: p.id, nTrials: 50),
        spawnWorkers: _inline,
      );
      final vector = p.addTo != null;
      expect(f.unsupported != null, vector, reason: p.id);
      if (vector) continue;
      // Every non-input continuous node's ω is free by default.
      expect(f.free, isNotEmpty, reason: p.id);
      expect(f.free.every((id) => id.startsWith('omega_')), isTrue);
      // The response parameter is offered only for responses.
      expect(f.offered.any((c) => c.isResponse), isFalse);
      f.setTarget(FitTarget.responses);
      expect(f.offered.any((c) => c.isResponse), isTrue);
      expect(f.free.where((id) => !id.startsWith('omega_')), hasLength(1));
    }
  });

  test('parameter recovery from a simulated agent (MAP + Laplace)', () async {
    final pg = Playground(presetId: 'continuous2', nTrials: 400);
    final f = Fitting(pg, spawnWorkers: _inline)
      ..setAgent(noise: 0.05)
      ..simulateAgent(seed: 3);
    expect(f.target, FitTarget.responses);
    expect(f.truth!['omega_1'], -4);
    expect(f.truth!['response_precision'], closeTo(400, 1e-9));
    // Fit from elsewhere: the sliders don't set the free parameters.
    pg.setParam(1, NodeField.tonicVolatility, -1);
    expect(f.responses, isNotNull, reason: 'slider edits keep responses');
    await f.fit();
    expect(f.error, isNull);
    final r = f.result!;
    expect(r.method, FitMethod.map);
    expect(r.logEvidence!.isFinite, isTrue);
    for (final e in r.estimates) {
      expect(e.lower, lessThan(e.estimate), reason: e.param.id);
      expect(e.upper, greaterThan(e.estimate), reason: e.param.id);
      expect(e.truth, inInclusiveRange(e.lower, e.upper), reason: e.param.id);
      expect(e.density!(e.estimate), greaterThan(0));
    }
    // Apply moves the slider to the estimate.
    f.apply();
    final omega = r.estimates.firstWhere((e) => e.param.id == 'omega_1');
    expect(pg.param(1, NodeField.tonicVolatility), omega.estimate);
    // New data drop the responses and the result.
    pg.reseed();
    expect(f.responses, isNull);
    expect(f.result, isNull);
  });

  test('fit to the observations, binary, with NUTS', () async {
    final pg = Playground(presetId: 'binary3', nTrials: 200);
    final f = Fitting(pg, spawnWorkers: _inline)
      ..nutsWarmup = 60
      ..nutsDraws = 40
      ..toggle('omega_2', false)
      ..setMethod(FitMethod.nuts);
    expect(f.free, {'omega_1'});
    await f.fit();
    expect(f.error, isNull);
    final r = f.result!;
    expect(r.method, FitMethod.nuts);
    final e = r.estimates.single;
    expect(e.draws, hasLength(4 * 40));
    expect(e.rhat!.isFinite, isTrue);
    expect(e.lower, lessThanOrEqualTo(e.upper));
    expect(e.truth, isNull);
  });

  test('cancel and data changes stop a running fit', () async {
    final pg = Playground(presetId: 'binary3', nTrials: 300);
    final pool = await FitWorkers.spawn(count: 0)
      ..yieldInterval = Duration.zero;
    final f = Fitting(pg, spawnWorkers: () async => pool)
      ..setMethod(FitMethod.nuts);
    final job = f.fit();
    while (f.progress == null || f.progress!.completed < 5) {
      await Future<void>.delayed(Duration.zero);
    }
    expect(f.running, isTrue);
    f.cancel();
    await job;
    expect(f.running, isFalse);
    expect(f.result, isNull);
    expect(f.error, isNull);

    final again = f.fit();
    await Future<void>.delayed(Duration.zero);
    pg.load(Float64List.fromList(List.filled(50, 1.0)), 'ones.csv');
    await again;
    expect(f.running, isFalse);
    expect(f.result, isNull);
  });

  test('loaded responses must match the trials and the input kind', () {
    final pg = Playground(presetId: 'binary3', nTrials: 4);
    final f = Fitting(pg, spawnWorkers: _inline);
    expect(
      () => f.loadResponses(Float64List.fromList([1, 0]), 'r.csv'),
      throwsFormatException,
    );
    expect(
      () => f.loadResponses(Float64List.fromList([1, 0, 0.5, 1]), 'r.csv'),
      throwsFormatException,
    );
    f.loadResponses(Float64List.fromList([1, 0, double.nan, 1]), 'r.csv');
    expect(f.responsesSource, 'r.csv');
    expect(f.truth, isNull);
    expect(f.canFit, isTrue);
  });

  testWidgets('a coalescing playground: new data before its notification', (
    tester,
  ) async {
    // `coalesce` re-filters and notifies at the next frame; responses
    // simulated in between must survive that notification.
    final pg = Playground(
      presetId: 'continuous2',
      nTrials: 100,
      coalesce: true,
    );
    final f = Fitting(pg, spawnWorkers: _inline)..simulateAgent(seed: 1);
    final old = f.responses;
    var notified = false;
    pg.addListener(() => notified = true);
    pg.updateData((d) => d.nTrials = 120);
    expect(notified, isFalse); // no frame has run
    f.simulateAgent(seed: 1);
    final fresh = f.responses;
    expect(fresh, isNot(same(old)));
    await tester.pump();
    expect(notified, isTrue);
    expect(f.responses, same(fresh));
    expect(f.responses, hasLength(120));
  });

  testWidgets('the fit card shows MAP and NUTS results', (tester) async {
    tester.view.physicalSize = const Size(1400, 2400);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    final pg = Playground(presetId: 'binary3', nTrials: 150);
    final f = Fitting(pg, spawnWorkers: _inline)
      ..nutsWarmup = 40
      ..nutsDraws = 40;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: SizedBox(width: 360, child: FitCard(fitting: f)),
          ),
        ),
      ),
    );
    expect(find.text('Fit'), findsOneWidget);
    await tester.tap(find.text('to responses'));
    await tester.pump();
    expect(find.text('Simulate or load responses first.'), findsOneWidget);
    await tester.tap(find.text('Simulate'));
    await tester.pump();
    expect(find.textContaining('simulated agent'), findsOneWidget);
    for (final method in FitMethod.values) {
      f.setMethod(method);
      await tester.runAsync(f.fit);
      await tester.pump();
      expect(tester.takeException(), isNull);
      expect(f.result, isNotNull, reason: method.name);
      expect(find.text('Apply'), findsOneWidget);
      // ω x₁, ω x₂ and β, each with its prior/posterior plot.
      expect(
        find.byWidgetPredicate(
          (w) => w is CustomPaint && w.painter is PosteriorPainter,
        ),
        findsNWidgets(3),
      );
      expect(find.textContaining('true '), findsNWidgets(3));
    }
    expect(find.textContaining('R̂'), findsNWidgets(3));
  });

  testWidgets('vector node presets explain why they can\'t be fitted', (
    tester,
  ) async {
    final f = Fitting(Playground(presetId: 'dp', nTrials: 20));
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: FitCard(fitting: f)),
      ),
    );
    expect(find.textContaining('Dirichlet process'), findsOneWidget);
    expect(find.text('Fit'), findsNothing);
  });
}
