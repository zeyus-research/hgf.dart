import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_playground/src/model/fit_config.dart';
import 'package:hgf_playground/src/model/fitting.dart';
import 'package:hgf_playground/src/model/playground.dart';
import 'package:hgf_playground/src/ui/recovery_card.dart';

Future<FitWorkers> _inline([int? _]) => FitWorkers.spawn(count: 0);

void main() {
  test('continuous: ω and the response precision are recovered', () async {
    final pg = Playground(presetId: 'continuous2', nTrials: 300);
    final f =
        Fitting(pg, spawnWorkers: _inline, config: const FitConfig(restarts: 0))
          ..setTarget(FitTarget.responses)
          ..setRecoveryAgents(50);
    expect(f.canRecover, isTrue);
    expect(f.recoveryWantsGpu, isFalse); // 50 runs < autoGpuMapRuns
    await f.recover(seed: 4);
    expect(f.error, isNull);
    final r = f.recovery!;
    expect(r.agents + r.dropped, 50);
    expect(r.params.map((p) => p.id), ['omega_1', 'response_precision']);
    expect(r.where, 'the main thread');
    for (var i = 0; i < r.params.length; i++) {
      expect(r.truth[i], hasLength(r.agents));
      // Every truth is a draw from the prior: distinct, within its support.
      expect(r.truth[i].toSet(), hasLength(r.agents));
      expect(r.truth[i].every(r.params[i].prior.contains), isTrue);
      expect(r.stats[i].r, greaterThan(0.8), reason: r.params[i].id);
    }
    // ω on a linear axis, the precision on a log axis.
    expect(r.params.map((p) => p.logScale), [false, true]);
    // New data drop the recovery.
    pg.reseed();
    expect(f.recovery, isNull);
  });

  test('binary: diverging draws are redrawn; Auto picks the GPU', () async {
    final pg = Playground(presetId: 'binary3', nTrials: 200);
    final f = Fitting(
      pg,
      spawnWorkers: _inline,
      openGpu: () async => (device: null, reason: 'none in tests'),
      config: const FitConfig(restarts: 1),
    )..setTarget(FitTarget.responses);
    f.setRecoveryAgents(200);
    expect(f.recoveryWantsGpu, isTrue); // 400 runs
    f.setRecoveryAgents(50);
    expect(f.recoveryWantsGpu, isFalse);
    await f.recover(seed: 1);
    expect(f.error, isNull);
    final r = f.recovery!;
    expect(r.ok.where((o) => o).length, greaterThan(40));
    // β is identified from 200 choices, if loosely.
    final beta = r.params.indexWhere((p) => p.id == 'beta');
    expect(r.stats[beta].r, greaterThan(0.5));
  });

  test('cancel stops a recovery; a fit can follow', () async {
    final pg = Playground(presetId: 'continuous2', nTrials: 200);
    final pool = await FitWorkers.spawn(count: 0)
      ..yieldInterval = Duration.zero;
    final f =
        Fitting(
            pg,
            spawnWorkers: (_) async => pool,
            config: const FitConfig(device: FitDevice.cpu),
          )
          ..setTarget(FitTarget.responses)
          ..setRecoveryAgents(1000);
    final job = f.recover();
    while (f.stage == null || !f.stage!.startsWith('Fitting')) {
      await Future<void>.delayed(Duration.zero);
    }
    expect(f.runningJob, FitJob.recovery);
    // One job at a time.
    await f.fit();
    expect(f.result, isNull);
    f.cancel();
    await job;
    expect(f.running, isFalse);
    expect(f.recovery, isNull);
    expect(f.error, isNull);
    f.simulateAgent(seed: 1);
    await f.fit();
    expect(f.result, isNotNull);
  });

  testWidgets('the recovery card plots one panel per free parameter', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1400, 2400);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    final pg = Playground(presetId: 'continuous3', nTrials: 120);
    final f = Fitting(
      pg,
      spawnWorkers: _inline,
      config: const FitConfig(restarts: 0),
    )..setRecoveryAgents(50);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: SizedBox(width: 360, child: RecoveryCard(fitting: f)),
          ),
        ),
      ),
    );
    expect(find.text('Run recovery'), findsNothing);
    await tester.tap(find.text('Fit to responses'));
    await tester.pump();
    expect(find.text('Run recovery'), findsOneWidget);
    f.toggle('mu0_1', true);
    await tester.runAsync(f.recover);
    await tester.pump();
    expect(tester.takeException(), isNull);
    expect(f.recovery, isNotNull);
    final k = f.recovery!.params.length;
    expect(k, 4); // ω x₁, ω x₂, μ₀ x₁, π_r
    expect(
      find.byWidgetPredicate(
        (w) => w is CustomPaint && w.painter is RecoveryPainter,
      ),
      findsNWidgets(k),
    );
    expect(find.textContaining('r '), findsAtLeastNWidgets(k));
    expect(pg.network.getInitial(1, NodeField.tonicVolatility).isFinite, true);
  });
}
