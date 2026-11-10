@TestOn('vm')
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:hgf_playground/src/model/fit_config.dart';
import 'package:hgf_playground/src/model/fit_host.dart';
import 'package:hgf_playground/src/model/fitting.dart';
import 'package:hgf_playground/src/model/playground.dart';

void main() {
  const cpu = FitConfig(
    device: FitDevice.cpu,
    workers: 2,
    restarts: 1,
    warmup: 60,
    draws: 40,
  );

  test('fits, samples and recovers in the host isolate', () async {
    final pg = Playground(presetId: 'continuous2', nTrials: 200);
    final f = Fitting(pg, engine: spawnFitHost(), config: cpu)
      ..setAgent(noise: 0.05)
      ..simulateAgent(seed: 3);
    addTearDown(f.dispose);

    // MAP + Laplace: the outcome comes back whole, the density closure too.
    final stages = <String>[];
    f.addListener(() {
      if (f.stage != null && !stages.contains(f.stage)) stages.add(f.stage!);
    });
    await f.fit();
    expect(f.error, isNull);
    final r = f.result!;
    expect(r.where, '2 isolates');
    for (final e in r.estimates) {
      expect(e.truth, inInclusiveRange(e.lower, e.upper), reason: e.param.id);
      expect(e.density!(e.estimate), greaterThan(0));
    }
    expect(
      stages,
      containsAll(['Finding the mode (MAP)…', 'Laplace approximation…']),
    );

    // NUTS reports progress.
    final seen = <double>[];
    f.addListener(() {
      final p = f.progress;
      if (p != null) seen.add(p.fraction);
    });
    f.setMethod(FitMethod.nuts);
    await f.fit();
    expect(f.error, isNull);
    expect(f.result!.estimates.first.draws, hasLength(4 * 40));
    expect(seen, isNotEmpty);

    // Recovery.
    f.setRecoveryAgents(50);
    await f.recover(seed: 1);
    expect(f.error, isNull);
    expect(f.recovery!.agents, 50);
    expect(f.recovery!.stats.first.r, greaterThan(0.8));
  });

  test('cancel reaches the host; the next job runs', () async {
    final pg = Playground(presetId: 'binary3', nTrials: 300);
    final f = Fitting(
      pg,
      engine: spawnFitHost(),
      config: cpu.copyWith(warmup: 2000, draws: 2000),
    )..setMethod(FitMethod.nuts);
    addTearDown(f.dispose);
    final job = f.fit();
    while (f.progress == null || f.progress!.completed < 20) {
      await Future<void>.delayed(const Duration(milliseconds: 5));
    }
    final sw = Stopwatch()..start();
    f.cancel();
    await job;
    expect(f.running, isFalse);
    expect(f.result, isNull);
    expect(f.error, isNull);
    f
      ..configure(cpu)
      ..setMethod(FitMethod.map);
    await f.fit();
    expect(f.result, isNotNull);
    // The cancelled sampler stopped (the MAP fit didn't wait for it).
    expect(sw.elapsed, lessThan(const Duration(seconds: 10)));
  });

  test('errors in the host come back as errors', () async {
    final engine = spawnFitHost()!;
    final pg = Playground(presetId: 'continuous2', nTrials: 50);
    final f = Fitting(pg, engine: engine, config: cpu.copyWith(restarts: -1));
    await f.fit();
    expect(f.error, contains('restarts'));
    await engine.close();
    await expectLater(engine.probeGpu(), throwsStateError);
  });
}
