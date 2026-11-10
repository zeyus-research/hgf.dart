@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hgf_playground/src/model/fit_config.dart';
import 'package:hgf_playground/src/model/fit_engine.dart';
import 'package:hgf_playground/src/model/fit_wire.dart';
import 'package:hgf_playground/src/model/fitting.dart';
import 'package:hgf_playground/src/model/playground.dart';

/// Jobs, events and results through the web fit host's wire format (as the
/// page and the worker exchange them), run on a [LocalFitEngine].
final class _WireEngine implements FitEngine {
  _WireEngine(this.inner);

  final LocalFitEngine inner;
  var jobs = 0, events = 0;

  @override
  Future<GpuInfo> probeGpu() async =>
      decodeGpuInfo(encodeGpuInfo(await inner.probeGpu()));

  @override
  Future<R> run<R>(FitJobSpec<R> job, void Function(JobEvent) onEvent) async {
    final m = encodeJob(job);
    final there = decodeJob(m.json, _copied(m.arrays));
    jobs++;
    final result = await inner.run(there, (e) {
      events++;
      onEvent(decodeEvent(encodeEvent(e)));
    });
    final r = encodeResult(result);
    return decodeResult(job, r.json, _copied(r.arrays));
  }

  // Structured clone: copies.
  static List<Float64List> _copied(List<Float64List> a) => [
    for (final x in a) Float64List.fromList(x),
  ];

  @override
  void cancel() => inner.cancel();

  @override
  Future<void> close() => inner.close();
}

void main() {
  const cpu = FitConfig(
    device: FitDevice.cpu,
    workers: 2,
    restarts: 1,
    warmup: 60,
    draws: 40,
  );

  test('pack keeps arrays, non-finite numbers and nesting', () {
    final a = Float64List.fromList([1, double.nan, -0.0]);
    final value = {
      'a': a,
      'b': [double.infinity, double.negativeInfinity, double.nan, 1.5, 2],
      'c': {'d': null, 'e': true, 'f': 'x', 'g': a},
    };
    final p = pack(value);
    expect(p.arrays, hasLength(2));
    final back = unpack(p.json, p.arrays)! as Map<String, Object?>;
    expect(back['a'], same(a));
    final b = back['b']! as List<Object?>;
    expect(b[0], double.infinity);
    expect(b[1], double.negativeInfinity);
    expect((b[2]! as double).isNaN, isTrue);
    expect(b.sublist(3), [1.5, 2]);
    expect(back['c'], {'d': null, 'e': true, 'f': 'x', 'g': a});
  });

  for (final (preset, target) in [
    ('continuous2', FitTarget.responses),
    ('binary3', FitTarget.responses),
    ('categorical3', FitTarget.responses),
  ]) {
    test(
      '$preset: fits and recoveries through the wire match local ones',
      () async {
        Fitting make(FitEngine engine) {
          final pg = Playground(presetId: preset, nTrials: 150);
          final f = Fitting(pg, engine: engine, config: cpu)..setTarget(target);
          if (target == FitTarget.responses) {
            f
              ..setAgent(noise: 0.05)
              ..simulateAgent(seed: 3);
          }
          addTearDown(f.dispose);
          return f;
        }

        final wire = _WireEngine(LocalFitEngine());
        final local = make(LocalFitEngine()), remote = make(wire);
        final stages = <String>[];
        remote.addListener(() {
          if (remote.stage case final s? when !stages.contains(s)) {
            stages.add(s);
          }
        });

        for (final method in FitMethod.values) {
          local.setMethod(method);
          remote.setMethod(method);
          await local.fit();
          await remote.fit();
          expect(remote.error, isNull);
          final a = local.result!, b = remote.result!;
          expect(b.where, a.where);
          expect(b.converged, a.converged);
          expect(b.note, a.note);
          expect(b.logEvidence, a.logEvidence);
          expect(b.chains, a.chains);
          expect(b.divergences, a.divergences);
          for (final (i, e) in b.estimates.indexed) {
            final f = a.estimates[i];
            expect(e.param.id, f.param.id);
            expect(
              [e.estimate, e.lower, e.upper, e.truth, e.rhat, e.ess],
              [f.estimate, f.lower, f.upper, f.truth, f.rhat, f.ess],
            );
            expect(e.draws, f.draws);
            expect(e.density?.call(e.estimate), f.density?.call(f.estimate));
          }
        }
        expect(stages, contains('Finding the mode (MAP)…'));

        local.setRecoveryAgents(50);
        remote.setRecoveryAgents(50);
        await local.recover(seed: 1);
        await remote.recover(seed: 1);
        expect(remote.error, isNull);
        final a = local.recovery!, b = remote.recovery!;
        expect(b.truth, a.truth);
        expect(b.estimate, a.estimate);
        expect(b.ok, a.ok);
        expect(
          [b.dropped, b.redrawn, b.unconverged],
          [a.dropped, a.redrawn, a.unconverged],
        );
        expect(b.params.map((p) => p.id), a.params.map((p) => p.id));
        expect(wire.jobs, 3);
        expect(wire.events, greaterThan(3));
      },
    );
  }
}
