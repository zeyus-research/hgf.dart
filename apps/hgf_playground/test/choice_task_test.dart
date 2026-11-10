import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_playground/src/model/data.dart';
import 'package:hgf_playground/src/model/fit_config.dart';
import 'package:hgf_playground/src/model/fitting.dart';
import 'package:hgf_playground/src/model/fit_wire.dart';
import 'package:hgf_playground/src/model/playground.dart';
import 'package:hgf_playground/src/model/presets.dart';

Future<FitWorkers> _inline([int? _]) => FitWorkers.spawn(count: 0);

void main() {
  group('choice task fitting', () {
    test('IGT: the agent\'s choices, a shared ω and β', () async {
      final pg = Playground(presetId: 'igt', nTrials: 300);
      final f = Fitting(pg, spawnWorkers: _inline);
      expect(f.target, FitTarget.responses);
      expect(f.observationsUnavailable, isNotNull);
      f.setTarget(FitTarget.observations);
      expect(f.target, FitTarget.responses);
      expect(f.free, {'omega', 'beta'});
      expect(f.responses, same(pg.choices));
      expect(f.truth, {
        'omega': -2,
        for (final i in [4, 5, 6, 7]) 'omega_$i': -2,
        'pi_u': 1,
        'beta': pg.agentBeta,
      });
      // The shared ω and the per-deck ones exclude each other.
      f.toggle('omega_5', true);
      expect(f.free, {'omega_5', 'beta'});
      f.toggle('omega', true);
      expect(f.free, {'omega', 'beta'});

      await f.fit();
      expect(f.error, isNull);
      final r = f.result!;
      for (final e in r.estimates) {
        expect(e.truth, inInclusiveRange(e.lower, e.upper), reason: e.param.id);
      }
      // Simulated choices would change under the applied values.
      expect(f.applyUnavailable, isNotNull);

      // A slider edit that changes the choices is new data.
      pg.setAgentBeta(1);
      expect(f.result, isNull);
      expect(f.responses, same(pg.choices));
      expect(f.truth!['beta'], 1);
    });

    test('recorded choices: fit, then apply β and the shared ω', () async {
      final pg = Playground(presetId: 'igt', nTrials: 200);
      final f = Fitting(pg, spawnWorkers: _inline);
      final choices = Float64List.fromList(pg.choices!);
      final dollars = Float64List.fromList([
        for (final p in pg.observations) p * 100,
      ]);
      pg.loadChoices(choices, [dollars], 'record.csv');
      expect(f.responses, same(choices));
      expect(f.responsesSource, 'record.csv');
      expect(f.truth, isNull);
      expect(f.applyUnavailable, isNull);
      await f.fit();
      expect(f.error, isNull);
      final r = f.result!;
      f.apply();
      final omega = r.estimates.firstWhere((e) => e.param.id == 'omega');
      final beta = r.estimates.firstWhere((e) => e.param.id == 'beta');
      for (final i in [4, 5, 6, 7]) {
        expect(pg.param(i, NodeField.tonicVolatility), omega.estimate);
      }
      expect(pg.agentBeta, beta.estimate);
      expect(pg.choices, same(choices), reason: 'recorded choices stay');
    });

    for (final id in ['igt', 'bandit']) {
      test('$id: recovery plays the task per agent', () async {
        final pg = Playground(presetId: id, nTrials: 200);
        final f = Fitting(pg, spawnWorkers: _inline)..setRecoveryAgents(50);
        expect(f.canRecover, isTrue);
        await f.recover(seed: 2);
        expect(f.error, isNull);
        final r = f.recovery!;
        final beta = r.params.indexWhere((p) => p.id == 'beta');
        expect(r.stats[beta].r, greaterThan(0.6));
      });
    }

    test('the recovery job carries the reveals over the wire', () {
      final job = RecoveryJob(
        config: const FitConfig(),
        gpu: false,
        params: const [],
        model: HgfModel(
          network: IowaGamblingTask.network(),
          response: IowaGamblingTask.readout(),
          parameters: [
            Parameter(
              'omega',
              targets: [
                for (final n in IowaGamblingTask.valueNodes)
                  NodeTarget(n, NodeField.tonicVolatility),
              ],
              prior: const Normal(-2, 1),
            ),
          ],
        ),
        base: Subject(inputs: Float64List(8)),
        agents: 3,
        seed: 1,
        reveals: IowaGamblingTask.reveals,
      );
      final packed = encodeJob(job);
      final back = decodeJob(packed.json, packed.arrays) as RecoveryJob;
      expect(back.reveals, IowaGamblingTask.reveals);
    });
  });

  group('choice tasks', () {
    for (final id in ['igt', 'bandit']) {
      test('$id: the agent sees only what its choice reveals', () {
        final pg = Playground(presetId: id);
        final task = pg.task!;
        final nIn = pg.network.inputIdxs.length;
        final c = pg.choices!;
        expect(c.length, pg.preset.defaultTrials);
        expect(pg.observed!.length, c.length * nIn);
        expect(pg.error, isNull);
        for (var t = 0; t < c.length; t++) {
          final k = c[t].toInt();
          final seen = [
            for (var i = 0; i < nIn; i++)
              if (pg.observed![t * nIn + i] == 1) i,
          ];
          expect(seen, task.reveals[k], reason: 'trial $t');
        }
        // Every option gets chosen at some point.
        expect(c.toSet().length, task.options.length);
      });

      test('$id: the choices are what the shown beliefs would choose', () {
        // The playground's own filter (Network) agrees with the agent's
        // loop (BatchNetwork): the agent's log likelihood of its choices,
        // from the plotted choice probabilities, matches the fit model's.
        final pg = Playground(presetId: id);
        final series = pg.choiceSeries()!;
        var nll = 0.0;
        for (var t = 0; t < pg.choices!.length; t++) {
          final k = pg.choices![t].toInt();
          nll -= math.log(series.probabilities[k][t]);
        }
        final model = HgfModel(
          network: Network.fromJson(pg.network.toJson()),
          response: pg.task!.readout(pg.agentBeta),
          parameters: [
            Parameter.response(
              'beta',
              responseName: 'inverseTemperature',
              prior: const LogNormal(0, 1),
            ),
          ],
        );
        final subject = Subject(
          inputs: pg.networkInput,
          observed: pg.observed,
          responses: pg.choices,
        );
        final out = Float64List(1), lik = Float64List(1);
        model.negLogPosterior(
          [subject],
          Int32List(1),
          model.toUnconstrained([pg.agentBeta]),
          out,
          negLogLikelihood: lik,
        );
        expect(nll, closeTo(lik[0], 1e-9 * lik[0].abs()));
      });
    }

    test('edits play the task again; same seed, same draws', () {
      final pg = Playground(presetId: 'igt');
      final first = Float64List.fromList(pg.choices!);
      final obs = pg.observations;
      pg.setParam(0, NodeField.precision, 1.0); // no change in value
      expect(pg.choices, first);
      // Nothing changed: the same data (so a fit result stays).
      expect(identical(pg.observations, obs), isTrue);
      pg.setAgentBeta(0.01);
      expect(pg.choices, isNot(first));
      pg.setAgentBeta(pg.preset.task!.beta);
      expect(pg.choices, first);
      // Higher volatility: different beliefs, different choices.
      pg.setParam(4, NodeField.tonicVolatility, 1.0);
      expect(pg.choices, isNot(first));
    });

    test('recorded choices and payoffs (IGT, in dollars)', () {
      final pg = Playground(presetId: 'igt');
      final choices = Float64List.fromList([0, 2, double.nan, 3, 1]);
      final payoffs = Float64List.fromList([100, -200, double.nan, 50, 100]);
      pg.loadChoices(choices, [payoffs], 'record.csv');
      expect(pg.source, DataSource.loaded);
      expect(pg.observed, [
        ...[1, 0, 0, 0],
        ...[0, 0, 1, 0],
        ...[0, 0, 0, 0],
        ...[0, 0, 0, 1],
        ...[0, 1, 0, 0],
      ]);
      expect(pg.networkInput.sublist(4, 8), [0, 0, -2, 0]);
      expect(pg.observations.sublist(0, 2), [1, -2]);
      expect(pg.trajectories, isNotNull);
      // Edits don't touch recorded choices.
      pg.setParam(4, NodeField.tonicVolatility, 0.0);
      expect(pg.choices, same(choices));
      expect(
        () => pg.loadChoices(Float64List.fromList([4]), [
          Float64List.fromList([1]),
        ], 'bad'),
        throwsFormatException,
      );
      // Another preset: back to simulated data.
      pg.setPreset('binary3');
      expect(pg.source, DataSource.simulated);
      expect(pg.choices, isNull);
      expect(pg.observed, isNull);
    });

    test('bandit outcomes: the good arm swaps every block', () {
      final o = simulateOutcomes(
        DataModel.bandit,
        DataSettings(nTrials: 100, block: 50, pHigh: 0.9),
      );
      expect(o.nTrials, 100);
      expect(o.hidden[0][0], closeTo(0.8, 1e-12));
      expect(o.hidden[0][50], closeTo(-0.8, 1e-12));
    });
  });
}
