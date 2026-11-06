// FitWorkers on Web Workers, with the worker compiled by both web
// compilers. Needs the compiled workers next to this file (see
// fit_worker.dart; `melos run test:web-workers` builds them first).
@TestOn('browser')
@Tags(['web-workers'])
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

import 'test_codec.dart';

const _wasmPage = bool.fromEnvironment('dart.tool.dart2wasm');

HgfModel _model([Prior omegaPrior = const Normal(-4, 2)]) => HgfModel(
  network: Network()
    ..addNodes(kind: NodeKind.binaryState)
    ..addNodes(valueChildren: [0]),
  response: const BinarySoftmaxReadout(),
  parameters: [
    Parameter.node('omega2', 1, NodeField.tonicVolatility, prior: omegaPrior),
    Parameter.response('inverseTemperature', prior: const LogNormal(0.5, 1)),
  ],
);

Float64List _inputs(math.Random rng, int n) {
  final out = Float64List(n);
  var p = 0.8;
  for (var t = 0; t < n; t++) {
    if (t % 40 == 0) p = 1 - p;
    out[t] = rng.nextDouble() < p ? 1.0 : 0.0;
  }
  return out;
}

void _same(List<double> a, List<double> b, [String? reason]) {
  expect(a.length, b.length, reason: reason);
  for (var i = 0; i < a.length; i++) {
    final x = a[i], y = b[i];
    if (x.isNaN && y.isNaN) continue;
    expect(x, y, reason: '${reason ?? ''}[$i]');
  }
}

void _close(List<double> a, List<double> b, double tol, [String? reason]) {
  expect(a.length, b.length, reason: reason);
  for (var i = 0; i < a.length; i++) {
    expect(a[i], closeTo(b[i], tol * (1 + b[i].abs())), reason: reason);
  }
}

void _sameChains(PosteriorDraws a, PosteriorDraws b) {
  expect(a.nChains, b.nChains);
  for (var c = 0; c < a.nChains; c++) {
    _same(a.chains[c].y, b.chains[c].y, 'chain $c y');
    expect(a.chains[c].leapfrogSteps, b.chains[c].leapfrogSteps);
  }
}

void main() {
  final model = _model();
  final rng = math.Random(3);
  final shared = _inputs(rng, 150);
  final truth = <List<double>>[];
  final inputs = <Float64List>[];
  final steps = <Float64List?>[];
  for (var s = 0; s < 8; s++) {
    truth.add([-4 + rng.nextDouble() * 2, 1 + rng.nextDouble() * 3]);
    inputs.add(s < 3 ? shared : _inputs(rng, s.isEven ? 150 : 90));
    steps.add(
      s == 7
          ? Float64List.fromList([for (var t = 0; t < 90; t++) 1 + t % 3])
          : null,
    );
  }
  final responses = model.simulate(
    [
      for (var s = 0; s < 8; s++)
        Subject(inputs: inputs[s], timeSteps: steps[s]),
    ],
    truth,
    math.Random(5),
  );
  final subjects = [
    for (var s = 0; s < 8; s++)
      Subject(
        inputs: inputs[s],
        timeSteps: steps[s],
        responses: responses[s]..[3] = double.nan,
        observed: s == 4
            ? (Float64List(inputs[s].length)
                ..fillRange(0, inputs[s].length, 1.0)
                ..[10] = 0.0)
            : null,
        id: 's$s',
      ),
  ];

  for (final wasmWorker in [false, true]) {
    final exact = wasmWorker == _wasmPage;
    group(wasmWorker ? 'Wasm worker' : 'JS worker', () {
      late FitWorkers workers;
      setUpAll(() async {
        workers = await FitWorkers.spawn(
          count: 3,
          minSliceTrials: 0,
          script: wasmWorker
              ? const FitWorkerScript.wasm('fit_worker.wasm')
              : const FitWorkerScript.js('fit_worker.js'),
        );
        expect(workers.count, 3);
      });
      tearDownAll(() => workers.close());

      test('fitMap', () async {
        final a = fitMap(model, subjects, restarts: 1);
        final b = await workers.fitMap(model, subjects, restarts: 1);
        expect(workers.workerSlices, greaterThan(0));
        expect(workers.localSlices, 0);
        for (var s = 0; s < subjects.length; s++) {
          if (exact) {
            _same(b.results[s].y, a.results[s].y, 'y');
            expect(b.results[s].negLogPosterior, a.results[s].negLogPosterior);
          } else {
            // exp/log may differ in the last bit between compilers.
            _close(b.results[s].y, a.results[s].y, 1e-5, 'y');
          }
        }
      });

      test('laplace, NUTS, hierarchical NUTS and empirical Bayes', () async {
        final fit = fitMap(model, subjects);
        final la = laplace(model, fit);
        final lb = await workers.laplace(model, fit);
        if (exact) {
          expect(lb.logEvidence, la.logEvidence);
        } else {
          expect(lb.logEvidence, closeTo(la.logEvidence, 1e-4));
        }
        if (!exact) return; // chains diverge from a last-bit difference
        final na = sampleNuts(
          model,
          subjects,
          chains: 2,
          warmup: 30,
          draws: 20,
        );
        final nb = await workers.sampleNuts(
          model,
          subjects,
          chains: 2,
          warmup: 30,
          draws: 20,
        );
        for (var s = 0; s < subjects.length; s++) {
          _sameChains(nb.results[s], na.results[s]);
        }
        final hier = HierarchicalModel(
          model,
          pooled: {
            'omega2': const Population(mean: Normal(-4, 3), sd: HalfNormal(2)),
          },
        );
        final ha = sampleHierarchical(
          hier,
          subjects,
          chains: 2,
          warmup: 30,
          draws: 15,
        );
        final hb = await workers.sampleHierarchical(
          hier,
          subjects,
          chains: 2,
          warmup: 30,
          draws: 15,
        );
        _sameChains(hb.draws, ha.draws);
        // Refits with package-internal population priors, which cross as
        // FitCodec JSON.
        final ea = fitEmpiricalBayes(hier, subjects);
        final eb = await workers.fitEmpiricalBayes(hier, subjects);
        expect(eb.history, ea.history);
        expect(eb.populations, ea.populations);
      });

      test('custom prior decoded by the worker\'s codec', () async {
        final custom = _model(const LaplacePrior(-4, 1.5));
        final a = fitMap(custom, subjects);
        final b = await workers.fitMap(custom, subjects);
        for (var s = 0; s < subjects.length; s++) {
          if (exact) {
            _same(b.results[s].y, a.results[s].y);
          } else {
            _close(b.results[s].y, a.results[s].y, 1e-5);
          }
        }
      });

      test('errors reach the caller; the pool keeps working', () async {
        // Can't be serialised: fails before anything is sent.
        final fn = HgfModel(
          network: Network()
            ..addNodes(kind: NodeKind.binaryState)
            ..addNodes(valueChildren: [0]),
          response: ResponseModel.fn(surprise: (n, e, t, r, p, o) => 0.0),
          parameters: [
            Parameter.node(
              'omega2',
              1,
              NodeField.tonicVolatility,
              prior: const Normal(-4, 2),
            ),
          ],
        );
        await expectLater(
          workers.fitMap(fn, subjects),
          throwsA(isA<UnsupportedError>()),
        );
        // The worker can't decode it.
        await expectLater(
          workers.fitMap(_model(const UnknownToWorker()), subjects),
          throwsA(
            predicate(
              (e) => '$e'.contains('Unknown prior type "UnknownToWorker"'),
            ),
          ),
        );
        // Throws while evaluating in the worker.
        await expectLater(
          workers.fitMap(
            _model(const ThrowsInWorker()),
            subjects,
            gradients: Gradients.central,
          ),
          throwsA(predicate((e) => '$e'.contains('boom in the worker'))),
        );
        final ok = await workers.fitMap(model, subjects.sublist(0, 2));
        expect(ok.results, hasLength(2));
      });
    });
  }

  test('a missing script fails spawn', () async {
    for (final script in const [
      FitWorkerScript.js('missing.js'),
      FitWorkerScript.wasm('missing.wasm'),
    ]) {
      await expectLater(
        FitWorkers.spawn(count: 2, script: script),
        throwsA(isA<StateError>()),
      );
    }
  });

  test('no script: no workers', () async {
    final w = await FitWorkers.spawn(count: 2);
    expect(w.count, 0);
    await w.fitMap(model, subjects.sublist(0, 1));
    expect(w.localSlices, greaterThan(0));
    await w.close();
  });
}
