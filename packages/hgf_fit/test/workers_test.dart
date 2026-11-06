// Runs on every platform: real isolates on the VM, the inline pool on the
// web. Either way the results must equal the synchronous functions' bit for
// bit.
import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

HgfModel _model() => HgfModel(
  network: Network()
    ..addNodes(kind: NodeKind.binaryState)
    ..addNodes(valueChildren: [0]),
  response: const BinarySoftmaxReadout(),
  parameters: [
    Parameter.node(
      'omega2',
      1,
      NodeField.tonicVolatility,
      prior: const Normal(-4, 2),
    ),
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

void _sameChains(PosteriorDraws a, PosteriorDraws b) {
  expect(a.nChains, b.nChains);
  for (var c = 0; c < a.nChains; c++) {
    final x = a.chains[c], y = b.chains[c];
    _same(x.y, y.y, 'chain $c y');
    _same(x.logDensity, y.logDensity, 'chain $c logDensity');
    _same(x.energy, y.energy, 'chain $c energy');
    expect(x.leapfrogSteps, y.leapfrogSteps);
    expect(x.stepSize, y.stepSize);
  }
}

void main() {
  final model = _model();
  final rng = math.Random(3);
  // Subjects with shared and own series, two lengths and irregular time
  // steps, so a pass spans several batch groups.
  final shared = _inputs(rng, 150);
  final truth = <List<double>>[];
  final inputs = <Float64List>[];
  final steps = <Float64List?>[];
  for (var s = 0; s < 10; s++) {
    truth.add([-4 + rng.nextDouble() * 2, 1 + rng.nextDouble() * 3]);
    inputs.add(s < 4 ? shared : _inputs(rng, s.isEven ? 150 : 90));
    steps.add(
      s == 9
          ? Float64List.fromList([for (var t = 0; t < 90; t++) 1 + t % 3])
          : null,
    );
  }
  final base = [
    for (var s = 0; s < 10; s++)
      Subject(inputs: inputs[s], timeSteps: steps[s]),
  ];
  final responses = model.simulate(base, truth, math.Random(5));
  final subjects = [
    for (var s = 0; s < 10; s++)
      Subject(
        inputs: inputs[s],
        timeSteps: steps[s],
        responses: responses[s],
        id: 's$s',
      ),
  ];

  late FitWorkers workers;
  setUpAll(() async {
    // Split even the smallest passes, so every path crosses isolates.
    workers = await FitWorkers.spawn(count: 3, minSliceTrials: 0);
  });
  tearDownAll(() => workers.close());

  test('fitMap with restarts', () async {
    final a = fitMap(model, subjects, restarts: 2, random: math.Random(1));
    final b = await workers.fitMap(
      model,
      subjects,
      restarts: 2,
      random: math.Random(1),
    );
    // Streams change how elements are grouped into passes, not how many
    // are evaluated.
    expect(b.elements, a.elements);
    for (var s = 0; s < subjects.length; s++) {
      _same(b.results[s].y, a.results[s].y, 'y');
      expect(b.results[s].negLogPosterior, a.results[s].negLogPosterior);
      expect(b.results[s].negLogLikelihood, a.results[s].negLogLikelihood);
    }
    if (workers.count > 0) expect(workers.workerSlices, greaterThan(0));
  });

  test('fitMap with Nelder-Mead and finite differences', () async {
    final a = fitMap(model, subjects, optimizer: const NelderMead());
    final b = await workers.fitMap(
      model,
      subjects,
      optimizer: const NelderMead(),
    );
    for (var s = 0; s < subjects.length; s++) {
      _same(b.results[s].y, a.results[s].y);
    }
    final c = fitMap(model, subjects, gradients: Gradients.central);
    final d = await workers.fitMap(
      model,
      subjects,
      gradients: Gradients.central,
    );
    for (var s = 0; s < subjects.length; s++) {
      _same(d.results[s].y, c.results[s].y);
    }
  });

  test('laplace', () async {
    final fit = fitMap(model, subjects);
    final a = laplace(model, fit);
    final b = await workers.laplace(model, fit);
    for (var s = 0; s < subjects.length; s++) {
      _same(b.results[s].hessian, a.results[s].hessian);
    }
    expect(b.logEvidence, a.logEvidence);
  });

  test('sampleNuts', () async {
    final a = sampleNuts(model, subjects, chains: 2, warmup: 60, draws: 40);
    final b = await workers.sampleNuts(
      model,
      subjects,
      chains: 2,
      warmup: 60,
      draws: 40,
    );
    for (var s = 0; s < subjects.length; s++) {
      _sameChains(b.results[s], a.results[s]);
    }
    expect(b.elements, a.elements);
  });

  test('sampleHierarchical and fitEmpiricalBayes', () async {
    final hier = HierarchicalModel(
      model,
      pooled: {
        'omega2': const Population(mean: Normal(-4, 3), sd: HalfNormal(2)),
      },
    );
    final a = sampleHierarchical(
      hier,
      subjects,
      chains: 2,
      warmup: 40,
      draws: 20,
    );
    final b = await workers.sampleHierarchical(
      hier,
      subjects,
      chains: 2,
      warmup: 40,
      draws: 20,
    );
    _sameChains(b.draws, a.draws);

    final c = fitEmpiricalBayes(hier, subjects);
    final d = await workers.fitEmpiricalBayes(hier, subjects);
    expect(d.history, c.history);
    expect(d.populations, c.populations);
    for (var s = 0; s < subjects.length; s++) {
      _same(d.subjects[s].y, c.subjects[s].y);
    }
  });

  test('custom response functions cross isolates', () async {
    const beta = 2.5;
    final custom = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0]),
      response: ResponseModel.fn(
        surprise: (net, e, t, r, params, o) {
          final m = net.get(e, 0, NodeField.expectedMean).clamp(1e-9, 1 - 1e-9);
          final p = 1 / (1 + math.pow((1 - m) / m, beta));
          return r.isNaN ? 0.0 : -math.log(r > 0.5 ? p : 1 - p);
        },
      ),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-4, 2),
        ),
      ],
    );
    final a = fitMap(custom, subjects);
    final b = await workers.fitMap(custom, subjects);
    for (var s = 0; s < subjects.length; s++) {
      _same(b.results[s].y, a.results[s].y);
    }
  });

  test('errors in a pass reach the caller', () async {
    final failing = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0]),
      response: ResponseModel.fn(
        surprise: (net, e, t, r, params, o) =>
            t == 50 ? throw StateError('boom') : 0.0,
      ),
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
      workers.fitMap(failing, subjects, gradients: Gradients.central),
      throwsA(anything),
    );
    // The pool still works afterwards.
    final ok = await workers.fitMap(model, subjects.sublist(0, 2));
    expect(ok.results, hasLength(2));
  });

  test('progress counts the transitions; cancel stops a job', () async {
    final progress = SamplingProgress();
    final job = workers.sampleNuts(
      model,
      subjects.sublist(0, 2),
      chains: 2,
      warmup: 30,
      draws: 20,
      progress: progress,
    );
    await job;
    expect(progress.total, 2 * 2 * 50);
    expect(progress.completed, progress.total);
    expect(progress.fraction, 1);

    // Inline (on the web), a job only gives way with a yield interval.
    workers.yieldInterval = Duration.zero;
    final long = SamplingProgress();
    final cancelled = workers.sampleNuts(
      model,
      subjects,
      warmup: 2000,
      draws: 2000,
      progress: long,
    );
    while (long.completed < 10) {
      await Future<void>.delayed(const Duration(milliseconds: 1));
    }
    workers.cancel();
    await expectLater(cancelled, throwsA(isA<FitCancelled>()));
    workers.yieldInterval = null;
    expect(long.fraction, lessThan(1));
    // Jobs started after the cancel run.
    final ok = await workers.fitMap(model, subjects.sublist(0, 2));
    expect(ok.results, hasLength(2));
  });

  test('without workers, yieldInterval lets other events run', () async {
    final inline = await FitWorkers.spawn(count: 0)
      ..yieldInterval = Duration.zero;
    var ticks = 0;
    final timer = Timer.periodic(Duration.zero, (_) => ticks++);
    final a = sampleNuts(model, subjects.sublist(0, 2), warmup: 20, draws: 20);
    final b = await inline.sampleNuts(
      model,
      subjects.sublist(0, 2),
      warmup: 20,
      draws: 20,
    );
    timer.cancel();
    expect(ticks, greaterThan(10));
    for (var s = 0; s < 2; s++) {
      _sameChains(b.results[s], a.results[s]);
    }
    // And it can be cancelled too.
    final job = inline.sampleNuts(model, subjects, warmup: 500, draws: 500);
    Timer.run(inline.cancel);
    await expectLater(job, throwsA(isA<FitCancelled>()));
    await inline.close();
  });
}
