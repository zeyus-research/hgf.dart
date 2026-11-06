// Speed-up of FitWorkers (isolate-parallel passes) over the single-isolate
// functions, on pyhgf's parameter-recovery problem (binary 2-level HGF,
// sigmoid decisions, ω₂ and β per agent, 320 trials), checking that the
// results are bit-identical.
//
//   dart compile exe benchmark/bin/parallel_benchmark.dart -o /tmp/pb && /tmp/pb
//   /tmp/pb --workers 2,4,8 --agents 20,200 --draws 300
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

Future<void> main(List<String> args) async {
  var workerCounts = [2, 4, 8];
  var agentCounts = [20, 200];
  var draws = 300, repeat = 5;
  var minSlice = FitWorkers.defaultMinSliceTrials;
  var only = <String>{};
  for (var i = 0; i < args.length; i++) {
    List<int> ints() => [for (final s in args[++i].split(',')) int.parse(s)];
    switch (args[i]) {
      case '--workers':
        workerCounts = ints();
      case '--agents':
        agentCounts = ints();
      case '--draws':
        draws = int.parse(args[++i]);
      case '--repeat':
        repeat = int.parse(args[++i]);
      case '--min-slice':
        minSlice = int.parse(args[++i]);
      case '--only':
        only = args[++i].split(',').toSet();
    }
  }
  bool want(String s) => only.isEmpty || only.contains(s);

  final u = Float64List.fromList([
    for (final l in File('data/binary_input.txt').readAsLinesSync())
      if (l.trim().isNotEmpty) double.parse(l),
  ]);
  final model = HgfModel(
    network: Network()
      ..addNodes(kind: NodeKind.binaryState)
      ..addNodes(valueChildren: [0]),
    response: const BinarySoftmaxReadout(),
    parameters: [
      Parameter.node(
        'omega2',
        1,
        NodeField.tonicVolatility,
        prior: TruncatedNormal(-3, 5, lower: -8, upper: 2),
      ),
      Parameter.response(
        'inverseTemperature',
        prior: const Uniform(0.2, 20),
        initial: 1.0,
      ),
    ],
  );
  List<Subject> subjectsFor(int n) {
    final responses = model.simulate(
      [for (var i = 0; i < n; i++) Subject(inputs: u)],
      [
        for (var i = 0; i < n; i++)
          [-6.0 + 5.0 * i / (n - 1), 0.5 + 5.5 * i / (n - 1)],
      ],
      math.Random(123),
    );
    return [
      for (var i = 0; i < n; i++)
        Subject(inputs: u, responses: responses[i], id: i),
    ];
  }

  final pools = <int, FitWorkers>{
    for (final w in workerCounts)
      w: await FitWorkers.spawn(count: w, minSliceTrials: minSlice),
  };
  stdout.writeln(
    'cores ${Platform.numberOfProcessors}, minSliceTrials $minSlice; '
    'time in ms (speed-up over one isolate)',
  );
  stdout.writeln(
    '| job | 1 isolate | ${workerCounts.map((w) => '$w workers').join(' | ')} |',
  );
  stdout.writeln('|---|---|${workerCounts.map((_) => '---|').join()}');

  Future<void> row<T>(
    String name,
    int reps,
    T Function() sync,
    Future<T> Function(FitWorkers) parallel,
    List<double> Function(T) fingerprint,
  ) async {
    // Warm up both paths (JIT-free under AOT, but caches and workers).
    final ref = sync();
    for (final p in pools.values) {
      await parallel(p);
    }
    double best(Iterable<double> xs) => xs.reduce(math.min);
    final ts = <double>[];
    for (var r = 0; r < reps; r++) {
      final sw = Stopwatch()..start();
      sync();
      ts.add(sw.elapsedMicroseconds / 1000);
    }
    final t1 = best(ts);
    final cells = <String>[];
    for (final w in workerCounts) {
      final pool = pools[w]!;
      final tp = <double>[];
      late T got;
      for (var r = 0; r < reps; r++) {
        final sw = Stopwatch()..start();
        got = await parallel(pool);
        tp.add(sw.elapsedMicroseconds / 1000);
      }
      final a = fingerprint(ref), b = fingerprint(got);
      var same = a.length == b.length;
      for (var i = 0; same && i < a.length; i++) {
        same = a[i] == b[i] || (a[i].isNaN && b[i].isNaN);
      }
      if (!same) throw StateError('$name on $w workers differs from sync');
      final t = best(tp);
      cells.add('${_fmt(t)} (${(t1 / t).toStringAsFixed(2)}×)');
    }
    stdout.writeln('| $name | ${_fmt(t1)} | ${cells.join(' | ')} |');
  }

  for (final n in agentCounts) {
    final subjects = subjectsFor(n);
    if (want('map')) {
      await row(
        'MAP, $n agents',
        repeat,
        () => fitMap(model, subjects),
        (p) => p.fitMap(model, subjects),
        (f) => [for (final r in f.results) ...r.y],
      );
    }
    if (want('laplace')) {
      final fit = fitMap(model, subjects);
      await row(
        'Laplace, $n agents',
        repeat,
        () => laplace(model, fit),
        (p) => p.laplace(model, fit),
        (f) => [for (final r in f.results) ...r.hessian],
      );
    }
    if (want('nuts')) {
      await row(
        'NUTS, $n agents, 4 × ($draws + $draws)',
        1,
        () => sampleNuts(model, subjects, warmup: draws, draws: draws),
        (p) => p.sampleNuts(model, subjects, warmup: draws, draws: draws),
        (f) => [
          for (final r in f.results)
            for (final c in r.chains) ...c.y,
        ],
      );
    }
  }

  if (want('hier')) {
    final subjects = subjectsFor(10);
    final hier = HierarchicalModel(
      model.withPriors({
        'omega2': const Normal(-4, 1),
        'inverseTemperature': const LogNormal(0.5, 0.5),
      }),
      pooled: {
        'omega2': const Population(mean: Normal(-5, 5), sd: HalfNormal(10)),
        'inverseTemperature': const Population(
          mean: Normal(0, 2),
          sd: HalfNormal(2),
        ),
      },
      centered: true,
    );
    await row(
      'hierarchical NUTS, 10 agents, 4 × ($draws + $draws)',
      1,
      () => sampleHierarchical(hier, subjects, warmup: draws, draws: draws),
      (p) => p.sampleHierarchical(hier, subjects, warmup: draws, draws: draws),
      (f) => [for (final c in f.draws.chains) ...c.y],
    );
    await row(
      'empirical Bayes, 10 agents',
      repeat,
      () => fitEmpiricalBayes(hier, subjects),
      (p) => p.fitEmpiricalBayes(hier, subjects),
      (f) => [for (final s in f.subjects) ...s.y],
    );
  }
  for (final p in pools.values) {
    stdout.writeln(
      '${p.count} workers: ${p.workerSlices} slices on workers, '
      '${p.localSlices} here',
    );
    await p.close();
  }
}

String _fmt(double ms) => ms >= 1000
    ? '${(ms / 1000).toStringAsFixed(2)} s'
    : '${ms.toStringAsFixed(1)} ms';
