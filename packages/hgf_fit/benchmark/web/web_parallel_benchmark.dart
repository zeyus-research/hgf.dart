// Speed-up of FitWorkers on Web Workers over the page's main thread, in a
// browser: bench:parallel's jobs on pyhgf's parameter-recovery problem
// (binary 2-level HGF, ω₂ and β per agent, 320 trials). The page and its
// worker are compiled alike (both dart2wasm -O4, or both dart2js -O2), so
// the results are bit-identical (checked). Run by `melos run
// bench:web-parallel` (tool/web/fit_workers_bench.py), which prints the
// `HGF_BENCH` line this logs.
import 'dart:convert';
import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

const _wasm = bool.fromEnvironment('dart.tool.dart2wasm');

Future<double> _time(Future<Object?> Function() f, int repeat) async {
  var best = double.infinity;
  for (var r = 0; r < repeat; r++) {
    final sw = Stopwatch()..start();
    await f();
    best = math.min(best, sw.elapsedMicroseconds / 1000);
  }
  return best;
}

Future<void> main() async {
  final query = Uri.base.queryParameters;
  final counts = [
    0,
    for (final s in (query['workers'] ?? '2,4,8').split(',')) int.parse(s),
  ];
  final draws = int.parse(query['draws'] ?? '150');
  final text = await _fetch('binary_input.txt');
  final u = Float64List.fromList([
    for (final l in const LineSplitter().convert(text))
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

  final s10 = subjectsFor(10), s20 = subjectsFor(20), s200 = subjectsFor(200);
  final hier = HierarchicalModel(
    model,
    pooled: {
      'omega2': const Population(mean: Normal(-3, 3), sd: HalfNormal(2)),
      'inverseTemperature': const Population(
        mean: Normal(0, 1.5),
        sd: HalfNormal(1),
      ),
    },
  );
  const script = _wasm
      ? FitWorkerScript.wasm('fit_worker.wasm')
      : FitWorkerScript.js('fit_worker.js');
  final pools = [
    for (final c in counts) await FitWorkers.spawn(count: c, script: script),
  ];
  final jobs = <String, (int, Future<Object?> Function(FitWorkers))>{
    'MAP, 20 agents': (5, (w) => w.fitMap(model, s20)),
    'Laplace, 20 agents': (
      5,
      (w) async => w.laplace(model, await w.fitMap(model, s20)),
    ),
    'MAP, 200 agents': (3, (w) => w.fitMap(model, s200)),
    'Laplace, 200 agents': (
      3,
      (w) async => w.laplace(model, await w.fitMap(model, s200)),
    ),
    'NUTS, 20 agents': (
      1,
      (w) => w.sampleNuts(model, s20, chains: 4, warmup: draws, draws: draws),
    ),
    'hierarchical NUTS, 10 agents': (
      1,
      (w) => w.sampleHierarchical(
        hier,
        s10,
        chains: 4,
        warmup: draws,
        draws: draws,
      ),
    ),
    'empirical Bayes, 10 agents': (3, (w) => w.fitEmpiricalBayes(hier, s10)),
  };
  final rows = <Map<String, Object?>>[];
  for (final MapEntry(key: name, value: (repeat, job)) in jobs.entries) {
    final times = [for (final w in pools) await _time(() => job(w), repeat)];
    rows.add({'job': name, 'ms': times});
    print('$name: ${times.map((t) => t.toStringAsFixed(1)).join(' / ')} ms');
  }
  // Bit-identical to the main thread.
  final a = await pools.first.fitMap(model, s200);
  final b = await pools.last.fitMap(model, s200);
  var identical = true;
  for (var s = 0; s < s200.length; s++) {
    for (var i = 0; i < model.dimension; i++) {
      if (a.results[s].y[i] != b.results[s].y[i]) identical = false;
    }
  }
  for (final w in pools) {
    await w.close();
  }
  print(
    'HGF_BENCH ${jsonEncode({'target': _wasm ? 'dart2wasm' : 'dart2js', 'workers': counts, 'draws': draws, 'identical': identical, 'rows': rows})}',
  );
}

Future<String> _fetch(String url) async {
  final response = await _jsFetch(url).toDart;
  return (await response.text().toDart).toDart;
}

@JS('fetch')
external JSPromise<_Response> _jsFetch(String url);

extension type _Response._(JSObject _) implements JSObject {
  external JSPromise<JSString> text();
}
