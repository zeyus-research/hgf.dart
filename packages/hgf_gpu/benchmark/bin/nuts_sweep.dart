// NUTS across network types, agent counts and chain counts: hgf_fit on one
// isolate, on a FitWorkers pool and on the GPU (GpuFitter, wgpu-native).
// It also writes each problem (data, model JSON) and the model's pyhgf code
// (hgf_fit's PythonExport), which tool/parity/nuts_sweep.py samples with
// NumPyro (CPU and GPU) and PyMC, so both sides fit the same model by
// construction. From packages/hgf_gpu:
//
//   dart compile exe benchmark/bin/nuts_sweep.dart -o .dart_tool/nuts_sweep
//   .dart_tool/nuts_sweep --out ../../.dart_tool/nuts_sweep
//
// Models: b2/b3 binary 2/3-level HGF with a sigmoid decision (ω₂ and β per
// agent, pyhgf's recovery notebook); c2/c3 continuous 2/3-level HGF on a
// random walk with Gaussian read-out of the prediction (ω₁ and ω₂ per
// agent). Every agent has 320 trials.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/codegen.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';

const trials = 320;

HgfModel binary(int levels) => HgfModel(
  network: levels == 3
      ? (Network()
          ..addNodes(kind: NodeKind.binaryState)
          ..addNodes(valueChildren: [0])
          ..addNodes(volatilityChildren: [1], tonicVolatility: -6))
      : (Network()
          ..addNodes(kind: NodeKind.binaryState)
          ..addNodes(valueChildren: [0])),
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

HgfModel continuous(int levels) {
  final net = Network()
    ..addNodes(precision: 1e4)
    ..addNodes(valueChildren: [0], mean: 1.0)
    ..addNodes(volatilityChildren: [1]);
  if (levels == 3) net.addNodes(volatilityChildren: [2], tonicVolatility: -6);
  return HgfModel(
    network: net,
    response: const GaussianReadout(precision: 4e4),
    parameters: [
      Parameter.node(
        'omega1',
        1,
        NodeField.tonicVolatility,
        prior: TruncatedNormal(-10, 4, lower: -16, upper: -2),
        initial: -10,
      ),
      Parameter.node(
        'omega2',
        2,
        NodeField.tonicVolatility,
        prior: TruncatedNormal(-4, 4, lower: -10, upper: 2),
        initial: -4,
      ),
    ],
  );
}

/// Each model, its inputs and the range its agents' true values span.
final models = <String, (HgfModel Function(), String, List<(double, double)>)>{
  'b2': (() => binary(2), 'binary', [(-6, -1), (0.5, 6)]),
  'b3': (() => binary(3), 'binary', [(-6, -1), (0.5, 6)]),
  'c2': (() => continuous(2), 'continuous', [(-12, -8), (-6, -2)]),
  'c3': (() => continuous(3), 'continuous', [(-12, -8), (-6, -2)]),
};

Future<void> main(List<String> args) async {
  var modelNames = models.keys.toList();
  var agentCounts = [20, 200];
  var chainCounts = [4];
  var chainSweep = <int>[1, 4, 16];
  var warmup = 500, draws = 500;
  var cpu1Max = 20;
  var runners = {'cpu1', 'pool', 'gpu'};
  var workers = Platform.numberOfProcessors;
  var outDir = '../../.dart_tool/nuts_sweep';
  for (var i = 0; i < args.length; i++) {
    List<int> ints() => [for (final s in args[++i].split(',')) int.parse(s)];
    switch (args[i]) {
      case '--models':
        modelNames = args[++i].split(',');
      case '--agents':
        agentCounts = ints();
      case '--chains':
        chainCounts = ints();
      case '--chain-sweep':
        chainSweep = args[i + 1] == 'none' ? (i++, <int>[]).$2 : ints();
      case '--warmup':
        warmup = int.parse(args[++i]);
      case '--draws':
        draws = int.parse(args[++i]);
      case '--cpu1-max-agents':
        cpu1Max = int.parse(args[++i]);
      case '--runners':
        runners = args[++i].split(',').toSet();
      case '--workers':
        workers = int.parse(args[++i]);
      case '--out':
        outDir = args[++i];
      default:
        stderr.writeln('Unknown argument ${args[i]}');
        exit(64);
    }
  }
  final out = Directory(outDir)..createSync(recursive: true);
  final results = File('${out.path}/dart.jsonl')..writeAsStringSync('');

  final binaryInputs = Float64List.fromList([
    for (final l in File('../hgf_fit/data/binary_input.txt').readAsLinesSync())
      if (l.trim().isNotEmpty) double.parse(l),
  ]).sublist(0, trials);
  final benchInputs =
      jsonDecode(File('../hgf/benchmark/data/inputs.json').readAsStringSync())
          as Map<String, Object?>;
  final continuousInputs = Float64List.fromList([
    for (final v in (benchInputs['continuous']! as List<Object?>).take(trials))
      (v! as num).toDouble(),
  ]);

  final backend = runners.contains('gpu') ? WgpuBackend.create() : null;
  if (runners.contains('gpu') && backend == null) {
    stderr.writeln('No wgpu adapter');
    exit(1);
  }
  final gpuName = backend?.description.split(' / ').elementAtOrNull(1);
  final pool = runners.contains('pool') || backend != null
      ? await FitWorkers.spawn(count: workers)
      : null;
  final gpu = backend == null ? null : GpuFitter(backend, cpu: pool);

  // Cells: every model x agents at the default chains, plus the chain
  // sweep on the first model at the smallest agent count.
  final cells = <(String, int, int)>{
    for (final m in modelNames)
      for (final n in agentCounts)
        for (final c in chainCounts) (m, n, c),
    for (final c in chainSweep) (modelNames.first, agentCounts.first, c),
  };

  for (final name in modelNames) {
    final (make, kind, ranges) = models[name]!;
    final model = make();
    final encoded = const FitCodec().encodeModel(model);
    File('${out.path}/model_$name.py').writeAsStringSync(
      PythonExport(
        ExportSpec(model: encoded, fit: const ExportNuts(), title: name),
      ).script(),
    );
    final u = kind == 'binary' ? binaryInputs : continuousInputs;
    if (gpu != null) {
      // Compile this model's kernels outside the timings.
      await gpu.fitMap(model, _subjects(model, u, ranges, 4));
    }
    for (final n in {
      for (final c in cells)
        if (c.$1 == name) c.$2,
    }) {
      final subjects = _subjects(model, u, ranges, n);
      File('${out.path}/problem_${name}_$n.json').writeAsStringSync(
        jsonEncode({
          'model': name,
          'model_json': encoded,
          'inputs': u.toList(),
          'responses': [for (final s in subjects) s.responses!.toList()],
          'truth': [for (var i = 0; i < n; i++) _truth(ranges, i, n)],
        }),
      );
      for (final chains in [
        for (final c in cells)
          if (c.$1 == name && c.$2 == n) c.$3,
      ]) {
        for (final runner in ['cpu1', 'pool', 'gpu']) {
          if (!runners.contains(runner)) continue;
          if (runner == 'cpu1' && n > cpu1Max) continue;
          final sw = Stopwatch()..start();
          final fit = switch (runner) {
            'cpu1' => sampleNuts(
              model,
              subjects,
              chains: chains,
              warmup: warmup,
              draws: draws,
            ),
            'pool' => await pool!.sampleNuts(
              model,
              subjects,
              chains: chains,
              warmup: warmup,
              draws: draws,
            ),
            _ => await gpu!.sampleNuts(
              model,
              subjects,
              chains: chains,
              warmup: warmup,
              draws: draws,
            ),
          };
          final ms = sw.elapsedMicroseconds / 1e3;
          final row = _row(fit, ms)
            ..addAll({
              'model': name,
              'agents': n,
              'chains': chains,
              'warmup': warmup,
              'draws': draws,
              'backend': 'hgf_${runner == 'pool' ? 'pool$workers' : runner}',
              if (runner == 'gpu') 'device': gpuName,
            });
          results.writeAsStringSync(
            // R̂ is NaN for one chain: written as null.
            '${jsonEncode(row, toEncodable: (o) => o is double && !o.isFinite ? null : o)}\n',
            mode: FileMode.append,
          );
          print(
            '$name agents=$n chains=$chains ${row['backend']}: '
            '${(ms / 1e3).toStringAsFixed(1)} s, min ESS '
            '${(row['min_ess'] as double).toStringAsFixed(0)} '
            '(${(row['min_ess_per_s'] as double).toStringAsFixed(1)}/s), '
            'max R̂ ${(row['max_rhat'] as double).toStringAsFixed(3)}, '
            '${row['divergences']} divergences',
          );
        }
      }
    }
  }
  await pool?.close();
  gpu?.dispose();
}

List<double> _truth(List<(double, double)> ranges, int i, int n) => [
  for (final (lo, hi) in ranges) lo + (hi - lo) * i / math.max(1, n - 1),
];

List<Subject> _subjects(
  HgfModel model,
  Float64List u,
  List<(double, double)> ranges,
  int n,
) {
  final responses = model.simulate(
    [for (var i = 0; i < n; i++) Subject(inputs: u)],
    [for (var i = 0; i < n; i++) _truth(ranges, i, n)],
    math.Random(123),
  );
  return [
    for (var i = 0; i < n; i++)
      Subject(inputs: u, responses: responses[i], id: i),
  ];
}

Map<String, Object?> _row(NutsFit fit, double ms) {
  final summaries = [for (final r in fit.results) r.summary()];
  final ess = [
    for (final s in summaries)
      for (final p in s) p.essBulk,
  ]..sort();
  final rhat = [
    for (final s in summaries)
      for (final p in s) p.rhat,
  ];
  return {
    'ms': ms,
    'min_ess': ess.first,
    'median_ess': ess[ess.length ~/ 2],
    'min_ess_per_s': ess.first / ms * 1e3,
    'median_ess_per_s': ess[ess.length ~/ 2] / ms * 1e3,
    'max_rhat': rhat.reduce(math.max),
    'divergences': fit.divergences,
    'mean': [
      for (final s in summaries) [for (final p in s) p.mean],
    ],
    'sd': [
      for (final s in summaries) [for (final p in s) p.sd],
    ],
  };
}
