// hgf_fit's drivers with their passes on the GPU (GpuFitter, wgpu-native)
// against one isolate and a FitWorkers pool, on pyhgf's parameter-recovery
// problem (binary 2-level HGF, sigmoid decisions, ω₂ and β per agent, 320
// trials). From packages/hgf_gpu:
//
//   dart compile exe benchmark/bin/fit_gpu.dart -o .dart_tool/fit_gpu
//   .dart_tool/fit_gpu --agents 20,200,2000 --workers 8 --draws 300
//
// `--runners cpu1,pool,gpu,gpuiso` picks the runners; gpuiso is
// WgpuIsolateFitter (the GPU from a background isolate).
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';

Future<void> main(List<String> args) async {
  var agentCounts = [20, 200, 2000];
  var workers = Platform.numberOfProcessors;
  var draws = 300, chains = 4;
  var only = <String>{};
  var runners = {'cpu1', 'pool', 'gpu'};
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--agents':
        agentCounts = [for (final s in args[++i].split(',')) int.parse(s)];
      case '--workers':
        workers = int.parse(args[++i]);
      case '--draws':
        draws = int.parse(args[++i]);
      case '--chains':
        chains = int.parse(args[++i]);
      case '--only':
        only = args[++i].split(',').toSet();
      case '--runners':
        runners = args[++i].split(',').toSet();
    }
  }
  bool want(String s) => only.isEmpty || only.contains(s);

  final backend = WgpuBackend.create();
  if (backend == null) {
    stderr.writeln('No wgpu adapter');
    exitCode = 1;
    return;
  }
  print('GPU: ${backend.description}');
  final u = Float64List.fromList([
    for (final l in File('../hgf_fit/data/binary_input.txt').readAsLinesSync())
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

  final pool = await FitWorkers.spawn(count: workers);
  final gpu = GpuFitter(backend, cpu: pool);
  // The same on a background isolate (what a UI uses).
  final iso = runners.contains('gpuiso')
      ? (await WgpuIsolateFitter.spawn()
          ..cpu = pool)
      : null;
  // Compile the kernels and warm up the JIT-free paths once.
  await gpu.fitMap(model, subjectsFor(4));

  Future<void> time(
    String job,
    int agents,
    String runner,
    Future<Object?> Function() f,
  ) async {
    final p0 = gpu.gpuPasses;
    for (final s in [gpu.prepareTime, gpu.waitTime, gpu.finishTime]) {
      s.reset();
    }
    final sw = Stopwatch()..start();
    await f();
    final ms = sw.elapsedMicroseconds / 1e3;
    String t(Stopwatch s) => (s.elapsedMicroseconds / 1e3).toStringAsFixed(0);
    final extra = runner == 'gpu'
        ? ', ${gpu.gpuPasses - p0} GPU passes (prepare ${t(gpu.prepareTime)}'
              ', wait ${t(gpu.waitTime)}, finish ${t(gpu.finishTime)} ms)'
        : '';
    print(
      '$job agents=$agents ${runner.padRight(5)} '
      '${ms.toStringAsFixed(1).padLeft(9)} ms$extra',
    );
  }

  for (final n in agentCounts) {
    final subjects = subjectsFor(n);
    if (want('map')) {
      if (runners.contains('cpu1')) {
        await time('map ', n, 'cpu1', () async => fitMap(model, subjects));
      }
      if (runners.contains('pool')) {
        await time('map ', n, 'pool', () => pool.fitMap(model, subjects));
      }
      if (runners.contains('gpu')) {
        await time('map ', n, 'gpu', () => gpu.fitMap(model, subjects));
      }
      if (iso != null) {
        await time('map ', n, 'gpuiso', () => iso.fitMap(model, subjects));
      }
    }
    if (want('nuts')) {
      Future<NutsFit> on(PassRunner? r) async => r == null
          ? sampleNuts(
              model,
              subjects,
              chains: chains,
              warmup: draws,
              draws: draws,
            )
          : r.sampleNuts(
              model,
              subjects,
              chains: chains,
              warmup: draws,
              draws: draws,
            );
      if (runners.contains('cpu1') && n <= 200) {
        await time('nuts', n, 'cpu1', () => on(null));
      }
      if (runners.contains('pool')) {
        await time('nuts', n, 'pool', () => on(pool));
      }
      if (runners.contains('gpu')) {
        await time('nuts', n, 'gpu', () => on(gpu));
      }
      if (iso != null) {
        await time('nuts', n, 'gpuiso', () => on(iso));
      }
    }
  }
  await iso?.close();
  await pool.close();
  gpu.dispose();
}
