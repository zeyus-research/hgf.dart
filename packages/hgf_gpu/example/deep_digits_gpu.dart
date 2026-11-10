// Handwritten digits (UCI optical digits, see
// ../hgf_deep/example/data/README.md) learned by a deep HGF network on the
// GPU through wgpu-native, in the two ways hgf_gpu runs DeepNetwork:
//
//  1. Minibatch learning (GpuDeepNetwork.trainEpoch, f32), next to the same
//     minibatch steps on the CPU (DeepNetwork.batchUpdate, f64): accuracy and
//     time per epoch.
//  2. The sequential, one-sample-at-a-time fit for an ensemble of networks
//     that differ only in their initial weights (GpuDeepEnsemble.fit, one
//     workgroup per network): the spread of test accuracy over seeds.
//
// Needs the wgpu-native library (Homebrew's `wgpu-native`, or WGPU_LIB);
// without it the example says so and stops.
//
//   dart run example/deep_digits_gpu.dart [epochs] [ensemble size]
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_gpu/hgf_gpu.dart';
import 'package:hgf_gpu/native.dart';

Dataset _digits() {
  final file = File('../hgf_deep/example/data/digits.csv.gz');
  if (!file.existsSync()) {
    throw StateError('Run from packages/hgf_gpu (${file.path} not found).');
  }
  final lines = const LineSplitter()
      .convert(String.fromCharCodes(gzip.decode(file.readAsBytesSync())))
      .where((l) => l.trim().isNotEmpty)
      .toList();
  final x = Matrix(lines.length, 64);
  final labels = <int>[];
  for (var i = 0; i < lines.length; i++) {
    final cells = lines[i].split(',');
    for (var j = 0; j < 64; j++) {
      x.data[i * 64 + j] = double.parse(cells[j]);
    }
    labels.add(int.parse(cells[64]));
  }
  return Dataset(
    'digits',
    x,
    encodeLabels(labels, 10),
    labels: labels,
    classes: 10,
  );
}

DeepNetwork _net(int seed) => DeepNetwork()
  ..addLayer(10, kind: LayerKind.categorical)
  ..addLayer(32, coupling: CouplingFn.tanh, volatilityParent: false)
  ..addLayer(64)
  ..weightInitialisation(const XavierInit(), seed: seed);

String _f(double v) => v.toStringAsFixed(3);

Future<void> main(List<String> args) async {
  final epochs = args.isEmpty ? 15 : int.parse(args[0]);
  final members = args.length > 1 ? int.parse(args[1]) : 64;

  WgpuBackend? created;
  try {
    created = WgpuBackend.create();
  } on ArgumentError catch (e) {
    print('wgpu-native is not available ($e).');
    print('Install it (e.g. `brew install wgpu-native`) or set WGPU_LIB.');
    return;
  }
  if (created == null) {
    print('wgpu-native found no GPU adapter.');
    return;
  }
  final backend = created;

  final (:train, :test) = split(_digits(), testFraction: 0.25, seed: 1);
  final scale = Standardiser.fit(train.x);
  final tr = train.withX(scale.apply(train.x));
  final te = test.withX(scale.apply(test.x));
  print(
    'digits: ${tr.length} train / ${te.length} test, '
    '64 → 32 tanh → 10 categorical\n',
  );

  // 1. Minibatches: GPU (f32) and CPU (f64), same initial weights, same
  // batches in the same order.
  const batch = 16;
  final cpu = _net(1);
  final gpuNet = await GpuDeepNetwork.create(backend, _net(1), capacity: 256);
  final cpuAdam = Adam(0.005), gpuAdam = Adam(0.005);
  print('minibatch learning, batches of $batch, Adam 0.005');
  print(
    '${'epoch'.padLeft(5)}  ${'GPU acc'.padLeft(8)}  ${'GPU ms'.padLeft(7)}'
    '  ${'CPU acc'.padLeft(8)}  ${'CPU ms'.padLeft(7)}',
  );
  for (var e = 1; e <= epochs; e++) {
    final b = shuffled(tr, seed: e);
    final g = Stopwatch()..start();
    await gpuNet.trainEpoch(b.x, b.y, batchSize: batch, optimiser: gpuAdam);
    final gpuPred = await gpuNet.predict(te.x);
    g.stop();
    final c = Stopwatch()..start();
    for (var s = 0; s < b.length; s += batch) {
      final rows = [for (var i = s; i < math.min(s + batch, b.length); i++) i];
      final sub = b.subset(rows);
      cpu.batchUpdate(sub.x, sub.y, optimiser: cpuAdam);
    }
    final cpuPred = cpu.predict(te.x);
    c.stop();
    print(
      '${'$e'.padLeft(5)}  ${_f(accuracy(gpuPred, te.labels!)).padLeft(8)}'
      '  ${'${g.elapsedMilliseconds}'.padLeft(7)}'
      '  ${_f(accuracy(cpuPred, te.labels!)).padLeft(8)}'
      '  ${'${c.elapsedMilliseconds}'.padLeft(7)}',
    );
  }
  gpuNet.dispose();

  // 2. Sequential fit of an ensemble: one network per seed.
  print(
    '\nsequential fit, $members networks (seeds 1..$members) at once, '
    'Adam 0.005',
  );
  final nets = [for (var s = 1; s <= members; s++) _net(s)];
  final ensemble = await GpuDeepEnsemble.create(backend, nets);
  final sw = Stopwatch()..start();
  for (var e = 1; e <= epochs; e++) {
    final b = shuffled(tr, seed: e);
    // A fresh optimiser per call: its moments live on the GPU between the
    // samples of one call.
    await ensemble.fit(
      [b.x],
      [b.y],
      optimiser: Adam(0.005),
      readPredictions: false,
    );
  }
  // fit without predictions only queues the work: the download waits for
  // it.
  await ensemble.downloadAll(nets);
  sw.stop();
  final accs = [for (final n in nets) accuracy(n.predict(te.x), te.labels!)]
    ..sort();
  final mean = accs.reduce((a, b) => a + b) / accs.length;
  print(
    '$epochs epochs in ${sw.elapsedMilliseconds} ms '
    '(${(sw.elapsedMilliseconds / members).toStringAsFixed(1)} ms per network)',
  );
  print(
    'test accuracy over seeds: mean ${_f(mean)}, worst ${_f(accs.first)}, '
    'median ${_f(accs[accs.length ~/ 2])}, best ${_f(accs.last)}',
  );
}
