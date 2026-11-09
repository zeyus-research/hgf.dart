/// Per-phase cost of one sequential fit step (width 1) and of a batched
/// step, on the benchmark's medium network.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_deep/hgf_deep.dart';
import 'package:hgf_deep/src/kernels.dart';

void main(List<String> args) {
  final sizes = args.isEmpty
      ? [10, 64, 64, 16]
      : [for (final s in args.first.split(',')) int.parse(s)];
  final rng = math.Random(1);
  final net = DeepNetwork(coupling: CouplingFn.tanh);
  for (final s in sizes) {
    net.addLayer(s);
  }
  net.weightInitialisation(const XavierInit(), seed: 1);
  final layers = net.debugLayers;
  final x = Matrix(
    1,
    sizes.last,
    Float64List.fromList([
      for (var i = 0; i < sizes.last; i++) rng.nextDouble(),
    ]),
  );
  final y = Matrix(
    1,
    sizes.first,
    Float64List.fromList([
      for (var i = 0; i < sizes.first; i++) rng.nextDouble(),
    ]),
  );
  final states = [for (var l = 0; l < sizes.length; l++) net.state(l).copy()];
  final ws = Workspace(1);
  final grads = [
    for (final l in layers)
      l.weights == null ? null : Float64List(l.weights!.data.length),
  ];
  final weights = [for (final l in layers) l.weights];
  final adam = Adam(1e-3);
  double time(String name, void Function() f) {
    for (var i = 0; i < 200; i++) {
      f();
    }
    const n = 2000;
    final sw = Stopwatch()..start();
    for (var i = 0; i < n; i++) {
      f();
    }
    final us = sw.elapsedMicroseconds / n;
    stdout.writeln('${name.padRight(22)} ${us.toStringAsFixed(1)} µs');
    return us;
  }

  final s = net.settings;
  time(
    'predictionSweep',
    () => predictionSweep(layers, states, x, 0, ws, 1.0, s),
  );
  time('updateSweep', () => updateSweep(layers, states, y, 0, ws, 1.0, s));
  time(
    'accumulateGradients',
    () => accumulateGradients(
      layers,
      states,
      ws,
      LearningKind.precisionWeighted,
      grads,
    ),
  );
  time('adam step', () => adam.step(weights, grads));
  for (final l in states) {
    l.precision.fillRange(0, l.precision.length, 1.0);
  }
  time(
    'predictionError (vol)',
    () => predictionError(states[1], layers[1], 1.0, s),
  );
  time(
    'posteriorUpdate',
    () => posteriorUpdate(states[1], layers[1], states[0], layers[0], ws, s),
  );
  time('tanh x128', () {
    var a = 0.0;
    for (var i = 0; i < 128; i++) {
      a += CouplingFn.tanh.f(i / 64 - 1);
    }
    if (a == 42) print(a);
  });
  final xb = Matrix(2000, sizes.last, Float64List(2000 * sizes.last));
  final sw = Stopwatch()..start();
  for (var i = 0; i < 10; i++) {
    net.predict(xb);
  }
  stdout.writeln(
    'predict 2000           ${(sw.elapsedMicroseconds / 10).toStringAsFixed(0)} µs',
  );
}
