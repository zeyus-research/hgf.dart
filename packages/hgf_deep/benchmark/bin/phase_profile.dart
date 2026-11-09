/// Per-phase cost of one sequential fit step (width 1, the default) or of a
/// batched step (`<sizes> <width>`), on the benchmark's medium network
/// unless sizes are given: `phase_profile 16,256,256,32 256`.
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
  final width = args.length > 1 ? int.parse(args[1]) : 1;
  final rng = math.Random(1);
  final net = DeepNetwork(coupling: CouplingFn.tanh);
  for (final s in sizes) {
    net.addLayer(s);
  }
  net.weightInitialisation(const XavierInit(), seed: 1);
  final layers = net.debugLayers;
  final x = Matrix(
    width,
    sizes.last,
    Float64List.fromList([
      for (var i = 0; i < width * sizes.last; i++) rng.nextDouble(),
    ]),
  );
  final y = Matrix(
    width,
    sizes.first,
    Float64List.fromList([
      for (var i = 0; i < width * sizes.first; i++) rng.nextDouble(),
    ]),
  );
  final states = [
    for (var l = 0; l < sizes.length; l++) net.state(l).copy().tile(width),
  ];
  final ws = Workspace(width);
  final grads = [
    for (final l in layers)
      l.weights == null ? null : Float64List(l.weights!.data.length),
  ];
  final weights = [for (final l in layers) l.weights];
  final adam = Adam(1e-3);
  double time(String name, void Function() f) {
    final n = math.max(20, 2000 ~/ width);
    for (var i = 0; i < n ~/ 10; i++) {
      f();
    }
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
