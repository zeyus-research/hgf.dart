import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/hgf_gpu.dart';

import '../../../hgf_fit/test/src/fit_cases.dart';

export '../../../hgf_fit/test/src/fit_cases.dart';

bool _none(String _) => false;

/// Checks every fit case (but those [skip] names) on [fitter] against the
/// CPU; returns the failures.
Future<List<String>> checkFitCases(
  GpuFitter fitter, {
  double valueTol = 2e-5,
  double gradientTol = 2e-3,
  void Function(String)? log,
  bool Function(String name) skip = _none,
}) async {
  final failures = <String>[];
  for (final c in fitCases()) {
    if (skip(c.name)) {
      log?.call('${c.name}: skipped');
      continue;
    }
    final model = c.model, k = model.dimension;
    final n = c.subjects.length * c.points.length;
    final subjectOf = Int32List(n);
    final y = Float64List(n * k);
    var e = 0;
    for (final x in c.points) {
      final yy = model.toUnconstrained(x);
      for (var s = 0; s < c.subjects.length; s++) {
        subjectOf[e] = s;
        y.setRange(e * k, (e + 1) * k, yy);
        e++;
      }
    }
    var maxV = 0.0, maxG = 0.0;
    for (final gradient in [false, true]) {
      for (final jacobian in [false, true]) {
        Pass make() => Pass(
          model,
          c.subjects,
          subjectOf,
          y,
          jacobian: jacobian,
          negLogLikelihood: true,
          gradient: gradient,
        );
        final cpu = make()..run();
        final gpu = make();
        await fitter.evaluatePass(gpu);
        for (var j = 0; j < n; j++) {
          final a = gpu.out[j], b = cpu.out[j];
          final errV = (a - b).abs() / (1 + b.abs());
          if (errV > maxV) maxV = errV;
          if (!(errV <= valueTol)) {
            failures.add('${c.name}: element $j value $a vs $b');
          }
          if (!gradient) continue;
          // Gradient errors relative to the gradient's scale.
          var scale = 1.0;
          for (var i = 0; i < k; i++) {
            scale = math.max(scale, cpu.gradient![j * k + i].abs());
          }
          for (var i = 0; i < k; i++) {
            final ga = gpu.gradient![j * k + i], gb = cpu.gradient![j * k + i];
            final errG = (ga - gb).abs() / scale;
            if (errG > maxG) maxG = errG;
            if (!(errG <= gradientTol)) {
              failures.add(
                '${c.name}: element $j ${model.names[i]} gradient $ga vs $gb',
              );
            }
          }
        }
      }
    }
    log?.call(
      '${c.name}: max value err ${maxV.toStringAsExponential(2)}, '
      'max gradient err ${maxG.toStringAsExponential(2)}',
    );
  }
  return failures;
}

/// Draws from three normal clusters (centres −6, 0.5, 7) in random order.
Float64List dpMixture(int nTime, int seed) {
  final rng = math.Random(seed);
  const centres = [-6.0, 0.5, 7.0];
  return Float64List.fromList([
    for (var t = 0; t < nTime; t++)
      centres[rng.nextInt(3)] +
          math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
              math.cos(2 * math.pi * rng.nextDouble()),
  ]);
}

/// A dp-state model (six clusters) fitted to its observations' surprise
/// (`DpSurprise`): α and the sensory precision.
HgfModel dpModel() => HgfModel(
  network: Network()
    ..addDpState(batchSize: 6, alpha: 1.0)
    ..addEfState(nNodes: 6, valueChildren: [0], nus: 15.0),
  response: const ObservationSurprise(DpSurprise()),
  parameters: [
    Parameter.node('alpha', 0, NodeField.alpha, prior: const LogNormal(-2, 2)),
    Parameter.node(
      'sensoryPrecision',
      0,
      NodeField.sensoryPrecision,
      prior: const LogNormal(0, 1),
    ),
  ],
);

/// Two subjects (one with missing observations and time steps) of [nTime]
/// trials for [dpModel].
List<Subject> dpSubjects(int nTime) => [
  Subject(inputs: dpMixture(nTime, 0)),
  Subject(
    inputs: dpMixture(nTime, 1),
    observed: Float64List.fromList([
      for (var t = 0; t < nTime; t++) t % 7 == 3 ? 0.0 : 1.0,
    ]),
    timeSteps: Float64List.fromList([
      for (var t = 0; t < nTime; t++) 1.0 + (t % 3) * 0.5,
    ]),
  ),
];

/// [dpModel]'s likelihood passes on [fitter] against the CPU, at α from
/// 0.01 to 10 and two sensory precisions; returns the largest error
/// relative to `1 + |value|`. The cluster assignments are discrete, so a
/// trial whose f32 weights order differently would show as a jump.
Future<double> checkDpPasses(GpuFitter fitter, {int nTime = 120}) async {
  final model = dpModel();
  final subjects = dpSubjects(nTime);
  final points = [
    for (final alpha in [0.01, 0.3, 2.0, 10.0])
      for (final sp in [1.0, 2.5]) [alpha, sp],
  ];
  final n = subjects.length * points.length, k = model.dimension;
  final subjectOf = Int32List(n);
  final y = Float64List(n * k);
  var e = 0;
  for (final x in points) {
    final yy = model.toUnconstrained(x);
    for (var s = 0; s < subjects.length; s++) {
      subjectOf[e] = s;
      y.setRange(e * k, (e + 1) * k, yy);
      e++;
    }
  }
  Pass make() => Pass(
    model,
    subjects,
    subjectOf,
    y,
    jacobian: true,
    negLogLikelihood: true,
  );
  final cpu = make()..run();
  final gpu = make();
  await fitter.evaluatePass(gpu);
  var worst = 0.0;
  for (var j = 0; j < n; j++) {
    final b = cpu.out[j];
    worst = math.max(worst, (gpu.out[j] - b).abs() / (1 + b.abs()));
  }
  return worst;
}
