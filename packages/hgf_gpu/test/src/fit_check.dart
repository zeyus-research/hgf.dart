import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/hgf_gpu.dart';

import '../../../hgf_fit/test/src/fit_cases.dart';

export '../../../hgf_fit/test/src/fit_cases.dart';

Future<List<String>> checkFitCases(
  GpuFitter fitter, {
  double valueTol = 2e-5,
  double gradientTol = 2e-3,
  void Function(String)? log,
}) async {
  final failures = <String>[];
  for (final c in fitCases()) {
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
