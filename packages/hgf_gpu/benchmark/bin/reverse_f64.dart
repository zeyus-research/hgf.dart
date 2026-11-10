// Reverse-mode spike: which f32 gradient is closer to the truth? For the
// elements where the GPU's forward and reverse gradients disagree most (and
// a random sample), the f64 CPU engine's exact gradient (hgf_fit, forward
// tangents in f64) at the same parameters and series. From packages/hgf_gpu:
//
//   dart run benchmark/bin/reverse_f64.dart --batch 65536 --trials 320
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/native.dart';
import 'package:hgf_gpu/src/fit/grad_kernel.dart';
import 'package:hgf_gpu/src/trial.dart';

import '../src/reverse_cases.dart';

Future<void> main(List<String> args) async {
  var batch = 65536, nT = 320, worstN = 20, randomN = 20;
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--batch':
        batch = int.parse(args[++i]);
      case '--trials':
        nT = int.parse(args[++i]);
    }
  }
  final backend = WgpuBackend.create();
  if (backend == null) {
    stderr.writeln('No wgpu adapter');
    exit(1);
  }
  for (final rc in reverseCases) {
    if (rc.name.startsWith('wide')) continue;
    final p = rc.params.length;
    final fk = GradKernel(
      rc.network(),
      rc.response,
      params: rc.params,
      directions: p,
    );
    final rk = GradKernel(
      rc.network(),
      rc.response,
      params: rc.params,
      mode: GradMode.reverseCheckpoint,
      checkpoint: math.sqrt(nT).ceil(),
    );
    final rng = math.Random(11); // as reverse_gpu.dart
    final slots = rc.slots(fk, batch, rng);
    final data = rc.data(nT, rng);
    final resp = rc.responses(nT, batch, rng);
    Future<Float32List> run(GradKernel k) async {
      final job = await GradJob.create(
        backend,
        k,
        batch: batch,
        nT: nT,
        slots: slots,
        data: data,
        responses: resp,
      );
      final out = await job.run();
      job.release();
      return out;
    }

    final f = await run(fk), r = await run(rk);
    double relDiff(int e) {
      var w = 0.0;
      for (var i = 0; i < p; i++) {
        final a = f[(1 + i) * batch + e], b = r[(1 + i) * batch + e];
        w = math.max(w, (a - b).abs() / math.max(1.0, a.abs()));
      }
      return w;
    }

    final finite = [
      for (var e = 0; e < batch; e++)
        if (f[e] < 3.3e38) e,
    ];
    final byDiff = [...finite]
      ..sort((a, b) => relDiff(b).compareTo(relDiff(a)));
    final pick = <int>{
      ...byDiff.take(worstN),
      for (var i = 0; i < randomN; i++) finite[rng.nextInt(finite.length)],
    };

    final model = _model(rc);
    final subject = Subject(
      inputs: Float64List.fromList(data),
      responses: resp == null
          ? null
          : Float64List.fromList([
              for (var t = 0; t < nT; t++) resp[t * batch], // replaced below
            ]),
    );
    print(
      '== ${rc.name} (B = $batch, $nT trials): ${pick.length} elements, '
      'error relative to max(1, |f64|), worst parameter per element',
    );
    print(
      '   ${'element'.padLeft(7)} ${'fwd−rev'.padLeft(9)} '
      '${'fwd−f64'.padLeft(9)} ${'rev−f64'.padLeft(9)}  largest |∂| (f64)',
    );
    var sumF = 0.0, sumR = 0.0, fWins = 0, rWins = 0;
    for (final e in pick) {
      final s = resp == null
          ? subject
          : Subject(
              inputs: subject.inputs,
              responses: Float64List.fromList([
                for (var t = 0; t < nT; t++) resp[t * batch + e],
              ]),
            );
      final y = Float64List.fromList([
        for (final v in rc.params) slots[fk.slotOfVar[v]! * batch + e],
      ]);
      final out = Float64List(1), g = Float64List(p);
      model.negLogPosterior(
        [s],
        Int32List(1),
        y,
        out,
        gradient: g,
        includePrior: List.filled(p, false),
      );
      var ef = 0.0, er = 0.0, big = 0.0;
      for (var i = 0; i < p; i++) {
        final scale = math.max(1.0, g[i].abs());
        ef = math.max(ef, (f[(1 + i) * batch + e] - g[i]).abs() / scale);
        er = math.max(er, (r[(1 + i) * batch + e] - g[i]).abs() / scale);
        big = math.max(big, g[i].abs());
      }
      sumF += ef;
      sumR += er;
      if (ef < er) fWins++;
      if (er < ef) rWins++;
      String x(double v) => v.toStringAsExponential(1).padLeft(9);
      print(
        '   ${'$e'.padLeft(7)} ${x(relDiff(e))} ${x(ef)} ${x(er)}  '
        '${big.toStringAsExponential(1)}',
      );
    }
    print(
      '   mean error: forward ${(sumF / pick.length).toStringAsExponential(1)}, '
      'reverse ${(sumR / pick.length).toStringAsExponential(1)}; '
      'closer to f64: forward $fWins, reverse $rWins\n',
    );
  }
  exit(0);
}

/// The case's likelihood as an [HgfModel] with one identity-transformed
/// parameter per slot variable (priors excluded when evaluating).
HgfModel _model(ReverseCase rc) {
  const prior = Normal(0, 1);
  Parameter param(String v) {
    final node = RegExp(r'\d+$');
    if (v.startsWith('rp')) {
      return Parameter.response('inverseTemperature', prior: prior);
    }
    if (v.startsWith('k')) {
      final [parent, child] = v.substring(1).split('_').map(int.parse).toList();
      return Parameter.volatilityCoupling(v, parent, child, prior: prior);
    }
    final i = int.parse(node.stringMatch(v)!);
    final field = switch (v.substring(0, v.length - '$i'.length)) {
      'tv' => NodeField.tonicVolatility,
      'td' => NodeField.tonicDrift,
      'ac' => NodeField.autoconnectionStrength,
      'm' => NodeField.mean,
      'p' => NodeField.precision,
      final f => throw ArgumentError('No mapping for $f'),
    };
    return Parameter.node(v, i, field, prior: prior);
  }

  return HgfModel(
    network: rc.network(),
    response: switch (rc.response) {
      GpuBinarySoftmax() => const BinarySoftmaxReadout(),
      _ => const ObservationSurprise(TotalGaussianSurprise()),
    },
    parameters: [for (final v in rc.params) param(v)],
  );
}
