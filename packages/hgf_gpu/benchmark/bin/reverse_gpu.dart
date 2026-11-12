// Reverse-mode spike: the cost of a gradient in forward mode (k tangent
// directions) against reverse mode (full tape, √T checkpoints), and the
// primal alone, on wgpu-native. From packages/hgf_gpu:
//
//   dart compile exe benchmark/bin/reverse_gpu.dart -o .dart_tool/reverse_gpu
//   .dart_tool/reverse_gpu --batch 2000,16384,65536 --trials 320
//
// `--cases binary2-softmax,...` picks models; `--reps 15`.
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_gpu/native.dart';
import 'package:hgf_gpu/src/fit/grad_kernel.dart';

import '../src/reverse_cases.dart';

Future<void> main(List<String> args) async {
  var batches = [2000, 16384, 65536];
  var nT = 320, reps = 15;
  Set<String>? only;
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--batch':
        batches = [for (final s in args[++i].split(',')) int.parse(s)];
      case '--trials':
        nT = int.parse(args[++i]);
      case '--reps':
        reps = int.parse(args[++i]);
      case '--cases':
        only = args[++i].split(',').toSet();
    }
  }
  final backend = WgpuBackend.create();
  if (backend == null) {
    stderr.writeln('No wgpu adapter');
    exitCode = 1;
    return;
  }
  final maxBytes = backend.maxStorageBufferBytes;
  print('GPU: ${backend.description}');
  print('max storage binding: ${(maxBytes / 1048576).toStringAsFixed(0)} MiB');
  print('trials: $nT, best/median of $reps\n');
  final checkpoint = math.sqrt(nT).ceil();

  for (final rc in reverseCases) {
    if (only != null && !only.contains(rc.name)) continue;
    final p = rc.params.length;
    final kernels = <String, GradKernel>{};
    final compile = Stopwatch()..start();
    kernels['primal'] = GradKernel(
      rc.network(),
      rc.response,
      params: rc.params,
      mode: GradMode.primal,
    );
    for (final k in {2, 5, 10, 20, p}) {
      kernels['fwd k=$k'] = GradKernel(
        rc.network(),
        rc.response,
        params: rc.params,
        directions: k,
      );
    }
    kernels['rev tape'] = GradKernel(
      rc.network(),
      rc.response,
      params: rc.params,
      mode: GradMode.reverseTape,
    );
    kernels['rev ckpt $checkpoint'] = GradKernel(
      rc.network(),
      rc.response,
      params: rc.params,
      mode: GradMode.reverseCheckpoint,
      checkpoint: checkpoint,
    );
    kernels['rev ckpt, no 0-guard'] = GradKernel(
      rc.network(),
      rc.response,
      params: rc.params,
      mode: GradMode.reverseCheckpoint,
      checkpoint: checkpoint,
      zeroGuard: false,
    );
    print(
      '== ${rc.name}: $p parameters, '
      '${kernels['rev tape']!.tapeCarried} carried slots '
      '(codegen ${compile.elapsedMilliseconds} ms)',
    );
    for (final MapEntry(:key, :value) in kernels.entries) {
      print('   ${key.padRight(14)} WGSL ${value.source.length ~/ 1024} KiB');
    }

    for (final batch in batches) {
      final rng = math.Random(11);
      final slots = rc.slots(kernels['primal']!, batch, rng);
      final data = rc.data(nT, rng);
      final resp = rc.responses(nT, batch, rng);
      final results = <String, Float32List>{};
      print('-- B = $batch');
      double? primalMs;
      // A forward run this slow means more directions would run long enough
      // for the driver to reset the GPU (k = 33 at B = 65536 did).
      var forwardTooSlow = false;
      for (final MapEntry(:key, :value) in kernels.entries) {
        if (forwardTooSlow && value.mode == GradMode.forward) {
          print('   ${key.padRight(14)} skipped: a smaller k took > 300 ms');
          continue;
        }
        final tape = value.tapeBytes(batch, nT);
        if (tape > maxBytes) {
          print(
            '   ${key.padRight(14)} skipped: tape '
            '${(tape / 1048576).toStringAsFixed(0)} MiB > binding limit',
          );
          continue;
        }
        final sw = Stopwatch()..start();
        final job = await GradJob.create(
          backend,
          value,
          batch: batch,
          nT: nT,
          slots: slots,
          data: data,
          responses: resp,
        );
        final pipelineMs = sw.elapsedMilliseconds;
        results[key] = await job.run();
        for (var i = 0; i < 3; i++) {
          job.dispatch();
          await backend.idle();
        }
        final times = <double>[];
        for (var i = 0; i < reps; i++) {
          final t = Stopwatch()..start();
          job.dispatch();
          await backend.idle();
          times.add(t.elapsedMicroseconds / 1e3);
        }
        job.release();
        times.sort();
        final best = times.first, median = times[times.length ~/ 2];
        primalMs ??= best;
        if (value.mode == GradMode.forward && best > 300) forwardTooSlow = true;
        final tapeNote = tape == 0
            ? ''
            : ', tape ${(tape / 1048576).toStringAsFixed(1)} MiB';
        print(
          '   ${key.padRight(14)} ${best.toStringAsFixed(2).padLeft(8)} ms '
          '(median ${median.toStringAsFixed(2)}) '
          '${(best / primalMs).toStringAsFixed(2).padLeft(5)}× primal'
          '$tapeNote (pipeline $pipelineMs ms)',
        );
      }
      _agreement(results, 'fwd k=$p', p, batch);
    }
    print('');
  }
  exit(0);
}

/// Reverse against forward with one direction per parameter.
void _agreement(Map<String, Float32List> r, String fwd, int p, int batch) {
  final f = r[fwd];
  if (f == null) return;
  for (final MapEntry(:key, :value) in r.entries) {
    if (!key.startsWith('rev')) continue;
    var worst = 0.0, sumAbs = 0.0, n = 0, bad = 0, zeros = 0;
    for (var e = 0; e < batch; e++) {
      if (f[e] >= 3.3e38) {
        bad++;
        continue;
      }
      for (var i = 0; i < p; i++) {
        final a = f[(1 + i) * batch + e], b = value[(1 + i) * batch + e];
        worst = math.max(worst, (a - b).abs() / math.max(1.0, a.abs()));
        sumAbs += a.abs();
        if (a == 0) zeros++;
        n++;
      }
    }
    print(
      '   $key vs $fwd: max rel diff ${worst.toStringAsExponential(1)}, '
      'mean |∂| ${(sumAbs / n).toStringAsFixed(2)}, $zeros/$n exact zeros, '
      '$bad non-finite elements',
    );
  }
}
