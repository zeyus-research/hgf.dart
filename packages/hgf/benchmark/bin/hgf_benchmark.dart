/// hgf.dart benchmark (Dart VM / AOT).
///
///     dart run benchmark/bin/hgf_benchmark.dart --repeat 7 --out bench.json
///     dart compile exe benchmark/bin/hgf_benchmark.dart -o hgf_bench && ./hgf_bench
///
/// Writes `customSmallerIsBetter` JSON (github-action-benchmark), like
/// liblsl.dart's benchmark.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import '../src/scenarios.dart';

Future<void> main(List<String> args) async {
  var repeat = 7;
  String? outPath;
  String? runtime;
  var inputsPath = 'benchmark/data/inputs.json';
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--repeat':
        repeat = int.parse(args[++i]);
      case '--out':
        outPath = args[++i];
      case '--runtime':
        runtime = args[++i];
      case '--inputs':
        inputsPath = args[++i];
      default:
        stderr.writeln('Unknown argument ${args[i]}');
        exit(64);
    }
  }
  // `dart compile exe` binaries report themselves as AOT.
  runtime ??= const bool.fromEnvironment('dart.vm.product')
      ? 'dart_aot'
      : 'dart_jit';

  final inputs =
      jsonDecode(File(inputsPath).readAsStringSync()) as Map<String, Object?>;

  final results = <BenchResult>[
    ...seriesScenarios(inputs, runtime: runtime, repeat: repeat),
    ...batchScenario(inputs, runtime: runtime, repeat: repeat),
    ...await _isolateBatch(inputs, runtime: runtime, repeat: repeat),
    ...await _isolateBatch(
      inputs,
      runtime: runtime,
      repeat: repeat,
      lockStep: true,
    ),
  ];

  for (final r in results) {
    stdout.writeln(
      '${r['name']}: ${(r['value'] as double).toStringAsFixed(3)} ms',
    );
  }
  if (outPath != null) {
    File(
      outPath,
    ).writeAsStringSync(const JsonEncoder.withIndent(' ').convert(results));
  }
}

/// The batch split across one isolate per core, each looping a `Network`
/// over its share of the ω (or, with [lockStep], running it as one
/// `BatchNetwork`).
Future<List<BenchResult>> _isolateBatch(
  Map<String, Object?> inputs, {
  required String runtime,
  required int repeat,
  bool lockStep = false,
}) async {
  final workers = Platform.numberOfProcessors;
  final spec =
      (inputs['specs'] as Map<String, Object?>)['continuous_2level']
          as Map<String, Object?>;
  final data = Float64List.sublistView(
    floats(inputs['continuous']),
    0,
    batchTrials,
  );
  final omegas = floats(inputs['omegas']);
  final chunk = (omegas.length + workers - 1) ~/ workers;

  Future<void> once() async {
    await Future.wait([
      for (var w = 0; w < workers; w++)
        if (w * chunk < omegas.length)
          Isolate.run(() {
            final end = (w + 1) * chunk;
            final mine = Float64List.sublistView(
              omegas,
              w * chunk,
              end < omegas.length ? end : omegas.length,
            );
            return lockStep
                ? batchNetworkRun(spec, data, mine)
                : batchRun(spec, data, mine);
          }),
    ]);
  }

  await once(); // warm-up
  final times = <double>[];
  for (var i = 0; i < repeat; i++) {
    final sw = Stopwatch()..start();
    await once();
    times.add(sw.elapsedMicroseconds / 1000.0);
  }
  times.sort();
  return [
    result(
      'batch_1000x1000_continuous_2level/'
      '${runtime}_${lockStep ? 'batch_isolates' : 'isolates'}',
      times[times.length ~/ 2],
      extra: '$workers isolates',
    ),
  ];
}
