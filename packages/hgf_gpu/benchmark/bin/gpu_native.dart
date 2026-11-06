/// Phase 7 GPU probe on desktop (see `../src/probe.dart`): wgpu-native
/// through `dart:ffi`, against `BatchNetwork` on one isolate. From
/// `packages/hgf_gpu`:
///
///     tool/fetch_wgpu.sh   # once: downloads libwgpu_native into .dart_tool
///     dart compile exe benchmark/bin/gpu_native.dart -o .dart_tool/gpu_native
///     WGPU_LIB=.dart_tool/wgpu/lib/libwgpu_native.so .dart_tool/gpu_native \
///         sizes=1000,64000 scenarios=c2_unbounded
///
/// Arguments are `key=value` pairs, as the web page's URL parameters.
/// Prints one JSON object per line.
library;

import 'dart:convert';
import 'dart:io';

import 'package:hgf_gpu/native.dart';

import '../src/probe.dart';

Future<void> main(List<String> args) async {
  final backend = WgpuBackend.create();
  if (backend == null) {
    stderr.writeln('No wgpu adapter');
    exitCode = 1;
    return;
  }
  final inputs =
      jsonDecode(File('../hgf/benchmark/data/inputs.json').readAsStringSync())
          as Map<String, Object?>;
  final q = {
    for (final a in args)
      if (a.contains('=')) a.split('=')[0]: a.split('=')[1],
  };
  await runProbe(
    backend,
    inputs,
    q,
    (m) => print(jsonEncode(m, toEncodable: (o) => o is double ? '$o' : o)),
  );
}
