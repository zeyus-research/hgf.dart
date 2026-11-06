/// Writes the WGSL of the benchmark scenarios to DIR (for `naga` validation
/// and inspection): `dart run tool/dump_kernels.dart DIR`.
library;

import 'dart:convert';
import 'dart:io';

import 'package:hgf/hgf.dart';
import 'package:hgf_gpu/hgf_gpu.dart';

void main(List<String> args) {
  final dir = Directory(args.isEmpty ? '.dart_tool/wgsl' : args.first)
    ..createSync(recursive: true);
  final inputs =
      jsonDecode(File('../hgf/benchmark/data/inputs.json').readAsStringSync())
          as Map<String, Object?>;
  final specs = inputs['specs'] as Map<String, Object?>;
  for (final name in specs.keys) {
    for (final mode in ['standard', 'eHGF', 'unbounded']) {
      for (final mf in [false, true]) {
        final s = jsonDecode(jsonEncode(specs[name])) as Map<String, Object?>;
        (s['network'] as Map<String, Object?>)
          ..['volatility_updates'] = mode
          ..['mean_field_updates'] = mf;
        final net = NetworkSpec.fromJson(s).build();
        final k = HgfKernel(
          net,
          response: name.startsWith('binary')
              ? GpuResponse.firstLevelBinarySurprise
              : GpuResponse.totalGaussianSurprise,
        );
        final f = File('${dir.path}/${name}_$mode${mf ? '_mf' : ''}.wgsl')
          ..writeAsStringSync(k.source);
        print(
          '${f.path}: ${k.nSlots} slots, ${k.loaded} loaded, ${k.stored} stored',
        );
      }
    }
  }
}
