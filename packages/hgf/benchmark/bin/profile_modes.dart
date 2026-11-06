/// Per-trial cost by volatility update mode (diagnostic, not part of the gate).
///
/// Runs unchanged on every runtime, from `packages/hgf`:
///
///     dart compile exe benchmark/bin/profile_modes.dart -o /tmp/pm && /tmp/pm
///     dart compile wasm -O4 benchmark/bin/profile_modes.dart -o /tmp/pm.wasm
///     node ../../tool/web/run_wasm.mjs /tmp/pm.wasm
///
/// For a CPU profile of the Wasm build (dart2wasm keeps a name section):
///
///     node --cpu-prof --cpu-prof-dir=/tmp/prof \
///       ../../tool/web/run_wasm.mjs /tmp/pm.wasm unbounded
///
/// An optional first argument picks one mode (`standard`, `eHGF`,
/// `unbounded`, `floor`, `response`, `record`, `ef`, `categorical`) and
/// runs it longer, which
/// keeps the profile focused.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import '../src/read_text.dart';
import '../src/scenarios.dart';

void main(List<String> args) {
  final only = args.isEmpty ? null : args.first;
  final repeat = only == null ? 21 : 400;
  final inputs =
      jsonDecode(readText('benchmark/data/inputs.json'))
          as Map<String, Object?>;
  // The first 3000 trials, on the 3-level spec with ω2 = ω3 = -6: every mode
  // stays finite there. With the benchmark's ω2 = ω3 = -2, relaxed eHGF
  // goes NaN after 48 trials and relaxed standard after 2229 (as in pyhgf),
  // and the timings would measure NaN propagation instead.
  final data = Float64List.sublistView(floats(inputs['continuous']), 0, 3000);
  final spec =
      jsonDecode(
            jsonEncode(
              (inputs['specs'] as Map<String, Object?>)['continuous_3level'],
            ),
          )
          as Map<String, Object?>;
  for (final i in [2, 3]) {
    final node = (spec['nodes'] as List<Object?>)[i] as Map<String, Object?>;
    (node['params'] as Map<String, Object?>)['tonic_volatility'] = -6.0;
  }
  String nsPerTrial(double ms) =>
      '${(ms * 1e6 / data.length).toStringAsFixed(0)} ns/trial';

  Network build(String mode, bool mf) {
    final s = jsonDecode(jsonEncode(spec)) as Map<String, Object?>;
    (s['network'] as Map<String, Object?>)
      ..['volatility_updates'] = mode
      ..['mean_field_updates'] = mf;
    final net = NetworkSpec.fromJson(s).build()..build();
    _checkFinite(net, data, '$mode meanField=$mf');
    return net;
  }

  for (final mode in ['standard', 'eHGF', 'unbounded']) {
    if (only != null && only != mode) continue;
    for (final mf in [false, true]) {
      final net = build(mode, mf);
      final ms = medianMs(
        () => net.inputData(data, record: false),
        repeat: repeat,
        warmup: 5,
      );
      print('${mode.padRight(10)} meanField=$mf  ${nsPerTrial(ms)}');
    }
  }

  if (only == null || only == 'floor') {
    // The engine floor: a single continuous node with no volatility coupling.
    final net = Network()
      ..addNodes(precision: 1e4)
      ..addNodes(valueChildren: [0])
      ..build();
    _checkFinite(net, data, '2-node');
    final ms = medianMs(
      () => net.inputData(data, record: false),
      repeat: repeat,
    );
    print('2-node value-only       ${nsPerTrial(ms)}');
  }

  if (only == null || only == 'record') {
    // Trajectory recording on top of the filter (default fields).
    final net = build('standard', false);
    final ms = medianMs(() => net.inputData(data), repeat: repeat, warmup: 5);
    print('standard + record       ${nsPerTrial(ms)}');
    // The same, reusing one Trajectories (no fresh pages to fault in).
    var traj = net.inputData(data)!;
    final msReuse = medianMs(
      () => traj = net.inputData(data, into: traj)!,
      repeat: repeat,
      warmup: 5,
    );
    print('standard + record(into) ${nsPerTrial(msReuse)}');
  }

  if (only == null || only == 'ef') {
    // ef-state nodes: the fixed-rate update alone, and hgf-2 learning (one
    // 3-node continuous chain per sufficient statistic, unbounded updates).
    for (final (label, learning, dimension) in [
      ('ef fixed, d=1          ', EfLearning.fixed, 1),
      ('ef fixed, d=3          ', EfLearning.fixed, 3),
      ('ef hgf-2, d=1          ', EfLearning.hgf2, 1),
    ]) {
      final net = Network()
        ..addEfState(dimension: dimension, learning: learning)
        ..build();
      final u = Float64List(data.length * dimension);
      for (var t = 0; t < data.length; t++) {
        for (var j = 0; j < dimension; j++) {
          u[t * dimension + j] = data[t] * (j + 1);
        }
      }
      _checkFinite(net, u, label.trim());
      final ms = medianMs(
        () => net.inputData(u, record: false),
        repeat: repeat,
        warmup: 5,
      );
      print('$label${nsPerTrial(ms)}');
    }
  }

  if (only == null || only == 'categorical') {
    // A categorical node: n binary HGFs (unbounded updates on the shared
    // volatility parent) plus the Dirichlet posterior (2n lgamma/digamma
    // terms for the KL divergence).
    for (final (label, n) in [
      ('categorical, n=3       ', 3),
      ('categorical, n=8       ', 8),
    ]) {
      final net = Network()
        ..addCategoricalState(nCategories: n)
        ..build();
      final u = Float64List(data.length * n);
      for (var t = 0; t < data.length; t++) {
        // A deterministic, uneven category sequence driven by the series.
        u[t * n + ((data[t] * 1e4).floor() % n + t % 2) % n] = 1.0;
      }
      _checkFinite(net, u, label.trim());
      final ms = medianMs(
        () => net.inputData(u, record: false),
        repeat: repeat,
        warmup: 5,
      );
      print('$label${nsPerTrial(ms)}');
    }
  }

  if (only == null || only == 'response') {
    // Response-model overhead on top of the filter (the fitting objective).
    final hgf = build('unbounded', false);
    final responses = Float64List.fromList(data);
    final perTrial = Float64List(data.length);
    final models = <String, ResponseFunction?>{
      'no response': null,
      'FirstLevelGaussian': const FirstLevelGaussianSurprise(),
      'TotalGaussian': const TotalGaussianSurprise(),
      'GaussianResponse': GaussianResponse(responses),
      'GaussianResponse+perTrial': GaussianResponse(responses),
    };
    for (final MapEntry(key: name, value: rf) in models.entries) {
      final ms = medianMs(
        () => rf == null
            ? hgf.inputData(data, record: false)
            : hgf.surprise(
                data,
                rf,
                perTrial: name.endsWith('perTrial') ? perTrial : null,
              ),
        repeat: repeat,
        warmup: 5,
      );
      print('unbounded 3-level, ${name.padRight(26)} ${nsPerTrial(ms)}');
    }
  }
}

/// Warn when [net] goes non-finite on [data]: the timing would then measure
/// NaN propagation (on Wasm noticeably cheaper than the real update).
void _checkFinite(Network net, Float64List data, String label) {
  final traj = net.inputData(data, fields: const [NodeField.mean])!;
  for (var t = 0; t < traj.nTime; t++) {
    for (var i = 0; i < traj.nNodes; i++) {
      if (!traj.at(i, NodeField.mean, t).isFinite) {
        print('WARNING: $label is non-finite from trial $t; timing is off');
        return;
      }
    }
  }
  for (final f in traj.vectorFields) {
    if (!traj.rawVector(f).every((v) => v.isFinite)) {
      print('WARNING: $label has non-finite ${f.pyhgfName}; timing is off');
    }
  }
}
