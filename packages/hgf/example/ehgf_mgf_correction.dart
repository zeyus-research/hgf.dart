/// Demonstrates `Network.ehgfMgfCorrection`, a deliberate divergence from
/// pyhgf (0.3.2): the relaxed eHGF update with the moment-generating-function
/// terms added to its volatility re-prediction.
///
/// pyhgf's eHGF precision update re-predicts a volatility child's volatility
/// from the parent's new mean as `t·exp(κ μ_b + ω̃)`. The relaxed prediction
/// of that child also had the terms `Σ_p κ_p² / (2 π̂_p)` in the exponent,
/// so the two disagree even when μ_b = μ̂_b. The increment then floors at
/// zero almost every trial, the volatility parents never gain precision, and
/// their means take ever-larger steps until the filter goes NaN.
///
/// From `packages/hgf`:
///
///     dart run example/ehgf_mgf_correction.dart [out.csv]
///
/// The optional CSV has the trajectories of both runs, for plotting.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

void main(List<String> args) {
  // pyhgf's USD/CHF example series, as the benchmarks use it.
  final inputs =
      jsonDecode(File('benchmark/data/inputs.json').readAsStringSync())
          as Map<String, Object?>;
  final u = Float64List.fromList([
    for (final v in inputs['usdchf'] as List<Object?>) (v as num).toDouble(),
  ]);

  // The 3-level continuous HGF of pyhgf's tutorials, relaxed eHGF updates.
  Network hgf({required bool correction}) =>
      Network(
          volatilityUpdates: VolatilityUpdates.eHGF,
          ehgfMgfCorrection: correction,
        )
        ..addNodes(precision: 1e4)
        ..addNodes(valueChildren: [0], mean: 1.04, tonicVolatility: -8.0)
        ..addNodes(volatilityChildren: [1], tonicVolatility: -2.0)
        ..addNodes(volatilityChildren: [2], tonicVolatility: -2.0);

  final runs = {
    'pyhgf (stock)': hgf(correction: false).inputData(u)!,
    'MGF-corrected': hgf(correction: true).inputData(u)!,
  };

  print('3-level continuous HGF, relaxed eHGF, USD/CHF (${u.length} trials)');
  print('');
  for (final MapEntry(key: name, value: traj) in runs.entries) {
    int? firstNaN;
    var gains = 0;
    for (var t = 0; t < traj.nTime; t++) {
      if (firstNaN == null && !traj.at(2, NodeField.mean, t).isFinite) {
        firstNaN = t;
      }
      if (traj.at(2, NodeField.precision, t) >
          traj.at(2, NodeField.expectedPrecision, t)) {
        gains++;
      }
    }
    var total = 0.0, finite = 0;
    for (final v in traj.totalSurprise()) {
      if (v.isFinite) {
        total += v;
        finite++;
      }
    }
    print(name);
    print('  first non-finite trial:          ${firstNaN ?? 'none'}');
    print(
      '  x2 gains precision from x1 on:   $gains/${traj.nTime} trials '
      '(the eHGF increment is floored at 0 on the rest)',
    );
    print(
      '  total surprise:                  ${total.toStringAsFixed(1)} '
      '(over the $finite finite trials)',
    );
  }

  print('');
  print(
    'trial   x2 mean (stock / corrected)   x2 precision (stock / corrected)',
  );
  final a = runs['pyhgf (stock)']!, b = runs['MGF-corrected']!;
  for (final t in [0, 10, 20, 30, 40, 42, 43, 44, 100, 300, u.length - 1]) {
    String f(Trajectories x, NodeField field) =>
        x.at(2, field, t).toStringAsPrecision(4).padLeft(10);
    print(
      '${'$t'.padLeft(5)}   ${f(a, NodeField.mean)} / ${f(b, NodeField.mean)}'
      '        ${f(a, NodeField.precision)} / ${f(b, NodeField.precision)}',
    );
  }

  if (args.isNotEmpty) {
    final out = StringBuffer(
      'run,time,node,mean,precision,expected_precision\n',
    );
    for (final MapEntry(key: name, value: traj) in runs.entries) {
      for (var i = 0; i < traj.nNodes; i++) {
        for (var t = 0; t < traj.nTime; t++) {
          out.writeln(
            '$name,$t,$i,${traj.at(i, NodeField.mean, t)},'
            '${traj.at(i, NodeField.precision, t)},'
            '${traj.at(i, NodeField.expectedPrecision, t)}',
          );
        }
      }
    }
    File(args.first).writeAsStringSync(out.toString());
    print('');
    print('Wrote ${args.first}');
  }
}
