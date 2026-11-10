import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'fit_jobs.dart';

/// CSV exports of what the playground shows: the trajectories, a fit's
/// estimates, and a recovery's truth and estimates per agent.

/// The trajectories in wide format, one row per trial, with the column
/// names of pyhgf's `to_pandas()`: `observation` (empty where missing),
/// `response` (if any), then `x_<node>_<field>` for every recorded field,
/// `x_<node>_surprise` where pyhgf defines a node surprise (continuous and
/// binary nodes) and `total_surprise`.
String trajectoriesCsv(
  Trajectories traj, {
  required Float64List observations,
  Float64List? observed,
  Float64List? responses,
}) {
  final cols = <(String, Float64List)>[];
  final nodeSurprise = traj.kinds != null;
  for (var node = 0; node < traj.nNodes; node++) {
    for (final f in traj.fields) {
      cols.add(('x_${node}_${f.pyhgfName}', traj.series(node, f)));
    }
    if (nodeSurprise) {
      final kind = traj.kinds![node];
      if (kind == NodeKind.continuousState || kind == NodeKind.binaryState) {
        cols.add(('x_${node}_surprise', traj.nodeSurprise(node)));
      }
    }
  }
  if (nodeSurprise) cols.add(('total_surprise', traj.totalSurprise()));
  final b = StringBuffer('trial,observation');
  if (responses != null) b.write(',response');
  for (final (name, _) in cols) {
    b.write(',$name');
  }
  b.writeln();
  for (var t = 0; t < traj.nTime; t++) {
    b
      ..write(t)
      ..write(',')
      ..write(observed != null && observed[t] == 0 ? '' : _v(observations[t]));
    if (responses != null) b.write(',${_v(responses[t])}');
    for (final (_, s) in cols) {
      b.write(',${_v(s[t])}');
    }
    b.writeln();
  }
  return b.toString();
}

/// A fit's estimates, one row per parameter: `parameter`, `symbol`,
/// `estimate` (MAP, or the posterior mean for NUTS), `lower` and `upper`
/// (the 95 % interval), `truth` (simulated responses), and for NUTS `rhat`
/// and `ess`. Comment lines (`#`) before the header give the method, the
/// target and the log evidence.
String fitResultsCsv(FitOutcome r) {
  final nuts = r.method == FitMethod.nuts;
  final b = StringBuffer()
    ..writeln('# method: ${nuts ? 'NUTS' : 'MAP + Laplace'}')
    ..writeln(
      '# fitted to: '
      '${r.target == FitTarget.observations ? 'observations' : 'responses'}',
    )
    ..writeln('# ran on: ${r.where}, ${r.millis} ms');
  if (r.logEvidence != null) b.writeln('# log evidence: ${r.logEvidence}');
  if (nuts) {
    b.writeln('# chains: ${r.chains}, divergences: ${r.divergences ?? 0}');
  }
  if (r.note != null) b.writeln('# note: ${r.note}');
  b.writeln(
    'parameter,symbol,estimate,lower,upper,truth${nuts ? ',rhat,ess' : ''}',
  );
  for (final e in r.estimates) {
    b.write(
      '${_q(e.param.id)},${_q(e.param.symbol)},${_v(e.estimate)},'
      '${_v(e.lower)},${_v(e.upper)},${e.truth == null ? '' : _v(e.truth!)}',
    );
    if (nuts) {
      b.write(',${_v(e.rhat ?? double.nan)},${_v(e.ess ?? double.nan)}');
    }
    b.writeln();
  }
  return b.toString();
}

/// A fit's pooled NUTS draws, one column per parameter (null without
/// draws).
String? fitDrawsCsv(FitOutcome r) {
  final draws = [for (final e in r.estimates) e.draws];
  if (draws.isEmpty || draws.any((d) => d == null)) return null;
  final n = draws.first!.length;
  final b = StringBuffer()
    ..writeln(['draw', for (final e in r.estimates) _q(e.param.id)].join(','));
  for (var i = 0; i < n; i++) {
    b.writeln([i, for (final d in draws) _v(d![i])].join(','));
  }
  return b.toString();
}

/// A recovery, one row per agent: `agent`, `fitted` (whether the fit found
/// a finite posterior), then `<parameter>_true` and `<parameter>_estimate`
/// for each parameter.
String recoveryCsv(RecoveryOutcome r) {
  final b = StringBuffer('agent,fitted');
  for (final p in r.params) {
    b.write(',${_q('${p.id}_true')},${_q('${p.id}_estimate')}');
  }
  b.writeln();
  final m = r.ok.length;
  for (var j = 0; j < m; j++) {
    b.write('$j,${r.ok[j] ? 1 : 0}');
    for (var i = 0; i < r.params.length; i++) {
      b.write(',${_v(r.truth[i][j])},${_v(r.estimate[i][j])}');
    }
    b.writeln();
  }
  return b.toString();
}

/// A number for CSV: empty when missing (NaN), `inf`/`-inf` as in pandas.
String _v(double v) => v.isNaN
    ? ''
    : v == double.infinity
    ? 'inf'
    : v == double.negativeInfinity
    ? '-inf'
    : '$v';

/// A text cell, quoted if it needs to be.
String _q(String s) =>
    s.contains(RegExp('[",\n]')) ? '"${s.replaceAll('"', '""')}"' : s;
