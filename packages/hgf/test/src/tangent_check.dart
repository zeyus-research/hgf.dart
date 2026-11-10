import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'fixtures.dart';

/// Elements per batch, and the block size (3 elements in blocks of 2 also
/// covers a partial block).
const int _elements = 3;
const int _block = 2;

/// One parameter direction: what a unit change of θ moves.
final class _Direction {
  _Direction(this.name);

  final String name;

  /// (node, field) pairs whose initial value moves by `θ`.
  final List<(int, NodeField)> fields = [];

  /// Value and volatility edges (parent, child) whose strength moves by `θ`.
  final List<(int, int)> valueEdges = [], volatilityEdges = [];

  bool get isEmpty =>
      fields.isEmpty && valueEdges.isEmpty && volatilityEdges.isEmpty;
}

/// The directions checked on [net]: every continuous node's tonic
/// volatility, tonic drift, prior mean and prior precision (each as one
/// direction), the first value edge and the first volatility edge.
List<_Direction> _directions(Network net) {
  final continuous = [
    for (var i = 0; i < net.nNodes; i++)
      if (net.kindOf(i) == NodeKind.continuousState) i,
  ];
  final hidden = [
    for (final i in continuous)
      if (!net.inputIdxs.contains(i)) i,
  ];
  _Direction fieldDir(String name, NodeField f, List<int> nodes) =>
      _Direction(name)..fields.addAll([for (final i in nodes) (i, f)]);
  final dirs = [
    fieldDir('tonicVolatility', NodeField.tonicVolatility, continuous),
    fieldDir('tonicDrift', NodeField.tonicDrift, hidden),
    fieldDir('mean', NodeField.mean, hidden),
    fieldDir('precision', NodeField.precision, continuous),
  ];
  final value = _Direction('valueCoupling');
  final vol = _Direction('volatilityCoupling');
  outer:
  for (var p = 0; p < net.nNodes; p++) {
    for (final c in net.valueChildrenOf(p)) {
      value.valueEdges.add((p, c));
      break outer;
    }
  }
  outer2:
  for (var p = 0; p < net.nNodes; p++) {
    for (final c in net.volatilityChildrenOf(p)) {
      vol.volatilityEdges.add((p, c));
      break outer2;
    }
  }
  return [...dirs, value, vol];
}

/// A batch of [_elements] perturbed copies of [template], moved by
/// `theta[e * k + d]` along each direction, with tangents if [tangents].
BatchNetwork _batch(
  Network template,
  List<_Direction> dirs,
  Float64List theta, {
  required bool tangents,
}) {
  final k = dirs.length;
  final batch = BatchNetwork(
    template,
    _elements,
    blockSize: _block,
    tangents: tangents ? k : 0,
  );
  final values = Float64List(_elements);
  // Element-specific base parameters, as in batch_check.dart.
  for (var i = 0; i < template.nNodes; i++) {
    if (template.kindOf(i) != NodeKind.continuousState) continue;
    for (final (f, step) in [
      (NodeField.tonicVolatility, -0.25),
      (NodeField.mean, 0.1),
    ]) {
      for (var e = 0; e < _elements; e++) {
        values[e] = template.getInitial(i, f) + step * e;
      }
      batch.setInitial(i, f, values);
    }
  }
  for (var d = 0; d < k; d++) {
    final dir = dirs[d];
    for (final (i, f) in dir.fields) {
      for (var e = 0; e < _elements; e++) {
        values[e] = batch.getInitial(e, i, f) + theta[e * k + d];
      }
      batch.setInitial(i, f, values);
    }
    for (final (p, c) in dir.valueEdges) {
      for (var e = 0; e < _elements; e++) {
        values[e] = 1.0 + 0.1 * e + theta[e * k + d];
      }
      batch.setValueCoupling(p, c, values);
    }
    for (final (p, c) in dir.volatilityEdges) {
      for (var e = 0; e < _elements; e++) {
        values[e] = 1.0 + 0.2 * e + theta[e * k + d];
      }
      batch.setVolatilityCoupling(p, c, values);
    }
  }
  if (tangents) {
    final ones = Float64List(_elements)..fillRange(0, _elements, 1.0);
    for (var d = 0; d < k; d++) {
      final dir = dirs[d];
      for (final (i, f) in dir.fields) {
        batch.setInitialTangent(i, f, d, ones);
      }
      for (final (p, c) in dir.valueEdges) {
        batch.setValueCouplingTangent(p, c, d, ones);
      }
      for (final (p, c) in dir.volatilityEdges) {
        batch.setVolatilityCouplingTangent(p, c, d, ones);
      }
    }
  }
  return batch..reset();
}

bool _same(double a, double b) => a == b || (a.isNaN && b.isNaN);

/// Checks [fixture]'s network with forward-mode tangents:
///
/// * a batch with tangents computes bit for bit what one without computes
///   (final state and per-trial surprise);
/// * the tangents of the total surprise and of every tangent field of every
///   node at the end of the series match Richardson-extrapolated central
///   differences, along each direction of [_directions], at one of the
///   steps [steps] (some fixtures are so sensitive, e.g. the binary 3-level
///   ones in pyhgf's runaway regime, that large steps leave the linear
///   range and small ones drown in rounding). The absolute tolerance scales
///   with the largest tangent of the element and direction: a difference of
///   two large tangents (`vape = μ − μ̂`) can't be more exact than they are.
///
/// Returns the failures (empty if none) and the number of finite-difference
/// comparisons made; null if the network has no continuous node (nothing
/// to differentiate).
(List<String>, int)? tangentMismatches(
  Map<String, Object?> fixture, {
  double relTol = 2e-5,
  List<double> steps = const [1e-4, 1e-5, 1e-6, 1e-7],
  int? maxTrials,
}) {
  final name = fixture['name'] as String;
  final spec = NetworkSpec.fromJson(fixture['spec'] as Map<String, Object?>);
  final template = spec.build(couplings: testCouplings)..build();
  // Nothing to differentiate without continuous nodes (a lone fixed-rate
  // ef-state node, a dp-state node and its clusters).
  if (!List.generate(
    template.nNodes,
    template.kindOf,
  ).contains(NodeKind.continuousState)) {
    return null;
  }
  final nIn = template.inputIdxs.length, nCols = template.nInputColumns;
  var data = flatten(fixture['inputs']);
  var observed = fixture['observed'] == null
      ? null
      : flatten(fixture['observed']);
  var timeSteps = fixture['time_steps'] == null
      ? null
      : Float64List.fromList(series(fixture['time_steps']));
  if (maxTrials != null) {
    if (maxTrials * nCols < data.length) {
      data = Float64List.sublistView(data, 0, maxTrials * nCols);
      if (observed != null) {
        observed = Float64List.sublistView(observed, 0, maxTrials * nIn);
      }
      if (timeSteps != null) {
        timeSteps = Float64List.sublistView(timeSteps, 0, maxTrials);
      }
    }
  }
  final response = switch (template.kindOf(template.inputIdxs.first)) {
    NodeKind.binaryState => const FirstLevelBinarySurprise(),
    NodeKind.categoricalState => const CategoricalSurprise(),
    _ => const TotalGaussianSurprise() as ResponseFunction,
  };
  final nTime = data.length ~/ nCols;

  final dirs = _directions(template);
  final k = dirs.length;
  final zero = Float64List(_elements * k);
  final failures = <String>[];

  // Primal bit-identity.
  final plain = _batch(template, dirs, zero, tangents: false);
  final dual = _batch(template, dirs, zero, tangents: true);
  final perTrialPlain = Float64List(nTime * _elements);
  final perTrialDual = Float64List(nTime * _elements);
  final totals = plain.surprise(
    data,
    response,
    observed: observed,
    timeSteps: timeSteps,
    perTrial: perTrialPlain,
  );
  final gradient = Float64List(_elements * k);
  final dualTotals = dual.surprise(
    data,
    response,
    observed: observed,
    timeSteps: timeSteps,
    perTrial: perTrialDual,
    gradient: gradient,
  );
  for (var e = 0; e < _elements; e++) {
    if (!_same(totals[e], dualTotals[e])) {
      failures.add('$name: total e=$e ${totals[e]} vs ${dualTotals[e]}');
    }
  }
  for (var j = 0; j < perTrialPlain.length; j++) {
    if (!_same(perTrialPlain[j], perTrialDual[j])) {
      failures.add('$name: per-trial surprise differs at $j');
      break;
    }
  }
  for (var i = 0; i < template.nNodes; i++) {
    for (final f in NodeField.values) {
      for (var e = 0; e < _elements; e++) {
        if (!_same(plain.get(e, i, f), dual.get(e, i, f))) {
          failures.add('$name: final ${f.name} node $i e=$e differs');
        }
      }
    }
  }

  // Tangents vs Richardson central differences.
  final n = template.nNodes;
  final fields = BatchNetwork.tangentFields;
  // Final state of a run at theta, as [e][node][field].
  (Float64List, Float64List) run(Float64List theta) {
    final b = _batch(template, dirs, theta, tangents: false);
    final tot = b.surprise(
      data,
      response,
      observed: observed,
      timeSteps: timeSteps,
    );
    final state = Float64List(_elements * n * fields.length);
    for (var e = 0; e < _elements; e++) {
      for (var i = 0; i < n; i++) {
        for (var f = 0; f < fields.length; f++) {
          state[(e * n + i) * fields.length + f] = b.get(e, i, fields[f]);
        }
      }
    }
    return (tot, state);
  }

  var compared = 0;
  void check(
    String what,
    double exact,
    List<double> fds,
    double scale,
    double value,
  ) {
    if (fds.any((fd) => !fd.isFinite) || !exact.isFinite) return;
    compared++;
    for (final fd in fds) {
      final err = (exact - fd).abs();
      final big = fd.abs() > exact.abs() ? fd.abs() : exact.abs();
      final floor = 1e-12 * scale + 1e-9 * (1 + value.abs());
      if (err <= relTol * big + floor) return;
    }
    failures.add('$name: $what: tangent $exact, FD $fds');
  }

  for (var d = 0; d < k; d++) {
    if (dirs[d].isEmpty) continue;
    Float64List at(double s) {
      final th = Float64List(_elements * k);
      for (var e = 0; e < _elements; e++) {
        th[e * k + d] = s;
      }
      return th;
    }

    // Richardson-extrapolated central differences at each step.
    final totalFd = <Float64List>[], stateFd = <Float64List>[];
    for (final h in steps) {
      final (tp1, sp1) = run(at(h));
      final (tm1, sm1) = run(at(-h));
      final (tp2, sp2) = run(at(2 * h));
      final (tm2, sm2) = run(at(-2 * h));
      Float64List rich(
        Float64List p1,
        Float64List m1,
        Float64List p2,
        Float64List m2,
      ) => Float64List.fromList([
        for (var o = 0; o < p1.length; o++)
          (8 * (p1[o] - m1[o]) - (p2[o] - m2[o])) / (12 * h),
      ]);
      totalFd.add(rich(tp1, tm1, tp2, tm2));
      stateFd.add(rich(sp1, sm1, sp2, sm2));
    }
    for (var e = 0; e < _elements; e++) {
      // The largest finite tangent of this element and direction.
      var scale = gradient[e * k + d].abs();
      if (!scale.isFinite) scale = 0.0;
      for (var i = 0; i < n; i++) {
        for (final f in fields) {
          final v = dual.getTangent(e, i, f, d).abs();
          if (v.isFinite && v > scale) scale = v;
        }
      }
      check(
        '${dirs[d].name} e=$e total surprise',
        gradient[e * k + d],
        [for (final fd in totalFd) fd[e]],
        scale,
        totals[e],
      );
      for (var i = 0; i < n; i++) {
        for (var f = 0; f < fields.length; f++) {
          final o = (e * n + i) * fields.length + f;
          check(
            '${dirs[d].name} e=$e node $i ${fields[f].name}',
            dual.getTangent(e, i, fields[f], d),
            [for (final fd in stateFd) fd[o]],
            scale,
            dual.get(e, i, fields[f]),
          );
        }
      }
    }
  }
  return (failures, compared);
}

/// Fixtures too ill-conditioned for the default check, with what is checked
/// instead:
///
/// * `continuous_2level_eHGF`: after ~150 trials the posterior precision
///   explodes (tangents ~1e10) and the trajectory turns chaotic: central
///   differences disagree with each other at every step from 1e-4 to 1e-7.
///   Checked over the first 120 trials, where they converge onto the
///   tangents as the step shrinks (to ~1e-4).
/// * `continuous_2level_unbounded_overflow`: the predicted variances reach
///   1e300; finite differences are good to ~1e-4 there.
/// * `dp_mixed`: its dp-state node costs ~1 ms per element-trial on the JIT
///   VM, and the check runs dozens of filters; 40 trials keep it to seconds.
const _overrides = {
  'continuous_2level_eHGF': (maxTrials: 120, relTol: 2e-4),
  'continuous_2level_unbounded_overflow': (maxTrials: null, relTol: 1e-4),
  'dp_mixed': (maxTrials: 40, relTol: 2e-5),
};

/// [tangentMismatches] with the per-fixture settings of [_overrides].
(List<String>, int)? checkTangents(Map<String, Object?> fixture) {
  final o = _overrides[fixture['name'] as String];
  return o == null
      ? tangentMismatches(fixture)
      : tangentMismatches(fixture, maxTrials: o.maxTrials, relTol: o.relTol);
}
