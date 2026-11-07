import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import '../fit.dart';
import '../model.dart';
import '../optim/optimizer.dart';
import '../pass.dart';
import 'diagnostics.dart';
import 'nuts.dart';

/// Posterior summary of one parameter.
typedef ParameterSummary = ({
  String name,
  double mean,
  double sd,
  double q5,
  double median,
  double q95,
  double mcseMean,
  double essBulk,
  double essTail,
  double rhat,
});

/// The draws of several chains for one target (one subject), with the
/// sampler statistics and convergence diagnostics.
final class PosteriorDraws {
  PosteriorDraws({
    required this.names,
    required List<ChainOutput> chains,
    required Float64List Function(Float64List y) toConstrained,
    this.subject,
  }) : chains = [
         for (final c in chains)
           if (c.error == null) c,
       ],
       failedChains = [for (final c in chains) ?c.error] {
    final k = names.length;
    _x = [
      for (final c in this.chains)
        () {
          final out = Float64List(c.y.length);
          for (var d = 0; d * k < c.y.length; d++) {
            out.setRange(
              d * k,
              (d + 1) * k,
              toConstrained(Float64List.sublistView(c.y, d * k, (d + 1) * k)),
            );
          }
          return out;
        }(),
    ];
  }

  /// The subject sampled, if any.
  final Subject? subject;

  final List<String> names;

  /// The chains that ran (each holds its draws on the unconstrained scale
  /// and the sampler statistics).
  final List<ChainOutput> chains;

  /// Why the other chains failed to start.
  final List<String> failedChains;

  late final List<Float64List> _x;

  int get dimension => names.length;
  int get nChains => chains.length;
  int get nDraws => chains.isEmpty ? 0 : chains.first.logDensity.length;

  /// Draw [d] of chain [c] on the parameters' natural scale.
  Float64List draw(int c, int d) =>
      Float64List.sublistView(_x[c], d * dimension, (d + 1) * dimension);

  /// Parameter [i]'s draws on the natural scale, one list per chain.
  List<Float64List> parameter(int i) => [
    for (final x in _x)
      Float64List.fromList([
        for (var d = i; d < x.length; d += dimension) x[d],
      ]),
  ];

  /// All of parameter [i]'s draws, pooled and sorted.
  Float64List _sorted(int i) {
    final p = parameter(i);
    final out = Float64List(p.fold(0, (s, c) => s + c.length));
    var o = 0;
    for (final c in p) {
      out.setRange(o, o + c.length, c);
      o += c.length;
    }
    return out..sort();
  }

  /// Posterior mean of parameter [i].
  double mean(int i) {
    final a = _sorted(i);
    return a.fold(0.0, (s, v) => s + v) / a.length;
  }

  /// Posterior standard deviation of parameter [i].
  double sd(int i) {
    final a = _sorted(i), m = mean(i);
    return math.sqrt(
      a.fold(0.0, (s, v) => s + (v - m) * (v - m)) / (a.length - 1),
    );
  }

  /// The [p] quantile of parameter [i].
  double quantile(int i, double p) => quantileOf(_sorted(i), p);

  /// Divergent transitions after warmup, over all chains.
  int get divergences =>
      chains.fold(0, (s, c) => s + c.divergent.fold(0, (a, v) => a + v));

  /// Transitions that hit the maximum tree depth.
  int maxDepthHits(int maxTreeDepth) => chains.fold(
    0,
    (s, c) => s + c.treeDepth.where((d) => d >= maxTreeDepth).length,
  );

  /// Leapfrog steps after warmup, over all chains.
  int get leapfrogSteps =>
      chains.fold(0, (s, c) => s + c.leapfrogSteps.fold(0, (a, v) => a + v));

  /// E-BFMI of each chain (`Σ (E_d − E_{d−1})² / Σ (E_d − Ē)²`); below 0.3
  /// suggests the momentum resampling explores the energy poorly.
  List<double> get eBfmi => [
    for (final c in chains)
      () {
        final e = c.energy;
        if (e.length < 2) return double.nan;
        final m = e.fold(0.0, (s, v) => s + v) / e.length;
        var num = 0.0, den = 0.0;
        for (var d = 0; d < e.length; d++) {
          if (d > 0) num += (e[d] - e[d - 1]) * (e[d] - e[d - 1]);
          den += (e[d] - m) * (e[d] - m);
        }
        return num / den;
      }(),
  ];

  /// Mean, sd, 5/50/95% quantiles, MCSE of the mean, bulk and tail ESS and
  /// R̂ of each parameter. These are ArviZ's estimators; the tail ESS uses
  /// the 5% and 95% quantiles (Stan's convention; ArviZ 1.x's `summary`
  /// uses the bounds of its 89% interval instead).
  List<ParameterSummary> summary() => [
    for (var i = 0; i < dimension; i++)
      () {
        final p = parameter(i), s = _sorted(i);
        return (
          name: names[i],
          mean: mean(i),
          sd: sd(i),
          q5: quantileOf(s, 0.05),
          median: quantileOf(s, 0.5),
          q95: quantileOf(s, 0.95),
          mcseMean: mcseMean(p),
          essBulk: essBulk(p),
          essTail: essTail(p),
          rhat: rhat(p),
        );
      }(),
  ];

  /// Whether every parameter has R̂ ≤ [maxRhat] and bulk and tail ESS of
  /// at least [minEss], with no divergences.
  bool converged({double maxRhat = 1.01, double minEss = 400}) =>
      divergences == 0 &&
      summary().every(
        (s) => s.rhat <= maxRhat && s.essBulk >= minEss && s.essTail >= minEss,
      );

  /// The draws as an ArviZ `from_dict` payload: `posterior` (each
  /// parameter as `[chain][draw]`) and `sample_stats` (`lp`,
  /// `acceptance_rate`, `diverging`, `energy`, `tree_depth`, `n_steps`,
  /// `step_size`).
  Map<String, Object> toArviz() => {
    'posterior': {for (var i = 0; i < dimension; i++) names[i]: parameter(i)},
    'sample_stats': _stats(),
  };

  Map<String, Object> _stats() => {
    'lp': [for (final c in chains) c.logDensity],
    'acceptance_rate': [for (final c in chains) c.acceptStat],
    'diverging': [
      for (final c in chains) [for (final v in c.divergent) v == 1],
    ],
    'energy': [for (final c in chains) c.energy],
    'tree_depth': [for (final c in chains) c.treeDepth],
    'n_steps': [for (final c in chains) c.leapfrogSteps],
    'step_size': [
      for (final c in chains) List.filled(c.logDensity.length, c.stepSize),
    ],
  };

  /// The draws as CSV in the column layout of Stan's output: `chain`,
  /// `draw`, `lp__`, `accept_stat__`, `stepsize__`, `treedepth__`,
  /// `n_leapfrog__`, `divergent__`, `energy__`, then the parameters
  /// (natural scale).
  String toCsv() {
    final b = StringBuffer()
      ..writeln(
        [
          'chain',
          'draw',
          'lp__',
          'accept_stat__',
          'stepsize__',
          'treedepth__',
          'n_leapfrog__',
          'divergent__',
          'energy__',
          ...names,
        ].join(','),
      );
    for (var c = 0; c < nChains; c++) {
      final ch = chains[c];
      for (var d = 0; d < nDraws; d++) {
        b.writeln(
          [
            c,
            d,
            ch.logDensity[d],
            ch.acceptStat[d],
            ch.stepSize,
            ch.treeDepth[d],
            ch.leapfrogSteps[d],
            ch.divergent[d],
            ch.energy[d],
            ...draw(c, d),
          ].join(','),
        );
      }
    }
    return b.toString();
  }

  @override
  String toString() {
    final b = StringBuffer(
      'PosteriorDraws(${subject?.id ?? ''} $nChains chains x $nDraws draws, '
      '$divergences divergences)\n',
    );
    final w = names.fold(18, (m, n) => math.max(m, n.length)) + 2;
    b.writeln(
      '${'parameter'.padRight(w)}${'mean'.padLeft(10)}${'sd'.padLeft(10)}'
      '${'q5'.padLeft(10)}${'q95'.padLeft(10)}${'ess_bulk'.padLeft(10)}'
      '${'ess_tail'.padLeft(10)}${'r_hat'.padLeft(8)}',
    );
    String f(double v, [int d = 3]) => v.toStringAsFixed(d).padLeft(10);
    for (final s in summary()) {
      b.writeln(
        '${s.name.padRight(w)}${f(s.mean)}${f(s.sd)}${f(s.q5)}${f(s.q95)}'
        '${f(s.essBulk, 0)}${f(s.essTail, 0)}'
        '${s.rhat.toStringAsFixed(3).padLeft(8)}',
      );
    }
    return b.toString();
  }
}

/// How far a [sampleNuts] run has got: pass one in and read it while the
/// run goes on (it is updated on the calling isolate, between passes).
final class SamplingProgress {
  List<ChainOutput> _chains = const [];
  int _perChain = 0;

  void _attach(List<ChainOutput> chains, int perChain) {
    _chains = chains;
    _perChain = perChain;
  }

  /// Transitions to run, warmup included, over all chains (0 until the
  /// run has started).
  int get total => _chains.length * _perChain;

  /// Transitions done; a chain that failed to start counts as done.
  int get completed {
    var n = 0;
    for (final c in _chains) {
      n += c.error != null ? _perChain : c.iterations;
    }
    return n;
  }

  /// [completed] / [total], 0 before the start.
  double get fraction => total == 0 ? 0 : completed / total;
}

/// The posterior draws of a [sampleNuts] call, one per subject, with what
/// they cost.
final class NutsFit {
  NutsFit(this.results, this.passes, this.elements);

  /// One per subject, in order.
  final List<PosteriorDraws> results;

  /// Batched passes over the data (one per leapfrog step of the slowest
  /// chain, plus starting points and step-size searches).
  final int passes;

  /// Batch elements evaluated in total (one per leapfrog step with exact
  /// gradients, `k + 1` with forward differences, `2k + 1` with central
  /// ones).
  final int elements;

  /// Divergent transitions over all subjects.
  int get divergences => results.fold(0, (s, r) => s + r.divergences);

  /// All subjects' draws as one ArviZ `from_dict` payload, with a
  /// `subject` dimension: each parameter is `[chain][draw][subject]`, and
  /// so is each sampler statistic. Pass the returned `dims` and `coords`
  /// on to `from_dict` as well. Every subject must have the same number of
  /// chains and draws.
  Map<String, Object> toArviz() {
    if (results.isEmpty) return {};
    final nC = results.first.nChains, nD = results.first.nDraws;
    for (final r in results) {
      if (r.nChains != nC || r.nDraws != nD) {
        throw StateError('Subjects have different numbers of chains/draws');
      }
    }
    final per = [for (final r in results) r.toArviz()];
    List<Object> stack(Object Function(Map<String, Object>) pick) => [
      for (var c = 0; c < nC; c++)
        [
          for (var d = 0; d < nD; d++)
            [
              for (final p in per)
                ((pick(p) as List<Object?>)[c] as List<Object?>)[d],
            ],
        ],
    ];
    final first = per.first;
    final names = results.first.names;
    final stats = (first['sample_stats'] as Map<String, Object>).keys;
    return {
      'posterior': {
        for (final n in names)
          n: stack((p) => (p['posterior'] as Map<String, Object>)[n]!),
      },
      'sample_stats': {
        for (final s in stats)
          s: stack((p) => (p['sample_stats'] as Map<String, Object>)[s]!),
      },
      'dims': {
        for (final n in names) n: ['subject'],
        for (final s in stats) s: ['subject'],
      },
      'coords': {
        'subject': [
          for (var i = 0; i < results.length; i++)
            '${results[i].subject?.id ?? i}',
        ],
      },
    };
  }

  /// [toArviz] as JSON. In Python (ArviZ 1.x):
  ///
  /// ```python
  /// d = json.load(open("draws.json"))
  /// groups = {g: {k: np.asarray(v) for k, v in d[g].items()}
  ///           for g in ("posterior", "sample_stats")}
  /// idata = az.from_dict(groups, dims=d["dims"], coords=d["coords"])
  /// ```
  String toJson() => jsonEncode(toArviz());
}

/// Posterior draws of [model]'s parameters for each of [subjects] by NUTS,
/// [chains] chains per subject, all advanced in lock step: each batched pass
/// evaluates one leapfrog step (point and gradient) of every running chain
/// of every subject.
///
/// The target is the posterior density on the unconstrained scale (with the
/// transforms' Jacobians). Chain `c` of subject `s` starts at `start[s]`
/// (natural scale; default the parameters' starting values, e.g. pass the
/// MAP estimates) moved by a uniform `±jitter` on the unconstrained scale,
/// redrawn until the density is finite. [seed] makes runs reproducible.
///
/// Gradients are exact where the model allows it
/// (`HgfModel.hasExactGradients`), otherwise forward differences (`k + 1`
/// filters per leapfrog step): the sampler stays exact with any
/// deterministic gradient approximation, so a finite-difference gradient
/// only changes how far a step size can stretch.
///
/// [progress], if given, counts the transitions as they are done.
NutsFit sampleNuts(
  HgfModel model,
  List<Subject> subjects, {
  Nuts sampler = const Nuts(),
  int chains = 4,
  int warmup = 1000,
  int draws = 1000,
  List<List<double>>? start,
  double jitter = 1,
  int seed = 0,
  double fdStep = 1e-5,
  Gradients? gradients,
  SamplingProgress? progress,
}) => runInline(
  (o) => _sampleNuts(
    model,
    subjects,
    sampler,
    chains,
    warmup,
    draws,
    start,
    jitter,
    seed,
    fdStep,
    gradients,
    progress,
    1,
    o,
  ),
);

/// [sampleNuts] with each leapfrog pass split across the pool's isolates.
extension SampleNutsOnWorkers on PassRunner {
  /// [sampleNuts] on this pool; the same draws, bit for bit.
  Future<NutsFit> sampleNuts(
    HgfModel model,
    List<Subject> subjects, {
    Nuts sampler = const Nuts(),
    int chains = 4,
    int warmup = 1000,
    int draws = 1000,
    List<List<double>>? start,
    double jitter = 1,
    int seed = 0,
    double fdStep = 1e-5,
    Gradients? gradients,
    SamplingProgress? progress,
  }) => run(
    (o) => _sampleNuts(
      model,
      subjects,
      sampler,
      chains,
      warmup,
      draws,
      start,
      jitter,
      seed,
      fdStep,
      gradients,
      progress,
      streams,
      o,
    ),
  );
}

Iterable<Step> _sampleNuts(
  HgfModel model,
  List<Subject> subjects,
  Nuts sampler,
  int chains,
  int warmup,
  int draws,
  List<List<double>>? start,
  double jitter,
  int seed,
  double fdStep,
  Gradients? gradients,
  SamplingProgress? progress,
  int streams,
  Out<NutsFit> result,
) sync* {
  if (start != null && start.length != subjects.length) {
    throw ArgumentError('start must have one vector per subject');
  }
  if (chains < 1) throw ArgumentError.value(chains, 'chains');
  final objective = BatchedObjective(
    model,
    subjects,
    fdStep: fdStep,
    jacobian: true,
    gradients:
        gradients ??
        (model.hasExactGradients ? Gradients.exact : Gradients.forward),
  );
  final k = model.dimension;
  final outs = <List<ChainOutput>>[];
  final tasks = <({int subject, Iterator<EvalRequest> it})>[];
  for (var s = 0; s < subjects.length; s++) {
    final x0 = start?[s] ?? model.start();
    if (x0.length != k) {
      throw ArgumentError('Start vectors must have $k values');
    }
    final y0 = model.toUnconstrained(x0);
    final row = <ChainOutput>[];
    for (var c = 0; c < chains; c++) {
      final rng = math.Random(chainSeed(seed, s, c));
      final out = ChainOutput();
      row.add(out);
      final it = sampler
          .chain(
            init: (_) => Float64List.fromList([
              for (final v in y0) v + jitter * (2 * rng.nextDouble() - 1),
            ]),
            warmup: warmup,
            draws: draws,
            random: rng,
            out: out,
          )
          .iterator;
      tasks.add((subject: s, it: it));
    }
    outs.add(row);
  }
  progress?._attach([for (final r in outs) ...r], warmup + draws);
  yield* driveChainPasses(tasks, streams: streams, (
    subjectOf,
    points,
    o,
  ) sync* {
    final evaluated = Out<(Float64List, List<Float64List?>)>();
    yield* objective.evaluatePasses(
      subjectOf,
      points,
      List.filled(points.length, true),
      evaluated,
    );
    final (v, g) = evaluated.value;
    o.value = (v, [for (final x in g) x!]);
  });
  result.value = NutsFit(
    [
      for (var s = 0; s < subjects.length; s++)
        PosteriorDraws(
          names: model.names,
          chains: outs[s],
          toConstrained: model.toConstrained,
          subject: subjects[s],
        ),
    ],
    objective.passes,
    objective.elements,
  );
}

/// NUTS on a plain log density, for testing and for targets outside
/// `HgfModel`: [potential] is `U(y) = −log p(y)` and [gradient] its
/// gradient. Chains start at [y0] moved by a uniform `±jitter`.
PosteriorDraws sampleFunction(
  double Function(Float64List y) potential,
  Float64List Function(Float64List y) gradient,
  Float64List y0, {
  Nuts sampler = const Nuts(),
  int chains = 4,
  int warmup = 1000,
  int draws = 1000,
  double jitter = 1,
  int seed = 0,
  List<String>? names,
}) {
  final outs = <ChainOutput>[];
  final tasks = <({int subject, Iterator<EvalRequest> it})>[];
  for (var c = 0; c < chains; c++) {
    final rng = math.Random(chainSeed(seed, 0, c));
    final out = ChainOutput();
    outs.add(out);
    tasks.add((
      subject: 0,
      it: sampler
          .chain(
            init: (_) => Float64List.fromList([
              for (final v in y0) v + jitter * (2 * rng.nextDouble() - 1),
            ]),
            warmup: warmup,
            draws: draws,
            random: rng,
            out: out,
          )
          .iterator,
    ));
  }
  driveChains(tasks, (_, points) {
    final v = Float64List(points.length);
    final g = <Float64List>[];
    for (var j = 0; j < points.length; j++) {
      final u = potential(points[j]);
      v[j] = u.isNaN ? double.infinity : u;
      g.add(gradient(points[j]));
    }
    return (v, g);
  });
  return PosteriorDraws(
    names: names ?? [for (var i = 0; i < y0.length; i++) 'y$i'],
    chains: outs,
    toConstrained: (y) => Float64List.fromList(y),
  );
}

/// Advances every task's generator in lock step, answering all pending
/// requests with one [evaluate] call per round.
void driveChains(
  List<({int subject, Iterator<EvalRequest> it})> tasks,
  (Float64List, List<Float64List>) Function(
    List<int> subjectOf,
    List<Float64List> points,
  )
  evaluate,
) {
  for (final _ in driveChainPasses(tasks, (subjectOf, points, o) sync* {
    o.value = evaluate(subjectOf, points);
  })) {}
}

/// [driveChains] as a program: [evaluate] is a program too. With
/// [streams] > 1 the tasks are split into that many [Concurrent] streams
/// (each in lock step).
Iterable<Step> driveChainPasses(
  List<({int subject, Iterator<EvalRequest> it})> tasks,
  Iterable<Step> Function(
    List<int> subjectOf,
    List<Float64List> points,
    Out<(Float64List, List<Float64List>)> result,
  )
  evaluate, {
  int streams = 1,
}) sync* {
  if (streams > 1 && tasks.length > 1) {
    yield Concurrent([
      for (final group in splitTasks(tasks.length, streams))
        _driveChains([for (final t in group) tasks[t]], evaluate),
    ]);
  } else {
    yield* _driveChains(tasks, evaluate);
  }
}

Iterable<Step> _driveChains(
  List<({int subject, Iterator<EvalRequest> it})> tasks,
  Iterable<Step> Function(
    List<int> subjectOf,
    List<Float64List> points,
    Out<(Float64List, List<Float64List>)> result,
  )
  evaluate,
) sync* {
  final evaluated = Out<(Float64List, List<Float64List>)>();
  var active = [
    for (final t in tasks)
      if (t.it.moveNext()) t,
  ];
  while (active.isNotEmpty) {
    final subjectOf = <int>[], points = <Float64List>[];
    for (final t in active) {
      for (final p in t.it.current.points) {
        subjectOf.add(t.subject);
        points.add(p);
      }
    }
    yield* evaluate(subjectOf, points, evaluated);
    final (values, grads) = evaluated.value;
    var j = 0;
    for (final t in active) {
      final r = t.it.current;
      final n = r.points.length;
      r.values = Float64List.sublistView(values, j, j + n);
      r.gradients = grads.sublist(j, j + n);
      j += n;
    }
    active = [
      for (final t in active)
        if (t.it.moveNext()) t,
    ];
  }
}

/// A per-chain seed from the run seed, subject and chain (`math.Random`
/// scrambles its seed, so nearby seeds give unrelated streams). Kept below
/// 2^53 so it is the same on the web.
int chainSeed(int seed, int subject, int chain) =>
    ((seed % 0x100000) * 0x100000 + subject * 4099 + chain) % 0x20000000000000;
