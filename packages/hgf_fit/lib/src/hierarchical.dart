import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'fit.dart';
import 'model.dart';
import 'pass.dart';
import 'prior.dart';
import 'sample/nuts.dart';
import 'sample/sample.dart';
import 'transform.dart';

const double _logSqrt2Pi = 0.9189385332046727;

/// The group-level (population) distribution of a pooled parameter.
///
/// Each subject's value of the parameter, on the parameter's unconstrained
/// scale `y = T(x)` (its `Parameter.transform`), is drawn from `N(μ, σ²)`,
/// with the hyperpriors [mean] on `μ` and [sd] on `σ`. So the subjects'
/// values are normal for an unbounded parameter (identity transform),
/// log-normal for a positive one (log transform, e.g. a `HalfNormal` or
/// `LogNormal` prior on the parameter), logit-normal for one on an interval.
///
/// pyhgf's multilevel notebook, as hgf_fit populations:
///
/// ```dart
/// // volatility ~ N(μ, σ), μ ~ N(−5, 5), σ ~ HalfNormal(10)
/// 'omega2': Population(mean: Normal(-5, 5), sd: HalfNormal(10)),
/// // inverse temperature ~ LogNormal(μ, σ), μ ~ N(0, 2), σ ~ HalfNormal(2)
/// 'inverseTemperature': Population(mean: Normal(0, 2), sd: HalfNormal(2)),
/// ```
final class Population {
  const Population({
    this.mean = const Normal(0, 10),
    this.sd = const HalfNormal(5),
  });

  /// Prior on the population mean `μ` (unconstrained scale of the
  /// parameter).
  final Prior mean;

  /// Prior on the population standard deviation `σ`; its support must be
  /// positive.
  final Prior sd;

  @override
  String toString() => 'Population(μ ~ $mean, σ ~ $sd)';
}

/// What a hierarchical model does with one of the subject model's
/// parameters.
enum _Role { independent, pooled, shared }

/// A hierarchical (multilevel) model over many subjects, built on a
/// subject-level [HgfModel]:
///
/// - **pooled** parameters ([pooled]) vary by subject around a group-level
///   [Population] whose mean and standard deviation are estimated too
///   (the subject-level prior of such a parameter is replaced; its
///   transform is kept and defines the population's scale);
/// - **shared** parameters ([shared]) take one value for all subjects,
///   with the parameter's own prior;
/// - the others are **independent**: one value per subject, each with the
///   parameter's own prior, as in `fitMap` and `sampleNuts`.
///
/// Unlike those, the posterior doesn't factor over subjects: the group
/// parameters couple them. [sampleHierarchical] samples it jointly by NUTS,
/// every subject of every chain in one batched pass per leapfrog step.
///
/// The joint parameter vector (unconstrained) is laid out as: for each
/// pooled parameter (in the subject model's order) `μ` and `σ` (each on its
/// hyperprior's unconstrained scale), then the shared parameters, then per
/// subject its pooled and independent parameters in model order. With the
/// default non-centred parametrisation ([centered] false) a pooled entry is
/// the standardised `z = (y − μ) / σ`, which avoids the funnel between `σ`
/// and the subjects' values when the data say little about each subject;
/// centred, it is `y` itself (better when every subject is well
/// identified).
final class HierarchicalModel {
  HierarchicalModel(
    this.model, {
    Map<String, Population> pooled = const {},
    Iterable<String> shared = const [],
    this.centered = false,
  }) : pooled = Map.unmodifiable(pooled),
       shared = Set.unmodifiable(shared) {
    final names = model.names;
    for (final n in [...this.pooled.keys, ...this.shared]) {
      if (!names.contains(n)) {
        throw ArgumentError('The model has no parameter "$n"');
      }
    }
    for (final n in this.pooled.keys) {
      if (this.shared.contains(n)) {
        throw ArgumentError('"$n" cannot be both pooled and shared');
      }
      final sd = this.pooled[n]!.sd;
      if (sd.lower < 0) {
        throw ArgumentError(
          'The population sd prior of "$n" ($sd) must have positive support',
        );
      }
    }
    for (var i = 0; i < model.dimension; i++) {
      final n = names[i];
      if (this.pooled.containsKey(n)) {
        _role.add(_Role.pooled);
        _slot.add(_pooledIdx.length);
        _pooledIdx.add(i);
      } else if (this.shared.contains(n)) {
        _role.add(_Role.shared);
        _slot.add(_sharedIdx.length);
        _sharedIdx.add(i);
        continue;
      } else {
        _role.add(_Role.independent);
        _slot.add(-1);
      }
      _localIdx.add(i);
    }
    _local = List.filled(model.dimension, -1);
    for (var l = 0; l < _localIdx.length; l++) {
      _local[_localIdx[l]] = l;
    }
    _includePrior = [for (final r in _role) r == _Role.independent];
    _pops = [for (final i in _pooledIdx) this.pooled[names[i]]!];
    _muT = [for (final p in _pops) p.mean.defaultTransform];
    _sdT = [for (final p in _pops) p.sd.defaultTransform];
  }

  /// The subject-level model.
  final HgfModel model;

  /// The pooled parameters by name, with their populations.
  final Map<String, Population> pooled;

  /// The parameters shared by all subjects.
  final Set<String> shared;

  /// Centred (`y`) rather than non-centred (`z`) subject entries for
  /// pooled parameters.
  final bool centered;

  final List<_Role> _role = [];
  final List<int> _slot = []; // pooled or shared index, per parameter
  final List<int> _pooledIdx = [], _sharedIdx = [], _localIdx = [];
  late final List<int> _local; // local index per parameter (-1: shared)
  late final List<bool> _includePrior;
  late final List<Population> _pops;
  late final List<Transform> _muT, _sdT;

  int get _nGroup => 2 * _pooledIdx.length + _sharedIdx.length;
  int get _nLocal => _localIdx.length;

  /// Length of the joint parameter vector for [nSubjects] subjects.
  int dimension(int nSubjects) => _nGroup + nSubjects * _nLocal;

  /// Names of the joint parameters: `name_mu` and `name_sigma` for each
  /// pooled parameter, shared parameters by name, then `name[id]` per
  /// subject (`id` is `Subject.id`, or the subject's index).
  List<String> names(List<Subject> subjects) {
    final pn = model.names;
    return [
      for (final i in _pooledIdx) ...['${pn[i]}_mu', '${pn[i]}_sigma'],
      for (final i in _sharedIdx) pn[i],
      for (var s = 0; s < subjects.length; s++)
        for (final i in _localIdx) '${pn[i]}[${subjects[s].id ?? s}]',
    ];
  }

  /// Index in the joint vector of the group mean (`'mu'`) or sd
  /// (`'sigma'`) of pooled parameter [name], of shared parameter [name]
  /// (no [subject]), or of [subject]'s value of [name].
  int indexOf(String name, {int? subject, String? group}) {
    final i = model.names.indexOf(name);
    if (i < 0) throw ArgumentError('The model has no parameter "$name"');
    if (group != null) {
      if (_role[i] != _Role.pooled) {
        throw ArgumentError('"$name" isn\'t pooled');
      }
      return 2 * _slot[i] +
          switch (group) {
            'mu' => 0,
            'sigma' => 1,
            _ => throw ArgumentError.value(group, 'group', "'mu' or 'sigma'"),
          };
    }
    if (_role[i] == _Role.shared) return 2 * _pooledIdx.length + _slot[i];
    if (subject == null) {
      throw ArgumentError('"$name" varies by subject: pass subject or group');
    }
    return _nGroup + subject * _nLocal + _local[i];
  }

  /// The joint vector [theta] (unconstrained) on the natural scale: group
  /// means as they are (they live on the parameter's unconstrained scale),
  /// group sds, shared and subject values on the parameters' natural scale.
  Float64List toConstrained(Float64List theta) {
    final n = _subjectsIn(theta);
    final out = Float64List(theta.length);
    final mu = Float64List(_pops.length), sd = Float64List(_pops.length);
    for (var p = 0; p < _pops.length; p++) {
      mu[p] = out[2 * p] = _muT[p].toConstrained(theta[2 * p]);
      sd[p] = out[2 * p + 1] = _sdT[p].toConstrained(theta[2 * p + 1]);
    }
    final o = 2 * _pops.length;
    for (var q = 0; q < _sharedIdx.length; q++) {
      out[o + q] = model.parameters[_sharedIdx[q]].transform.toConstrained(
        theta[o + q],
      );
    }
    for (var s = 0; s < n; s++) {
      for (var l = 0; l < _nLocal; l++) {
        final i = _localIdx[l], e = _nGroup + s * _nLocal + l;
        out[e] = model.parameters[i].transform.toConstrained(
          _subjectY(theta, i, e, mu, sd),
        );
      }
    }
    return out;
  }

  int _subjectsIn(Float64List theta) {
    final r = theta.length - _nGroup;
    if (r < 0 || (_nLocal == 0 ? r != 0 : r % _nLocal != 0)) {
      throw ArgumentError(
        'A joint vector has $_nGroup + nSubjects x $_nLocal entries',
      );
    }
    return _nLocal == 0 ? 0 : r ~/ _nLocal;
  }

  /// Subject value `y` (unconstrained) of local parameter [i] stored at
  /// joint index [e].
  double _subjectY(
    Float64List theta,
    int i,
    int e,
    Float64List mu,
    Float64List sd,
  ) {
    final v = theta[e];
    if (_role[i] != _Role.pooled || centered) return v;
    final p = _slot[i];
    return mu[p] + sd[p] * v;
  }

  /// A joint starting point (unconstrained) for [nSubjects] subjects.
  ///
  /// With [start] (one natural-scale vector per subject, e.g. MAP
  /// estimates), each population starts at the mean and sd of the
  /// subjects' values (the sd at least 0.1) and shared parameters at their
  /// mean; otherwise every parameter at its starting value and population
  /// sds at 1.
  Float64List initialPoint(int nSubjects, {List<List<double>>? start}) {
    if (start != null && start.length != nSubjects) {
      throw ArgumentError('start must have one vector per subject');
    }
    final k = model.dimension;
    final ys = [
      for (var s = 0; s < nSubjects; s++)
        model.toUnconstrained(
          start == null ? model.start() : _checked(start[s], k),
        ),
    ];
    final theta = Float64List(dimension(nSubjects));
    final mu = Float64List(_pops.length), sd = Float64List(_pops.length);
    double meanOf(int i) => nSubjects == 0
        ? model.toUnconstrained(model.start())[i]
        : ys.fold(0.0, (a, y) => a + y[i]) / nSubjects;
    for (var p = 0; p < _pops.length; p++) {
      final i = _pooledIdx[p];
      mu[p] = meanOf(i);
      var v = 1.0;
      if (start != null && nSubjects > 1) {
        v = math.sqrt(
          ys.fold(0.0, (a, y) => a + (y[i] - mu[p]) * (y[i] - mu[p])) /
              (nSubjects - 1),
        );
        v = math.max(v, 0.1);
      }
      sd[p] = v;
      theta[2 * p] = _muT[p].toUnconstrained(mu[p]);
      theta[2 * p + 1] = _sdT[p].toUnconstrained(v);
    }
    for (var q = 0; q < _sharedIdx.length; q++) {
      theta[2 * _pops.length + q] = meanOf(_sharedIdx[q]);
    }
    for (var s = 0; s < nSubjects; s++) {
      for (var l = 0; l < _nLocal; l++) {
        final i = _localIdx[l];
        var v = ys[s][i];
        if (_role[i] == _Role.pooled && !centered) {
          v = (v - mu[_slot[i]]) / sd[_slot[i]];
        }
        theta[_nGroup + s * _nLocal + l] = v;
      }
    }
    return theta;
  }

  static List<double> _checked(List<double> x, int k) {
    if (x.length != k) throw ArgumentError('Start vectors must have $k values');
    return x;
  }

  /// The objective (negative log joint density on the unconstrained scale,
  /// Jacobians included) of each joint vector in [thetas], for [subjects],
  /// with its gradient, in one batched pass of `thetas.length x
  /// subjects.length` elements. Gradients are exact where the subject model
  /// allows it, otherwise forward differences per subject.
  (Float64List values, List<Float64List> gradients) potential(
    List<Subject> subjects,
    List<Float64List> thetas, {
    Gradients? gradients,
    double fdStep = 1e-5,
  }) => _objective(subjects, gradients, fdStep).evaluate(thetas);

  HierarchicalObjective _objective(
    List<Subject> subjects,
    Gradients? gradients,
    double fdStep,
  ) => HierarchicalObjective._(
    this,
    subjects,
    BatchedObjective(
      model,
      subjects,
      fdStep: fdStep,
      jacobian: true,
      includePrior: _includePrior,
      gradients:
          gradients ??
          (model.hasExactGradients ? Gradients.exact : Gradients.forward),
    ),
  );

  @override
  String toString() =>
      'HierarchicalModel(${centered ? 'centred' : 'non-centred'}, '
      'pooled: $pooled, shared: $shared)';
}

/// [HierarchicalModel]'s joint objective for one set of subjects; counts
/// the batched passes it runs.
final class HierarchicalObjective {
  HierarchicalObjective._(this.model, this.subjects, this.subjectObjective);

  final HierarchicalModel model;
  final List<Subject> subjects;

  /// Evaluates the subjects' likelihoods (with the independent parameters'
  /// priors).
  final BatchedObjective subjectObjective;

  int get passes => subjectObjective.passes;
  int get elements => subjectObjective.elements;

  /// The objective `U` and its gradient at each joint vector of [thetas].
  /// `U` is `+∞` (and the gradient NaN) where the density isn't finite.
  (Float64List values, List<Float64List> gradients) evaluate(
    List<Float64List> thetas,
  ) => runInline((o) => evaluatePasses(thetas, o));

  /// [evaluate] as a program.
  Iterable<Step> evaluatePasses(
    List<Float64List> thetas,
    Out<(Float64List, List<Float64List>)> result,
  ) sync* {
    final hm = model, m = hm.model;
    final n = subjects.length, k = m.dimension, nP = thetas.length;
    final dim = hm.dimension(n), nG = hm._nGroup, nL = hm._nLocal;
    final nPop = hm._pops.length;
    final subjectOf = <int>[];
    final points = <Float64List>[];
    final mus = <Float64List>[], sds = <Float64List>[];
    for (final theta in thetas) {
      if (theta.length != dim) {
        throw ArgumentError('Joint vectors must have $dim entries');
      }
      final mu = Float64List(nPop), sd = Float64List(nPop);
      for (var p = 0; p < nPop; p++) {
        mu[p] = hm._muT[p].toConstrained(theta[2 * p]);
        sd[p] = hm._sdT[p].toConstrained(theta[2 * p + 1]);
      }
      mus.add(mu);
      sds.add(sd);
      for (var s = 0; s < n; s++) {
        final y = Float64List(k);
        for (var i = 0; i < k; i++) {
          y[i] = switch (hm._role[i]) {
            _Role.shared => theta[2 * nPop + hm._slot[i]],
            _ => hm._subjectY(theta, i, nG + s * nL + hm._local[i], mu, sd),
          };
        }
        subjectOf.add(s);
        points.add(y);
      }
    }
    final subjectValues = Out<(Float64List, List<Float64List?>)>()
      ..value = (Float64List(0), const <Float64List?>[]);
    if (n > 0) {
      yield* subjectObjective.evaluatePasses(
        subjectOf,
        points,
        List.filled(points.length, true),
        subjectValues,
      );
    }
    final (v, g) = subjectValues.value;

    final values = Float64List(nP);
    final grads = <Float64List>[];
    final dMu = Float64List(nPop), dSd = Float64List(nPop);
    for (var j = 0; j < nP; j++) {
      final theta = thetas[j], mu = mus[j], sd = sds[j];
      final grad = Float64List(dim);
      dMu.fillRange(0, nPop, 0);
      dSd.fillRange(0, nPop, 0);
      var u = 0.0;
      for (var s = 0; s < n; s++) {
        final e = j * n + s;
        u += v[e];
        final gs = g[e]!;
        for (var i = 0; i < k; i++) {
          final gi = gs[i];
          switch (hm._role[i]) {
            case _Role.shared:
              grad[2 * nPop + hm._slot[i]] += gi;
            case _Role.independent:
              grad[nG + s * nL + hm._local[i]] = gi;
            case _Role.pooled:
              final p = hm._slot[i], idx = nG + s * nL + hm._local[i];
              final w = theta[idx];
              if (hm.centered) {
                final r = (w - mu[p]) / sd[p];
                u += 0.5 * r * r + math.log(sd[p]) + _logSqrt2Pi;
                grad[idx] = gi + r / sd[p];
                dMu[p] -= r / sd[p];
                dSd[p] += (1 - r * r) / sd[p];
              } else {
                u += 0.5 * w * w + _logSqrt2Pi;
                grad[idx] = gi * sd[p] + w;
                dMu[p] += gi;
                dSd[p] += gi * w;
              }
          }
        }
      }
      for (var p = 0; p < nPop; p++) {
        final pop = hm._pops[p];
        final eMu = theta[2 * p], eSd = theta[2 * p + 1];
        final tMu = hm._muT[p], tSd = hm._sdT[p];
        u -=
            pop.mean.logDensity(mu[p]) +
            tMu.logAbsJacobian(eMu) +
            pop.sd.logDensity(sd[p]) +
            tSd.logAbsJacobian(eSd);
        grad[2 * p] =
            (dMu[p] - pop.mean.dLogDensity(mu[p])) * tMu.dToConstrained(eMu) -
            tMu.dLogAbsJacobian(eMu);
        grad[2 * p + 1] =
            (dSd[p] - pop.sd.dLogDensity(sd[p])) * tSd.dToConstrained(eSd) -
            tSd.dLogAbsJacobian(eSd);
      }
      for (var q = 0; q < hm._sharedIdx.length; q++) {
        final par = m.parameters[hm._sharedIdx[q]], t = par.transform;
        final y = theta[2 * nPop + q], x = t.toConstrained(y);
        u -= par.prior.logDensity(x) + t.logAbsJacobian(y);
        grad[2 * nPop + q] +=
            -par.prior.dLogDensity(x) * t.dToConstrained(y) -
            t.dLogAbsJacobian(y);
      }
      if (u.isFinite) {
        values[j] = u;
      } else {
        values[j] = double.infinity;
        grad.fillRange(0, dim, double.nan);
      }
      grads.add(grad);
    }
    result.value = (values, grads);
  }
}

/// The joint posterior draws of a [sampleHierarchical] call.
final class HierarchicalFit {
  HierarchicalFit(
    this.model,
    this.subjects,
    this.draws,
    this.passes,
    this.elements,
  );

  final HierarchicalModel model;
  final List<Subject> subjects;

  /// Draws of every joint parameter (natural scale: see
  /// [HierarchicalModel.toConstrained]), with sampler statistics and
  /// diagnostics; names as [HierarchicalModel.names].
  final PosteriorDraws draws;

  /// Batched passes over the data, and batch elements (one subject's filter
  /// each) evaluated.
  final int passes, elements;

  int get divergences => draws.divergences;

  /// Summary of the group mean (`group: 'mu'`) or sd (`'sigma'`) of a
  /// pooled parameter, of a shared parameter, or of one subject's value.
  ParameterSummary summary(String name, {int? subject, String? group}) =>
      draws.summary()[model.indexOf(name, subject: subject, group: group)];

  /// Whether every joint parameter has R̂ ≤ [maxRhat] and bulk and tail ESS
  /// of at least [minEss], with no divergences.
  bool converged({double maxRhat = 1.01, double minEss = 400}) =>
      draws.converged(maxRhat: maxRhat, minEss: minEss);

  /// The draws as an ArviZ `from_dict` payload: `name_mu`, `name_sigma`
  /// and shared parameters as `[chain][draw]`, subject-level parameters as
  /// `[chain][draw][subject]` (dims and coords included), and the sampler
  /// statistics.
  Map<String, Object> toArviz() {
    final hm = model, pn = hm.model.names, n = subjects.length;
    final nC = draws.nChains, nD = draws.nDraws;
    List<Object> byChain(int idx) => draws.parameter(idx);
    final posterior = <String, Object>{
      for (final i in hm._pooledIdx) ...{
        '${pn[i]}_mu': byChain(hm.indexOf(pn[i], group: 'mu')),
        '${pn[i]}_sigma': byChain(hm.indexOf(pn[i], group: 'sigma')),
      },
      for (final i in hm._sharedIdx) pn[i]: byChain(hm.indexOf(pn[i])),
    };
    for (final i in hm._localIdx) {
      final per = [
        for (var s = 0; s < n; s++)
          draws.parameter(hm.indexOf(pn[i], subject: s)),
      ];
      posterior[pn[i]] = [
        for (var c = 0; c < nC; c++)
          [
            for (var d = 0; d < nD; d++)
              [for (var s = 0; s < n; s++) per[s][c][d]],
          ],
      ];
    }
    return {
      'posterior': posterior,
      'sample_stats': draws.toArviz()['sample_stats']!,
      'dims': {
        for (final i in hm._localIdx) pn[i]: ['subject'],
      },
      'coords': {
        'subject': [for (var s = 0; s < n; s++) '${subjects[s].id ?? s}'],
      },
    };
  }

  /// [toArviz] as JSON (see `NutsFit.toJson` for loading it).
  String toJson() => jsonEncode(toArviz());

  @override
  String toString() => draws.toString();
}

/// Joint posterior draws of a [HierarchicalModel] for [subjects] by NUTS:
/// group means and sds, shared parameters and every subject's parameters
/// in one chain, [chains] chains advanced in lock step. Each leapfrog step
/// of every chain is one batched pass over all subjects
/// (`chains x subjects` elements).
///
/// Chains start at [HierarchicalModel.initialPoint] (from [start], natural
/// scale per subject, e.g. MAP estimates, if given) moved by a uniform
/// `±jitter` on the unconstrained scale, redrawn until the density is
/// finite. Gradients are exact where the subject model allows it, otherwise
/// forward differences (see `sampleNuts`).
HierarchicalFit sampleHierarchical(
  HierarchicalModel model,
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
}) => runInline(
  (o) => _sampleHierarchical(
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
    1,
    o,
  ),
);

/// [sampleHierarchical] with each leapfrog pass split across the pool's
/// isolates.
extension SampleHierarchicalOnWorkers on PassRunner {
  /// [sampleHierarchical] on this pool; the same draws, bit for bit.
  Future<HierarchicalFit> sampleHierarchical(
    HierarchicalModel model,
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
  }) => run(
    (o) => _sampleHierarchical(
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
      streams,
      o,
    ),
  );
}

Iterable<Step> _sampleHierarchical(
  HierarchicalModel model,
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
  int streams,
  Out<HierarchicalFit> result,
) sync* {
  if (chains < 1) throw ArgumentError.value(chains, 'chains');
  final objective = model._objective(subjects, gradients, fdStep);
  final theta0 = model.initialPoint(subjects.length, start: start);
  final outs = <ChainOutput>[];
  final ensembles = nutsEnsembles(
    sampler,
    theta0.length,
    warmup: warmup,
    draws: draws,
    chains: chains,
    streams: streams,
  );
  for (var c = 0; c < chains; c++) {
    final rng = math.Random(chainSeed(seed, 0, c));
    final out = ChainOutput();
    outs.add(out);
    ensembles.add(
      c,
      init: (_) => Float64List.fromList([
        for (final v in theta0) v + jitter * (2 * rng.nextDouble() - 1),
      ]),
      random: rng,
      out: out,
    );
  }
  final dim = theta0.length;
  yield* ensembles.drive((_, y, o) sync* {
    final n = y.length ~/ dim;
    final joint = Out<(Float64List, List<Float64List>)>();
    yield* objective.evaluatePasses([
      for (var j = 0; j < n; j++)
        Float64List.sublistView(y, j * dim, (j + 1) * dim),
    ], joint);
    final (v, g) = joint.value;
    final flat = Float64List(n * dim);
    for (var j = 0; j < n; j++) {
      flat.setRange(j * dim, (j + 1) * dim, g[j]);
    }
    o.value = (v, flat);
  });
  result.value = HierarchicalFit(
    model,
    subjects,
    PosteriorDraws(
      names: model.names(subjects),
      chains: outs,
      toConstrained: model.toConstrained,
    ),
    objective.passes,
    objective.elements,
  );
}
