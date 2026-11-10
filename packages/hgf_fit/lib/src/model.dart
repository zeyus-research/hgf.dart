import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'parameter.dart';
import 'prior.dart';
import 'response_model.dart';

/// One subject's data: the observations the network filters and, for
/// response models that score behaviour, the subject's responses.
final class Subject {
  /// [inputs] is laid out as for `Network.inputData` (row-major
  /// `nTime x nInputColumns`: one column per input node, `dimension` for an
  /// ef-state input and one per category for a categorical one);
  /// [observed] has one flag per trial and input node (`0` marks a missing
  /// observation; the shape of [inputs] when every input is scalar);
  /// [timeSteps] has one entry per trial; [responses] has at least one per
  /// trial (NaN marks a missed response).
  Subject({
    required this.inputs,
    this.responses,
    this.observed,
    this.timeSteps,
    this.id,
  });

  final Float64List inputs;
  final Float64List? responses;
  final Float64List? observed;
  final Float64List? timeSteps;

  /// A label for reports.
  final Object? id;

  @override
  String toString() => 'Subject(${id ?? '#$hashCode'})';
}

/// A generative model to fit: a [network] (the perceptual model, compiled
/// once and copied into batches), a [response] model, and the free
/// [parameters] estimated per subject.
///
/// The objective for a subject with parameters `x` is the negative log
/// posterior `−log p(responses | x) − Σ log prior(xᵢ)`. Batched evaluation
/// ([negLogPosterior]) runs many (subject, parameter) pairs through one
/// `BatchNetwork` pass; the optimisers in `fitMap` use it to fit every
/// subject at once. Gradients are exact where the model allows it
/// ([hasExactGradients]: forward-mode tangents through the filter, one
/// direction per parameter), finite differences in the same pass otherwise.
final class HgfModel {
  HgfModel({
    required Network network,
    required this.response,
    required List<Parameter> parameters,
    this.blockSize = 256,
  }) : network = network..build(),
       parameters = List.unmodifiable(parameters),
       _batches = {},
       _responseNames = response.parameters.keys.toList(growable: false),
       _responseDefaults = Float64List.fromList(
         response.parameters.values.toList(),
       ) {
    final names = <String>{};
    for (final p in parameters) {
      if (!names.add(p.name)) {
        throw ArgumentError('Duplicate parameter name "${p.name}"');
      }
      for (final t in p.targets) {
        switch (t) {
          case NodeTarget(:final node):
            _checkNode(node);
          case ValueCouplingTarget(:final parent, :final child):
            _checkNode(parent);
            _checkNode(child);
            if (!network.valueChildrenOf(parent).contains(child)) {
              throw ArgumentError('$child is not a value child of $parent');
            }
          case VolatilityCouplingTarget(:final parent, :final child):
            _checkNode(parent);
            _checkNode(child);
            if (!network.volatilityChildrenOf(parent).contains(child)) {
              throw ArgumentError(
                '$child is not a volatility child of $parent',
              );
            }
          case ResponseTarget(:final name):
            if (!_responseNames.contains(name)) {
              throw ArgumentError(
                '${response.runtimeType} has no parameter "$name" '
                '(it has ${_responseNames.join(', ')})',
              );
            }
        }
      }
    }
    response.prepare(network);
    // Fail early on node kinds the batched engine doesn't support.
    BatchNetwork(network, 1);
  }

  HgfModel._withParameters(HgfModel base, this.parameters)
    : network = base.network,
      response = base.response,
      blockSize = base.blockSize,
      _responseNames = base._responseNames,
      _responseDefaults = base._responseDefaults,
      _batches = base._batches;

  /// This model with the priors of the parameters named in [priors]
  /// replaced (transforms and starting values kept). The copy shares the
  /// compiled network and its batches, so it is cheap to make: for prior
  /// sensitivity checks, or empirical Bayes, which refits with updated
  /// group-level priors.
  HgfModel withPriors(Map<String, Prior> priors) {
    for (final n in priors.keys) {
      if (!names.contains(n)) {
        throw ArgumentError('The model has no parameter "$n"');
      }
    }
    return HgfModel._withParameters(this, [
      for (final p in parameters)
        switch (priors[p.name]) {
          null => p,
          final prior => Parameter(
            p.name,
            targets: p.targets,
            prior: prior,
            transform: p.transform,
            initial: p.initial != null && prior.contains(p.initial!)
                ? p.initial
                : null,
          ),
        },
    ]);
  }

  /// The perceptual model.
  final Network network;

  /// How responses arise from beliefs.
  final ResponseModel response;

  /// The free parameters, in the order of parameter vectors.
  final List<Parameter> parameters;

  /// `BatchNetwork.blockSize` for the batches.
  final int blockSize;

  final List<String> _responseNames;
  final Float64List _responseDefaults;
  final Map<(int, bool), BatchNetwork> _batches;

  /// Whether [negLogPosterior] can return exact gradients: the response
  /// model supports tangents and every parameter targets something the
  /// tangents reach (a field in `BatchNetwork.tangentFields`, a coupling
  /// strength or a response parameter).
  late final bool hasExactGradients =
      response.supportsGradients &&
      parameters.every(
        (p) => p.targets.every(
          (t) => switch (t) {
            NodeTarget(:final field) => BatchNetwork.tangentFields.contains(
              field,
            ),
            _ => true,
          },
        ),
      );

  /// Number of free parameters.
  int get dimension => parameters.length;

  /// The parameter names, in order.
  List<String> get names => [for (final p in parameters) p.name];

  void _checkNode(int node) {
    if (node < 0 || node >= network.nNodes) {
      throw ArgumentError.value(node, 'node', 'No such node');
    }
  }

  /// The starting point on the constrained scale (each parameter's
  /// `initial` or prior typical value).
  Float64List start() =>
      Float64List.fromList([for (final p in parameters) p.start]);

  /// [x] (constrained) mapped to the unconstrained scale.
  Float64List toUnconstrained(List<double> x) => Float64List.fromList([
    for (var i = 0; i < dimension; i++)
      parameters[i].transform.toUnconstrained(x[i]),
  ]);

  /// [y] (unconstrained) mapped to the constrained scale.
  Float64List toConstrained(List<double> y) => Float64List.fromList([
    for (var i = 0; i < dimension; i++)
      parameters[i].transform.toConstrained(y[i]),
  ]);

  /// `Σ log prior(xᵢ)` on the constrained scale.
  double logPrior(List<double> x) {
    var s = 0.0;
    for (var i = 0; i < dimension; i++) {
      s += parameters[i].prior.logDensity(x[i]);
    }
    return s;
  }

  /// [x] as a name → value map.
  Map<String, double> named(List<double> x) => {
    for (var i = 0; i < dimension; i++) parameters[i].name: x[i],
  };

  /// A parameter vector drawn from the priors.
  Float64List samplePrior(math.Random rng) =>
      Float64List.fromList([for (final p in parameters) p.prior.sample(rng)]);

  /// Negative log posterior of each element, in one batched pass per group
  /// of subjects sharing a trial count and time steps.
  ///
  /// Element `e` is subject `subjects[subjectOf[e]]` with unconstrained
  /// parameters `y[e * dimension + i]`. [out] receives
  /// `−log p(responses | x) − log p(x)`, and with [jacobian] also
  /// `− Σ log |dxᵢ/dyᵢ|` (the density on the unconstrained scale, for
  /// samplers). Non-finite values become `+∞`. [negLogLikelihood], if given,
  /// receives the first term alone.
  ///
  /// With [gradient] (length `nE * dimension`, needs [hasExactGradients]),
  /// `gradient[e * dimension + i]` receives the exact derivative of `out[e]`
  /// with respect to `y[e * dimension + i]`, from forward-mode tangents
  /// through the filter (NaN where `out[e]` is `+∞`). A pass with gradients
  /// costs more per element than one without, but far less than the
  /// `dimension + 1` elements of a finite-difference gradient.
  ///
  /// [includePrior] (one entry per parameter) leaves out the prior and
  /// Jacobian terms of the parameters marked `false`: a hierarchical model
  /// puts its own group-level terms in their place.
  void negLogPosterior(
    List<Subject> subjects,
    Int32List subjectOf,
    Float64List y,
    Float64List out, {
    bool jacobian = false,
    Float64List? negLogLikelihood,
    Float64List? gradient,
    List<bool>? includePrior,
  }) {
    final k = dimension, nE = subjectOf.length;
    if (y.length != nE * k) {
      throw ArgumentError('y must have ${nE * k} entries ($nE x $k)');
    }
    if (out.length != nE) throw ArgumentError('out must have $nE entries');
    if (includePrior != null && includePrior.length != k) {
      throw ArgumentError('includePrior must have $k entries');
    }
    final exact = gradient != null;
    if (exact) {
      if (!hasExactGradients) {
        throw UnsupportedError(
          'This model has no exact gradients (see hasExactGradients)',
        );
      }
      if (gradient.length != nE * k) {
        throw ArgumentError('gradient must have ${nE * k} entries');
      }
    }
    final groups = _group(subjects, subjectOf);
    final x = Float64List(nE * k);
    final dxdy = exact ? Float64List(nE * k) : null;
    for (var e = 0; e < nE; e++) {
      for (var i = 0; i < k; i++) {
        final t = parameters[i].transform, yi = y[e * k + i];
        x[e * k + i] = t.toConstrained(yi);
        dxdy?[e * k + i] = t.dToConstrained(yi);
      }
    }
    for (final g in groups) {
      final b = g.elements.length;
      final batch = _batch(b, tangents: exact);
      final (params, paramTangents) = _applyParameters(
        batch,
        g.elements,
        x,
        dxdy,
      );
      final (data, observed, perElement) = _data(subjects, subjectOf, g);
      final bound = _BoundResponse(response, params, _responseNames.length, [
        for (final e in g.elements) subjects[subjectOf[e]].responses,
      ], paramTangents);
      final grad = exact ? Float64List(b * k) : null;
      final totals = batch.surprise(
        data,
        bound,
        observed: observed,
        timeSteps: g.timeSteps,
        perElement: perElement,
        gradient: grad,
      );
      for (var j = 0; j < b; j++) {
        final e = g.elements[j];
        final nll = totals[j];
        negLogLikelihood?[e] = nll;
        _complete(
          e,
          nll,
          x,
          y,
          dxdy,
          out,
          grad,
          j,
          gradient,
          jacobian,
          includePrior,
        );
      }
    }
  }

  /// Element [e]'s posterior from its likelihood [nll]: the prior and
  /// Jacobian terms, and with [gradient] their derivatives added to the
  /// likelihood's (`likelihoodGradient[j * k + i]`, with respect to y).
  @pragma('vm:prefer-inline')
  void _complete(
    int e,
    double nll,
    Float64List x,
    Float64List y,
    Float64List? dxdy,
    Float64List out,
    Float64List? likelihoodGradient,
    int j,
    Float64List? gradient,
    bool jacobian,
    List<bool>? includePrior,
  ) {
    final k = dimension;
    var v = nll;
    for (var i = 0; i < k; i++) {
      if (includePrior != null && !includePrior[i]) continue;
      final p = parameters[i];
      v -= p.prior.logDensity(x[e * k + i]);
      if (jacobian) v -= p.transform.logAbsJacobian(y[e * k + i]);
    }
    final value = v.isNaN ? double.infinity : v;
    out[e] = value;
    if (gradient == null) return;
    for (var i = 0; i < k; i++) {
      final p = parameters[i], ei = e * k + i;
      var gi = likelihoodGradient![j * k + i];
      if (includePrior == null || includePrior[i]) {
        gi -= p.prior.dLogDensity(x[ei]) * dxdy![ei];
        if (jacobian) gi -= p.transform.dLogAbsJacobian(y[ei]);
      }
      gradient[ei] = value.isFinite ? gi : double.nan;
    }
  }

  /// [negLogPosterior] from likelihoods computed elsewhere (e.g. on a GPU):
  /// [negLogLikelihood] holds each element's `−log p(responses | x)` and,
  /// with [gradient], [likelihoodGradient] its derivatives with respect to
  /// `y` (`[e * dimension + i]`). Adds the prior and Jacobian terms the way
  /// [negLogPosterior] does.
  void completeNegLogPosterior(
    Float64List y,
    Float64List negLogLikelihood,
    Float64List out, {
    bool jacobian = false,
    Float64List? likelihoodGradient,
    Float64List? gradient,
    List<bool>? includePrior,
  }) {
    final k = dimension, nE = negLogLikelihood.length;
    if (y.length != nE * k || out.length != nE) {
      throw ArgumentError('y, out and negLogLikelihood disagree in size');
    }
    if ((gradient == null) != (likelihoodGradient == null)) {
      throw ArgumentError('gradient needs likelihoodGradient');
    }
    final x = Float64List(nE * k);
    final dxdy = gradient != null ? Float64List(nE * k) : null;
    for (var e = 0; e < nE; e++) {
      for (var i = 0; i < k; i++) {
        final t = parameters[i].transform, yi = y[e * k + i];
        x[e * k + i] = t.toConstrained(yi);
        dxdy?[e * k + i] = t.dToConstrained(yi);
      }
    }
    for (var e = 0; e < nE; e++) {
      _complete(
        e,
        negLogLikelihood[e],
        x,
        y,
        dxdy,
        out,
        likelihoodGradient,
        e,
        gradient,
        jacobian,
        includePrior,
      );
    }
  }

  /// The negative log posterior of one [subject] at constrained parameters
  /// [x].
  double negLogPosteriorAt(Subject subject, List<double> x) {
    final out = Float64List(1);
    negLogPosterior([subject], Int32List(1), toUnconstrained(x), out);
    return out[0];
  }

  /// Simulated responses of each subject (its inputs; responses ignored)
  /// under the constrained parameters `x[s]`.
  List<Float64List> simulate(
    List<Subject> subjects,
    List<List<double>> x,
    math.Random rng,
  ) {
    if (x.length != subjects.length) {
      throw ArgumentError('One parameter vector per subject');
    }
    final k = dimension, n = subjects.length;
    final subjectOf = Int32List.fromList([for (var s = 0; s < n; s++) s]);
    final flat = Float64List(n * k);
    for (var s = 0; s < n; s++) {
      if (x[s].length != k) throw ArgumentError('x[$s] must have $k values');
      flat.setRange(s * k, (s + 1) * k, x[s]);
    }
    final out = <Float64List>[
      for (final s in subjects) Float64List(s.inputs.length ~/ _nColumns),
    ];
    for (final g in _group(subjects, subjectOf, needResponses: false)) {
      final b = g.elements.length;
      final batch = _batch(b);
      final (params, _) = _applyParameters(batch, g.elements, flat);
      final (data, observed, perElement) = _data(subjects, subjectOf, g);
      final per = perElement ? b : 1;
      final width = _nColumns * per, flags = _nInputs * per;
      final row = Float64List(width);
      final rowObs = observed == null ? null : Float64List(flags);
      batch.reset();
      final p = _responseNames.length;
      for (var t = 0; t < g.nTime; t++) {
        row.setRange(0, width, data, t * width);
        rowObs?.setRange(0, flags, observed!, t * flags);
        batch.step(
          row,
          observed: rowObs,
          timeStep: g.timeSteps?[t] ?? 1.0,
          perElement: perElement,
        );
        for (var j = 0; j < b; j++) {
          out[subjectOf[g.elements[j]]][t] = response.sample(
            batch,
            j,
            t,
            params,
            j * p,
            rng,
          );
        }
      }
    }
    return out;
  }

  /// Input nodes (observed flags per trial).
  int get _nInputs => network.inputIdxs.length;

  /// Data columns per trial.
  int get _nColumns => network.nInputColumns;

  BatchNetwork _batch(int b, {bool tangents = false}) {
    final key = (b, tangents);
    if (_batches.length > 16 && !_batches.containsKey(key)) _batches.clear();
    return _batches[key] ??= BatchNetwork(
      network,
      b,
      blockSize: blockSize,
      tangents: tangents ? dimension : 0,
    );
  }

  /// Sets the batch's per-element node and coupling parameters from [x]
  /// (`x[e * dimension + i]` for the group's elements) and returns the
  /// response parameters, element-major.
  ///
  /// With [dxdy] (`dx/dy` per entry of [x]), the batch carries one tangent
  /// direction per parameter, seeded with `dxᵢ/dyᵢ` on each of parameter
  /// i's targets (and zero on the others, so a later parameter that sets the
  /// same target replaces it, as it does the value); the response
  /// parameters' tangents are returned too, `[(j * p + slot) * k + d]`.
  (Float64List, Float64List?) _applyParameters(
    BatchNetwork batch,
    List<int> elements,
    Float64List x, [
    Float64List? dxdy,
  ]) {
    final b = elements.length, k = dimension, p = _responseNames.length;
    final params = Float64List(b * p);
    for (var j = 0; j < b; j++) {
      params.setRange(j * p, (j + 1) * p, _responseDefaults);
    }
    final paramTangents = dxdy == null ? null : Float64List(b * p * k);
    if (dxdy != null) batch.clearTangents();
    final values = Float64List(b);
    final seed = Float64List(b), zero = Float64List(b);
    for (var i = 0; i < k; i++) {
      for (var j = 0; j < b; j++) {
        values[j] = x[elements[j] * k + i];
        if (dxdy != null) seed[j] = dxdy[elements[j] * k + i];
      }
      for (final t in parameters[i].targets) {
        switch (t) {
          case NodeTarget(:final node, :final field):
            batch.setInitial(node, field, values);
            if (dxdy != null) {
              for (var d = 0; d < k; d++) {
                batch.setInitialTangent(node, field, d, d == i ? seed : zero);
              }
            }
          case ValueCouplingTarget(:final parent, :final child):
            batch.setValueCoupling(parent, child, values);
            if (dxdy != null) {
              for (var d = 0; d < k; d++) {
                batch.setValueCouplingTangent(
                  parent,
                  child,
                  d,
                  d == i ? seed : zero,
                );
              }
            }
          case VolatilityCouplingTarget(:final parent, :final child):
            batch.setVolatilityCoupling(parent, child, values);
            if (dxdy != null) {
              for (var d = 0; d < k; d++) {
                batch.setVolatilityCouplingTangent(
                  parent,
                  child,
                  d,
                  d == i ? seed : zero,
                );
              }
            }
          case ResponseTarget(:final name):
            final slot = _responseNames.indexOf(name);
            for (var j = 0; j < b; j++) {
              params[j * p + slot] = values[j];
              if (paramTangents != null) {
                final o = (j * p + slot) * k;
                paramTangents.fillRange(o, o + k, 0.0);
                paramTangents[o + i] = seed[j];
              }
            }
        }
      }
    }
    return (params, paramTangents);
  }

  /// The group's observations: shared if every element filters the same
  /// series, otherwise laid out per element.
  (Float64List, Float64List?, bool) _data(
    List<Subject> subjects,
    Int32List subjectOf,
    _Group g,
  ) {
    final first = subjects[subjectOf[g.elements.first]];
    final shared = g.elements.every((e) {
      final s = subjects[subjectOf[e]];
      return identical(s.inputs, first.inputs) &&
          identical(s.observed, first.observed);
    });
    if (shared) return (first.inputs, first.observed, false);
    final b = g.elements.length;
    final n = g.nTime * _nColumns, nObs = g.nTime * _nInputs;
    final anyObserved = g.elements.any(
      (e) => subjects[subjectOf[e]].observed != null,
    );
    final data = Float64List(n * b);
    final observed = anyObserved ? Float64List(nObs * b) : null;
    for (var j = 0; j < b; j++) {
      final s = subjects[subjectOf[g.elements[j]]];
      final u = s.inputs, o = s.observed;
      for (var r = 0; r < n; r++) {
        data[r * b + j] = u[r];
      }
      if (observed != null) {
        for (var r = 0; r < nObs; r++) {
          observed[r * b + j] = o == null ? 1.0 : o[r];
        }
      }
    }
    return (data, observed, true);
  }

  /// Elements grouped by trial count and time steps (one batch each).
  List<_Group> _group(
    List<Subject> subjects,
    Int32List subjectOf, {
    bool needResponses = true,
  }) {
    final nCols = _nColumns;
    final checked = <int>{};
    final groups = <_Group>[];
    for (var e = 0; e < subjectOf.length; e++) {
      final si = subjectOf[e];
      if (si < 0 || si >= subjects.length) {
        throw RangeError.index(si, subjects, 'subjectOf[$e]');
      }
      final s = subjects[si];
      if (checked.add(si)) _validate(s, nCols, needResponses);
      final nTime = s.inputs.length ~/ nCols;
      _Group? g;
      for (final c in groups) {
        if (c.nTime == nTime && _sameSteps(c.timeSteps, s.timeSteps)) {
          g = c;
          break;
        }
      }
      (g ?? (groups..add(_Group(nTime, s.timeSteps))).last).elements.add(e);
    }
    return groups;
  }

  void _validate(Subject s, int nCols, bool needResponses) {
    if (s.inputs.length % nCols != 0) {
      throw ArgumentError(
        '$s: ${s.inputs.length} inputs is not a multiple of $nCols input '
        'columns',
      );
    }
    final nTime = s.inputs.length ~/ nCols;
    if (s.observed case final o? when o.length != nTime * _nInputs) {
      throw ArgumentError(
        '$s: observed must have one flag per trial and input node',
      );
    }
    if (s.timeSteps case final ts? when ts.length != nTime) {
      throw ArgumentError('$s: timeSteps must have one entry per trial');
    }
    if (needResponses && response.usesResponses) {
      final r = s.responses;
      if (r == null) throw ArgumentError('$s has no responses');
      if (r.length < nTime) {
        throw ArgumentError('$s: ${r.length} responses for $nTime trials');
      }
    }
  }

  static bool _sameSteps(Float64List? a, Float64List? b) {
    if (identical(a, b)) return true;
    if (a == null || b == null || a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// The token shared by [model] and every copy made from it by
/// `withPriors` (they share compiled batches). Package-internal.
Object modelFamily(HgfModel model) => model._batches;

/// [base] with [parameters] in place of its own, sharing its compiled
/// batches (what `withPriors` makes). Package-internal.
HgfModel modelWithParameters(HgfModel base, List<Parameter> parameters) =>
    HgfModel._withParameters(base, parameters);

final class _Group {
  _Group(this.nTime, this.timeSteps);

  final int nTime;
  final Float64List? timeSteps;
  final List<int> elements = [];
}

/// A [ResponseModel] bound to one batch: per-element parameters and
/// responses.
final class _BoundResponse extends ResponseFunction {
  _BoundResponse(
    this.model,
    this.params,
    this.nParams,
    this.responses, [
    this.paramTangents,
  ]);

  final ResponseModel model;
  final Float64List params;
  final int nParams;
  final List<Float64List?> responses;
  final Float64List? paramTangents;

  ResponsePartials? _partials;

  @override
  bool get supportsGradients => model.supportsGradients;

  @override
  double batchSurprisePartials(
    BatchNetwork network,
    int t,
    int element,
    SurprisePartials partials,
  ) {
    final r = responses[element];
    final offset = element * nParams;
    final p = _partials = bindResponsePartials(
      _partials,
      paramTangents!,
      partials,
      offset,
      network.tangents,
    );
    return model.surprisePartials(
      network,
      element,
      t,
      r == null ? double.nan : r[t],
      params,
      offset,
      p,
    );
  }

  @override
  double surprise(Network network, int t) =>
      throw UnsupportedError('batch only');

  @override
  double batchSurprise(BatchNetwork network, int t, int element) {
    final r = responses[element];
    return model.surprise(
      network,
      element,
      t,
      r == null ? double.nan : r[t],
      params,
      element * nParams,
    );
  }
}
