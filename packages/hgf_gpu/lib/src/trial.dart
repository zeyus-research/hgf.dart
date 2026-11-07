import 'package:hgf/hgf.dart' hide lambertW0, logAddExp;

import 'ir.dart';

/// The fields every node has a slot for, in slot order.
const List<NodeField> gpuNodeFields = [
  NodeField.mean,
  NodeField.expectedMean,
  NodeField.precision,
  NodeField.expectedPrecision,
  NodeField.conditionalExpectedPrecision,
  NodeField.currentVariance,
  NodeField.effectivePrecision,
  NodeField.valuePredictionError,
  NodeField.volatilityPredictionError,
  NodeField.observed,
  NodeField.tonicVolatility,
  NodeField.tonicDrift,
  NodeField.autoconnectionStrength,
];

const List<String> _fieldVar = [
  'm', 'em', 'p', 'ep', 'cep', 'cv', 'eff', 'vape', 'vope', 'ob', //
  'tv', 'td', 'ac',
];

/// What a kernel adds to each element's total after every trial.
///
/// The observation surprises read the network alone; the readouts score a
/// response per trial and have parameters, which live in slots (so they can
/// differ per element).
sealed class GpuResponse {
  const GpuResponse();

  /// `TotalGaussianSurprise`: the Gaussian surprise of every continuous
  /// node.
  static const totalGaussianSurprise = _ObservationResponse('total');

  /// `FirstLevelGaussianSurprise`: node 0's observation under node 1's
  /// prediction.
  static const firstLevelGaussianSurprise = _ObservationResponse('first');

  /// `FirstLevelBinarySurprise`: the binary surprise of the input nodes.
  static const firstLevelBinarySurprise = _ObservationResponse('binary');

  /// Parameter defaults, in slot order.
  List<double> get defaults => const [];

  /// Whether it scores a response per trial.
  bool get usesResponses => false;
}

final class _ObservationResponse extends GpuResponse {
  const _ObservationResponse(this.which);

  final String which;

  @override
  String toString() => 'GpuResponse.$which';
}

/// `GaussianReadout`: `response ~ N(belief, 1 / precision)`, the belief
/// being [field] of [node]. Parameter: `precision`.
final class GpuGaussianReadout extends GpuResponse {
  const GpuGaussianReadout({
    this.node = 0,
    this.field = NodeField.expectedMean,
    this.precision = 1.0,
  });

  final int node;
  final NodeField field;
  final double precision;

  @override
  List<double> get defaults => [precision];

  @override
  bool get usesResponses => true;
}

/// `BinarySoftmaxReadout`: binary responses with probability
/// `σ(β logit μ̂_node)`. Parameter: `inverseTemperature`.
final class GpuBinarySoftmax extends GpuResponse {
  const GpuBinarySoftmax({this.node = 0, this.inverseTemperature = 1.0});

  final int node;
  final double inverseTemperature;

  @override
  List<double> get defaults => [inverseTemperature];

  @override
  bool get usesResponses => true;
}

/// `f32` versions of the engine's `f64` constants.
abstract final class F32 {
  /// Smallest normal `f32`; the `f64` engine floors at 1e-128, below
  /// `f32`'s range (JAX in `float32` flushes it to 0).
  static const minNormal = Lit('1.17549435e-38f');

  /// `ln(f32::MAX)`, for `logMaxFinite`.
  static const logMaxFinite = Lit('88.7228391f');

  /// Largest `logS` for which the unbounded update forms `exp(logS)`
  /// directly (700 in `f64`).
  static const linearLogSMax = Lit('80.0f');
}

/// Where a trial's observations and response come from (WGSL
/// expressions, valid inside the trial loop where `t` is the trial).
abstract interface class TrialSource {
  /// Observation of input column [c] (`f32`).
  String input(int c);

  /// Whether column [c] is observed (`f32`, 0 or 1).
  String observed(int c);

  /// Whether this trial has a response (`bool`).
  String responsePresent();

  /// The response (`f32`; only read where [responsePresent]).
  String response();
}

/// The result of [TrialCompiler.compile].
final class CompiledTrial {
  CompiledTrial._(
    this.body,
    this.carried,
    this.written,
    this.active,
    this.slotOfVar,
  );

  /// The trial's statements (inside the trial loop).
  final String body;

  /// Slot variables read before they are written in a trial: loaded
  /// before the loop.
  final Set<String> carried;

  /// Slot variables the trial writes.
  final Set<String> written;

  /// Slot variables that carry tangents.
  final Set<String> active;

  /// Every slot variable's slot.
  final Map<String, int> slotOfVar;
}

/// Compiles one trial of a [Network] (predictions, observations, updates,
/// response) into WGSL, with forward-mode tangents along [directions]
/// directions for the slot variables in `seeds` and everything they reach.
///
/// The slot layout: every node field of [gpuNodeFields] (`field * n +
/// node`), the value edges, the volatility edges, the response's
/// parameters ([GpuResponse.defaults]), then the running total, its Kahan
/// compensation, the non-finite flag and the result (`resultSlot`).
///
/// Throws [UnsupportedError] for node kinds and update steps without a GPU
/// kernel (vector-valued nodes, custom couplings).
final class TrialCompiler {
  TrialCompiler(this.net, this.response, {this.directions = 0})
    : n = net.nNodes {
    net.build();
    for (var i = 0; i < n; i++) {
      switch (net.kindOf(i)) {
        case NodeKind.continuousState ||
            NodeKind.binaryState ||
            NodeKind.constantState:
          break;
        default:
          throw UnsupportedError(
            '${net.kindOf(i).pyhgfName} has no GPU kernel yet',
          );
      }
    }
    for (final i in net.inputIdxs) {
      if (net.nInputColumns != net.inputIdxs.length) {
        throw UnsupportedError('Multi-column inputs have no GPU kernel yet');
      }
      if (net.kindOf(i) != NodeKind.continuousState &&
          net.kindOf(i) != NodeKind.binaryState) {
        throw UnsupportedError('Input node $i: ${net.kindOf(i).pyhgfName}');
      }
    }
    for (var c = 0; c < n; c++) {
      for (final p in net.valueParentsOf(c)) {
        valueEdges.add((p, c));
      }
      for (final p in net.volatilityParentsOf(c)) {
        volatilityEdges.add((p, c));
      }
    }
    switch (response) {
      case GpuGaussianReadout(:final node, :final field):
        _checkNode(node);
        if (!gpuNodeFields.contains(field)) {
          throw UnsupportedError('No GPU slot for $field');
        }
      case GpuBinarySoftmax(:final node):
        _checkNode(node);
      case _ObservationResponse(:final which):
        if (which == 'first' &&
            (n < 2 ||
                net.kindOf(0) != NodeKind.continuousState ||
                net.kindOf(1) != NodeKind.continuousState)) {
          throw ArgumentError(
            'FirstLevelGaussianSurprise needs continuous nodes 0 and 1',
          );
        }
        if (which == 'binary' &&
            net.inputIdxs.any((i) => net.kindOf(i) != NodeKind.binaryState)) {
          throw ArgumentError('FirstLevelBinarySurprise needs binary inputs');
        }
    }
    var s = 0;
    for (var f = 0; f < gpuNodeFields.length; f++) {
      for (var i = 0; i < n; i++) {
        slotOfVar['${_fieldVar[f]}$i'] = s++;
      }
    }
    for (final (p, c) in valueEdges) {
      slotOfVar['a${p}_$c'] = s++;
    }
    for (final (p, c) in volatilityEdges) {
      slotOfVar['k${p}_$c'] = s++;
    }
    for (var j = 0; j < response.defaults.length; j++) {
      slotOfVar['rp$j'] = s++;
    }
    totalSlot = s;
  }

  void _checkNode(int node) {
    if (node < 0 || node >= n) {
      throw ArgumentError.value(node, 'node', 'No such node');
    }
  }

  final Network net;
  final GpuResponse response;

  /// Tangent directions (0: none).
  final int directions;
  final int n;

  final valueEdges = <(int, int)>[];
  final volatilityEdges = <(int, int)>[];
  final slotOfVar = <String, int>{};

  /// Slot of the running total (then compensation, flag, result).
  late final int totalSlot;

  int get resultSlot => totalSlot + 3;

  /// Slots per element, including the result.
  int get nSlots => totalSlot + 4;

  /// Name of the slot variable of [field] of [node].
  static String fieldVar(int node, NodeField field) =>
      '${_fieldVar[gpuNodeFields.indexOf(field)]}$node';

  /// Slot of [field] of [node].
  int slot(int node, NodeField field) {
    final f = gpuNodeFields.indexOf(field);
    if (f < 0) throw ArgumentError.value(field, 'field', 'No GPU slot');
    return f * n + node;
  }

  int valueCouplingSlot(int parent, int child) {
    final v = slotOfVar['a${parent}_$child'];
    if (v == null) throw ArgumentError('No value edge $parent → $child');
    return v;
  }

  int volatilityCouplingSlot(int parent, int child) {
    final v = slotOfVar['k${parent}_$child'];
    if (v == null) throw ArgumentError('No volatility edge $parent → $child');
    return v;
  }

  /// Slot of response parameter [j].
  int responseSlot(int j) => totalSlot - response.defaults.length + j;

  /// The network's initial value of every slot, the response's defaults,
  /// and zero for the total and its bookkeeping.
  List<double> initialSlots() {
    final out = List<double>.filled(nSlots, 0.0);
    for (var f = 0; f < gpuNodeFields.length; f++) {
      for (var i = 0; i < n; i++) {
        out[f * n + i] = net.getInitial(i, gpuNodeFields[f]);
      }
    }
    var s = gpuNodeFields.length * n;
    for (var c = 0; c < n; c++) {
      for (final x in net.valueParentStrengthsOf(c)) {
        out[s++] = x;
      }
    }
    for (var c = 0; c < n; c++) {
      for (final x in net.volatilityParentStrengthsOf(c)) {
        out[s++] = x;
      }
    }
    for (final d in response.defaults) {
      out[s++] = d;
    }
    return out;
  }

  /// Number of tangent chunks (vectors of up to 4 directions).
  int get chunks => (directions + 3) ~/ 4;

  /// Directions in chunk [c].
  int chunkWidth(int c) {
    final w = directions - 4 * c;
    return w > 4 ? 4 : w;
  }

  /// WGSL type of chunk [c].
  String chunkType(int c) {
    final w = chunkWidth(c);
    return w == 1 ? 'f32' : 'vec$w<f32>';
  }

  /// Compiles the trial, with [seeds] (slot variables) carrying tangents
  /// on entry; adds whatever they reach until the set is closed.
  CompiledTrial compile(TrialSource source, {Set<String> seeds = const {}}) {
    if (directions == 0 && seeds.isNotEmpty) {
      throw ArgumentError('Seeds without directions');
    }
    var active = {...seeds};
    for (;;) {
      final pass = _Pass(this, source, active);
      final needs = pass.run();
      if (needs.isEmpty) {
        final carried = {
          for (final e in pass.firstAccessRead.entries)
            if (e.value) e.key,
        };
        return CompiledTrial._(
          pass.body.toString(),
          carried,
          pass.written,
          active,
          slotOfVar,
        );
      }
      active = {...active, ...needs};
    }
  }
}

/// One code-generation pass over the trial, for a fixed set of active slot
/// variables.
final class _Pass {
  _Pass(this.c, this.source, this.activeSlots)
    : net = c.net,
      n = c.n,
      k = c.directions;

  final TrialCompiler c;
  final TrialSource source;
  final Set<String> activeSlots;
  final Network net;
  final int n, k;

  final body = StringBuffer();
  var _indent = 2;

  /// First access per slot variable: true if read before written.
  final firstAccessRead = <String, bool>{};
  final written = <String>{};

  /// Slot variables that would need tangents (to add, then recompile).
  final needs = <String>{};

  /// Local variables' activity, innermost scope last.
  final _scopes = <Map<String, bool>>[{}];

  late final tangents = Tangents(_isActive);

  bool _isActive(String name) {
    for (var s = _scopes.length - 1; s >= 0; s--) {
      final a = _scopes[s][name];
      if (a != null) return a;
    }
    if (c.slotOfVar.containsKey(name)) return activeSlots.contains(name);
    // Locals declared without tangents, and builtins.
    return false;
  }

  Set<String> run() {
    final seq = net.updateSequence;
    for (final (i, step) in seq.predictions) {
      _step(i, step);
    }
    line('// observations');
    final inputs = net.inputIdxs;
    for (var q = 0; q < inputs.length; q++) {
      final i = inputs[q];
      w(i, NodeField.mean, Lit(source.input(q)));
      w(i, NodeField.observed, Lit(source.observed(q)));
    }
    for (final (i, step) in seq.updates) {
      _step(i, step);
    }
    _response();
    return needs;
  }

  // ---------------------------------------------------------------------------
  // Emission
  // ---------------------------------------------------------------------------

  void line(String s) {
    body
      ..write('  ' * _indent)
      ..writeln(s);
  }

  void open(String s) {
    line('$s {');
    _indent++;
    _scopes.add({});
  }

  void close([String s = '}']) {
    _indent--;
    _scopes.removeLast();
    line(s);
    if (s.endsWith('{')) {
      _indent++;
      _scopes.add({});
    }
  }

  Ex? _d(Ex e, {Ex? self}) => k == 0 ? null : tangents.d(e, self: self);

  void _tangentLines(String prefix, String name, String op, Ex? dt) {
    for (var ch = 0; ch < c.chunks; ch++) {
      final type = c.chunkType(ch);
      final rhs = dt == null ? '$type()' : dt.emit(Emit(ch, type));
      line('$prefix${name}_t$ch $op $rhs;');
    }
  }

  /// `let name = e;` (and its tangent).
  Ex let(String name, Object e) {
    final x = ex(e);
    line('let $name = ${x.w};');
    final dt = _d(x, self: Ref(name));
    if (dt != null) _tangentLines('let ', name, '=', dt);
    _scopes.last[name] = dt != null;
    return Ref(name);
  }

  /// A boolean `let`.
  Cond letCond(String name, Cond c) {
    line('let $name = ${c.text};');
    return Cond(name);
  }

  /// `var name = e;`, with a tangent if [e] has one or [active].
  Ex variable(String name, Object e, {bool active = false}) {
    final x = ex(e);
    line('var $name = ${x.w};');
    final dt = _d(x);
    final has = k > 0 && (dt != null || active);
    if (has) _tangentLines('var ', name, '=', dt);
    _scopes.last[name] = has;
    return Ref(name);
  }

  /// `name = e;` for a local or slot variable.
  void set(String name, Object e) {
    final x = ex(e);
    final dt = _d(x);
    if (_isActive(name)) {
      _tangentLines('', name, '=', dt);
    } else if (dt != null) {
      if (c.slotOfVar.containsKey(name)) {
        needs.add(name);
      } else {
        throw StateError('Active value assigned to inactive local $name');
      }
    }
    line('$name = ${x.w};');
  }

  /// Whether any of [es] carries a tangent.
  bool anyActive(Iterable<Ex> es) => k > 0 && es.any((e) => _d(e) != null);

  var _tmp = 0;
  String tmp([String stem = 't']) => '$stem${_tmp++}';

  // ---------------------------------------------------------------------------
  // Variables
  // ---------------------------------------------------------------------------

  /// Reads node field [f] of [node].
  Ex r(int node, NodeField f) {
    final v = TrialCompiler.fieldVar(node, f);
    firstAccessRead.putIfAbsent(v, () => true);
    return Ref(v);
  }

  /// Writes node field [f] of [node].
  void w(int node, NodeField f, Object e) {
    final x = ex(e); // reads recorded first
    final v = TrialCompiler.fieldVar(node, f);
    firstAccessRead.putIfAbsent(v, () => false);
    written.add(v);
    set(v, x);
  }

  Ex _slotRead(String v) {
    firstAccessRead.putIfAbsent(v, () => true);
    return Ref(v);
  }

  Ex a(int parent, int child) => _slotRead('a${parent}_$child');
  Ex kap(int parent, int child) => _slotRead('k${parent}_$child');
  Ex rp(int j) => _slotRead('rp$j');

  static const dt = Ref('dt');
  static const logT = Ref('logT');

  // ---------------------------------------------------------------------------
  // Graph helpers
  // ---------------------------------------------------------------------------

  NodeKind kind(int i) => net.kindOf(i);
  bool isLeaf(int i) =>
      net.valueChildrenOf(i).isEmpty && net.volatilityChildrenOf(i).isEmpty;
  bool get meanField => net.meanFieldUpdates;

  CouplingFn couplingFn(int parent) {
    final cf = net.couplingOf(parent);
    if (cf is CouplingFn) return cf;
    throw UnsupportedError(
      'Custom coupling functions have no GPU kernel (node $parent)',
    );
  }

  /// Left-to-right sum of [terms] (0 if empty).
  static Ex sum(Iterable<Ex> terms) {
    Ex? out;
    for (final t in terms) {
      out = out == null ? t : out + t;
    }
    return out ?? zero;
  }

  // ---------------------------------------------------------------------------
  // Steps
  // ---------------------------------------------------------------------------

  void _step(int i, UpdateStep step) {
    line('// ${step.name} $i');
    open('');
    switch (step) {
      case UpdateStep.predictionContinuous:
        _predictionContinuous(i, meanField: false);
      case UpdateStep.predictionContinuousMeanField:
        _predictionContinuous(i, meanField: true);
      case UpdateStep.predictionBinary:
        _predictionBinary(i);
      case UpdateStep.predictionErrorContinuous:
        _predictionErrorContinuous(i);
      case UpdateStep.predictionErrorBinary:
        _predictionErrorBinary(i);
      case UpdateStep.posteriorContinuous:
        _posteriorStandard(i, meanField: false);
      case UpdateStep.posteriorContinuousMeanField:
        _posteriorStandard(i, meanField: true);
      case UpdateStep.posteriorContinuousEhgf:
        _posteriorEhgf(i, meanField: false);
      case UpdateStep.posteriorContinuousEhgfMeanField:
        _posteriorEhgf(i, meanField: true);
      case UpdateStep.posteriorContinuousUnbounded:
        _posteriorUnbounded(i);
      default:
        throw UnsupportedError('${step.name} has no GPU kernel yet');
    }
    close();
  }

  /// `g(x)`, `g'(x)`, `g''(x)` of [fn].
  Ex g(CouplingFn fn, Ex x) => switch (fn) {
    CouplingFn.linear => x,
    CouplingFn.relu => max(x, zero),
    CouplingFn.sigmoid => sigm(x),
    CouplingFn.tanh => tanh(x),
    CouplingFn.leakyRelu => sel(Lit('0.01f') * x, x, ge(x, zero)),
    CouplingFn.gelu => call('gelu', [x]),
  };

  Ex dg(CouplingFn fn, Ex x) => switch (fn) {
    CouplingFn.linear => one,
    CouplingFn.relu => sel(zero, one, gt(x, zero)),
    CouplingFn.sigmoid => call('dsigm', [x]),
    CouplingFn.tanh => call('dtanh', [x]),
    CouplingFn.leakyRelu => sel(Lit('0.01f'), one, ge(x, zero)),
    CouplingFn.gelu => call('dgelu', [x]),
  };

  Ex d2g(CouplingFn fn, Ex x) => switch (fn) {
    CouplingFn.linear || CouplingFn.relu || CouplingFn.leakyRelu => zero,
    CouplingFn.sigmoid => call('d2sigm', [x]),
    CouplingFn.tanh => call('d2tanh', [x]),
    CouplingFn.gelu => call('d2gelu', [x]),
  };

  void _predictionContinuous(int i, {required bool meanField}) {
    final driftTerms = <Ex>[r(i, NodeField.tonicDrift)];
    final vcvTerms = <Ex>[];
    for (final p in net.valueParentsOf(i)) {
      final alpha = a(p, i);
      final pm = r(p, NodeField.expectedMean);
      if (kind(p) == NodeKind.constantState) {
        // μ̂ = 1, π̂ = ∞: drift only.
        driftTerms.add(alpha * pm);
        continue;
      }
      final fn = couplingFn(p);
      driftTerms.add(alpha * g(fn, pm));
      if (!meanField) {
        final cc = let(tmp('c'), dt * alpha * dg(fn, pm));
        vcvTerms.add(cc * cc / r(p, NodeField.expectedPrecision));
      }
    }
    final volTerms = <Ex>[r(i, NodeField.tonicVolatility)];
    final volParents = net.volatilityParentsOf(i);
    for (final p in volParents) {
      final kappa = kap(p, i);
      volTerms.add(kappa * r(p, NodeField.expectedMean));
      if (!meanField) {
        volTerms.add(
          (kappa * kappa) / (Lit('2.0f') * r(p, NodeField.expectedPrecision)),
        );
      }
    }
    final drift = let('drift', sum(driftTerms));
    final totvol = let('totvol', sum(volTerms));
    final input = isLeaf(i);
    final fixedLeaf = input && volParents.isEmpty;
    final lambda = let('lambda', r(i, NodeField.autoconnectionStrength));
    final prec = let('prec', r(i, NodeField.precision));
    final expMean = let('expMean', lambda * r(i, NodeField.mean) + dt * drift);
    final pv = let('pv', dt * exp(totvol));
    // f64: pv ≤ 1e-128 becomes NaN; here pv ≤ 0 (underflow) flags the
    // element.
    line('bad = bad || !(pv > 0.0f);');
    final carried = let(
      'carried',
      input ? one / prec : (lambda * lambda) / prec,
    );
    Ex? cond;
    final Ex expected;
    if (!meanField) {
      final vcv = let('vcv', sum(vcvTerms));
      cond = let('cond', one / max(carried + pv, F32.minNormal));
      expected = let('expected', one / max(carried + pv + vcv, F32.minNormal));
    } else {
      expected = let('expected', one / max(carried + pv, F32.minNormal));
    }
    w(i, NodeField.currentVariance, carried);
    w(i, NodeField.expectedMean, expMean);
    w(i, NodeField.effectivePrecision, pv * expected);
    if (!fixedLeaf) {
      w(i, NodeField.expectedPrecision, expected);
      w(
        i,
        NodeField.conditionalExpectedPrecision,
        meanField ? expected : cond!,
      );
    } else {
      w(i, NodeField.expectedPrecision, prec);
      w(i, NodeField.conditionalExpectedPrecision, prec);
    }
  }

  void _predictionBinary(int i) {
    final parents = net.valueParentsOf(i);
    final s = sum([for (final p in parents) r(p, NodeField.expectedMean)]);
    final v = lit(net.precisionClippingValue);
    // 1 − μ̂ from σ(−s), not by subtraction, which loses f32 precision
    // where μ̂ is near 1; π̂ = μ̂ (1 − μ̂) keeps it (see [oneMinusMean]).
    final pr = let('pr', clamp(sigm(s), v, one - v));
    final qr = let('qr', clamp(sigm(-s), v, one - v));
    w(i, NodeField.expectedMean, pr);
    w(i, NodeField.expectedPrecision, pr * qr);
  }

  /// `1 − μ̂` of [node]: for a binary node `π̂ / μ̂` (exact to rounding,
  /// where `1 − μ̂` cancels catastrophically in f32 near 1).
  Ex oneMinusMean(int node) => kind(node) == NodeKind.binaryState
      ? r(node, NodeField.expectedPrecision) / r(node, NodeField.expectedMean)
      : one - r(node, NodeField.expectedMean);

  Cond anyObserved(int i) {
    final children = [
      ...net.valueChildrenOf(i),
      ...net.volatilityChildrenOf(i),
    ];
    if (children.isEmpty) return no;
    return children
        .map((ch) => ne(r(ch, NodeField.observed), zero))
        .reduce((x, y) => x | y);
  }

  void _predictionErrorContinuous(int i) {
    final leaf = isLeaf(i);
    final epx = let('epx', r(i, NodeField.expectedPrecision));
    final delta = let(
      'delta',
      r(i, NodeField.mean) - r(i, NodeField.expectedMean),
    );
    w(i, NodeField.valuePredictionError, delta);
    w(
      i,
      NodeField.volatilityPredictionError,
      (epx / r(i, NodeField.precision)) + epx * delta * delta - one,
    );
    if (!leaf) {
      w(i, NodeField.observed, sel(zero, one, anyObserved(i)));
    }
  }

  void _predictionErrorBinary(int i) {
    final epx = let('epx', r(i, NodeField.expectedPrecision));
    // u − μ̂ as u (1 − μ̂) − (1 − u) μ̂: exact for u ∈ {0, 1}.
    final u = r(i, NodeField.mean);
    final q = let('q', oneMinusMean(i));
    w(
      i,
      NodeField.valuePredictionError,
      (u * q - (one - u) * r(i, NodeField.expectedMean)) *
          r(i, NodeField.observed) /
          epx,
    );
    w(i, NodeField.precision, epx);
  }

  // ---- posterior building blocks ----

  Ex relaxedGain(int ch, {required bool meanField}) {
    final cep = r(ch, NodeField.expectedPrecision);
    if (meanField || kind(ch) != NodeKind.continuousState) return cep;
    final cond = r(ch, NodeField.conditionalExpectedPrecision);
    if (isLeaf(ch)) return cond;
    final cp = r(ch, NodeField.precision);
    return cond * cp / (cond + (cp - cep));
  }

  Ex relaxedEffective(int ch, {required bool meanField}) {
    final cep = r(ch, NodeField.expectedPrecision);
    if (meanField || kind(ch) != NodeKind.continuousState) return cep;
    final cond = r(ch, NodeField.conditionalExpectedPrecision);
    if (isLeaf(ch)) return cond;
    final cp = r(ch, NodeField.precision);
    return cond * (cp - cep) / (cond + (cp - cep));
  }

  /// The value-children precision increment of [i].
  Ex valuePrecisionWpe(int i, {required bool meanField}) {
    final children = net.valueChildrenOf(i);
    if (children.isEmpty) return zero;
    final fn = couplingFn(i);
    return sum([
      for (final ch in children)
        if (fn == CouplingFn.linear)
          relaxedEffective(ch, meanField: meanField) *
              a(i, ch) *
              a(i, ch) *
              r(ch, NodeField.observed)
        else
          (relaxedEffective(ch, meanField: meanField) *
                      a(i, ch) *
                      a(i, ch) *
                      dg(fn, r(i, NodeField.mean)) *
                      dg(fn, r(i, NodeField.mean)) -
                  relaxedGain(ch, meanField: meanField) *
                      a(i, ch) *
                      d2g(fn, r(i, NodeField.mean)) *
                      r(ch, NodeField.valuePredictionError)) *
              r(ch, NodeField.observed),
    ]);
  }

  /// The value-children mean increment of [i] at precision [prec].
  Ex valueMeanWpe(int i, Ex prec, {required bool meanField}) {
    final children = net.valueChildrenOf(i);
    if (children.isEmpty) return zero;
    final fn = couplingFn(i);
    return sum([
      for (final ch in children)
        (a(i, ch) *
                dg(fn, r(i, NodeField.mean)) *
                relaxedGain(ch, meanField: meanField) /
                prec) *
            (r(ch, NodeField.valuePredictionError) * r(ch, NodeField.observed)),
    ]);
  }

  /// The standard volatility precision increments of [i].
  List<Ex> volatilityPrecisionWpe(int i) => [
    for (final ch in net.volatilityChildrenOf(i))
      () {
        final kappa = kap(i, ch);
        final e = r(ch, NodeField.effectivePrecision);
        final v = r(ch, NodeField.volatilityPredictionError);
        final ke = let(tmp('ke'), kappa * e);
        return (Lit('0.5f') * ke * ke +
                ke * ke * v -
                Lit('0.5f') * kappa * kappa * e * v) *
            r(ch, NodeField.observed);
      }(),
  ];

  Ex volatilityMeanWpe(int i, Ex prec) => sum([
    for (final ch in net.volatilityChildrenOf(i))
      (kap(i, ch) *
              r(ch, NodeField.effectivePrecision) *
              r(ch, NodeField.volatilityPredictionError)) /
          (Lit('2.0f') * prec) *
          r(ch, NodeField.observed),
  ]);

  Ex observedOrPredicted(int i, Cond hasObs, Ex wpe) {
    final ep = r(i, NodeField.expectedPrecision);
    return min(
      sel(ep, max(ep + wpe, F32.minNormal), hasObs),
      lit(net.maxPosteriorPrecision),
    );
  }

  /// ω̃ of volatility child [ch] seen from parent [i].
  Ex effectiveTonic(int i, int ch) => sum([
    r(ch, NodeField.tonicVolatility),
    for (final p in net.volatilityParentsOf(ch))
      if (p != i) kap(p, ch) * r(p, NodeField.expectedMean),
  ]);

  Ex mgfTerms(int ch) => sum([
    for (final p in net.volatilityParentsOf(ch))
      kap(p, ch) *
          kap(p, ch) /
          (Lit('2.0f') * r(p, NodeField.expectedPrecision)),
  ]);

  Ex valueCouplingVariance(int ch) => max(
    one / r(ch, NodeField.expectedPrecision) -
        one / r(ch, NodeField.conditionalExpectedPrecision),
    zero,
  );

  void _posteriorStandard(int i, {required bool meanField}) {
    final hasObs = letCond('hasObs', anyObserved(i));
    final wpe = let(
      'wpe',
      sum([
        valuePrecisionWpe(i, meanField: meanField),
        ...volatilityPrecisionWpe(i),
      ]),
    );
    final prec = let('prec', observedOrPredicted(i, hasObs, wpe));
    final vm = let('vm', valueMeanWpe(i, prec, meanField: meanField));
    final volm = let('volm', volatilityMeanWpe(i, prec));
    w(i, NodeField.precision, prec);
    w(i, NodeField.mean, r(i, NodeField.expectedMean) + (vm + volm));
  }

  /// The eHGF precision increment of volatility child [ch] (lets named by
  /// [q]).
  Ex ehgfIncrement(int i, int ch, int q, Ex tonic) {
    final kappa = kap(i, ch);
    final prevVar = let(
      'pvv_$q',
      r(ch, NodeField.currentVariance) + valueCouplingVariance(ch),
    );
    final pv = let('pve_$q', dt * exp(kappa * r(i, NodeField.mean) + tonic));
    final ep = let('epe_$q', one / (prevVar + pv));
    final eff = let('effe_$q', pv * ep);
    final weight = let('wte_$q', (pv - prevVar) * ep);
    final d = r(ch, NodeField.mean) - r(ch, NodeField.expectedMean);
    final vope = let(
      'vpe_$q',
      (one / r(ch, NodeField.precision) + d * d) * ep - one,
    );
    final inc = let(
      'inc_$q',
      Lit('0.5f') * kappa * kappa * eff * (eff + weight * vope),
    );
    return max(inc, zero) * r(ch, NodeField.observed);
  }

  void _posteriorEhgf(int i, {required bool meanField}) {
    final epi = let('epi', r(i, NodeField.expectedPrecision));
    final vm = let('vm', valueMeanWpe(i, epi, meanField: meanField));
    final volm = let('volm', volatilityMeanWpe(i, epi));
    w(i, NodeField.mean, r(i, NodeField.expectedMean) + vm + volm);
    final hasObs = letCond('hasObs', anyObserved(i));
    final mgf = !meanField && net.ehgfMgfCorrection;
    final children = net.volatilityChildrenOf(i);
    final incs = <Ex>[
      for (var q = 0; q < children.length; q++)
        ehgfIncrement(
          i,
          children[q],
          q,
          mgf
              ? effectiveTonic(i, children[q]) + mgfTerms(children[q])
              : effectiveTonic(i, children[q]),
        ),
    ];
    final wpe = let(
      'wpe',
      sum([valuePrecisionWpe(i, meanField: meanField), ...incs]),
    );
    w(i, NodeField.precision, observedOrPredicted(i, hasObs, wpe));
  }

  void _posteriorUnbounded(int i) {
    final mf = meanField;
    final children = net.volatilityChildrenOf(i);
    final emi = let('emi', r(i, NodeField.expectedMean));
    final Ex pp, pm;
    if (net.valueChildrenOf(i).isEmpty) {
      pp = let('pp', r(i, NodeField.expectedPrecision));
      pm = let('pm', emi);
    } else {
      final vpw = let('vpw', valuePrecisionWpe(i, meanField: mf));
      pp = let(
        'pp',
        max(r(i, NodeField.expectedPrecision) + vpw, F32.minNormal),
      );
      final vmw = let('vmw', valueMeanWpe(i, pp, meanField: mf));
      pm = let('pm', emi + vmw);
    }
    final nc = children.length;
    // Per child: observed, coupling, ω̃, previous variance, b, vope.
    final ob = <Cond>[], ck = <Ex>[], to = <Ex>[], pv = <Ex>[];
    final b = <Ex>[], vo = <Ex>[];
    for (var q = 0; q < nc; q++) {
      final ch = children[q];
      ob.add(letCond('ob_$q', ne(r(ch, NodeField.observed), zero)));
      ck.add(let('ck_$q', kap(i, ch)));
      to.add(let('to_$q', effectiveTonic(i, ch)));
      pv.add(
        let(
          'pv_$q',
          max(
            r(ch, NodeField.currentVariance) + valueCouplingVariance(ch),
            F32.minNormal,
          ),
        ),
      );
      final d = let(
        'd_$q',
        r(ch, NodeField.mean) - r(ch, NodeField.expectedMean),
      );
      b.add(let('b_$q', one / r(ch, NodeField.precision) + d * d));
      vo.add(let('vo_$q', r(ch, NodeField.expectedPrecision) * b[q] - one));
    }
    const half = Lit('0.5f');
    // logS of child q at x, and w = S / (S + pv) in the linear or the log
    // regime.
    (Ex logS, Cond linear, Ex wq) weightAt(int q, Ex x, String name) {
      final logS = let('logS_$name', logT + ck[q] * x + to[q]);
      final linear = letCond('lin_$name', le(logS, F32.linearLogSMax));
      final wq = let(
        'w_$name',
        sel(sigm(logS - log(pv[q])), exp(logS) / (exp(logS) + pv[q]), linear),
      );
      return (logS, linear, wq);
    }

    // Expansion 0: at the prediction, concave part only.
    final p0 = <Ex>[pp], s0 = <Ex>[pp * (pm - emi)];
    for (var q = 0; q < nc; q++) {
      final (_, _, wq) = weightAt(q, emi, '0_$q');
      p0.add(sel(zero, half * ck[q] * ck[q] * wq * (one - wq), ob[q]));
      s0.add(sel(zero, half * ck[q] * wq * vo[q], ob[q]));
    }
    final pList = <Ex>[let('P_0', sum(p0))];
    final mList = <Ex>[let('M_0', emi + let('S_0', sum(s0)) / pList[0])];
    final vList = <Cond>[yes];
    // Expansions 1..nc: at each child's Lambert W0 mode.
    final act = anyActive([pp, pm, emi, ...ck, ...to, ...pv, ...b]);
    for (var x = 1; x <= nc; x++) {
      final q = x - 1;
      final px = variable('P_$x', zero, active: act);
      final mx = variable('M_$x', zero, active: act);
      line('var V_$x = false;');
      open('if (${ob[q].text} && ${ne(ck[q], zero).text})');
      final pihat = let('pihat', pp / (ck[q] * ck[q]));
      final gam = let('gam', logT + ck[q] * pm + to[q]);
      final w0 = let(
        'w0',
        b[q] / (Lit('2.0f') * pihat) * exp(half / pihat - gam),
      );
      final logW = let(
        'logW',
        log(b[q]) - log(Lit('2.0f') * pihat) + half / pihat - gam,
      );
      final warg = let(
        'warg',
        sel(exp(min(logW, F32.logMaxFinite)), w0, gt(w0, zero) & isFinite(w0)),
      );
      final lw = let('lw', lambertW0(warg));
      final xs = let('xs', (gam + lw - half / pihat - logT - to[q]) / ck[q]);
      final full = <Ex>[pp], conc = <Ex>[pp], grad = <Ex>[-pp * (xs - pm)];
      for (var r2 = 0; r2 < nc; r2++) {
        final (logS, linear, wr) = weightAt(r2, xs, 'm$r2');
        final invD = let(
          'invD_$r2',
          sel(
            exp(-logAddExp(log(pv[r2]), logS)),
            one / (exp(logS) + pv[r2]),
            linear,
          ),
        );
        final da = let('da_$r2', b[r2] * invD - one);
        full.add(
          sel(
            zero,
            half * ck[r2] * ck[r2] * wr * (wr + (Lit('2.0f') * wr - one) * da),
            ob[r2],
          ),
        );
        conc.add(sel(zero, half * ck[r2] * ck[r2] * wr * (one - wr), ob[r2]));
        grad.add(sel(zero, half * ck[r2] * wr * da, ob[r2]));
      }
      final fullV = let('full', sum(full));
      final concV = let('conc', sum(conc));
      final gradV = let('grad', sum(grad));
      set('P_$x', sel(fullV, concV, le(fullV, zero)));
      set('M_$x', xs + gradV / px);
      line('V_$x = ${(isFinite(px) & isFinite(mx)).text};');
      close();
      pList.add(px);
      mList.add(mx);
      vList.add(Cond('V_$x'));
    }
    // Energies.
    final energies = <Ex>[];
    for (var x = 0; x <= nc; x++) {
      final mx = mList[x];
      final terms = <Ex>[-half * pp * (mx - pm) * (mx - pm)];
      for (var q = 0; q < nc; q++) {
        final logS = let('logS_e${x}_$q', logT + ck[q] * mx + to[q]);
        final den = exp(logS) + pv[q];
        final logD = logAddExp(log(pv[q]), logS);
        terms.add(
          sel(
            zero,
            sel(
              -half * logD - half * b[q] * exp(-logD),
              -half * log(den) - half * b[q] / den,
              le(logS, F32.linearLogSMax),
            ),
            ob[q],
          ),
        );
      }
      energies.add(let('E_$x', sum(terms)));
    }
    // Softmax and moment matching over the valid expansions.
    var maxE = energies[0];
    for (var x = 1; x <= nc; x++) {
      maxE = let(
        'maxE_$x',
        sel(maxE, energies[x], vList[x] & gt(energies[x], maxE)),
      );
    }
    final u = <Ex>[
      for (var x = 0; x <= nc; x++)
        let('u_$x', sel(zero, exp(energies[x] - maxE), vList[x])),
    ];
    final tot = let('tot', sum(u));
    final pmean = let(
      'pmean',
      sum([
        for (var x = 0; x <= nc; x++)
          sel(zero, (u[x] / tot) * mList[x], vList[x]),
      ]),
    );
    final pvar = let(
      'pvar',
      sum([
        for (var x = 0; x <= nc; x++)
          sel(
            zero,
            (u[x] / tot) *
                (one / pList[x] + (mList[x] - pmean) * (mList[x] - pmean)),
            vList[x],
          ),
      ]),
    );
    final hasObs = letCond('hasObs', anyObserved(i));
    w(
      i,
      NodeField.precision,
      min(
        sel(r(i, NodeField.expectedPrecision), one / pvar, hasObs),
        lit(net.maxPosteriorPrecision),
      ),
    );
    w(i, NodeField.mean, pmean);
  }

  // ---------------------------------------------------------------------------
  // Response
  // ---------------------------------------------------------------------------

  static Ex gaussianSurprise(Ex x, Ex mu, Ex pi) {
    final d = x - mu;
    return Lit('0.5f') * (Lit('1.83787707f') - log(pi) + pi * d * d);
  }

  /// `binarySurprise` with the probability [p] of a 1 and [q] = 1 − p
  /// given separately (each clipped to [1e-6, 1 − 1e-6]).
  static Ex binarySurprise(Cond isOne, Ex p, Ex q) {
    const lo = Lit('1e-6f'), hi = Lit('0.999999f');
    return sel(-log(clamp(q, lo, hi)), -log(clamp(p, lo, hi)), isOne);
  }

  void _response() {
    line('// response');
    open('');
    final Ex s;
    switch (c.response) {
      case _ObservationResponse(:final which) when which == 'total':
        s = sum([
          for (var i = 0; i < n; i++)
            if (kind(i) == NodeKind.continuousState)
              gaussianSurprise(
                r(i, NodeField.mean),
                r(i, NodeField.expectedMean),
                r(i, NodeField.expectedPrecision),
              ),
        ]);
      case _ObservationResponse(:final which) when which == 'first':
        s = gaussianSurprise(
          r(0, NodeField.mean),
          r(1, NodeField.expectedMean),
          r(1, NodeField.expectedPrecision),
        );
      case _ObservationResponse():
        s = sum([
          for (final i in net.inputIdxs)
            binarySurprise(
              ne(r(i, NodeField.mean), zero),
              r(i, NodeField.expectedMean),
              oneMinusMean(i),
            ),
        ]);
      case GpuGaussianReadout(:final node, :final field):
        final present = letCond('present', Cond(source.responsePresent()));
        final resp = let('resp', Lit(source.response()));
        s = sel(zero, gaussianSurprise(resp, r(node, field), rp(0)), present);
      case GpuBinarySoftmax(:final node):
        final present = letCond('present', Cond(source.responsePresent()));
        final resp = let('resp', Lit(source.response()));
        final m = r(node, NodeField.expectedMean);
        final z = let('z', rp(0) * log(m / oneMinusMean(node)));
        final p = let('prob', one / (one + exp(-z)));
        final q = let('qprob', one / (one + exp(z)));
        s = sel(zero, binarySurprise(ne(resp, zero), p, q), present);
    }
    final sv = let('s', s);
    line('bad = bad || !isFinite(s);');
    // Kahan summation (the tangent is summed plainly).
    final ds = _d(sv);
    if (ds != null) _tangentLines('', 'total', '+=', ds);
    line('let yk = s - comp;');
    line('let tk = total + yk;');
    line('comp = (tk - total) - yk;');
    line('total = tk;');
    close();
  }
}

/// Functions the generated kernels call.
const String prelude = r'''
fn isFinite(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
fn sigm(x: f32) -> f32 { return 1.0f / (1.0f + exp(-x)); }
fn dsigm(x: f32) -> f32 { let s = sigm(x); return s * (1.0f - s); }
fn d2sigm(x: f32) -> f32 {
  let s = sigm(x);
  return s * (1.0f - s) * (1.0f - 2.0f * s);
}
fn d3sigm(x: f32) -> f32 {
  let s = sigm(x);
  return s * (1.0f - s) * (1.0f - 6.0f * s + 6.0f * s * s);
}
fn dtanh(x: f32) -> f32 { let t = tanh(x); return 1.0f - t * t; }
fn d2tanh(x: f32) -> f32 { let t = tanh(x); return -2.0f * t * (1.0f - t * t); }
fn d3tanh(x: f32) -> f32 {
  let t = tanh(x);
  return -2.0f * (1.0f - t * t) * (1.0f - 3.0f * t * t);
}
fn erfcApprox(x: f32) -> f32 {
  let t = 1.0f / (1.0f + 0.3275911f * abs(x));
  let poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f +
      t * (-1.453152027f + t * 1.061405429f))));
  let approx = poly * exp(-(x * x));
  return select(2.0f - approx, approx, x >= 0.0f);
}
fn gelu(x: f32) -> f32 { return x * 0.5f * erfcApprox(-x / 1.41421356f); }
fn dgelu(x: f32) -> f32 {
  let phi = 0.5f * erfcApprox(-x / 1.41421356f);
  let pdf = exp(-(x * x) / 2.0f) / 2.50662827f;
  return phi + x * pdf;
}
fn d2gelu(x: f32) -> f32 {
  let pdf = exp(-(x * x) / 2.0f) / 2.50662827f;
  return pdf * (2.0f - x * x);
}
fn d3gelu(x: f32) -> f32 {
  let pdf = exp(-(x * x) / 2.0f) / 2.50662827f;
  return pdf * (x * x * x - 4.0f * x);
}
fn logAddExp(a: f32, b: f32) -> f32 {
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}
fn lambertW0(z: f32) -> f32 {
  var w = log(z + 1.0f);
  for (var i = 0; i < 6; i++) {
    let ew = exp(w);
    let f = w * ew - z;
    let f1 = (w + 1.0f) * ew;
    let f2 = (w + 2.0f) * ew;
    let st = (2.0f * f * f1) / (2.0f * f1 * f1 - f * f2);
    w -= st;
    // 1e-12 in f64.
    if (abs(st) <= 1e-6f * abs(w)) { break; }
  }
  return w;
}
''';
