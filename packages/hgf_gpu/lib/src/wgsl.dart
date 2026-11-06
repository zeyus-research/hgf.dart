import 'dart:typed_data';

import 'package:hgf/hgf.dart';

/// The per-trial response a kernel accumulates into each element's total.
enum GpuResponse {
  /// [TotalGaussianSurprise]: the Gaussian surprise of every continuous node.
  totalGaussianSurprise,

  /// [FirstLevelBinarySurprise]: the binary surprise of the input nodes.
  firstLevelBinarySurprise,
}

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

/// A [Network] compiled into a WGSL compute shader: one invocation per
/// batch element (parameter set), looping over a range of trials with the
/// element's whole state in registers.
///
/// Everything per element lives in one `f32` storage buffer, `[slot][element]`
/// (element innermost, so a workgroup's loads coalesce): every node field of
/// [gpuNodeFields], every value and volatility coupling strength, and the
/// running response total. A dispatch loads the slots it reads, runs its
/// trials and stores the slots it carries, so any field or coupling can
/// differ per element without recompiling, and a long series can be split
/// over several dispatches.
///
/// The arithmetic is the [BatchNetwork]'s in `f32`, with the `f64`
/// constants moved into `f32` range (see [HgfKernel.new]). WGSL does not
/// guarantee NaN or infinity semantics, so the kernel does not rely on
/// them: an element whose state stops being finite is flagged and reports
/// an infinite total, like the CPU engine's `_nanToInf`.
final class HgfKernel {
  /// Compiles [network] (built if needed).
  ///
  /// [perElementData]: each element gets its own observations
  /// (`data[t][input][element]`) instead of sharing one series
  /// (`data[t][input]`). [observedMask]: the kernel reads a 0/1 mask with
  /// the same layout as the data (missing observations).
  ///
  /// Throws [UnsupportedError] for node kinds and update steps without a
  /// GPU kernel (vector-valued nodes, custom couplings).
  factory HgfKernel(
    Network network, {
    GpuResponse response = GpuResponse.totalGaussianSurprise,
    bool perElementData = false,
    bool observedMask = false,
    bool timeSteps = false,
    int workgroupSize = 64,
  }) => _Compiler(
    network..build(),
    response,
    perElementData,
    observedMask,
    timeSteps,
    workgroupSize,
  ).compile();

  HgfKernel._({
    required this.source,
    required this.nNodes,
    required this.nSlots,
    required this.valueEdges,
    required this.volatilityEdges,
    required this.initial,
    required this.inputs,
    required this.response,
    required this.perElementData,
    required this.observedMask,
    required this.timeSteps,
    required this.workgroupSize,
    required this.loaded,
    required this.stored,
  });

  /// The WGSL source. Entry point `main`; bindings in group 0:
  /// 0 `state` (read_write), 1 `data` (read), 2 `params` (uniform:
  /// batch size, first trial, trial count, time step), then `mask` (read,
  /// only with [observedMask]) and `steps` (read, only with [timeSteps]).
  final String source;

  final int nNodes;

  /// Slots per element in the state buffer.
  final int nSlots;

  /// `(parent, child)` per value edge, in slot order after the node fields.
  final List<(int, int)> valueEdges;

  /// `(parent, child)` per volatility edge, after the value edges.
  final List<(int, int)> volatilityEdges;

  /// The network's initial value of every slot (the total starts at 0).
  final Float64List initial;

  /// The input nodes, in data-column order.
  final List<int> inputs;

  final GpuResponse response;
  final bool perElementData;
  final bool observedMask;

  /// The kernel reads one time step per trial (`steps[t]`) instead of the
  /// dispatch's uniform one.
  final bool timeSteps;
  final int workgroupSize;

  /// How many slots a dispatch loads and stores (diagnostic).
  final int loaded, stored;

  /// Slot of [field] of [node].
  int slot(int node, NodeField field) {
    final f = gpuNodeFields.indexOf(field);
    if (f < 0) throw ArgumentError.value(field, 'field', 'No GPU slot');
    return f * nNodes + node;
  }

  /// Slot of the value coupling `parent → child`.
  int valueCouplingSlot(int parent, int child) {
    final k = valueEdges.indexOf((parent, child));
    if (k < 0) throw ArgumentError('No value edge $parent → $child');
    return gpuNodeFields.length * nNodes + k;
  }

  /// Slot of the volatility coupling `parent → child`.
  int volatilityCouplingSlot(int parent, int child) {
    final k = volatilityEdges.indexOf((parent, child));
    if (k < 0) throw ArgumentError('No volatility edge $parent → $child');
    return gpuNodeFields.length * nNodes + valueEdges.length + k;
  }

  /// Slot of the running response total (Kahan sum; the next slot holds
  /// its compensation, the one after that the non-finite flag).
  int get totalSlot => nSlots - 4;

  /// Slot of the result: the total, or `f32` max where the flag is set
  /// (WGSL has no infinity literal; Tint rejects even a bitcast one).
  int get resultSlot => nSlots - 1;

  /// A state buffer for [batchSize] elements, every element at the
  /// network's initial values.
  Float32List initialState(int batchSize) {
    final out = Float32List(nSlots * batchSize);
    for (var s = 0; s < nSlots; s++) {
      out.fillRange(s * batchSize, (s + 1) * batchSize, initial[s]);
    }
    return out;
  }

  /// Width of one trial's row of shared data.
  int get nInputs => inputs.length;
}

/// `f32` versions of the engine's `f64` constants.
abstract final class _F32 {
  /// Smallest normal `f32`; the `f64` engine floors at 1e-128, below
  /// `f32`'s range (JAX in `float32` flushes it to 0).
  static const minNormal = '1.17549435e-38f';

  /// `ln(f32::MAX)`, for `logMaxFinite`.
  static const logMaxFinite = '88.7228391f';

  /// Largest `logS` for which the unbounded update forms `exp(logS)`
  /// directly (700 in `f64`).
  static const linearLogSMax = '80.0f';
}

final class _Compiler {
  _Compiler(
    this.net,
    this.response,
    this.perElementData,
    this.observedMask,
    this.timeSteps,
    this.workgroupSize,
  ) : n = net.nNodes;

  final Network net;
  final GpuResponse response;
  final bool perElementData, observedMask, timeSteps;
  final int workgroupSize;
  final int n;

  final _body = StringBuffer();
  var _indent = 2;

  /// First access per variable: true if read before written.
  final _firstAccessRead = <String, bool>{};
  final _written = <String>{};

  final valueEdges = <(int, int)>[];
  final volatilityEdges = <(int, int)>[];

  // ---------------------------------------------------------------------------
  // Variables
  // ---------------------------------------------------------------------------

  String _v(int node, NodeField f) =>
      '${_fieldVar[gpuNodeFields.indexOf(f)]}$node';

  /// Reads node field [f] of [node] (records the access).
  String r(int node, NodeField f) {
    final v = _v(node, f);
    _firstAccessRead.putIfAbsent(v, () => true);
    return v;
  }

  /// Writes node field [f] of [node].
  String w(int node, NodeField f) {
    final v = _v(node, f);
    _firstAccessRead.putIfAbsent(v, () => false);
    _written.add(v);
    return v;
  }

  String a(int parent, int child) {
    final v = 'a${parent}_$child';
    _firstAccessRead.putIfAbsent(v, () => true);
    return v;
  }

  String k(int parent, int child) {
    final v = 'k${parent}_$child';
    _firstAccessRead.putIfAbsent(v, () => true);
    return v;
  }

  void line(String s) {
    _body
      ..write('  ' * _indent)
      ..writeln(s);
  }

  void open(String s) {
    line('$s {');
    _indent++;
  }

  void close([String s = '}']) {
    _indent--;
    line(s);
  }

  var _tmp = 0;
  String tmp([String stem = 't']) => '$stem${_tmp++}';

  // ---------------------------------------------------------------------------
  // Graph helpers
  // ---------------------------------------------------------------------------

  NodeKind kind(int i) => net.kindOf(i);
  bool isLeaf(int i) =>
      net.valueChildrenOf(i).isEmpty && net.volatilityChildrenOf(i).isEmpty;
  bool get meanField => net.meanFieldUpdates;

  CouplingFn couplingFn(int parent) {
    final c = net.couplingOf(parent);
    if (c is CouplingFn) return c;
    throw UnsupportedError(
      'Custom coupling functions have no GPU kernel (node $parent)',
    );
  }

  // ---------------------------------------------------------------------------
  // Compile
  // ---------------------------------------------------------------------------

  HgfKernel compile() {
    for (var i = 0; i < n; i++) {
      switch (kind(i)) {
        case NodeKind.continuousState ||
            NodeKind.binaryState ||
            NodeKind.constantState:
          break;
        default:
          throw UnsupportedError('${kind(i).pyhgfName} has no GPU kernel yet');
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
    final nSlots =
        gpuNodeFields.length * n +
        valueEdges.length +
        volatilityEdges.length +
        4;
    final initial = Float64List(nSlots);
    for (var f = 0; f < gpuNodeFields.length; f++) {
      for (var i = 0; i < n; i++) {
        initial[f * n + i] = net.getInitial(i, gpuNodeFields[f]);
      }
    }
    var s = gpuNodeFields.length * n;
    for (var c = 0; c < n; c++) {
      for (final x in net.valueParentStrengthsOf(c)) {
        initial[s++] = x;
      }
    }
    for (var c = 0; c < n; c++) {
      for (final x in net.volatilityParentStrengthsOf(c)) {
        initial[s++] = x;
      }
    }

    final seq = net.updateSequence;
    final inputs = net.inputIdxs;
    for (final i in inputs) {
      if (net.nInputColumns != inputs.length) {
        throw UnsupportedError('Multi-column inputs have no GPU kernel yet');
      }
      if (kind(i) != NodeKind.continuousState &&
          kind(i) != NodeKind.binaryState) {
        throw UnsupportedError('Input node $i: ${kind(i).pyhgfName}');
      }
    }

    // ---- one trial ----
    for (final (i, step) in seq.predictions) {
      _step(i, step);
    }
    line('// observations');
    for (var c = 0; c < inputs.length; c++) {
      final i = inputs[c];
      final at = perElementData
          ? '(t * ${inputs.length}u + ${c}u) * B + e'
          : 't * ${inputs.length}u + ${c}u';
      line('${w(i, NodeField.mean)} = data[$at];');
      line(
        '${w(i, NodeField.observed)} = '
        '${observedMask ? 'mask[$at]' : '1.0f'};',
      );
    }
    for (final (i, step) in seq.updates) {
      _step(i, step);
    }
    _response(inputs);

    final carried = {
      for (final e in _firstAccessRead.entries)
        if (e.value) e.key,
    };
    // Slots stored at the end: those carried between trials, plus the
    // beliefs a caller reads back.
    final stored = <String>{
      for (final v in _written)
        if (carried.contains(v)) v,
      for (var i = 0; i < n; i++)
        for (final f in const [
          NodeField.mean,
          NodeField.precision,
          NodeField.expectedMean,
          NodeField.expectedPrecision,
        ])
          if (_written.contains(_v(i, f))) _v(i, f),
    };

    final slotOfVar = <String, int>{};
    for (var f = 0; f < gpuNodeFields.length; f++) {
      for (var i = 0; i < n; i++) {
        slotOfVar['${_fieldVar[f]}$i'] = f * n + i;
      }
    }
    var es = gpuNodeFields.length * n;
    for (final (p, c) in valueEdges) {
      slotOfVar['a${p}_$c'] = es++;
    }
    for (final (p, c) in volatilityEdges) {
      slotOfVar['k${p}_$c'] = es++;
    }
    final total = nSlots - 4;

    final src = StringBuffer()
      ..writeln('// Generated by package:hgf_gpu from a $n-node network.')
      ..writeln('struct Params { B: u32, t0: u32, nT: u32, dt: f32 }')
      ..writeln(
        '@group(0) @binding(0) var<storage, read_write> state: array<f32>;',
      )
      ..writeln('@group(0) @binding(1) var<storage, read> data: array<f32>;')
      ..writeln('@group(0) @binding(2) var<uniform> params: Params;');
    var binding = 3;
    if (observedMask) {
      src.writeln(
        '@group(0) @binding(${binding++}) var<storage, read> mask: array<f32>;',
      );
    }
    if (timeSteps) {
      src.writeln(
        '@group(0) @binding($binding) var<storage, read> steps: array<f32>;',
      );
    }
    src
      ..writeln(_prelude)
      ..writeln('@compute @workgroup_size($workgroupSize)')
      ..writeln('fn main(@builtin(global_invocation_id) gid: vec3<u32>) {')
      ..writeln('  let B = params.B;')
      ..writeln('  let e = gid.x;')
      ..writeln('  if (e >= B) { return; }')
      ..writeln(timeSteps ? '' : '  let dt = params.dt;')
      ..writeln(
        timeSteps ? '' : '  let logT = select(log(dt), 0.0f, dt == 1.0f);',
      );
    final loads = slotOfVar.keys.where(carried.contains).toList();
    for (final v in loads) {
      final kw = _written.contains(v) ? 'var' : 'let';
      src.writeln('  $kw $v = state[${slotOfVar[v]}u * B + e];');
    }
    // Written but not carried: declared, not loaded.
    for (final v in slotOfVar.keys) {
      if (_written.contains(v) && !carried.contains(v)) {
        src.writeln('  var $v = 0.0f;');
      }
    }
    src
      ..writeln('  var total = state[${total}u * B + e];')
      ..writeln('  var comp = state[${total + 1}u * B + e];')
      ..writeln('  var bad = state[${total + 2}u * B + e] != 0.0f;')
      ..writeln('  for (var t = params.t0; t < params.t0 + params.nT; t++) {')
      ..writeln(timeSteps ? '    let dt = steps[t];' : '')
      ..writeln(
        timeSteps ? '    let logT = select(log(dt), 0.0f, dt == 1.0f);' : '',
      )
      ..write(_body)
      ..writeln('  }');
    for (final v in slotOfVar.keys.where(stored.contains)) {
      src.writeln('  state[${slotOfVar[v]}u * B + e] = $v;');
    }
    src
      ..writeln('  state[${total}u * B + e] = total;')
      ..writeln('  state[${total + 1}u * B + e] = comp;')
      ..writeln('  state[${total + 2}u * B + e] = select(0.0f, 1.0f, bad);')
      ..writeln(
        '  state[${total + 3}u * B + e] = '
        'select(total, 3.4e38f, bad);',
      )
      ..writeln('}');

    return HgfKernel._(
      source: src.toString(),
      nNodes: n,
      nSlots: nSlots,
      valueEdges: List.unmodifiable(valueEdges),
      volatilityEdges: List.unmodifiable(volatilityEdges),
      initial: initial,
      inputs: List.unmodifiable(inputs),
      response: response,
      perElementData: perElementData,
      observedMask: observedMask,
      timeSteps: timeSteps,
      workgroupSize: workgroupSize,
      loaded: loads.length,
      stored: stored.length,
    );
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

  /// `g(x)`, `g'(x)`, `g''(x)` of [fn] as WGSL expressions.
  String g(CouplingFn fn, String x) => switch (fn) {
    CouplingFn.linear => x,
    CouplingFn.relu => 'max($x, 0.0f)',
    CouplingFn.sigmoid => 'sigm($x)',
    CouplingFn.tanh => 'tanh($x)',
    CouplingFn.leakyRelu => 'select(0.01f * $x, $x, $x >= 0.0f)',
    CouplingFn.gelu => 'gelu($x)',
  };

  String dg(CouplingFn fn, String x) => switch (fn) {
    CouplingFn.linear => '1.0f',
    CouplingFn.relu => 'select(0.0f, 1.0f, $x > 0.0f)',
    CouplingFn.sigmoid => 'dsigm($x)',
    CouplingFn.tanh => 'dtanh($x)',
    CouplingFn.leakyRelu => 'select(0.01f, 1.0f, $x >= 0.0f)',
    CouplingFn.gelu => 'dgelu($x)',
  };

  String d2g(CouplingFn fn, String x) => switch (fn) {
    CouplingFn.linear || CouplingFn.relu || CouplingFn.leakyRelu => '0.0f',
    CouplingFn.sigmoid => 'd2sigm($x)',
    CouplingFn.tanh => 'd2tanh($x)',
    CouplingFn.gelu => 'd2gelu($x)',
  };

  void _predictionContinuous(int i, {required bool meanField}) {
    line('var drift = ${r(i, NodeField.tonicDrift)};');
    if (!meanField) line('var vcv = 0.0f;');
    line('var totvol = ${r(i, NodeField.tonicVolatility)};');
    for (final p in net.valueParentsOf(i)) {
      final alpha = a(p, i);
      final pm = r(p, NodeField.expectedMean);
      if (kind(p) == NodeKind.constantState) {
        // μ̂ = 1, π̂ = ∞: drift only.
        line('drift += $alpha * $pm;');
        continue;
      }
      final fn = couplingFn(p);
      line('drift += $alpha * ${g(fn, pm)};');
      if (!meanField) {
        final c = tmp('c');
        line('let $c = dt * $alpha * ${dg(fn, pm)};');
        line('vcv += $c * $c / ${r(p, NodeField.expectedPrecision)};');
      }
    }
    final volParents = net.volatilityParentsOf(i);
    for (final p in volParents) {
      final kappa = k(p, i);
      line('totvol += $kappa * ${r(p, NodeField.expectedMean)};');
      if (!meanField) {
        line(
          'totvol += ($kappa * $kappa) / '
          '(2.0f * ${r(p, NodeField.expectedPrecision)});',
        );
      }
    }
    final input = isLeaf(i);
    final fixedLeaf = input && volParents.isEmpty;
    line('let lambda = ${r(i, NodeField.autoconnectionStrength)};');
    line('let prec = ${r(i, NodeField.precision)};');
    line('let expMean = lambda * ${r(i, NodeField.mean)} + dt * drift;');
    line('let pv = dt * exp(totvol);');
    // f64: pv ≤ 1e-128 becomes NaN; here pv ≤ 0 (underflow) flags the
    // element.
    line('bad = bad || !(pv > 0.0f);');
    line(
      'let carried = ${input ? '1.0f / prec' : '(lambda * lambda) / prec'};',
    );
    if (!meanField) {
      line('let cond = 1.0f / floorVar(carried + pv);');
      line('let expected = 1.0f / floorVar(carried + pv + vcv);');
    } else {
      line('let expected = 1.0f / floorVar(carried + pv);');
    }
    line('${w(i, NodeField.currentVariance)} = carried;');
    line('${w(i, NodeField.expectedMean)} = expMean;');
    line('${w(i, NodeField.effectivePrecision)} = pv * expected;');
    if (!fixedLeaf) {
      line('${w(i, NodeField.expectedPrecision)} = expected;');
      line(
        '${w(i, NodeField.conditionalExpectedPrecision)} = '
        '${meanField ? 'expected' : 'cond'};',
      );
    } else {
      line('${w(i, NodeField.expectedPrecision)} = prec;');
      line('${w(i, NodeField.conditionalExpectedPrecision)} = prec;');
    }
  }

  void _predictionBinary(int i) {
    final parents = net.valueParentsOf(i);
    final sum = parents.isEmpty
        ? '0.0f'
        : parents.map((p) => r(p, NodeField.expectedMean)).join(' + ');
    final v = _lit(net.precisionClippingValue);
    line('let pr = clamp(sigm($sum), $v, 1.0f - $v);');
    line('${w(i, NodeField.expectedMean)} = pr;');
    line('${w(i, NodeField.expectedPrecision)} = pr * (1.0f - pr);');
  }

  String anyObserved(int i) {
    final children = [
      ...net.valueChildrenOf(i),
      ...net.volatilityChildrenOf(i),
    ];
    if (children.isEmpty) return 'false';
    return children
        .map((c) => '(${r(c, NodeField.observed)} != 0.0f)')
        .join(' || ');
  }

  void _predictionErrorContinuous(int i) {
    final leaf = isLeaf(i);
    line('let epx = ${r(i, NodeField.expectedPrecision)};');
    line(
      'let delta = ${r(i, NodeField.mean)} - '
      '${r(i, NodeField.expectedMean)};',
    );
    line('${w(i, NodeField.valuePredictionError)} = delta;');
    line(
      '${w(i, NodeField.volatilityPredictionError)} = '
      '(epx / ${r(i, NodeField.precision)}) + epx * delta * delta - 1.0f;',
    );
    if (!leaf) {
      line(
        '${w(i, NodeField.observed)} = '
        'select(0.0f, 1.0f, ${anyObserved(i)});',
      );
    }
  }

  void _predictionErrorBinary(int i) {
    line('let epx = ${r(i, NodeField.expectedPrecision)};');
    line(
      '${w(i, NodeField.valuePredictionError)} = '
      '(${r(i, NodeField.mean)} - ${r(i, NodeField.expectedMean)}) * '
      '${r(i, NodeField.observed)} / epx;',
    );
    line('${w(i, NodeField.precision)} = epx;');
  }

  // ---- posterior building blocks ----

  String relaxedGain(int c, {required bool meanField}) {
    final cep = r(c, NodeField.expectedPrecision);
    if (meanField || kind(c) != NodeKind.continuousState) return cep;
    final cond = r(c, NodeField.conditionalExpectedPrecision);
    if (isLeaf(c)) return cond;
    final cp = r(c, NodeField.precision);
    return '($cond * $cp / ($cond + ($cp - $cep)))';
  }

  String relaxedEffective(int c, {required bool meanField}) {
    final cep = r(c, NodeField.expectedPrecision);
    if (meanField || kind(c) != NodeKind.continuousState) return cep;
    final cond = r(c, NodeField.conditionalExpectedPrecision);
    if (isLeaf(c)) return cond;
    final cp = r(c, NodeField.precision);
    return '($cond * ($cp - $cep) / ($cond + ($cp - $cep)))';
  }

  /// Declares [out] = the value-children precision increment of [i].
  void valuePrecisionWpe(int i, String out, {required bool meanField}) {
    line('var $out = 0.0f;');
    final fn = net.valueChildrenOf(i).isEmpty ? null : couplingFn(i);
    for (final c in net.valueChildrenOf(i)) {
      final alpha = a(i, c);
      final ob = r(c, NodeField.observed);
      if (fn == CouplingFn.linear) {
        line(
          '$out += ${relaxedEffective(c, meanField: meanField)} * '
          '$alpha * $alpha * $ob;',
        );
      } else {
        final pm = r(i, NodeField.mean);
        final gp = tmp('gp');
        line('let $gp = ${dg(fn!, pm)};');
        line(
          '$out += (${relaxedEffective(c, meanField: meanField)} * '
          '$alpha * $alpha * $gp * $gp - '
          '${relaxedGain(c, meanField: meanField)} * $alpha * '
          '${d2g(fn, pm)} * ${r(c, NodeField.valuePredictionError)}) * $ob;',
        );
      }
    }
  }

  /// Declares [out] = the value-children mean increment of [i] at [prec].
  void valueMeanWpe(int i, String prec, String out, {required bool meanField}) {
    line('var $out = 0.0f;');
    final fn = net.valueChildrenOf(i).isEmpty ? null : couplingFn(i);
    for (final c in net.valueChildrenOf(i)) {
      final gp = fn == CouplingFn.linear
          ? ''
          : ' * ${dg(fn!, r(i, NodeField.mean))}';
      line(
        '$out += (${a(i, c)}$gp * '
        '${relaxedGain(c, meanField: meanField)} / $prec) * '
        '(${r(c, NodeField.valuePredictionError)} * '
        '${r(c, NodeField.observed)});',
      );
    }
  }

  /// `acc +=` the standard volatility precision increment of [i].
  void volatilityPrecisionWpe(int i, String acc) {
    for (final c in net.volatilityChildrenOf(i)) {
      final kappa = k(i, c);
      final e = r(c, NodeField.effectivePrecision);
      final v = r(c, NodeField.volatilityPredictionError);
      final ke = tmp('ke');
      line('let $ke = $kappa * $e;');
      line(
        '$acc += (0.5f * $ke * $ke + $ke * $ke * $v - '
        '0.5f * $kappa * $kappa * $e * $v) * ${r(c, NodeField.observed)};',
      );
    }
  }

  void volatilityMeanWpe(int i, String prec, String out) {
    line('var $out = 0.0f;');
    for (final c in net.volatilityChildrenOf(i)) {
      line(
        '$out += (${k(i, c)} * ${r(c, NodeField.effectivePrecision)} * '
        '${r(c, NodeField.volatilityPredictionError)}) / (2.0f * $prec) * '
        '${r(c, NodeField.observed)};',
      );
    }
  }

  String observedOrPredicted(int i, String hasObs, String wpe) {
    final ep = r(i, NodeField.expectedPrecision);
    return 'min(select($ep, max($ep + $wpe, ${_F32.minNormal}), $hasObs), '
        '${_lit(net.maxPosteriorPrecision)})';
  }

  /// ω̃ of volatility child [c] seen from parent [i].
  String effectiveTonic(int i, int c) {
    final terms = [
      r(c, NodeField.tonicVolatility),
      for (final p in net.volatilityParentsOf(c))
        if (p != i) '${k(p, c)} * ${r(p, NodeField.expectedMean)}',
    ];
    return '(${terms.join(' + ')})';
  }

  String mgfTerms(int c) => [
    for (final p in net.volatilityParentsOf(c))
      '${k(p, c)} * ${k(p, c)} / (2.0f * ${r(p, NodeField.expectedPrecision)})',
  ].join(' + ');

  String valueCouplingVariance(int c) =>
      'max(1.0f / ${r(c, NodeField.expectedPrecision)} - '
      '1.0f / ${r(c, NodeField.conditionalExpectedPrecision)}, 0.0f)';

  void _posteriorStandard(int i, {required bool meanField}) {
    line('let hasObs = ${anyObserved(i)};');
    valuePrecisionWpe(i, 'wpe', meanField: meanField);
    volatilityPrecisionWpe(i, 'wpe');
    line('let prec = ${observedOrPredicted(i, 'hasObs', 'wpe')};');
    valueMeanWpe(i, 'prec', 'vm', meanField: meanField);
    volatilityMeanWpe(i, 'prec', 'volm');
    line('${w(i, NodeField.precision)} = prec;');
    line(
      '${w(i, NodeField.mean)} = ${r(i, NodeField.expectedMean)} + '
      '(vm + volm);',
    );
  }

  void _posteriorEhgf(int i, {required bool meanField}) {
    line('let epi = ${r(i, NodeField.expectedPrecision)};');
    valueMeanWpe(i, 'epi', 'vm', meanField: meanField);
    volatilityMeanWpe(i, 'epi', 'volm');
    line(
      '${w(i, NodeField.mean)} = ${r(i, NodeField.expectedMean)} + '
      'vm + volm;',
    );
    line('let hasObs = ${anyObserved(i)};');
    valuePrecisionWpe(i, 'wpe', meanField: meanField);
    final mgf = !meanField && net.ehgfMgfCorrection;
    for (final c in net.volatilityChildrenOf(i)) {
      final tonic = mgf
          ? '${effectiveTonic(i, c)} + ${mgfTerms(c)}'
          : effectiveTonic(i, c);
      line('wpe += ehgfIncrement(');
      line(
        '  ${r(c, NodeField.currentVariance)} + ${valueCouplingVariance(c)},',
      );
      line('  ${k(i, c)}, ${r(i, NodeField.mean)}, $tonic, dt,');
      line(
        '  ${r(c, NodeField.mean)} - ${r(c, NodeField.expectedMean)}, '
        '${r(c, NodeField.precision)}, ${r(c, NodeField.observed)});',
      );
    }
    line(
      '${w(i, NodeField.precision)} = '
      '${observedOrPredicted(i, 'hasObs', 'wpe')};',
    );
  }

  void _posteriorUnbounded(int i) {
    final mf = meanField;
    final children = net.volatilityChildrenOf(i);
    line('let emi = ${r(i, NodeField.expectedMean)};');
    if (net.valueChildrenOf(i).isEmpty) {
      line('let pp = ${r(i, NodeField.expectedPrecision)};');
      line('let pm = emi;');
    } else {
      valuePrecisionWpe(i, 'vpw', meanField: mf);
      line(
        'let pp = max(${r(i, NodeField.expectedPrecision)} + vpw, '
        '${_F32.minNormal});',
      );
      valueMeanWpe(i, 'pp', 'vmw', meanField: mf);
      line('let pm = emi + vmw;');
    }
    final nc = children.length;
    // Per child: observed, coupling, ω̃, previous variance, b, vope.
    for (var q = 0; q < nc; q++) {
      final c = children[q];
      line('let ob_$q = ${r(c, NodeField.observed)} != 0.0f;');
      line('let ck_$q = ${k(i, c)};');
      line('let to_$q = ${effectiveTonic(i, c)};');
      line(
        'let pv_$q = max(${r(c, NodeField.currentVariance)} + '
        '${valueCouplingVariance(c)}, ${_F32.minNormal});',
      );
      line(
        'let d_$q = ${r(c, NodeField.mean)} - '
        '${r(c, NodeField.expectedMean)};',
      );
      line('let b_$q = 1.0f / ${r(c, NodeField.precision)} + d_$q * d_$q;');
      line('let vo_$q = ${r(c, NodeField.expectedPrecision)} * b_$q - 1.0f;');
    }
    final nx = nc + 1;
    // Expansion 0: at the prediction, concave part only.
    line('var P_0 = pp;');
    line('var S_0 = pp * (pm - emi);');
    for (var q = 0; q < nc; q++) {
      open('if (ob_$q)');
      line('let logS = logT + ck_$q * emi + to_$q;');
      line(
        'let wq = select(sigm(logS - log(pv_$q)), '
        'exp(logS) / (exp(logS) + pv_$q), logS <= ${_F32.linearLogSMax});',
      );
      line('P_0 += 0.5f * ck_$q * ck_$q * wq * (1.0f - wq);');
      line('S_0 += 0.5f * ck_$q * wq * vo_$q;');
      close();
    }
    line('let M_0 = emi + S_0 / P_0;');
    line('let V_0 = true;');
    // Expansions 1..nc: at each child's Lambert W0 mode.
    for (var x = 1; x < nx; x++) {
      final q = x - 1;
      line('var P_$x = 0.0f;');
      line('var M_$x = 0.0f;');
      line('var V_$x = false;');
      open('if (ob_$q && ck_$q != 0.0f)');
      line('let pihat = pp / (ck_$q * ck_$q);');
      line('let gam = logT + ck_$q * pm + to_$q;');
      line('var warg = b_$q / (2.0f * pihat) * exp(0.5f / pihat - gam);');
      open('if (!(warg > 0.0f && isFinite(warg)))');
      line('let logW = log(b_$q) - log(2.0f * pihat) + 0.5f / pihat - gam;');
      line('warg = exp(min(logW, ${_F32.logMaxFinite}));');
      close();
      line(
        'let xs = (gam + lambertW0(warg) - 0.5f / pihat - logT - to_$q) '
        '/ ck_$q;',
      );
      line('var full = pp;');
      line('var conc = pp;');
      line('var grad = -pp * (xs - pm);');
      for (var r2 = 0; r2 < nc; r2++) {
        open('if (ob_$r2)');
        line('let logS = logT + ck_$r2 * xs + to_$r2;');
        line('var wr = 0.0f;');
        line('var invD = 0.0f;');
        open('if (logS <= ${_F32.linearLogSMax})');
        line('let sv = exp(logS);');
        line('invD = 1.0f / (sv + pv_$r2);');
        line('wr = sv * invD;');
        close('} else {');
        _indent++;
        line('let logP = log(pv_$r2);');
        line('wr = sigm(logS - logP);');
        line('invD = exp(-logAddExp(logP, logS));');
        close();
        line('let da = b_$r2 * invD - 1.0f;');
        line(
          'full += 0.5f * ck_$r2 * ck_$r2 * wr * '
          '(wr + (2.0f * wr - 1.0f) * da);',
        );
        line('conc += 0.5f * ck_$r2 * ck_$r2 * wr * (1.0f - wr);');
        line('grad += 0.5f * ck_$r2 * wr * da;');
        close();
      }
      line('P_$x = select(full, conc, full <= 0.0f);');
      line('M_$x = xs + grad / P_$x;');
      line('V_$x = isFinite(P_$x) && isFinite(M_$x);');
      close();
    }
    // Energies.
    for (var x = 0; x < nx; x++) {
      line('var E_$x = -0.5f * pp * (M_$x - pm) * (M_$x - pm);');
      for (var q = 0; q < nc; q++) {
        open('if (ob_$q)');
        line('let logS = logT + ck_$q * M_$x + to_$q;');
        open('if (logS <= ${_F32.linearLogSMax})');
        line('let den = exp(logS) + pv_$q;');
        line('E_$x += -0.5f * log(den) - 0.5f * b_$q / den;');
        close('} else {');
        _indent++;
        line('let logD = logAddExp(log(pv_$q), logS);');
        line('E_$x += -0.5f * logD - 0.5f * b_$q * exp(-logD);');
        close();
        close();
      }
    }
    // Softmax and moment matching over the valid expansions.
    line('var maxE = E_0;');
    for (var x = 1; x < nx; x++) {
      line('if (V_$x && E_$x > maxE) { maxE = E_$x; }');
    }
    line('var tot = 0.0f;');
    for (var x = 0; x < nx; x++) {
      line('let u_$x = select(0.0f, exp(E_$x - maxE), V_$x);');
      line('tot += u_$x;');
    }
    line('var pmean = 0.0f;');
    for (var x = 0; x < nx; x++) {
      line('if (V_$x) { pmean += (u_$x / tot) * M_$x; }');
    }
    line('var pvar = 0.0f;');
    for (var x = 0; x < nx; x++) {
      line(
        'if (V_$x) { pvar += (u_$x / tot) * '
        '(1.0f / P_$x + (M_$x - pmean) * (M_$x - pmean)); }',
      );
    }
    line('let hasObs = ${anyObserved(i)};');
    line(
      '${w(i, NodeField.precision)} = min(select('
      '${r(i, NodeField.expectedPrecision)}, 1.0f / pvar, hasObs), '
      '${_lit(net.maxPosteriorPrecision)});',
    );
    line('${w(i, NodeField.mean)} = pmean;');
  }

  void _response(List<int> inputs) {
    line('// response');
    switch (response) {
      case GpuResponse.totalGaussianSurprise:
        line('var s = 0.0f;');
        for (var i = 0; i < n; i++) {
          if (kind(i) != NodeKind.continuousState) continue;
          line(
            's += gaussianSurprise(${r(i, NodeField.mean)}, '
            '${r(i, NodeField.expectedMean)}, '
            '${r(i, NodeField.expectedPrecision)});',
          );
        }
      case GpuResponse.firstLevelBinarySurprise:
        line('var s = 0.0f;');
        for (final i in inputs) {
          line(
            's += binarySurprise(${r(i, NodeField.mean)}, '
            '${r(i, NodeField.expectedMean)});',
          );
        }
    }
    line('bad = bad || !isFinite(s);');
    // Kahan summation.
    line('let yk = s - comp;');
    line('let tk = total + yk;');
    line('comp = (tk - total) - yk;');
    line('total = tk;');
  }
}

/// A finite `f64` as an `f32` WGSL literal.
String _lit(double x) {
  if (!x.isFinite || x.abs() > 3.4e38) {
    throw ArgumentError.value(x, 'x', 'Not representable as an f32 literal');
  }
  var s = x.toString();
  if (!s.contains('.') && !s.contains('e')) s = '$s.0';
  return '${s}f';
}

const String _prelude = r'''
fn isFinite(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
fn floorVar(v: f32) -> f32 { return max(v, 1.17549435e-38f); }
fn sigm(x: f32) -> f32 { return 1.0f / (1.0f + exp(-x)); }
fn dsigm(x: f32) -> f32 { let s = sigm(x); return s * (1.0f - s); }
fn d2sigm(x: f32) -> f32 {
  let s = sigm(x);
  return s * (1.0f - s) * (1.0f - 2.0f * s);
}
fn dtanh(x: f32) -> f32 { let t = tanh(x); return 1.0f - t * t; }
fn d2tanh(x: f32) -> f32 { let t = tanh(x); return -2.0f * t * (1.0f - t * t); }
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
fn ehgfIncrement(prevVar: f32, kappa: f32, mean: f32, tonic: f32, dt: f32,
    d: f32, prec: f32, observed: f32) -> f32 {
  let pv = dt * exp(kappa * mean + tonic);
  let ep = 1.0f / (prevVar + pv);
  let eff = pv * ep;
  let weight = (pv - prevVar) * ep;
  let vope = (1.0f / prec + d * d) * ep - 1.0f;
  let inc = 0.5f * kappa * kappa * eff * (eff + weight * vope);
  return max(inc, 0.0f) * observed;
}
fn gaussianSurprise(x: f32, mu: f32, pi: f32) -> f32 {
  let d = x - mu;
  return 0.5f * (1.83787707f - log(pi) + pi * d * d);
}
fn binarySurprise(x: f32, mu: f32) -> f32 {
  let m = clamp(mu, 1e-6f, 1.0f - 1e-6f);
  return select(-log(1.0f - m), -log(m), x != 0.0f);
}
''';
