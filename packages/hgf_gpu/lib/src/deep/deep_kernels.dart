/// WGSL kernels for hgf_deep's `DeepNetwork`: one batched learning step
/// (`batchUpdate`) and the mean-chain `predict`, in f32.
///
/// Each layer is one fused kernel: the weight contraction is a tiled GEMM,
/// and the layer's elementwise belief updates run as its epilogue, so a
/// step is about `3·layers + 2` dispatches with no host round trip. The
/// arithmetic mirrors `hgf_deep`'s `kernels.dart` term for term (which
/// mirrors pyhgf's vectorised updates).
library;

import 'dart:math' as math;

import 'package:hgf_deep/hgf_deep.dart';

import '../trial.dart' show prelude;

part 'deep_fit_kernel.dart';

/// The buffers a deep kernel binds, in binding order.
enum DeepBinding {
  /// Parameters: weights, optimiser moments, per-node constants and the
  /// network's own (width-1) beliefs.
  params('P', 'read_write'),

  /// Batch state: every layer's beliefs for up to `capacity` samples, and
  /// scratch.
  state('S', 'read_write'),

  /// Predictors, `rows × topSize`.
  x('X', 'read'),

  /// Observations, `rows × bottomSize`.
  y('Y', 'read'),

  /// Results: input-layer errors (`rows × topSize`) or predictions
  /// (`rows × bottomSize`).
  out('OUT', 'read_write'),

  /// The per-call uniform.
  uniform('U', ''),

  /// Adam's bias corrections per step of a sequential fit (`2·steps`).
  biasCorrection('BC', 'read');

  const DeepBinding(this.name, this.access);
  final String name;
  final String access;
}

/// The optimiser a gradient kernel applies.
enum DeepOptimiserKind { sgd, adam }

/// One generated kernel: its WGSL, the bindings it uses, and how to size
/// its grid.
final class DeepKernelSource {
  DeepKernelSource(this.name, this.source, this.bindings, this.grid);

  final String name;
  final String source;

  /// Which [DeepBinding]s the kernel declares (the others are left out of
  /// its bind group).
  final Set<DeepBinding> bindings;

  /// Workgroups along x and y, given the batch width `b`.
  final (int, int) Function(int b) grid;
}

/// Offsets of a layer's fields in the batch state (`[node][capacity]`
/// each) and its template fields in the parameter buffer.
final class _LayerLayout {
  final state = <LayerField, int>{};
  final template = <LayerField, int>{};
  int? coupled, perParent, effc, gain, u, h, predicted;

  /// The known variance of the value-level prediction (DeepFixes.logSpaceVolatility).
  int? known;

  /// Weight-gradient partial sums (split-K over samples), and how many.
  int? partial;
  int splits = 1;
  int? tonicVol, tonic, prior;
  int? weights, mu, nu;
}

/// Generates the kernels of one network shape for a batch capacity.
///
/// Shapes, kinds, couplings and the network's settings are compiled in;
/// the batch width (up to [capacity]), the row offset, the time step,
/// the learning kind and the optimiser's hyperparameters are uniforms.
final class DeepKernels {
  DeepKernels(DeepNetwork net, {required this.capacity})
    : settings = net.settings,
      layers = net.debugLayers,
      _pb = '',
      _sb = '' {
    _check();
  }

  /// The layout and kernel of a sequential `fit` over an ensemble of
  /// networks of [net]'s shape ([sequentialFit]): every member has its own
  /// width-1 copy of the parameter and state layouts, `paramFloats` and
  /// `stateFloats` apart, and the generated code addresses them relative
  /// to the member's base offsets `pb` and `sb`.
  DeepKernels.sequential(DeepNetwork net)
    : capacity = 1,
      settings = net.settings,
      layers = net.debugLayers,
      _pb = 'pb + ',
      _sb = 'sb + ' {
    _check();
  }

  void _check() {
    if (layers.length < 2) {
      throw ArgumentError('A GPU deep network needs at least two layers.');
    }
    if (capacity < 1) throw ArgumentError.value(capacity, 'capacity');
    for (final l in layers) {
      if (l.coupling is! CouplingFn) {
        throw UnsupportedError(
          'Custom coupling functions have no GPU kernel (layer ${l.index}).',
        );
      }
    }
    _layout();
  }

  /// Largest batch width a call can use.
  final int capacity;
  final DeepSettings settings;
  final List<Layer> layers;

  /// Base-offset prefixes of `P` and `S` indices (empty for the batch
  /// kernels; the member's bases for [DeepKernels.sequential]).
  final String _pb, _sb;

  final _l = <_LayerLayout>[];
  late final int paramFloats, stateFloats;

  int get nLayers => layers.length;
  int get topSize => layers.last.size;
  int get bottomSize => layers.first.size;

  // --- Layout -------------------------------------------------------------

  void _layout() {
    var p = 0, s = 0;
    final bm = capacity;
    for (var i = 0; i < layers.length; i++) {
      final layer = layers[i];
      final n = layer.size;
      final ll = _LayerLayout();
      final fields = [
        for (final f in LayerField.values)
          if (!f.volatility || layer.hasVolatility) f,
      ];
      for (final f in fields) {
        ll.template[f] = p;
        p += n;
        ll.state[f] = s;
        s += n * bm;
      }
      ll.tonicVol = p;
      p += n;
      if (layer.tonicVolatility != null) {
        ll.tonic = p;
        p += n;
      }
      ll.prior = p;
      p += n;
      final w = layer.weights;
      if (w != null) {
        ll.splits = _gradSplits(w.rows, w.cols);
        if (ll.splits > 1) {
          ll.partial = s;
          s += ll.splits * w.data.length;
        }
        ll.weights = p;
        p += w.data.length;
        ll.mu = p;
        p += w.data.length;
        ll.nu = p;
        p += w.data.length;
      }
      final isTop = i == layers.length - 1;
      final rows = n + (layer.hasConstant ? 1 : 0);
      if (i >= 1) {
        // Parent of the layer below: g(μ̂) for the prediction, g(μ) for the
        // gradient, g(μ̂) of the mean chain for predict().
        ll.coupled = s;
        s += rows * bm;
        ll.h = s;
        s += rows * bm;
        ll.predicted = s;
        s += rows * bm;
        if (settings.feedforwardUncertainty) {
          ll.perParent = s;
          s += rows * bm;
        }
      }
      if (_storesKnownVariance(i)) {
        ll.known = s;
        s += n * bm;
      }
      if (!isTop) {
        ll.gain = s;
        s += n * bm;
        ll.u = s;
        s += n * bm;
        ll.effc = s;
        s += n * bm;
      }
      if (i == 0) {
        // Raw logits of predict() on a categorical output.
        ll.predicted = s;
        s += n * bm;
      }
      _l.add(ll);
    }
    paramFloats = p;
    stateFloats = s;
  }

  /// Offset of [layer]'s template [field] in the parameter buffer.
  int templateOffset(int layer, LayerField field) => _l[layer].template[field]!;

  /// Offset of [layer]'s incoming weights (`childSize × cols`) in the
  /// parameter buffer.
  int weightsOffset(int layer) => _l[layer].weights!;
  int muOffset(int layer) => _l[layer].mu!;
  int nuOffset(int layer) => _l[layer].nu!;
  int tonicVolOffset(int layer) => _l[layer].tonicVol!;
  int? tonicOffset(int layer) => _l[layer].tonic;
  int priorOffset(int layer) => _l[layer].prior!;

  /// Bias rows (constant 1) in the state buffer, as (offset, length), to
  /// fill once.
  List<(int, int)> get biasRows => [
    for (var i = 1; i < layers.length; i++)
      if (layers[i].hasConstant)
        for (final base in [_l[i].coupled!, _l[i].h!, _l[i].predicted!])
          (base + layers[i].size * capacity, capacity),
  ];

  // --- WGSL helpers ---------------------------------------------------------

  String get _bm => '${capacity}u';

  /// `S[...]` of field [f] of layer [l] at flat index [k] (`i·cap + s`).
  String _s(int l, LayerField f, String k) => 'S[$_sb${_l[l].state[f]!}u + $k]';
  String _t(int l, LayerField f, String i) =>
      'P[$_pb${_l[l].template[f]!}u + $i]';
  String _at(int base, String k) => 'S[$_sb${base}u + $k]';

  /// `P[...]` at [base] + [k] (member-relative in a sequential kernel).
  String _p(int base, String k) => 'P[$_pb${base}u + $k]';

  static String _f(double v) {
    if (!v.isFinite) throw ArgumentError('Non-finite constant $v');
    final s = v.toString();
    return s.contains('.') || s.contains('e') ? '${s}f' : '$s.0f';
  }

  String _g(CouplingFn fn, String x) => switch (fn) {
    CouplingFn.linear => x,
    CouplingFn.relu => 'select(0.0f, $x, $x > 0.0f)',
    CouplingFn.sigmoid => 'sigm($x)',
    CouplingFn.tanh => 'tanhS($x)',
    CouplingFn.leakyRelu => 'select(0.01f * $x, $x, $x >= 0.0f)',
    CouplingFn.gelu => 'gelu($x)',
  };

  String _dg(CouplingFn fn, String x) => switch (fn) {
    CouplingFn.linear => '1.0f',
    CouplingFn.relu => 'select(0.0f, 1.0f, $x > 0.0f)',
    CouplingFn.sigmoid => 'dsigm($x)',
    CouplingFn.tanh => 'dtanhS($x)',
    CouplingFn.leakyRelu => 'select(0.01f, 1.0f, $x >= 0.0f)',
    CouplingFn.gelu => 'dgelu($x)',
  };

  String _d2g(CouplingFn fn, String x) => switch (fn) {
    CouplingFn.linear || CouplingFn.relu || CouplingFn.leakyRelu => '0.0f',
    CouplingFn.sigmoid => 'd2sigm($x)',
    CouplingFn.tanh => 'd2tanhS($x)',
    CouplingFn.gelu => 'd2gelu($x)',
  };

  CouplingFn _fn(int l) => layers[l].coupling as CouplingFn;

  static const _extra = r'''
// f32 has no 1e-128: the smallest normal f32 stands in for pyhgf's floors.
const MINV: f32 = 1.17549435e-38f;
fn nan() -> f32 { return bitcast<f32>(U.nanBits); }
fn guardV(v: f32) -> f32 { return select(nan(), v, v > MINV); }
fn floorV(x: f32, lo: f32) -> f32 { return select(x, lo, x < lo); }
// min/max that propagate NaN like Dart's (and jnp's): WGSL's may not.
fn isNan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn minN(a: f32, b: f32) -> f32 { return select(min(a, b), nan(), isNan(a) || isNan(b)); }
fn maxN(a: f32, b: f32) -> f32 { return select(max(a, b), nan(), isNan(a) || isNan(b)); }
fn tanhS(x: f32) -> f32 {
  let e = exp(-2.0f * abs(x));
  let t = (1.0f - e) / (1.0f + e);
  return select(-t, t, x >= 0.0f);
}
fn dtanhS(x: f32) -> f32 { let t = tanhS(x); return 1.0f - t * t; }
fn d2tanhS(x: f32) -> f32 { let t = tanhS(x); return -2.0f * t * (1.0f - t * t); }
// W0(exp(l)) without forming exp(l), which overflows f32 above l ≈ 88:
// Newton on w + ln w = l.
fn lambertWExp(l: f32) -> f32 {
  if (l < 1.0f) { return lambertW0(exp(l)); }
  var w = l - log(l);
  for (var i = 0; i < 8; i++) {
    let st = (w + log(w) - l) / (1.0f + 1.0f / w);
    w -= st;
    if (abs(st) <= 1e-7f * abs(w)) { break; }
  }
  return w;
}
''';

  static const _uniformStruct = '''
struct Uniform {
  b: u32,
  start: u32,
  flags: u32,
  nanBits: u32,
  t: f32,
  lr: f32,
  b1: f32,
  b2: f32,
  eps: f32,
  epsRoot: f32,
  bc1: f32,
  bc2: f32,
}
''';

  /// Wraps [body] with the declarations of the bindings it uses.
  DeepKernelSource _kernel(
    String name,
    String body,
    (int, int) Function(int b) grid, {
    String uniformStruct = _uniformStruct,
  }) {
    final used = <DeepBinding>{DeepBinding.uniform};
    final decl = StringBuffer(uniformStruct);
    for (final b in DeepBinding.values) {
      if (b == DeepBinding.uniform) {
        decl.writeln('@group(0) @binding(${b.index}) var<uniform> U: Uniform;');
        continue;
      }
      if (!RegExp('\\b${b.name}\\[').hasMatch(body)) continue;
      used.add(b);
      decl.writeln(
        '@group(0) @binding(${b.index}) '
        'var<storage, ${b.access}> ${b.name}: array<f32>;',
      );
    }
    return DeepKernelSource(
      name,
      '// $name\n$decl\n$prelude\n$_extra\n$body',
      used,
      grid,
    );
  }

  /// An elementwise kernel over `rows × b`: `i` (node) and `s` (sample).
  DeepKernelSource _elementwise(String name, int rows, String body) =>
      _kernel(name, '''
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  let i = gid.y;
  if (s >= U.b) { return; }
  let k = i * $_bm + s;
$body
}
''', (b) => ((b + 63) ~/ 64, rows));

  /// Workgroup shape (x, y) of a GEMM over at most [mMax] × [nMax]
  /// outputs: 4×4 outputs per thread, at most 256 threads.
  static (int, int) _tile(int mMax, int nMax) {
    final wy = math.min(_pow2AtLeast((mMax + 3) ~/ 4), 16);
    final wx = math.min(_pow2AtLeast((nMax + 3) ~/ 4), math.min(256 ~/ wy, 64));
    return (wx, wy);
  }

  /// Sample chunks the weight gradient of a `rows × cols` layer is split
  /// into, so that small layers still fill the GPU: about 128 workgroups,
  /// each chunk at least 256 samples of the capacity (below that the extra
  /// dispatch costs more than it saves: measured at capacity 256).
  int _gradSplits(int rows, int cols) {
    final (wx, wy) = _tile(rows, cols);
    final tiles =
        ((rows + 4 * wy - 1) ~/ (4 * wy)) * ((cols + 4 * wx - 1) ~/ (4 * wx));
    return math.max(1, math.min((128 + tiles - 1) ~/ tiles, capacity ~/ 256));
  }

  static int _pow2AtLeast(int n) {
    var p = 1;
    while (p < n) {
      p <<= 1;
    }
    return p;
  }

  /// A tiled GEMM `acc[r][c] = Σ_k A(r, k)·B(k, c)` over `r < m`, `c < n`,
  /// `k < depth`, with a second accumulator when [b1] is given (its A is
  /// A(r, k)² when [squareA1]); each thread holds a 4×4 register tile and
  /// calls `epi(r, c, acc0, acc1)` for each in-range output.
  ///
  /// [a] and [b0]/[b1] are WGSL expressions in `r`/`k` and `k`/`c`;
  /// [aAlongK]/[bAlongK] say which index is contiguous in memory, for
  /// coalesced tile loads. [m], [n], [depth] are WGSL `u32` expressions;
  /// [mMax] and [nMax] bound them, to size the tile.
  DeepKernelSource _gemm(
    String name, {
    required String m,
    required int mMax,
    required String n,
    required int nMax,
    required String depth,
    required String a,
    required bool aAlongK,
    required String b0,
    String? b1,
    bool squareA1 = false,
    required bool bAlongK,
    required String epilogue,
    required (int, int) Function(int b) Function(int tm, int tn) grid,
    int splits = 1,
  }) {
    const tk = 16;
    final (wx, wy) = _tile(mMax, nMax);
    final tm = 4 * wy, tn = 4 * wx, nt = wx * wy;
    final two = b1 != null;
    final buf = StringBuffer()
      ..writeln('var<workgroup> As: array<f32, ${tk * tm}>;')
      ..writeln('var<workgroup> Bs0: array<f32, ${tk * tn}>;');
    if (two) buf.writeln('var<workgroup> Bs1: array<f32, ${tk * tn}>;');
    buf
      ..writeln('fn epi(r: u32, c: u32, acc0: f32, acc1: f32, split: u32) {')
      ..writeln(epilogue)
      ..writeln('}')
      ..writeln('fn loadA(r: u32, k: u32) -> f32 { return $a; }')
      ..writeln('fn loadB0(k: u32, c: u32) -> f32 { return $b0; }');
    if (two) buf.writeln('fn loadB1(k: u32, c: u32) -> f32 { return $b1; }');
    buf
      ..writeln('@compute @workgroup_size($wx, $wy)')
      ..writeln(
        'fn main(@builtin(local_invocation_id) lid: vec3<u32>, '
        '@builtin(workgroup_id) wid: vec3<u32>) {',
      )
      ..writeln('  let M = $m; let N = $n; let K = $depth;')
      ..writeln('  let tx = lid.x; let ty = lid.y;')
      ..writeln('  let li = ty * ${wx}u + tx;')
      ..writeln('  let r0 = wid.y * ${tm}u;');
    if (splits > 1) {
      // Split-K: x = split · xTiles + column tile; each split sums one
      // chunk of k (a multiple of the tile depth).
      final xTiles = (nMax + tn - 1) ~/ tn;
      buf
        ..writeln('  let split = wid.x / ${xTiles}u;')
        ..writeln('  let c0 = (wid.x % ${xTiles}u) * ${tn}u;')
        ..writeln(
          '  let chunk = ((K + ${splits - 1}u) / ${splits}u + ${tk - 1}u) '
          '/ ${tk}u * ${tk}u;',
        )
        ..writeln('  let kb = min(K, split * chunk);')
        ..writeln('  let ke = min(K, kb + chunk);');
    } else {
      buf
        ..writeln('  let split = 0u;')
        ..writeln('  let c0 = wid.x * ${tn}u;')
        ..writeln('  let kb = 0u; let ke = K;');
    }
    for (var acc = 0; acc < (two ? 2 : 1); acc++) {
      for (var ii = 0; ii < 4; ii++) {
        for (var jj = 0; jj < 4; jj++) {
          buf.writeln('  var a${acc}_${ii}_$jj = 0.0f;');
        }
      }
    }
    buf.writeln('  for (var k0 = kb; k0 < ke; k0 += ${tk}u) {');
    // A tile: tk × tm, stored As[kk * tm + rr].
    buf.writeln('    for (var e = li; e < ${tk * tm}u; e += ${nt}u) {');
    if (aAlongK) {
      buf.writeln('      let rr = e / ${tk}u; let kk = e % ${tk}u;');
    } else {
      buf.writeln('      let kk = e / ${tm}u; let rr = e % ${tm}u;');
    }
    buf
      ..writeln('      let r = r0 + rr; let k = k0 + kk;')
      ..writeln('      var v = 0.0f;')
      ..writeln('      if (r < M && k < ke) { v = loadA(r, k); }')
      ..writeln('      As[kk * ${tm}u + rr] = v;')
      ..writeln('    }')
      ..writeln('    for (var e = li; e < ${tk * tn}u; e += ${nt}u) {');
    if (bAlongK) {
      buf.writeln('      let cc = e / ${tk}u; let kk = e % ${tk}u;');
    } else {
      buf.writeln('      let kk = e / ${tn}u; let cc = e % ${tn}u;');
    }
    buf
      ..writeln('      let c = c0 + cc; let k = k0 + kk;')
      ..writeln('      var v0 = 0.0f;');
    if (two) buf.writeln('      var v1 = 0.0f;');
    buf.writeln('      if (c < N && k < ke) {');
    buf.writeln('        v0 = loadB0(k, c);');
    if (two) buf.writeln('        v1 = loadB1(k, c);');
    buf
      ..writeln('      }')
      ..writeln('      Bs0[kk * ${tn}u + cc] = v0;');
    if (two) buf.writeln('      Bs1[kk * ${tn}u + cc] = v1;');
    buf
      ..writeln('    }')
      ..writeln('    workgroupBarrier();')
      ..writeln('    for (var kk = 0u; kk < ${tk}u; kk++) {');
    for (var ii = 0; ii < 4; ii++) {
      buf.writeln('      let x$ii = As[kk * ${tm}u + ty + ${ii * wy}u];');
      if (two && squareA1) buf.writeln('      let q$ii = x$ii * x$ii;');
    }
    for (var jj = 0; jj < 4; jj++) {
      buf.writeln('      let y$jj = Bs0[kk * ${tn}u + tx + ${jj * wx}u];');
      if (two) {
        buf.writeln('      let z$jj = Bs1[kk * ${tn}u + tx + ${jj * wx}u];');
      }
    }
    for (var ii = 0; ii < 4; ii++) {
      for (var jj = 0; jj < 4; jj++) {
        buf.writeln('      a0_${ii}_$jj += x$ii * y$jj;');
        if (two) {
          buf.writeln(
            '      a1_${ii}_$jj += ${squareA1 ? 'q' : 'x'}$ii * z$jj;',
          );
        }
      }
    }
    buf
      ..writeln('    }')
      ..writeln('    workgroupBarrier();')
      ..writeln('  }');
    for (var ii = 0; ii < 4; ii++) {
      for (var jj = 0; jj < 4; jj++) {
        buf
          ..writeln('  {')
          ..writeln('    let r = r0 + ty + ${ii * wy}u;')
          ..writeln('    let c = c0 + tx + ${jj * wx}u;')
          ..writeln(
            '    if (r < M && c < N) { '
            'epi(r, c, a0_${ii}_$jj, ${two ? 'a1_${ii}_$jj' : '0.0f'}, '
            'split); }',
          )
          ..writeln('  }');
      }
    }
    buf.writeln('}');
    final g = grid(tm, tn);
    return _kernel(
      name,
      buf.toString(),
      splits == 1
          ? g
          : (b) {
              final (x, y) = g(b);
              return (x * splits, y);
            },
    );
  }

  static (int, int) Function(int b) _overSamples(int rows, int tm, int tn) =>
      (b) => ((b + tn - 1) ~/ tn, (rows + tm - 1) ~/ tm);

  // --- batchUpdate ----------------------------------------------------------

  /// The kernels of one `batchUpdate`, in dispatch order. [errors] adds the
  /// input-layer error kernel; [optimiser] the weight steps (none when
  /// `null`); [updatePrecisions] the carried-belief increment.
  List<DeepKernelSource> batchUpdate({
    required bool errors,
    DeepOptimiserKind? optimiser,
    required bool updatePrecisions,
  }) => [
    top(),
    for (var l = nLayers - 2; l >= 0; l--) ...[
      predictLayer(l),
      if (layers[l].kind == LayerKind.categorical) softmax(),
    ],
    updateBottom(),
    for (var l = 1; l < nLayers - 1; l++) posterior(l),
    if (errors) inputErrors(),
    if (optimiser != null)
      for (var l = 1; l < nLayers; l++) ...gradient(l, optimiser),
    if (updatePrecisions) carry(),
  ];

  /// Clamp `x` on the top layer: its means, its coupled outputs `g(x)` (the
  /// prediction's and the gradient's) and, with feedforward uncertainty,
  /// `g'(x)²/π̂`.
  DeepKernelSource top() => _elementwise(
    'top',
    topSize,
    _topBody('X[(U.start + s) * ${topSize}u + i]'),
  );

  /// Clamp predictor [x] (a WGSL expression) on top node `i` (expects `i`,
  /// `k` in scope).
  String _topBody(String x) {
    final l = nLayers - 1;
    final ll = _l[l];
    final fn = _fn(l);
    final body = StringBuffer()
      ..writeln('  let v = $x;')
      ..writeln('  ${_s(l, LayerField.expectedMean, 'k')} = v;')
      ..writeln('  ${_s(l, LayerField.mean, 'k')} = v;')
      ..writeln('  let gv = ${_g(fn, 'v')};')
      ..writeln('  ${_at(ll.coupled!, 'k')} = gv;')
      ..writeln('  ${_at(ll.h!, 'k')} = select(0.0f, gv, isFinite(gv));');
    if (ll.perParent != null) {
      body
        ..writeln('  let d = ${_dg(fn, 'v')};')
        ..writeln(
          '  ${_at(ll.perParent!, 'k')} = d * d / '
          '${_t(l, LayerField.expectedPrecision, 'i')};',
        );
    }
    return body.toString();
  }

  /// Predict layer [l] from the one above: `μ̂ = W·g(μ̂_parent)` and the
  /// layer's kind-specific epilogue.
  DeepKernelSource predictLayer(int l) {
    final c = layers[l];
    final pl = layers[l + 1];
    final pll = _l[l + 1];
    final cols = pl.weights!.cols;
    final w = pll.weights!;
    final ffu = _feedforward(l);
    return _gemm(
      'predict layer $l',
      m: '${c.size}u',
      mMax: c.size,
      n: 'U.b',
      nMax: capacity,
      depth: '${cols}u',
      a: 'P[${w}u + r * ${cols}u + k]',
      aAlongK: true,
      b0: _at(pll.coupled!, 'k * $_bm + c'),
      // The bias row of perParent is never written (zero-initialised).
      b1: ffu ? _at(pll.perParent!, 'k * $_bm + c') : null,
      squareA1: true,
      bAlongK: false,
      epilogue: _predictEpilogue(l),
      grid: (tm, tn) => _overSamples(c.size, tm, tn),
    );
  }

  /// Whether layer [l]'s prediction stores its known variance (the carried
  /// `1/π`) for the volatility posterior ([DeepFixes.logSpaceVolatility]).
  bool _storesKnownVariance(int l) =>
      settings.fixes.logSpaceVolatility &&
      settings.predictPrecision &&
      layers[l].kind == LayerKind.volatile &&
      layers[l].hasVolatility &&
      !layers[l].isInputLayer;

  /// Whether layer [l]'s prediction adds the feedforward uncertainty
  /// `Σ W²·g'²/π̂` of its parents (the second accumulator `acc1`).
  bool _feedforward(int l) =>
      settings.feedforwardUncertainty &&
      layers[l].kind == LayerKind.volatile &&
      !layers[l].isInputLayer &&
      settings.predictPrecision;

  /// Layer [l]'s prediction from `acc0 = (W·g(μ̂_parent))[r]` (and `acc1`)
  /// for node `r`, sample `c`.
  String _predictEpilogue(int l) {
    final c = layers[l];
    final predict = settings.predictPrecision;
    final ffu = _feedforward(l);
    final e = StringBuffer()
      ..writeln('  let i = r; let s = c; let k = i * $_bm + s;')
      ..writeln('  let mu = acc0;');
    String S(LayerField f) => _s(l, f, 'k');
    String T(LayerField f) => _t(l, f, 'i');
    switch (c.kind) {
      case LayerKind.volatile:
        final ll = _l[l];
        e
          ..writeln('  let pi = ${T(LayerField.precision)};')
          ..writeln('  ${S(LayerField.precision)} = pi;')
          ..writeln('  var predVol = 0.0f;');
        if (c.hasVolatility) {
          e
            ..writeln('  let mv = ${T(LayerField.meanVol)};')
            ..writeln('  let pv = ${T(LayerField.precisionVol)};')
            ..writeln('  ${S(LayerField.meanVol)} = mv;')
            ..writeln('  ${S(LayerField.precisionVol)} = pv;');
        }
        if (predict) {
          if (c.hasVolatility) {
            e
              ..writeln(
                '  let pvv = guardV(U.t * exp(${_p(ll.tonicVol!, 'i')}));',
              )
              ..writeln('  let ev = 1.0f / (1.0f / pv + pvv);')
              ..writeln('  ${S(LayerField.expectedMeanVol)} = mv;')
              ..writeln('  ${S(LayerField.expectedPrecisionVol)} = ev;')
              ..writeln('  ${S(LayerField.effectivePrecisionVol)} = pvv * ev;')
              ..writeln('  var total = mv;');
            if (!settings.meanFieldUpdates) {
              e.writeln('  total = total + 1.0f / (2.0f * ev);');
            }
            if (ll.tonic != null) {
              e.writeln('  total = total + ${_p(ll.tonic!, 'i')};');
            }
            e.writeln('  predVol = guardV(U.t * exp(total));');
          } else if (ll.tonic != null) {
            e.writeln('  predVol = guardV(U.t * exp(${_p(ll.tonic!, 'i')}));');
          }
        } else if (c.hasVolatility) {
          e
            ..writeln('  ${S(LayerField.expectedMeanVol)} = mv;')
            ..writeln('  ${S(LayerField.expectedPrecisionVol)} = pv;');
        }
        e.writeln('  ${S(LayerField.expectedMean)} = mu;');
        if (c.isInputLayer || !predict) {
          e
            ..writeln('  ${S(LayerField.conditionalExpectedPrecision)} = pi;')
            ..writeln('  ${S(LayerField.expectedPrecision)} = pi;')
            ..writeln('  ${S(LayerField.effectivePrecision)} = 0.0f;')
            ..writeln('  let ep = pi;');
        } else {
          e
            ..writeln('  let cond = 1.0f / floorV(1.0f / pi + predVol, MINV);')
            ..writeln(
              '  let ep = 1.0f / floorV(1.0f / cond + '
              '${ffu ? 'acc1' : '0.0f'}, MINV);',
            )
            ..writeln('  ${S(LayerField.conditionalExpectedPrecision)} = cond;')
            ..writeln('  ${S(LayerField.expectedPrecision)} = ep;');
          if (settings.fixes.logSpaceVolatility) {
            // γ = Ω/(a + Ω) = 1/(1 + a/Ω): the limit 1, not ∞·0, once Ω
            // overflows (DeepFixes.logSpaceVolatility).
            e.writeln(
              '  ${S(LayerField.effectivePrecision)} = select(predVol * ep, '
              '1.0f / (1.0f + (1.0f / pi + ${ffu ? 'acc1' : '0.0f'}) / predVol), '
              '1.0f / pi + predVol >= MINV);',
            );
          } else {
            e.writeln('  ${S(LayerField.effectivePrecision)} = predVol * ep;');
          }
          if (_l[l].known != null) {
            e.writeln('  ${_at(_l[l].known!, 'k')} = 1.0f / pi;');
          }
        }
        _coupledOut(e, l, 'mu');
      case LayerKind.binary:
        final lo = settings.precisionClippingValue;
        e
          ..writeln('  let m = minN(maxN(sigm(mu), ${_f(lo)}), ${_f(1 - lo)});')
          ..writeln('  ${S(LayerField.expectedMean)} = m;')
          ..writeln('  let ep = m * (1.0f - m);')
          ..writeln('  ${S(LayerField.expectedPrecision)} = ep;')
          ..writeln('  ${S(LayerField.conditionalExpectedPrecision)} = ep;');
        _coupledOut(e, l, 'm');
      case LayerKind.categorical:
        e.writeln('  ${S(LayerField.expectedMean)} = mu;');
    }
    return e.toString();
  }

  /// `g(μ̂)` (and `g'(μ̂)²/π̂`) of layer [l] for the layer below.
  void _coupledOut(StringBuffer e, int l, String mean) {
    if (l == 0) return;
    final ll = _l[l];
    final fn = _fn(l);
    e.writeln('  ${_at(ll.coupled!, 'k')} = ${_g(fn, mean)};');
    if (ll.perParent != null) {
      e
        ..writeln('  let dg = ${_dg(fn, mean)};')
        ..writeln('  ${_at(ll.perParent!, 'k')} = dg * dg / ep;');
    }
  }

  /// The categorical output's softmax over its nodes, per sample.
  DeepKernelSource softmax() {
    final n = bottomSize;
    String em(String i) => _s(0, LayerField.expectedMean, '$i * $_bm + s');
    return _kernel('softmax', '''
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= U.b) { return; }
  var mx = ${em('0u')};
  for (var i = 1u; i < ${n}u; i++) { mx = maxN(mx, ${em('i')}); }
  var sum = 0.0f;
  for (var i = 0u; i < ${n}u; i++) {
    let e = exp(${em('i')} - mx);
    ${em('i')} = e;
    sum += e;
  }
  for (var i = 0u; i < ${n}u; i++) {
    let k = i * $_bm + s;
    ${em('i')} = ${em('i')} / sum;
    ${_s(0, LayerField.expectedPrecision, 'k')} = 1.0f;
    ${_s(0, LayerField.conditionalExpectedPrecision, 'k')} = 1.0f;
    ${_s(0, LayerField.effectivePrecision, 'k')} = 0.0f;
  }
}
''', (b) => ((b + 63) ~/ 64, 1));
  }

  /// Clamp `y` on the bottom layer, then its prediction error.
  DeepKernelSource updateBottom() => _elementwise(
    'update layer 0',
    bottomSize,
    _bottomBody('Y[(U.start + s) * ${bottomSize}u + i]'),
  );

  /// Clamp observation [y] (a WGSL expression) on bottom node `i`, then its
  /// prediction error (expects `i`, `s`, `k` in scope).
  String _bottomBody(String y) {
    final e = StringBuffer()
      ..writeln('  let mean = $y;')
      ..writeln('  ${_s(0, LayerField.mean, 'k')} = mean;')
      ..writeln('  let prec = ${_s(0, LayerField.precision, 'k')};');
    _predictionError(e, 0);
    return e.toString();
  }

  /// The value-level posterior of layer [l] from its children's errors,
  /// then its own prediction error.
  DeepKernelSource posterior(int l) {
    final pl = layers[l];
    final cl = _l[l - 1];
    final cols = pl.weights!.cols;
    final w = _l[l].weights!;
    final nc = layers[l - 1].size;
    return _gemm(
      'posterior layer $l',
      m: '${pl.size}u',
      mMax: pl.size,
      n: 'U.b',
      nMax: capacity,
      depth: '${nc}u',
      // Wᵀ: A(r = parent j, k = child i) = W[i][j].
      a: 'P[${w}u + k * ${cols}u + r]',
      aAlongK: false,
      b0: _at(cl.gain!, 'k * $_bm + c'),
      b1: _at(cl.effc!, 'k * $_bm + c'),
      squareA1: true,
      bAlongK: false,
      epilogue: _posteriorEpilogue(l),
      grid: (tm, tn) => _overSamples(pl.size, tm, tn),
    );
  }

  /// Layer [l]'s value posterior from `acc0 = (Wᵀ·gain·δ)[r]` and
  /// `acc1 = ((W²)ᵀ·π_eff)[r]` for node `r`, sample `c`, then its prediction
  /// error and `g(μ)` for the gradient.
  String _posteriorEpilogue(int l) {
    final fn = _fn(l);
    final e = StringBuffer()
      ..writeln('  let i = r; let s = c; let k = i * $_bm + s;')
      ..writeln('  let msg = acc0; let pc1 = acc1;')
      ..writeln('  let x = ${_s(l, LayerField.expectedMean, 'k')};')
      ..writeln('  let g1 = ${_dg(fn, 'x')};')
      ..writeln('  let g2 = ${_d2g(fn, 'x')};')
      ..writeln('  let epx = ${_s(l, LayerField.expectedPrecision, 'k')};')
      ..writeln('  let info = pc1 * (g1 * g1);')
      ..writeln('  let raw = epx + info + (-g2 * msg);')
      ..writeln(
        // DeepFixes.curvatureFloor: floor at π̂ + information, not π̂.
        '  let prec = minN(maxN(raw, '
        '${settings.fixes.curvatureFloor ? 'epx + info' : 'epx'}), '
        '${_f(settings.maxPosteriorPrecision)});',
      )
      ..writeln('  ${_s(l, LayerField.precision, 'k')} = prec;')
      ..writeln('  let mean = x + msg * g1 / prec;')
      ..writeln('  ${_s(l, LayerField.mean, 'k')} = mean;');
    _predictionError(e, l);
    final gm = _g(fn, 'mean');
    e
      ..writeln('  let hv = $gm;')
      ..writeln('  ${_at(_l[l].h!, 'k')} = select(0.0f, hv, isFinite(hv));');
    return e.toString();
  }

  /// Layer [l]'s prediction error by kind (expects `mean` and `precision`
  /// in scope, and `k`), the volatility-level update, and the child-side
  /// factors its parent's posterior and gradient read.
  void _predictionError(StringBuffer e, int l) {
    final layer = layers[l];
    String S(LayerField f) => _s(l, f, 'k');
    e
      ..writeln('  let em = ${S(LayerField.expectedMean)};')
      ..writeln('  let ep = ${S(LayerField.expectedPrecision)};');
    var precision = 'prec';
    switch (layer.kind) {
      case LayerKind.binary:
        e
          ..writeln('  let vpe = (mean - em) / ep;')
          ..writeln('  ${S(LayerField.precision)} = ep;');
        precision = 'ep';
      case LayerKind.categorical:
        e
          ..writeln('  let vpe = mean - em;')
          ..writeln('  ${S(LayerField.precision)} = ep;');
        precision = 'ep';
      case LayerKind.volatile:
        e.writeln('  let vpe = mean - em;');
        if (layer.hasVolatility) {
          e
            ..writeln('  let d0 = mean - em;')
            ..writeln(
              '  ${S(LayerField.volatilityPredictionError)} = '
              'ep / prec + ep * (d0 * d0) - 1.0f;',
            );
          _volatility(e, l);
        }
    }
    e.writeln('  ${S(LayerField.valuePredictionError)} = vpe;');
    if (l == nLayers - 1) return;
    final ll = _l[l];
    final meanField = settings.meanFieldUpdates;
    e.writeln('  let cep = ${S(LayerField.conditionalExpectedPrecision)};');
    if (meanField) {
      e
        ..writeln('  let effc = ep;')
        ..writeln('  let gain = ep;');
    } else {
      e
        ..writeln('  let piY = $precision - ep;')
        ..writeln(
          '  let effc = ${layer.isInputLayer ? 'cep' : 'cep * piY / (cep + piY)'};',
        )
        ..writeln('  let gain = cep * $precision / (cep + piY);');
    }
    e
      ..writeln('  ${_at(ll.effc!, 'k')} = effc;')
      ..writeln('  ${_at(ll.gain!, 'k')} = gain * vpe;')
      ..writeln('  var uu = mean - em;');
    if (layer.kind != LayerKind.binary) {
      e.writeln('  if ((U.flags & 1u) != 0u) { uu = uu * $precision; }');
    }
    e.writeln('  ${_at(ll.u!, 'k')} = -select(0.0f, uu, isFinite(uu));');
  }

  /// The volatility-level posterior of volatile layer [l] (expects `mean`,
  /// `precision`, `em`, `ep` and `k` in scope).
  void _volatility(StringBuffer e, int l) {
    String S(LayerField f) => _s(l, f, 'k');
    final ll = _l[l];
    final maxPp = _f(settings.maxPosteriorPrecision);
    final omega = ll.tonic == null ? '0.0f' : _p(ll.tonic!, 'i');
    e
      ..writeln('  let effv0 = ${S(LayerField.effectivePrecision)};')
      ..writeln('  let dpe = ${S(LayerField.volatilityPredictionError)};')
      ..writeln('  let evp = ${S(LayerField.expectedPrecisionVol)};')
      ..writeln('  let evm = ${S(LayerField.expectedMeanVol)};');
    switch (settings.volatilityUpdates) {
      case DeepVolatilityUpdate.standard:
        e
          ..writeln('  let gg = effv0;')
          ..writeln(
            '  let contrib = 0.5f * (gg * gg) + (gg * gg) * dpe - 0.5f * gg * dpe;',
          )
          ..writeln('  let pvn = minN(evp + contrib, $maxPp);')
          ..writeln('  ${S(LayerField.precisionVol)} = pvn;')
          ..writeln(
            '  ${S(LayerField.meanVol)} = evm + gg * dpe / (2.0f * pvn);',
          );
        return;
      case DeepVolatilityUpdate.ehgf:
      case DeepVolatilityUpdate.unbounded:
    }
    e
      ..writeln('  let omega = $omega;')
      ..writeln('  var logVol = omega + evm;');
    if (!settings.meanFieldUpdates) {
      e.writeln('  logVol = logVol + 1.0f / (2.0f * evp);');
    }
    e.writeln('  let cepv = ${S(LayerField.conditionalExpectedPrecision)};');
    if (ll.known != null) {
      // The stored known variance, not 1/π̄ − Ω (which cancels).
      e.writeln('  var previous = ${_at(ll.known!, 'k')};');
    } else {
      e
        ..writeln('  let predicted = U.t * exp(logVol);')
        ..writeln('  var previous = floorV(1.0f / cepv - predicted, MINV);');
    }
    if (settings.feedforwardUncertainty) {
      e.writeln('  previous = previous + maxN(1.0f / ep - 1.0f / cepv, 0.0f);');
    }
    if (settings.volatilityUpdates == DeepVolatilityUpdate.ehgf) {
      e.writeln('  let postMean = evm + effv0 * dpe / (2.0f * evp);');
      if (settings.fixes.logSpaceVolatility) {
        e
          ..writeln(
            '  let logRep = select(log(U.t), 0.0f, U.t == 1.0f) + omega + postMean;',
          )
          ..writeln('  let logPrevE = log(previous);')
          ..writeln('  let eff2 = sigm(logRep - logPrevE);')
          ..writeln('  let expected = exp(-logAddExp(logPrevE, logRep));')
          ..writeln('  let weight = 2.0f * eff2 - 1.0f;');
      } else {
        e
          ..writeln('  let repredicted = U.t * exp(omega + postMean);')
          ..writeln('  let expected = 1.0f / (previous + repredicted);')
          ..writeln('  let eff2 = repredicted * expected;')
          ..writeln('  let weight = (repredicted - previous) * expected;');
      }
      e
        ..writeln('  let dd = mean - em;')
        ..writeln('  let vpe2 = (1.0f / prec + dd * dd) * expected - 1.0f;')
        ..writeln(
          '  let contrib = maxN(0.0f, 0.5f * eff2 * (eff2 + weight * vpe2));',
        )
        ..writeln(
          '  ${S(LayerField.precisionVol)} = minN(evp + contrib, $maxPp);',
        )
        ..writeln('  ${S(LayerField.meanVol)} = postMean;');
      return;
    }
    // Unbounded.
    e
      ..writeln('  let logT = select(log(U.t), 0.0f, U.t == 1.0f);')
      ..writeln('  let dd = mean - em;')
      ..writeln('  let beAux = 1.0f / prec + dd * dd;')
      ..writeln('  let logPrev = log(previous);')
      ..writeln('  let gammaC = logT + evm + omega;')
      ..writeln('  let wJm1 = sigm(gammaC - logPrev);')
      ..writeln('  let daJm1 = ep * beAux - 1.0f;')
      ..writeln('  let pi1 = evp + 0.5f * wJm1 * (1.0f - wJm1);')
      ..writeln('  let mu1 = evm + wJm1 / (2.0f * pi1) * daJm1;')
      ..writeln(
        '  let logWArg = log(beAux) - log(2.0f * evp) + 0.5f / evp - gammaC;',
      )
      // ln(f64::MAX), the CPU's cap; lambertWExp needs no f32 one.
      ..writeln('  let vW = lambertWExp(minN(logWArg, 709.782712893384f));')
      ..writeln('  let yStar = gammaC + vW - 0.5f / evp;')
      ..writeln('  let xStar = yStar - logT - omega;')
      ..writeln('  let logS2 = logT + xStar + omega;')
      ..writeln('  let logDenomS = logAddExp(logPrev, logS2);')
      ..writeln('  let w2 = sigm(logS2 - logPrev);')
      ..writeln('  let da2 = beAux * exp(-logDenomS) - 1.0f;')
      ..writeln(
        '  let pi2Full = evp + 0.5f * w2 * (w2 + (2.0f * w2 - 1.0f) * da2);',
      )
      ..writeln(
        '  let pi2Safe = select(pi2Full, evp + 0.5f * w2 * (1.0f - w2), '
        'pi2Full <= 0.0f);',
      )
      ..writeln(
        '  let mu2Safe = xStar + (0.5f * w2 * da2 - evp * (xStar - evm)) / pi2Safe;',
      )
      ..writeln('  let finite2 = isFinite(pi2Safe) && isFinite(mu2Safe);')
      ..writeln('  let pi2 = select(pi1, pi2Safe, finite2);')
      ..writeln('  let mu2 = select(mu1, mu2Safe, finite2);')
      ..writeln('  let logDenom1 = logAddExp(logPrev, logT + mu1 + omega);')
      ..writeln(
        '  let i1 = -0.5f * logDenom1 - 0.5f * beAux * exp(-logDenom1) - '
        '0.5f * evp * ((mu1 - evm) * (mu1 - evm));',
      )
      ..writeln('  let logDenom2 = logAddExp(logPrev, logT + mu2 + omega);')
      ..writeln(
        '  let i2 = -0.5f * logDenom2 - 0.5f * beAux * exp(-logDenom2) - '
        '0.5f * evp * ((mu2 - evm) * (mu2 - evm));',
      )
      ..writeln('  let blend = sigm(i2 - i1);')
      ..writeln(
        '  ${S(LayerField.meanVol)} = (1.0f - blend) * mu1 + blend * mu2;',
      )
      ..writeln(
        '  let sig2 = (1.0f - blend) / pi1 + blend / pi2 + '
        'blend * (1.0f - blend) * ((mu1 - mu2) * (mu1 - mu2));',
      )
      ..writeln('  ${S(LayerField.precisionVol)} = minN(1.0f / sig2, $maxPp);');
  }

  /// The error message the top layer receives, written to `OUT` rows
  /// `start + s` (`rows × topSize`).
  DeepKernelSource inputErrors() {
    final l = nLayers - 1;
    final cols = layers[l].weights!.cols;
    final nc = layers[l - 1].size;
    final w = _l[l].weights!;
    return _gemm(
      'input errors',
      m: '${topSize}u',
      mMax: topSize,
      n: 'U.b',
      nMax: capacity,
      depth: '${nc}u',
      a: 'P[${w}u + k * ${cols}u + r]',
      aAlongK: false,
      b0: _at(_l[l - 1].gain!, 'k * $_bm + c'),
      bAlongK: false,
      epilogue:
          '  let x = ${_s(l, LayerField.expectedMean, 'r * $_bm + c')};\n'
          '  OUT[(U.start + c) * ${topSize}u + r] = acc0 * ${_dg(_fn(l), 'x')};',
      grid: (tm, tn) => _overSamples(topSize, tm, tn),
    );
  }

  /// Layer [l]'s weight gradient (`Σ_s u[i][s]·h[j][s] / b`) and the
  /// optimiser step on its weights: one fused kernel, or with split-K a
  /// partial-sum GEMM and a kernel that adds the partials and steps.
  List<DeepKernelSource> gradient(int l, DeepOptimiserKind optimiser) {
    final w = layers[l].weights!;
    final nc = w.rows, cols = w.cols;
    final ll = _l[l];
    final splits = ll.splits;
    final step = _optimiserStep(l, optimiser);
    final gemm = _gemm(
      'gradient layer $l${splits > 1 ? ' partial' : ' (${optimiser.name})'}',
      m: '${nc}u',
      mMax: nc,
      n: '${cols}u',
      nMax: cols,
      depth: 'U.b',
      a: _at(_l[l - 1].u!, 'r * $_bm + k'),
      aAlongK: true,
      b0: _at(ll.h!, 'c * $_bm + k'),
      bAlongK: true,
      epilogue: splits > 1
          ? '  ${_at(ll.partial!, 'split * ${nc * cols}u + r * ${cols}u + c')}'
                ' = acc0;'
          : '  let g = ${_edge(l, 'acc0 / f32(U.b)')};\n'
                '  let wi = r * ${cols}u + c;\n$step',
      grid: (tm, tn) =>
          (_) => ((cols + tn - 1) ~/ tn, (nc + tm - 1) ~/ tm),
      splits: splits,
    );
    if (splits == 1) return [gemm];
    final n = nc * cols;
    return [
      gemm,
      _kernel('gradient layer $l apply (${optimiser.name})', '''
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let wi = gid.x;
  if (wi >= ${n}u) { return; }
  var sum = 0.0f;
  for (var sp = 0u; sp < ${splits}u; sp++) {
    sum += ${_at(ll.partial!, 'sp * ${n}u + wi')};
  }
  let r = wi / ${cols}u; let c = wi % ${cols}u;
  let g = ${_edge(l, 'sum / f32(U.b)')};
$step
}
''', (_) => ((n + 63) ~/ 64, 1)),
    ];
  }

  /// Gradient [g] of weight `(r, c)` of layer [l]: zero off the diagonal
  /// of a one-to-one layer, whose only edges are the diagonal (pyhgf #476).
  String _edge(int l, String g) =>
      layers[l].config.fullyConnected ? g : 'select(0.0f, $g, r == c)';

  /// The optimiser's update of weight `wi` of layer [l] from gradient `g`.
  /// [bc1] and [bc2] are Adam's bias corrections (WGSL expressions).
  String _optimiserStep(
    int l,
    DeepOptimiserKind optimiser, {
    String bc1 = 'U.bc1',
    String bc2 = 'U.bc2',
  }) {
    final ll = _l[l];
    final w = _p(ll.weights!, 'wi');
    final mu = _p(ll.mu!, 'wi'), nu = _p(ll.nu!, 'wi');
    return switch (optimiser) {
      DeepOptimiserKind.sgd => '  $w = $w + (-U.lr) * g;',
      DeepOptimiserKind.adam =>
        '  let m = (1.0f - U.b1) * g + U.b1 * $mu;\n'
            '  let v = (1.0f - U.b2) * (g * g) + U.b2 * $nu;\n'
            '  $mu = m;\n'
            '  $nu = v;\n'
            '  let upd = (m / $bc1) / (sqrt(v / $bc2 + U.epsRoot) + U.eps);\n'
            '  $w = $w + (-U.lr) * upd;',
    };
  }

  /// `carry_precisions` and the batch-mean increment of the carried beliefs
  /// (precision, and the volatility level of volatile layers) into the
  /// network's own state: one workgroup per node.
  DeepKernelSource carry() {
    final carryActive =
        !(settings.carryPrecision && !settings.feedforwardUncertainty);
    final body = StringBuffer()
      ..writeln('var<workgroup> red: array<f32, 768>;')
      ..writeln('@compute @workgroup_size(256)')
      ..writeln(
        'fn main(@builtin(local_invocation_id) lid: vec3<u32>, '
        '@builtin(workgroup_id) wid: vec3<u32>) {',
      )
      ..writeln('  let t = lid.x;')
      ..writeln('  let node = wid.x;');
    var first = 0;
    for (var l = 0; l < nLayers; l++) {
      final layer = layers[l];
      final n = layer.size;
      final volatile = layer.kind == LayerKind.volatile;
      final vol = volatile && layer.hasVolatility;
      final carries = volatile && carryActive;
      String carried(String p, String cep, String ep, String i) =>
          settings.carryPrecision
          ? '$p + maxN($cep - $ep, 0.0f)'
          : _p(_l[l].prior!, i);
      body.writeln('  ${first == 0 ? '' : 'else '}if (node < ${first + n}u) {');
      body.writeln('    let i = node - ${first}u;');
      if (l == nLayers - 1) {
        // The top layer's batch beliefs are its own (never predicted).
        if (carries) {
          final tp = _t(l, LayerField.precision, 'i');
          body
            ..writeln('    if (t == 0u) {')
            ..writeln('      let tp = $tp;')
            ..writeln(
              '      let v = ${carried('tp', _t(l, LayerField.conditionalExpectedPrecision, 'i'), _t(l, LayerField.expectedPrecision, 'i'), 'i')};',
            )
            ..writeln('      $tp = tp + (v - tp);')
            ..writeln('    }');
        }
        body.writeln('  }');
        first += n;
        continue;
      }
      final fields = [
        LayerField.precision,
        if (vol) LayerField.meanVol,
        if (vol) LayerField.precisionVol,
      ];
      for (var f = 0; f < fields.length; f++) {
        body
          ..writeln('    let t$f = ${_t(l, fields[f], 'i')};')
          ..writeln('    var sum$f = 0.0f;');
      }
      body
        ..writeln('    for (var s = t; s < U.b; s += 256u) {')
        ..writeln('      let k = i * $_bm + s;');
      for (var f = 0; f < fields.length; f++) {
        var v = _s(l, fields[f], 'k');
        if (f == 0 && carries) {
          v = carried(
            v,
            _s(l, LayerField.conditionalExpectedPrecision, 'k'),
            _s(l, LayerField.expectedPrecision, 'k'),
            'i',
          );
        }
        body.writeln('      sum$f += ($v) - t$f;');
      }
      body.writeln('    }');
      for (var f = 0; f < fields.length; f++) {
        body.writeln('    red[${f * 256}u + t] = sum$f;');
      }
      body
        ..writeln('    workgroupBarrier();')
        ..writeln('    for (var w = 128u; w > 0u; w >>= 1u) {')
        ..writeln('      if (t < w) {');
      for (var f = 0; f < fields.length; f++) {
        body.writeln(
          '        red[${f * 256}u + t] += red[${f * 256}u + t + w];',
        );
      }
      body
        ..writeln('      }')
        ..writeln('      workgroupBarrier();')
        ..writeln('    }')
        ..writeln('    if (t == 0u) {');
      for (var f = 0; f < fields.length; f++) {
        body.writeln(
          '      ${_t(l, fields[f], 'i')} = t$f + red[${f * 256}u] / f32(U.b);',
        );
      }
      body
        ..writeln('    }')
        ..writeln('  }');
      first += n;
    }
    body.writeln('}');
    final total = first;
    return _kernel('carry', body.toString(), (_) => (total, 1));
  }

  // --- predict --------------------------------------------------------------

  /// The kernels of `predict`: the mean chain only, into scratch, with the
  /// bottom layer's predictions in `OUT` rows `start + s`
  /// (`rows × bottomSize`).
  List<DeepKernelSource> predict() => [
    predictTop(),
    for (var l = nLayers - 2; l >= 0; l--) predictChain(l),
    if (layers.first.kind == LayerKind.categorical) predictSoftmax(),
  ];

  DeepKernelSource predictTop() {
    final l = nLayers - 1;
    return _elementwise('predict top', topSize, '''
  let v = X[(U.start + s) * ${topSize}u + i];
  ${_at(_l[l].predicted!, 'k')} = ${_g(_fn(l), 'v')};
''');
  }

  DeepKernelSource predictChain(int l) {
    final c = layers[l];
    final pl = layers[l + 1];
    final cols = pl.weights!.cols;
    final e = StringBuffer()
      ..writeln('  let k = r * $_bm + c;')
      ..writeln('  var v = acc0;');
    if (c.kind == LayerKind.binary) {
      final lo = settings.precisionClippingValue;
      e.writeln('  v = minN(maxN(sigm(v), ${_f(lo)}), ${_f(1 - lo)});');
    }
    if (l > 0) {
      e.writeln('  ${_at(_l[l].predicted!, 'k')} = ${_g(_fn(l), 'v')};');
    } else if (c.kind == LayerKind.categorical) {
      e.writeln('  ${_at(_l[0].predicted!, 'k')} = v;');
    } else {
      e.writeln('  OUT[(U.start + c) * ${bottomSize}u + r] = v;');
    }
    return _gemm(
      'predict chain $l',
      m: '${c.size}u',
      mMax: c.size,
      n: 'U.b',
      nMax: capacity,
      depth: '${cols}u',
      a: 'P[${_l[l + 1].weights}u + r * ${cols}u + k]',
      aAlongK: true,
      b0: _at(_l[l + 1].predicted!, 'k * $_bm + c'),
      bAlongK: false,
      epilogue: e.toString(),
      grid: (tm, tn) => _overSamples(c.size, tm, tn),
    );
  }

  DeepKernelSource predictSoftmax() {
    final n = bottomSize;
    String z(String i) => _at(_l[0].predicted!, '$i * $_bm + s');
    return _kernel('predict softmax', '''
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= U.b) { return; }
  var mx = ${z('0u')};
  for (var i = 1u; i < ${n}u; i++) { mx = maxN(mx, ${z('i')}); }
  var sum = 0.0f;
  for (var i = 0u; i < ${n}u; i++) {
    let e = exp(${z('i')} - mx);
    ${z('i')} = e;
    sum += e;
  }
  for (var i = 0u; i < ${n}u; i++) {
    OUT[(U.start + s) * ${n}u + i] = ${z('i')} / sum;
  }
}
''', (b) => ((b + 63) ~/ 64, 1));
  }
}
