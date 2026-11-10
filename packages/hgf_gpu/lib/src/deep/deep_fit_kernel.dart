part of 'deep_kernels.dart';

/// The uniform of [SequentialFitKernel.sequentialFit] (64 bytes).
const _fitUniformStruct = '''
struct Uniform {
  members: u32,
  start: u32,
  end: u32,
  flags: u32,
  nanBits: u32,
  xStride: u32,
  yStride: u32,
  outStride: u32,
  t: f32,
  lr: f32,
  b1: f32,
  b2: f32,
  eps: f32,
  epsRoot: f32,
  pad0: u32,
  pad1: u32,
}
''';

/// Flag bits of the sequential fit's uniform (bit 0, precision-weighted
/// learning, is shared with the batch kernels).
abstract final class DeepFitFlags {
  static const precisionWeighted = 1;
  static const updatePrecisions = 2;
  static const predictions = 4;
}

/// Workgroups along x before a grid wraps into y.
const deepFitGridWidth = 65535;

/// pyhgf's sequential `fit` for an ensemble of networks of one shape: one
/// workgroup per member runs the whole sample loop, its threads sharing
/// each phase's nodes (or weights) with a barrier between phases.
extension SequentialFitKernel on DeepKernels {
  /// The kernel of a sequential `fit` over rows `[U.start, U.end)` of `X`
  /// and `Y` for `U.members` networks, built by [DeepKernels.sequential].
  ///
  /// Per row, as `DeepNetwork.fit`: the prediction sweep (clamping x on top),
  /// the update sweep (clamping y at the bottom), with flag
  /// [DeepFitFlags.predictions] the bottom layer's expected means into
  /// `OUT`, the weight gradient and one [optimiser] step (none when
  /// `null`), and with [DeepFitFlags.updatePrecisions] the carried beliefs
  /// into the member's own state.
  ///
  /// Member `e` reads `X[e·xStride + row·topSize + i]` (stride 0: shared
  /// data), `Y` likewise, and writes `OUT[e·outStride + row·bottomSize + i]`.
  /// Adam's bias corrections for row `row` are `BC[2·row]` and
  /// `BC[2·row + 1]`.
  DeepKernelSource sequentialFit({
    DeepOptimiserKind? optimiser,
    int workgroupSize = 64,
  }) {
    if (!sequentialLayout) {
      throw StateError('Build the kernels with DeepKernels.sequential.');
    }
    final wg = '${workgroupSize}u';
    final top = nLayers - 1;
    final ws = workgroupState, wp = workgroupParams;
    final b = StringBuffer();
    if (ws) b.writeln('var<workgroup> WS: array<f32, ${stateFloats}u>;');
    if (wp) b.writeln('var<workgroup> WP: array<f32, ${paramFloats}u>;');
    b
      ..writeln('@compute @workgroup_size($workgroupSize)')
      ..writeln(
        'fn main(@builtin(local_invocation_id) lid3: vec3<u32>, '
        '@builtin(workgroup_id) wid: vec3<u32>) {',
      )
      ..writeln('  let member = wid.y * ${deepFitGridWidth}u + wid.x;')
      ..writeln('  if (member >= U.members) { return; }')
      ..writeln('  let lid = lid3.x;')
      ..writeln('  let pb = member * ${paramFloats}u;')
      ..writeln('  let sb = member * ${stateFloats}u;')
      ..writeln('  let xb = member * U.xStride;')
      ..writeln('  let yb = member * U.yStride;')
      ..writeln('  let ob = member * U.outStride;');
    if (ws) {
      b
        ..writeln('  for (var i = lid; i < ${stateFloats}u; i += $wg) {')
        ..writeln('    WS[i] = S[sb + i];')
        ..writeln('  }');
      if (wp) {
        b
          ..writeln('  for (var i = lid; i < ${paramFloats}u; i += $wg) {')
          ..writeln('    WP[i] = P[pb + i];')
          ..writeln('  }');
      }
      b.writeln('  workgroupBarrier();');
    }
    b.writeln('  for (var row = U.start; row < U.end; row++) {');

    void nodes(int n, String body) {
      b
        ..writeln('    for (var i = lid; i < ${n}u; i += $wg) {')
        ..writeln('      let s = 0u; let k = i;')
        ..writeln(body)
        ..writeln('    }');
    }

    // Phases hand data over through storage buffers, which only
    // storageBarrier orders (workgroupBarrier covers workgroup memory).
    // With the state in workgroup memory, the phases within a row hand
    // over only WS (the weights and carried beliefs in P are written by
    // the optimiser and carry phases and read in the next row, after the
    // row's final barrier, which orders both).
    void barrier() =>
        b.writeln(ws ? '    workgroupBarrier();' : '    storageBarrier();');

    // Prediction sweep.
    nodes(topSize, _topBody('X[xb + row * ${topSize}u + i]'));
    barrier();
    for (var l = nLayers - 2; l >= 0; l--) {
      final pll = _l[l + 1];
      final cols = layers[l + 1].weights!.cols;
      final ffu = _feedforward(l);
      b
        ..writeln('    for (var r = lid; r < ${layers[l].size}u; r += $wg) {')
        ..writeln('      var acc0 = 0.0f; var acc1 = 0.0f;')
        ..writeln('      for (var j = 0u; j < ${cols}u; j++) {')
        ..writeln('        let w = ${_p(pll.weights!, 'r * ${cols}u + j')};')
        ..writeln('        acc0 += w * ${_at(pll.coupled!, 'j')};');
      if (ffu) {
        // The bias row of perParent is never written (zero-initialised).
        b.writeln('        acc1 += w * w * ${_at(pll.perParent!, 'j')};');
      }
      b
        ..writeln('      }')
        ..writeln('      let c = 0u;')
        ..writeln('      {')
        ..writeln(_predictEpilogue(l))
        ..writeln('      }')
        ..writeln('    }');
      barrier();
      if (layers[l].kind == LayerKind.categorical) {
        String em(String i) => _s(0, LayerField.expectedMean, i);
        final n = bottomSize;
        b.writeln('''
    if (lid == 0u) {
      var mx = ${em('0u')};
      for (var i = 1u; i < ${n}u; i++) { mx = maxN(mx, ${em('i')}); }
      var sum = 0.0f;
      for (var i = 0u; i < ${n}u; i++) {
        let e = exp(${em('i')} - mx);
        ${em('i')} = e;
        sum += e;
      }
      for (var i = 0u; i < ${n}u; i++) {
        ${em('i')} = ${em('i')} / sum;
        ${_s(0, LayerField.expectedPrecision, 'i')} = 1.0f;
        ${_s(0, LayerField.conditionalExpectedPrecision, 'i')} = 1.0f;
        ${_s(0, LayerField.effectivePrecision, 'i')} = 0.0f;
      }
    }''');
        barrier();
      }
    }

    // Update sweep.
    nodes(bottomSize, _bottomBody('Y[yb + row * ${bottomSize}u + i]'));
    barrier();
    for (var l = 1; l < top; l++) {
      final cl = _l[l - 1];
      final cols = layers[l].weights!.cols;
      b
        ..writeln('    for (var r = lid; r < ${layers[l].size}u; r += $wg) {')
        ..writeln('      var acc0 = 0.0f; var acc1 = 0.0f;')
        ..writeln('      for (var j = 0u; j < ${layers[l - 1].size}u; j++) {')
        ..writeln('        let w = ${_p(_l[l].weights!, 'j * ${cols}u + r')};')
        ..writeln('        acc0 += w * ${_at(cl.gain!, 'j')};')
        ..writeln('        acc1 += w * w * ${_at(cl.effc!, 'j')};')
        ..writeln('      }')
        ..writeln('      let c = 0u;')
        ..writeln('      {')
        ..writeln(_posteriorEpilogue(l))
        ..writeln('      }')
        ..writeln('    }');
      barrier();
    }

    // The prediction made before seeing y.
    b
      ..writeln('    if ((U.flags & ${DeepFitFlags.predictions}u) != 0u) {')
      ..writeln('      for (var i = lid; i < ${bottomSize}u; i += $wg) {')
      ..writeln(
        '        OUT[ob + row * ${bottomSize}u + i] = '
        '${_s(0, LayerField.expectedMean, 'i')};',
      )
      ..writeln('      }')
      ..writeln('    }');

    // Weight gradient (one sample) and optimiser step.
    if (optimiser != null) {
      if (optimiser == DeepOptimiserKind.adam) {
        b
          ..writeln('    let bc1 = BC[2u * row];')
          ..writeln('    let bc2 = BC[2u * row + 1u];');
      }
      for (var l = 1; l < nLayers; l++) {
        final w = layers[l].weights!;
        final cols = w.cols;
        b
          ..writeln(
            '    for (var wi = lid; wi < ${w.rows * cols}u; wi += $wg) {',
          )
          ..writeln('      let r = wi / ${cols}u; let c = wi % ${cols}u;')
          ..writeln(
            '      let g = '
            '${_edge(l, '${_at(_l[l - 1].u!, 'r')} * ${_at(_l[l].h!, 'c')}')};',
          )
          ..writeln('      {')
          ..writeln(_optimiserStep(l, optimiser, bc1: 'bc1', bc2: 'bc2'))
          ..writeln('      }')
          ..writeln('    }');
      }
    }

    // Carry the beliefs into the member's own state.
    b.writeln('    if ((U.flags & ${DeepFitFlags.updatePrecisions}u) != 0u) {');
    _sequentialCarry(b, wg);
    b.writeln('    }');
    if (!wp) b.writeln('    storageBarrier();');
    if (ws) b.writeln('    workgroupBarrier();');
    b.writeln('  }');
    if (ws) {
      b
        ..writeln('  for (var i = lid; i < ${stateFloats}u; i += $wg) {')
        ..writeln('    S[sb + i] = WS[i];')
        ..writeln('  }');
    }
    if (wp) {
      b
        ..writeln('  for (var i = lid; i < ${paramFloats}u; i += $wg) {')
        ..writeln('    P[pb + i] = WP[i];')
        ..writeln('  }');
    }
    b.writeln('}');
    return _kernel(
      'sequential fit${optimiser == null ? '' : ' (${optimiser.name})'}',
      b.toString(),
      (members) => (
        math.min(members, deepFitGridWidth),
        (members + deepFitGridWidth - 1) ~/ deepFitGridWidth,
      ),
      uniformStruct: _fitUniformStruct,
    );
  }

  /// `carry_precisions` at width 1: the posterior's carried beliefs become
  /// the member's own (`carry()` without the batch mean).
  void _sequentialCarry(StringBuffer b, String wg) {
    final carryActive =
        !(settings.carryPrecision && !settings.feedforwardUncertainty);
    String carried(String p, String cep, String ep, String i, int l) =>
        settings.carryPrecision
        ? '$p + maxN($cep - $ep, 0.0f)'
        : _p(_l[l].prior!, i);
    for (var l = 0; l < nLayers; l++) {
      final layer = layers[l];
      final volatile = layer.kind == LayerKind.volatile;
      final vol = volatile && layer.hasVolatility;
      final carries = volatile && carryActive;
      String t(LayerField f) => _t(l, f, 'i');
      String s(LayerField f) => _s(l, f, 'i');
      final body = StringBuffer();
      if (l == nLayers - 1) {
        // The top layer's beliefs are its own (never predicted).
        if (!carries) continue;
        body.writeln(
          '        ${t(LayerField.precision)} = ${carried(t(LayerField.precision), t(LayerField.conditionalExpectedPrecision), t(LayerField.expectedPrecision), 'i', l)};',
        );
      } else {
        final p = carries
            ? carried(
                s(LayerField.precision),
                s(LayerField.conditionalExpectedPrecision),
                s(LayerField.expectedPrecision),
                'i',
                l,
              )
            : s(LayerField.precision);
        body.writeln('        ${t(LayerField.precision)} = $p;');
        if (vol) {
          for (final f in const [LayerField.meanVol, LayerField.precisionVol]) {
            body.writeln('        ${t(f)} = ${s(f)};');
          }
        }
      }
      b
        ..writeln('      for (var i = lid; i < ${layer.size}u; i += $wg) {')
        ..write(body)
        ..writeln('      }');
    }
  }
}
