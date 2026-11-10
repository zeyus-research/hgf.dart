import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_deep/hgf_deep.dart';

import '../batch.dart';
import 'deep_kernels.dart';

/// Many `DeepNetwork`s of one shape on the GPU, trained by pyhgf's
/// sequential `fit` (one sample at a time) all at once, in f32.
///
/// One sample of `fit` is a chain of small steps, each depending on the
/// last, which leaves a GPU mostly idle for a single network. Here every
/// member network gets one workgroup that runs the whole sample loop in a
/// single kernel ([SequentialFitKernel.sequentialFit]), so a fit costs no
/// more host round trips than one dispatch per [fit]'s `rowsPerDispatch`
/// rows, and the members fill the GPU. Typical uses: an ensemble over
/// weight seeds, a sweep over initial beliefs or tonic volatilities, or one
/// network per participant on their own data.
///
/// Members share the layer shapes, kinds, couplings and settings (they are
/// compiled in), and may differ in everything else: weights, beliefs,
/// tonic volatilities, prior precisions. Each keeps its own weights,
/// optimiser moments and carried beliefs on the GPU; [download] copies
/// them back. Results match the f64 `DeepNetwork.fit` to f32 accuracy
/// (`test/native_deep_test.dart`).
final class GpuDeepEnsemble {
  GpuDeepEnsemble._(
    this.backend,
    this.kernels,
    this.size,
    this.workgroupSize,
    this._signature0,
  ) : _params = backend.storage(size * kernels.paramFloats * 4),
      _state = backend.storage(size * kernels.stateFloats * 4),
      _uniform = backend.uniform(64);

  /// Uploads [members] (at least one; all built alike, see
  /// [GpuDeepEnsemble]). [workgroupSize] threads share each member's
  /// work; by default twice the largest layer's size rounded up to a power
  /// of two, between 32 and 256 (the fastest of 32–256 on an RX 7600 XT at
  /// 1024 members, `benchmark/bin/deep_ensemble.dart`).
  ///
  /// [workgroupState] keeps each member's working state in workgroup
  /// memory rather than in a storage buffer, and [workgroupParams] its
  /// weights, optimiser moments and beliefs too (see
  /// `DeepKernels.sequential`; `DeepKernels.fitsWorkgroup` tells whether
  /// they fit). By default the state goes there when that was measured
  /// faster (see [_workgroupPlan]), the parameters never.
  static Future<GpuDeepEnsemble> create(
    GpuBackend backend,
    List<DeepNetwork> members, {
    int? workgroupSize,
    bool? workgroupState,
    bool? workgroupParams,
  }) async {
    if (members.isEmpty) throw ArgumentError('No members.');
    final first = members.first;
    final signature = _signature(first);
    for (var e = 1; e < members.length; e++) {
      if (_signature(members[e]) != signature) {
        throw ArgumentError(
          'Member $e differs from member 0 in its layers or settings.',
        );
      }
    }
    final (ws, wp) = _workgroupPlan(
      first,
      members.length,
      workgroupState,
      workgroupParams,
    );
    final kernels = DeepKernels.sequential(
      first,
      workgroupState: ws,
      workgroupParams: wp,
    );
    final wg =
        workgroupSize ??
        math.max(
          32,
          math.min(256, 2 * _pow2AtLeast(first.layerSizes.reduce(math.max))),
        );
    final n = members.length;
    for (final (what, floats) in [
      ('weights and beliefs', kernels.paramFloats),
      ('working state', kernels.stateFloats),
    ]) {
      final bytes = n * floats * 4;
      if (bytes > backend.maxStorageBufferBytes) {
        throw ArgumentError(
          'The $what of $n members take $bytes bytes; the device allows '
          '${backend.maxStorageBufferBytes} per buffer.',
        );
      }
    }
    final gpu = GpuDeepEnsemble._(backend, kernels, n, wg, signature);
    gpu._initState();
    final host = backend.hostFloats(n * kernels.paramFloats);
    for (var e = 0; e < n; e++) {
      gpu._pack(members[e], host, e * kernels.paramFloats);
    }
    backend.write(gpu._params, host);
    return gpu;
  }

  /// What goes into workgroup memory: the requested choices, or by
  /// default the working state when it is small (≤ a quarter of the
  /// workgroup memory) or the ensemble is (< 128 members), and never the
  /// parameters. Measured on an RX 7600 XT (`deep_ensemble --state`): the
  /// small network's 1.7 KB state is 10–40 % faster at every size; the
  /// medium network's 12 KB state is 12–16 % faster up to 16 members and
  /// 10–17 % slower from 256 (it limits how many workgroups run at once);
  /// the parameters as well (small network) were no faster on balance.
  static (bool, bool) _workgroupPlan(
    DeepNetwork net,
    int members,
    bool? state,
    bool? params,
  ) {
    final k = DeepKernels.sequential(net);
    final limit = DeepKernels.workgroupStorageBytes;
    final stateBytes = k.stateFloats * 4;
    final allBytes = stateBytes + k.paramFloats * 4;
    final wp = params ?? false;
    final ws =
        state ??
        (wp ||
            (stateBytes <= limit &&
                (stateBytes * 4 <= limit || members < 128)));
    if (wp && !ws) {
      throw ArgumentError('workgroupParams needs workgroupState');
    }
    final need = wp ? allBytes : (ws ? stateBytes : 0);
    if (need > limit) {
      throw ArgumentError(
        'A member of ${net.layerSizes} needs $need bytes of workgroup '
        'memory for this; a workgroup has $limit.',
      );
    }
    return (ws, wp);
  }

  static int _pow2AtLeast(int n) {
    var p = 1;
    while (p < n) {
      p <<= 1;
    }
    return p;
  }

  static String _signature(DeepNetwork net) {
    final s = net.settings;
    return [
      s.volatilityUpdates,
      s.maxPosteriorPrecision,
      s.precisionClippingValue,
      s.predictPrecision,
      s.feedforwardUncertainty,
      s.carryPrecision,
      s.meanFieldUpdates,
      s.fixes,
      for (final l in net.debugLayers)
        '${l.size} ${l.kind} ${l.hasConstant} ${l.config.fullyConnected} '
            '${l.coupling} ${l.hasVolatility} ${l.tonicVolatility != null}',
    ].join('|');
  }

  final GpuBackend backend;
  final DeepKernels kernels;

  /// Number of member networks.
  final int size;

  /// Threads per member.
  final int workgroupSize;

  /// What every member shares (layers and settings).
  final String _signature0;

  final GpuBuffer _params, _state, _uniform;
  GpuBuffer? _x, _y, _out, _bc;
  int _xFloats = 0, _yFloats = 0, _outFloats = 0, _bcFloats = 0;

  final _pipelines = <DeepOptimiserKind?, Future<GpuPipeline>>{};

  Adam? _adam;
  int _adamCount = 0;

  /// Zeroed working state with the constant (bias) rows set to one.
  void _initState() {
    final k = kernels;
    final host = backend.hostFloats(size * k.stateFloats);
    for (var e = 0; e < size; e++) {
      for (final (offset, length) in k.biasRows) {
        final o = e * k.stateFloats + offset;
        host.fillRange(o, o + length, 1.0);
      }
    }
    backend.write(_state, host);
  }

  /// Writes [net]'s parameters and beliefs into [host] at [base].
  void _pack(DeepNetwork net, Float32List host, int base) {
    final k = kernels;
    if (_signature(net) != _signature0) {
      throw ArgumentError('The network differs in its layers or settings.');
    }
    for (var l = 0; l < k.nLayers; l++) {
      final layer = net.debugLayers[l];
      final s = net.state(l);
      for (final f in LayerField.values) {
        final v = s.field(f);
        if (v == null || f.volatility && !layer.hasVolatility) continue;
        host.setAll(base + k.templateOffset(l, f), v);
      }
      host.setAll(base + k.tonicVolOffset(l), layer.tonicVolatilityVol);
      final tonic = k.tonicOffset(l);
      if (tonic != null) host.setAll(base + tonic, layer.tonicVolatility!);
      host.setAll(base + k.priorOffset(l), layer.priorPrecision);
      final w = layer.weights;
      if (w != null) host.setAll(base + k.weightsOffset(l), w.data);
    }
  }

  /// Replaces member [member]'s weights and beliefs with [net]'s (built
  /// like the others). Optimiser moments are kept; pass a new `Adam` to
  /// [fit] to restart them for every member.
  void upload(int member, DeepNetwork net) {
    RangeError.checkValidIndex(member, this, 'member', size);
    final host = backend.hostFloats(kernels.paramFloats);
    _pack(net, host, 0);
    // The moments live in the same region: keep them.
    _writeExceptMoments(host, member);
  }

  void _writeExceptMoments(Float32List host, int member) {
    final k = kernels;
    final base = member * k.paramFloats;
    // Contiguous runs between the moment blocks.
    final moments = <(int, int)>[
      for (var l = 1; l < k.nLayers; l++) ...[
        (k.muOffset(l), k.layers[l].weights!.data.length),
        (k.nuOffset(l), k.layers[l].weights!.data.length),
      ],
    ]..sort((a, b) => a.$1.compareTo(b.$1));
    var at = 0;
    for (final (o, n) in [...moments, (k.paramFloats, 0)]) {
      if (o > at) {
        backend.write(
          _params,
          host,
          offset: (base + at) * 4,
          start: at,
          length: o - at,
        );
      }
      at = o + n;
    }
  }

  /// Copies member [member]'s weights and carried beliefs (precision, and
  /// the volatility level's mean and precision) into [net].
  Future<void> download(int member, DeepNetwork net) async {
    RangeError.checkValidIndex(member, this, 'member', size);
    final k = kernels;
    final p = await backend.read(
      _params,
      offset: member * k.paramFloats * 4,
      bytes: k.paramFloats * 4,
    );
    _unpack(p, 0, net);
  }

  /// [download] for every member, with one read: `nets[e]` gets member `e`.
  Future<void> downloadAll(List<DeepNetwork> nets) async {
    if (nets.length != size) {
      throw ArgumentError('Expected $size networks, got ${nets.length}.');
    }
    final k = kernels;
    final p = await backend.read(_params, bytes: size * k.paramFloats * 4);
    for (var e = 0; e < size; e++) {
      _unpack(p, e * k.paramFloats, nets[e]);
    }
  }

  void _unpack(Float32List p, int base, DeepNetwork net) {
    final k = kernels;
    final weights = <Matrix>[];
    for (var l = 0; l < k.nLayers; l++) {
      final s = net.state(l);
      for (final f in const [
        LayerField.precision,
        LayerField.meanVol,
        LayerField.precisionVol,
      ]) {
        final dst = s.field(f);
        if (dst == null || f.volatility && !k.layers[l].hasVolatility) {
          continue;
        }
        final o = base + k.templateOffset(l, f);
        for (var i = 0; i < dst.length; i++) {
          dst[i] = p[o + i];
        }
      }
      final w = net.debugLayers[l].weights;
      if (w != null) {
        final o = base + k.weightsOffset(l);
        final m = Matrix(w.rows, w.cols);
        for (var i = 0; i < m.data.length; i++) {
          m.data[i] = p[o + i];
        }
        weights.add(m);
      }
    }
    net.setWeights(weights);
  }

  GpuBuffer _ensure(GpuBuffer? buffer, int have, int floats) {
    if (buffer != null && have >= floats) return buffer;
    if (buffer != null) backend.release(buffer);
    return backend.storage(math.max(floats, 4) * 4);
  }

  /// Uploads [data] (one matrix for every member, or one per member) and
  /// returns the per-member stride in floats (0 when shared).
  int _uploadData(List<Matrix> data, {required bool isX}) {
    final per = data.first.data.length;
    final n = per * data.length;
    if (isX) {
      _x = _ensure(_x, _xFloats, n);
      _xFloats = math.max(_xFloats, n);
    } else {
      _y = _ensure(_y, _yFloats, n);
      _yFloats = math.max(_yFloats, n);
    }
    final host = backend.hostFloats(n);
    for (var e = 0; e < data.length; e++) {
      final src = data[e].data;
      final o = e * per;
      for (var i = 0; i < per; i++) {
        host[o + i] = src[i];
      }
    }
    backend.write(isX ? _x! : _y!, host);
    return data.length == 1 ? 0 : per;
  }

  void _checkData(List<Matrix> data, int cols, String name) {
    if (data.length != 1 && data.length != size) {
      throw ArgumentError(
        '$name: give one matrix for every member or one per member ($size), '
        'got ${data.length}.',
      );
    }
    for (final m in data) {
      if (m.cols != cols) {
        throw ArgumentError(
          '$name has ${m.cols} column(s) but its layer has $cols node(s).',
        );
      }
      if (m.rows != data.first.rows) {
        throw ArgumentError('Every $name matrix needs the same rows.');
      }
    }
  }

  Future<GpuPipeline> _pipeline(DeepOptimiserKind? kind) =>
      _pipelines.putIfAbsent(
        kind,
        () => backend.pipeline(
          kernels
              .sequentialFit(optimiser: kind, workgroupSize: workgroupSize)
              .source,
        ),
      );

  /// Trains every member on its rows of `(x, y)` one sample at a time, as
  /// `DeepNetwork.fit`. [x] and [y] hold one matrix shared by all members
  /// or one per member, all with the same rows.
  ///
  /// Returns each member's predictions (`rows × bottomSize`, each made
  /// before that row's `y`) when [readPredictions]; otherwise `null`
  /// without waiting for the GPU.
  ///
  /// The members keep their own Adam moments on the GPU: they start from
  /// zero the first time an `Adam` is passed and continue while the same
  /// one is, as in `GpuDeepNetwork`. Trajectories are not recorded.
  /// [rowsPerDispatch] bounds how long one dispatch runs: a GPU driver
  /// resets the device when a kernel runs for seconds (seen at 256 rows of
  /// 1024 wide members). By default it is sized for ~50 ms per dispatch on
  /// a mid-range GPU ([defaultRowsPerDispatch]).
  Future<List<Matrix>?> fit(
    List<Matrix> x,
    List<Matrix> y, {
    Optimiser? optimiser,
    LearningKind learningKind = LearningKind.precisionWeighted,
    bool weightUpdate = true,
    double timeStep = 1.0,
    bool updatePrecisions = true,
    bool readPredictions = true,
    int? rowsPerDispatch,
  }) async {
    if (weightUpdate && optimiser == null) {
      throw ArgumentError('fit needs an optimiser (or weightUpdate: false).');
    }
    rowsPerDispatch ??= defaultRowsPerDispatch;
    if (rowsPerDispatch < 1) {
      throw ArgumentError.value(rowsPerDispatch, 'rowsPerDispatch');
    }
    _checkData(x, kernels.topSize, 'x');
    _checkData(y, kernels.bottomSize, 'y');
    final rows = x.first.rows;
    if (y.first.rows != rows) {
      throw ArgumentError(
        'x and y must have the same number of samples, got $rows and '
        '${y.first.rows}.',
      );
    }
    final opt = weightUpdate ? optimiser : null;
    final kind = switch (opt) {
      Sgd() => DeepOptimiserKind.sgd,
      Adam() => DeepOptimiserKind.adam,
      null => null,
    };
    final pipeline = await _pipeline(kind);
    if (rows == 0) {
      return readPredictions
          ? [for (var e = 0; e < size; e++) Matrix(0, kernels.bottomSize)]
          : null;
    }
    final xStride = _uploadData(x, isX: true);
    final yStride = _uploadData(y, isX: false);
    final outStride = rows * kernels.bottomSize;
    // The kernel declares OUT whether or not it writes it (the predictions
    // flag decides at run time), so it is always bound: a placeholder of
    // one float when the predictions are not read.
    final outFloats = readPredictions ? size * outStride : 1;
    _out = _ensure(_out, _outFloats, outFloats);
    _outFloats = math.max(_outFloats, outFloats);
    if (opt is Adam) {
      _prepareAdam(opt);
      _bc = _ensure(_bc, _bcFloats, 2 * rows);
      _bcFloats = math.max(_bcFloats, 2 * rows);
      final bc = backend.hostFloats(2 * rows);
      for (var t = 0; t < rows; t++) {
        final count = _adamCount + t + 1;
        bc[2 * t] = 1 - math.pow(opt.b1, count).toDouble();
        bc[2 * t + 1] = 1 - math.pow(opt.b2, count).toDouble();
      }
      backend.write(_bc!, bc);
      _adamCount += rows;
    }
    final u = Uint32List(16);
    final uf = Float32List.view(u.buffer);
    u
      ..[0] = size
      ..[3] =
          (learningKind == LearningKind.precisionWeighted
              ? DeepFitFlags.precisionWeighted
              : 0) |
          (updatePrecisions ? DeepFitFlags.updatePrecisions : 0) |
          (readPredictions ? DeepFitFlags.predictions : 0)
      ..[4] = 0x7fc00000
      ..[5] = xStride
      ..[6] = yStride
      ..[7] = readPredictions ? outStride : 0;
    uf[8] = timeStep;
    switch (opt) {
      case Sgd(:final learningRate):
        uf[9] = learningRate;
      case Adam():
        uf
          ..[9] = opt.learningRate
          ..[10] = opt.b1
          ..[11] = opt.b2
          ..[12] = opt.eps
          ..[13] = opt.epsRoot;
      case null:
    }
    final bindings = <GpuBuffer?>[
      for (final b in DeepBinding.values)
        switch (b) {
          DeepBinding.params => _params,
          DeepBinding.state => _state,
          DeepBinding.x => _x,
          DeepBinding.y => _y,
          DeepBinding.out => _out,
          DeepBinding.uniform => _uniform,
          DeepBinding.biasCorrection => opt is Adam ? _bc : null,
        },
    ];
    final gx = math.min(size, deepFitGridWidth);
    final gy = (size + deepFitGridWidth - 1) ~/ deepFitGridWidth;
    for (var start = 0; start < rows; start += rowsPerDispatch) {
      u
        ..[1] = start
        ..[2] = math.min(rows, start + rowsPerDispatch);
      backend.writeUints(_uniform, u);
      backend.dispatch(pipeline, bindings, gx, gy);
    }
    if (!readPredictions) {
      backend.flush();
      return null;
    }
    final raw = await backend.read(_out!, bytes: size * outStride * 4);
    return [
      for (var e = 0; e < size; e++)
        Matrix(
          rows,
          kernels.bottomSize,
          Float64List.fromList(raw.sublist(e * outStride, (e + 1) * outStride)),
        ),
    ];
  }

  /// Rows per dispatch when [fit] is given none: about 5e8 weight visits
  /// (members × rows × weights), ~50 ms on an RX 7600 XT, at most 256.
  int get defaultRowsPerDispatch {
    var weights = 0;
    for (final l in kernels.layers) {
      weights += (l.weights?.data.length ?? 0) + l.size;
    }
    return math.max(1, math.min(256, 500000000 ~/ (size * weights)));
  }

  /// Fresh moments for every member when [adam] is new to the GPU.
  void _prepareAdam(Adam adam) {
    if (identical(adam, _adam)) return;
    _adam = adam;
    _adamCount = 0;
    final k = kernels;
    for (var l = 1; l < k.nLayers; l++) {
      final n = k.layers[l].weights!.data.length;
      final zeros = backend.hostFloats(n);
      for (var e = 0; e < size; e++) {
        final base = e * k.paramFloats;
        backend
          ..write(_params, zeros, offset: (base + k.muOffset(l)) * 4)
          ..write(_params, zeros, offset: (base + k.nuOffset(l)) * 4);
      }
    }
  }

  /// Waits for all queued work.
  Future<void> idle() => backend.idle();

  void dispose() {
    for (final b in [_params, _state, _uniform, _x, _y, _out, _bc]) {
      if (b != null) backend.release(b);
    }
  }
}
