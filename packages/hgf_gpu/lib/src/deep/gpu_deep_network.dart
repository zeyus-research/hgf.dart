import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_deep/hgf_deep.dart';

import '../batch.dart';
import 'deep_kernels.dart';

/// A copy of a `DeepNetwork` on the GPU, for batched learning
/// ([batchUpdate], [trainEpoch]) and [predict] in f32.
///
/// The weights, the optimiser's moments and the network's carried beliefs
/// stay on the GPU between calls; [download] copies them back into a
/// `DeepNetwork`, [upload] replaces them from one. Every step runs as a
/// chain of fused kernels with no host round trip (see [DeepKernels]), so
/// [trainEpoch] queues a whole epoch and reads back once.
///
/// Results match the f64 CPU engine to f32 accuracy (`test/deep_gpu_test`).
/// The sequential `fit` stays on the CPU.
final class GpuDeepNetwork {
  GpuDeepNetwork._(this.backend, this.kernels)
    : _params = backend.storage(kernels.paramFloats * 4),
      _state = backend.storage(kernels.stateFloats * 4),
      _uniform = backend.uniform(48);

  /// Builds the kernels for [net]'s shape and settings, for batches of up
  /// to [capacity] samples, and uploads its weights and beliefs.
  static Future<GpuDeepNetwork> create(
    GpuBackend backend,
    DeepNetwork net, {
    int capacity = 256,
  }) async {
    final kernels = DeepKernels(net, capacity: capacity);
    final bytes = kernels.stateFloats * 4;
    if (bytes > backend.maxStorageBufferBytes) {
      throw ArgumentError(
        'Batch state for capacity $capacity is $bytes bytes; the device '
        'allows ${backend.maxStorageBufferBytes}.',
      );
    }
    final gpu = GpuDeepNetwork._(backend, kernels);
    gpu._fillBias();
    gpu.upload(net);
    return gpu;
  }

  final GpuBackend backend;
  final DeepKernels kernels;
  final GpuBuffer _params, _state, _uniform;
  GpuBuffer? _x, _y, _out;
  int _xFloats = 0, _yFloats = 0, _outFloats = 0;

  final _pipelines = <String, Future<GpuPipeline>>{};
  final _plans = <String, List<DeepKernelSource>>{};

  Adam? _adam;
  int _adamCount = 0;

  /// Largest batch one step takes.
  int get capacity => kernels.capacity;

  void _fillBias() {
    for (final (offset, length) in kernels.biasRows) {
      final ones = backend.hostFloats(length)..fillRange(0, length, 1.0);
      backend.write(_state, ones, offset: offset * 4);
    }
  }

  /// Replaces the GPU's weights and beliefs with [net]'s (same shape and
  /// settings as the network the kernels were built from), and resets the
  /// optimiser state.
  void upload(DeepNetwork net) {
    final k = kernels;
    if (net.nLayers != k.nLayers) {
      throw ArgumentError('Expected ${k.nLayers} layers, got ${net.nLayers}.');
    }
    final host = backend.hostFloats(k.paramFloats);
    for (var l = 0; l < k.nLayers; l++) {
      final layer = net.debugLayers[l];
      if (layer.size != k.layers[l].size) {
        throw ArgumentError('Layer $l has a different size.');
      }
      final s = net.state(l);
      for (final f in LayerField.values) {
        final v = s.field(f);
        if (v == null || f.volatility && !k.layers[l].hasVolatility) continue;
        host.setAll(k.templateOffset(l, f), v);
      }
      host.setAll(k.tonicVolOffset(l), layer.tonicVolatilityVol);
      final tonic = k.tonicOffset(l);
      if (tonic != null) host.setAll(tonic, layer.tonicVolatility!);
      host.setAll(k.priorOffset(l), layer.priorPrecision);
      final w = layer.weights;
      if (w != null) host.setAll(k.weightsOffset(l), w.data);
    }
    backend.write(_params, host);
    _adam = null;
    _adamCount = 0;
  }

  /// Copies the weights and the carried beliefs (precision, and the
  /// volatility level's mean and precision) into [net].
  Future<void> download(DeepNetwork net) async {
    final k = kernels;
    final p = await backend.read(_params, bytes: k.paramFloats * 4);
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
        final o = k.templateOffset(l, f);
        for (var i = 0; i < dst.length; i++) {
          dst[i] = p[o + i];
        }
      }
      final w = net.debugLayers[l].weights;
      if (w != null) {
        final o = k.weightsOffset(l);
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

  void _uploadMatrix(Matrix m, {required bool isX}) {
    final n = m.data.length;
    if (isX) {
      _x = _ensure(_x, _xFloats, n);
      _xFloats = math.max(_xFloats, n);
    } else {
      _y = _ensure(_y, _yFloats, n);
      _yFloats = math.max(_yFloats, n);
    }
    final host = backend.hostFloats(n);
    for (var i = 0; i < n; i++) {
      host[i] = m.data[i];
    }
    backend.write(isX ? _x! : _y!, host);
  }

  void _ensureOut(int floats) {
    _out = _ensure(_out, _outFloats, floats);
    _outFloats = math.max(_outFloats, floats);
  }

  void _checkShapes(Matrix x, Matrix? y) {
    if (x.cols != kernels.topSize) {
      throw ArgumentError(
        'x has ${x.cols} column(s) but the top layer has ${kernels.topSize}.',
      );
    }
    if (y == null) return;
    if (y.cols != kernels.bottomSize) {
      throw ArgumentError(
        'y has ${y.cols} column(s) but the bottom layer has '
        '${kernels.bottomSize}.',
      );
    }
    if (x.rows != y.rows) {
      throw ArgumentError(
        'x and y must have the same number of samples, got ${x.rows} and '
        '${y.rows}.',
      );
    }
  }

  final _u = Uint32List(12);
  late final _uf = Float32List.view(_u.buffer);

  void _writeUniform({
    required int b,
    required int start,
    LearningKind learningKind = LearningKind.precisionWeighted,
    double timeStep = 1.0,
    Optimiser? optimiser,
  }) {
    _u
      ..[0] = b
      ..[1] = start
      ..[2] = learningKind == LearningKind.standard ? 0 : 1
      ..[3] = 0x7fc00000;
    _uf[4] = timeStep;
    switch (optimiser) {
      case Sgd(:final learningRate):
        _uf[5] = learningRate;
      case Adam():
        _adamCount++;
        _uf
          ..[5] = optimiser.learningRate
          ..[6] = optimiser.b1
          ..[7] = optimiser.b2
          ..[8] = optimiser.eps
          ..[9] = optimiser.epsRoot
          ..[10] = 1 - math.pow(optimiser.b1, _adamCount).toDouble()
          ..[11] = 1 - math.pow(optimiser.b2, _adamCount).toDouble();
      case null:
    }
    backend.writeUints(_uniform, _u);
  }

  /// Starts fresh moments for an Adam the GPU hasn't stepped yet (as a new
  /// CPU `Adam` would).
  void _prepareOptimiser(Optimiser? optimiser) {
    if (optimiser is! Adam || identical(optimiser, _adam)) return;
    _adam = optimiser;
    _adamCount = 0;
    final k = kernels;
    for (var l = 1; l < k.nLayers; l++) {
      final n = k.layers[l].weights!.data.length;
      final zeros = backend.hostFloats(n);
      backend
        ..write(_params, zeros, offset: k.muOffset(l) * 4)
        ..write(_params, zeros, offset: k.nuOffset(l) * 4);
    }
  }

  Future<GpuPipeline> _pipeline(DeepKernelSource k) =>
      _pipelines.putIfAbsent(k.name, () => backend.pipeline(k.source));

  List<GpuBuffer?> _bindings(DeepKernelSource k) => [
    for (final b in DeepBinding.values)
      k.bindings.contains(b)
          ? switch (b) {
              DeepBinding.params => _params,
              DeepBinding.state => _state,
              DeepBinding.x => _x,
              DeepBinding.y => _y,
              DeepBinding.out => _out,
              DeepBinding.uniform => _uniform,
              DeepBinding.biasCorrection => null,
            }
          : null,
  ];

  List<DeepKernelSource> _stepPlan({
    required bool errors,
    required Optimiser? optimiser,
    required bool updatePrecisions,
  }) {
    final kind = switch (optimiser) {
      Sgd() => DeepOptimiserKind.sgd,
      Adam() => DeepOptimiserKind.adam,
      null => null,
    };
    return _plans.putIfAbsent(
      'step $errors $kind $updatePrecisions',
      () => kernels.batchUpdate(
        errors: errors,
        optimiser: kind,
        updatePrecisions: updatePrecisions,
      ),
    );
  }

  /// Compiles every kernel of [plan] (once) before queueing any of it.
  Future<List<GpuPipeline>> _compile(List<DeepKernelSource> plan) =>
      Future.wait([for (final k in plan) _pipeline(k)]);

  void _queue(List<DeepKernelSource> plan, List<GpuPipeline> pipelines, int b) {
    for (var i = 0; i < plan.length; i++) {
      final (gx, gy) = plan[i].grid(b);
      if (gx == 0 || gy == 0) continue;
      backend.dispatch(pipelines[i], _bindings(plan[i]), gx, gy);
    }
  }

  /// One batch-synchronous learning step, as `DeepNetwork.batchUpdate`:
  /// every row of `(x, y)` (at most [capacity]) is swept from the network's
  /// state, the mean weight gradient drives one [optimiser] step, and with
  /// [updatePrecisions] the mean change of the carried beliefs is added to
  /// the state.
  ///
  /// Returns the per-sample error messages at the input layer
  /// (`samples × topSize`) when [readErrors]; otherwise `null` without
  /// waiting for the GPU.
  ///
  /// The GPU keeps its own Adam moments: they start from zero the first
  /// time an `Adam` is passed and continue while the same one is.
  Future<Matrix?> batchUpdate(
    Matrix x,
    Matrix y, {
    Optimiser? optimiser,
    LearningKind learningKind = LearningKind.precisionWeighted,
    double timeStep = 1.0,
    bool updatePrecisions = true,
    bool readErrors = true,
  }) async {
    _checkShapes(x, y);
    final b = x.rows;
    if (b < 1 || b > capacity) {
      throw ArgumentError('Batch of $b samples; capacity is $capacity.');
    }
    final plan = _stepPlan(
      errors: readErrors,
      optimiser: optimiser,
      updatePrecisions: updatePrecisions,
    );
    final pipelines = await _compile(plan);
    _uploadMatrix(x, isX: true);
    _uploadMatrix(y, isX: false);
    if (readErrors) _ensureOut(b * kernels.topSize);
    _prepareOptimiser(optimiser);
    _writeUniform(
      b: b,
      start: 0,
      learningKind: learningKind,
      timeStep: timeStep,
      optimiser: optimiser,
    );
    _queue(plan, pipelines, b);
    if (!readErrors) return null;
    return _readMatrix(b, kernels.topSize);
  }

  /// One pass over `(x, y)` in minibatches of [batchSize] (the last one
  /// may be smaller), each a [batchUpdate]. The data is uploaded once and
  /// the whole epoch is queued without waiting for the GPU.
  ///
  /// With [readErrors], returns every row's input-layer error message
  /// (`rows × topSize`, from the step that row was part of).
  Future<Matrix?> trainEpoch(
    Matrix x,
    Matrix y, {
    required int batchSize,
    Optimiser? optimiser,
    LearningKind learningKind = LearningKind.precisionWeighted,
    double timeStep = 1.0,
    bool updatePrecisions = true,
    bool readErrors = false,
  }) async {
    _checkShapes(x, y);
    if (batchSize < 1 || batchSize > capacity) {
      throw ArgumentError.value(
        batchSize,
        'batchSize',
        'must be in 1..$capacity',
      );
    }
    final plan = _stepPlan(
      errors: readErrors,
      optimiser: optimiser,
      updatePrecisions: updatePrecisions,
    );
    final pipelines = await _compile(plan);
    _uploadMatrix(x, isX: true);
    _uploadMatrix(y, isX: false);
    if (readErrors) _ensureOut(x.rows * kernels.topSize);
    _prepareOptimiser(optimiser);
    for (var start = 0; start < x.rows; start += batchSize) {
      final b = math.min(batchSize, x.rows - start);
      _writeUniform(
        b: b,
        start: start,
        learningKind: learningKind,
        timeStep: timeStep,
        optimiser: optimiser,
      );
      _queue(plan, pipelines, b);
    }
    if (!readErrors) {
      backend.flush();
      return null;
    }
    return _readMatrix(x.rows, kernels.topSize);
  }

  /// The output layer's predicted mean for every row of [x], from the
  /// current weights (the state is not modified). `samples × bottomSize`.
  Future<Matrix> predict(Matrix x) async {
    _checkShapes(x, null);
    final plan = _plans.putIfAbsent('predict', kernels.predict);
    final pipelines = await _compile(plan);
    _uploadMatrix(x, isX: true);
    _ensureOut(x.rows * kernels.bottomSize);
    for (var start = 0; start < x.rows; start += capacity) {
      final b = math.min(capacity, x.rows - start);
      _writeUniform(b: b, start: start);
      _queue(plan, pipelines, b);
    }
    return _readMatrix(x.rows, kernels.bottomSize);
  }

  Future<Matrix> _readMatrix(int rows, int cols) async {
    final raw = await backend.read(_out!, bytes: rows * cols * 4);
    final m = Matrix(rows, cols);
    for (var i = 0; i < m.data.length; i++) {
      m.data[i] = raw[i];
    }
    return m;
  }

  /// Waits for all queued work.
  Future<void> idle() => backend.idle();

  void dispose() {
    for (final b in [_params, _state, _uniform, _x, _y, _out]) {
      if (b != null) backend.release(b);
    }
  }
}
