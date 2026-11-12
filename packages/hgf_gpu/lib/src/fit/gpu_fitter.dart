import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

import '../batch.dart';
import 'fit_kernel.dart';

/// Runs hgf_fit's fitting functions with their likelihoods on a GPU:
/// `gpu.fitMap(...)`, `gpu.sampleNuts(...)`, `gpu.sampleHierarchical(...)`,
/// `gpu.laplace(...)`, `gpu.fitEmpiricalBayes(...)`.
///
/// Each batched pass is one kernel dispatch over its elements (subject ×
/// parameter vector), with exact gradients from forward-mode tangents
/// where the pass asks for them. The kernel works in `f32`, so the
/// objective is accurate to about 1e-6 relative, as in pyhgf's `float32`
/// mode; the priors and transforms are added on the host in `f64`.
///
/// Passes that need `f64` ([Pass.precise]: finite differences, Hessians,
/// the values `fitMap` reports) and models without a GPU kernel (vector
/// nodes, custom couplings or response models) run on the CPU: on [cpu]
/// if given, else on the calling isolate. So `laplace` and the final
/// estimates of `fitMap` match the CPU's to `f64` accuracy at the optimum
/// the GPU found.
///
/// The GPU pays off with many elements per pass: many subjects, chains or
/// restarts. A pass costs at least a round trip (~1 ms native, 2.5-4 ms in
/// a browser), and one GPU thread filters a trial more slowly than a CPU
/// core.
final class GpuFitter implements PassRunner {
  GpuFitter(
    this.backend, {
    this.cpu,
    this.trialsPerDispatch = 0,
    this.gradientMode = GpuGradientMode.auto,
    this.reverseAbove = defaultReverseAbove,
  });

  /// How gradients are computed: forward-mode tangents, reverse mode, or
  /// ([GpuGradientMode.auto]) reverse mode once the parameters times the
  /// carried slots exceed [reverseAbove].
  final GpuGradientMode gradientMode;

  /// [GpuGradientMode.auto] switches to reverse mode above this many
  /// `parameters × carried slots`: a forward-mode thread carries that many
  /// tangents, and past a register budget they spill and the cost jumps
  /// (DIFFERENTIATION.md §7). Measured (`benchmark/bin/fit_reverse_gpu.dart`,
  /// Apple M3 Pro): at 170 reverse mode is 2× faster on the unbounded
  /// update and ~1.4× slower on standard/eHGF; past ~300 it wins on every
  /// model (19× at 2013).
  final int reverseAbove;

  /// The default [reverseAbove].
  static const defaultReverseAbove = 150;

  final GpuBackend backend;

  /// Where passes that the GPU doesn't answer run (null: inline).
  final FitWorkers? cpu;

  /// Bounds the trials per dispatch (browsers reset the GPU on long
  /// dispatches); 0 runs each series in one.
  final int trialsPerDispatch;

  /// Passes answered on the GPU and on the CPU.
  int gpuPasses = 0, cpuPasses = 0;

  /// Elements filtered on the GPU.
  int gpuElements = 0;

  /// Time spent in GPU passes: preparing and queueing them on the host,
  /// waiting for the results, and turning them into posteriors.
  final Stopwatch prepareTime = Stopwatch(),
      waitTime = Stopwatch(),
      finishTime = Stopwatch();

  /// Why the last model that fell back to the CPU has no GPU kernel.
  String? fallbackReason;

  final _models = <(Network, ResponseModel, String, bool, int), _Model?>{};
  final _subjectData = Expando<Map<(int, bool), _SubjectData>>();
  bool _disposed = false;

  @override
  int get streams => 1;

  /// A long job gives way to the event loop (a timer event, so frames and
  /// input get their turn) whenever it has run this long. Matters where the
  /// backend completes synchronously (wgpu-native): there a job would
  /// otherwise hold the isolate until it ends (a UI should rather use
  /// `WgpuIsolateFitter`). Results don't change.
  Duration? yieldInterval;
  final _sinceYield = Stopwatch();

  // Bumped by [cancel]; each run stops once it differs from its own.
  int _generation = 0;

  /// Stops every running job: each one's future fails with [FitCancelled]
  /// before its next pass. The fitter stays usable.
  void cancel() {
    _generation++;
    cpu?.cancel();
  }

  @override
  Future<T> run<T>(Program<T> program) async {
    final result = Out<T>();
    await _drive(program(result), _generation);
    return result.value;
  }

  Future<void> _drive(Iterable<Step> steps, int generation) async {
    for (final step in steps) {
      if (generation != _generation) throw const FitCancelled();
      switch (step) {
        case Pass():
          await _maybeYield();
          if (generation != _generation) throw const FitCancelled();
          await evaluatePass(step);
        case Concurrent(:final streams):
          for (final s in streams) {
            await _drive(s, generation);
          }
      }
    }
  }

  Future<void> _maybeYield() async {
    final interval = yieldInterval;
    if (interval == null) return;
    if (!_sinceYield.isRunning) _sinceYield.start();
    if (_sinceYield.elapsed < interval) return;
    await Future<void>.delayed(Duration.zero);
    _sinceYield.reset();
  }

  Future<void> _cpu(Pass pass) async {
    cpuPasses++;
    final c = cpu;
    if (c != null) return c.evaluatePass(pass);
    pass.run();
  }

  /// Answers one [pass].
  Future<void> evaluatePass(Pass pass) async {
    if (_disposed) throw StateError('GpuFitter disposed');
    if (pass.precise || pass.length == 0) return _cpu(pass);
    final m = await _model(pass.model, pass.gradient != null, pass.subjects);
    if (m == null) return _cpu(pass);
    gpuPasses++;
    gpuElements += pass.length;
    await m.evaluate(this, pass);
  }

  Future<_Model?> _model(
    HgfModel model,
    bool gradients,
    List<Subject> subjects,
  ) async {
    final sig = [
      for (final p in model.parameters) p.targets.join('|'),
    ].join(';');
    // Reverse mode checkpoints every ~√T trials (rounded up to a multiple
    // of 8, so similar lengths share a kernel).
    var checkpoint = 0;
    if (gradients && gradientMode != GpuGradientMode.forward) {
      final t = _maxTime(subjects, model.network.inputIdxs.length);
      checkpoint = math.max(8, (math.sqrt(t).ceil() + 7) ~/ 8 * 8);
    }
    final key = (model.network, model.response, sig, gradients, checkpoint);
    if (_models.containsKey(key)) return _models[key];
    _Model? m;
    try {
      var reverse = false;
      if (gradients) {
        reverse = switch (gradientMode) {
          GpuGradientMode.forward => false,
          GpuGradientMode.reverse => true,
          GpuGradientMode.auto =>
            model.dimension * FitKernel(model).tapeCarried > reverseAbove,
        };
      }
      final kernel = FitKernel(
        model,
        gradients: gradients,
        reverse: reverse,
        checkpoint: reverse ? checkpoint : 32,
      );
      m = _Model(
        kernel,
        await backend.pipeline(kernel.source),
        reverse ? await backend.pipeline(kernel.reverseSource!) : null,
      );
    } on UnsupportedError catch (e) {
      fallbackReason = e.message;
    }
    return _models[key] = m;
  }

  /// The longest series among [subjects] (trials).
  static int _maxTime(List<Subject> subjects, int nIn) {
    var t = 0;
    for (final s in subjects) {
      t = math.max(t, nIn == 0 ? 0 : s.inputs.length ~/ nIn);
    }
    return t;
  }

  _SubjectData _data(List<Subject> subjects, HgfModel model) {
    final nIn = model.network.inputIdxs.length;
    final uses = model.response.usesResponses;
    final byLayout = _subjectData[subjects] ??= {};
    return byLayout[(nIn, uses)] ??= _SubjectData(backend, subjects, nIn, uses);
  }

  /// Releases the GPU buffers (the backend stays open).
  void dispose() {
    _disposed = true;
    for (final m in _models.values) {
      m?.release(backend);
    }
    _models.clear();
  }
}

/// The subjects' series on the GPU: one data buffer and a subject table.
final class _SubjectData {
  factory _SubjectData(
    GpuBackend backend,
    List<Subject> subjects,
    int nIn,
    bool usesResponses,
  ) {
    var words = 0;
    var maxTime = 0;
    for (final s in subjects) {
      if (s.inputs.length % nIn != 0) {
        throw ArgumentError(
          '$s: ${s.inputs.length} inputs is not a multiple of $nIn input '
          'nodes',
        );
      }
      final nTime = s.inputs.length ~/ nIn;
      if (s.observed case final o? when o.length != s.inputs.length) {
        throw ArgumentError('$s: observed must have the shape of inputs');
      }
      if (s.timeSteps case final ts? when ts.length != nTime) {
        throw ArgumentError('$s: timeSteps must have one entry per trial');
      }
      if (usesResponses) {
        final r = s.responses;
        if (r == null) throw ArgumentError('$s has no responses');
        if (r.length < nTime) {
          throw ArgumentError('$s: ${r.length} responses for $nTime trials');
        }
      }
      if (nTime > maxTime) maxTime = nTime;
      words += s.inputs.length + (s.observed?.length ?? 0);
      words += (s.timeSteps?.length ?? 0) + (usesResponses ? nTime : 0);
    }
    final data = Uint32List(words < 4 ? 4 : words);
    final f = Float32List.view(data.buffer);
    final table = Uint32List(subjects.length * subjectStride);
    var o = 0;
    int put(List<double>? values, int n) {
      if (values == null) return noData;
      final at = o;
      for (var i = 0; i < n; i++) {
        f[o++] = values[i];
      }
      return at;
    }

    for (var j = 0; j < subjects.length; j++) {
      final s = subjects[j];
      final nTime = s.inputs.length ~/ nIn;
      table
        ..[j * subjectStride] = nTime
        ..[j * subjectStride + 1] = put(s.inputs, s.inputs.length)
        ..[j * subjectStride + 2] = put(s.observed, s.inputs.length)
        ..[j * subjectStride + 3] = put(s.timeSteps, nTime)
        ..[j * subjectStride + 4] = usesResponses
            ? put(s.responses, nTime)
            : noData;
    }
    if (data.length * 4 > backend.maxStorageBufferBytes) {
      throw ArgumentError(
        'The subjects\' data is ${data.length * 4} bytes; the device allows '
        '${backend.maxStorageBufferBytes}',
      );
    }
    final dataBuffer = backend.storage(data.length * 4);
    final tableBuffer = backend.storage(
      table.length < 4 ? 16 : table.length * 4,
    );
    backend
      ..writeUints(dataBuffer, data)
      ..writeUints(tableBuffer, table);
    return _SubjectData._(dataBuffer, tableBuffer, maxTime);
  }

  _SubjectData._(this.data, this.table, this.maxTime);

  final GpuBuffer data, table;
  final int maxTime;
}

/// Which mode [GpuFitter] computes gradients in.
enum GpuGradientMode {
  /// Reverse mode for models whose forward tangents would not fit in
  /// registers ([GpuFitter.reverseAbove]), forward mode otherwise.
  auto,

  /// Forward-mode tangents: cost grows with the number of parameters.
  forward,

  /// Reverse mode (adjoints, √T checkpoints): a fixed multiple of the
  /// likelihood's cost.
  reverse,
}

/// A compiled model and its per-pass buffers (grown as needed).
final class _Model {
  _Model(this.kernel, this.pipeline, this.reversePipeline);

  final FitKernel kernel;
  final GpuPipeline pipeline;

  /// Reverse mode: the backward sweep.
  final GpuPipeline? reversePipeline;
  GpuBuffer? _state, _tangents, _job, _params, _tape, _seg, _adj;
  int _capacity = 0, _tapeSegments = 0;

  void _ensure(GpuBackend backend, int b, int maxTime) {
    final segments = kernel.segments(maxTime);
    if (b <= _capacity && segments <= _tapeSegments) return;
    release(backend);
    final cap = b < 64 ? 64 : b;
    final k = kernel.directions;
    final stateBytes = kernel.nSlots * cap * 4;
    final tanBytes = kernel.reverse ? 0 : kernel.tangentSlots * k * cap * 4;
    final nC = math.max(1, kernel.tapeCarried);
    final tapeBytes = segments * nC * cap * 4;
    final segBytes = kernel.checkpoint * nC * cap * 4;
    final limit = backend.maxStorageBufferBytes;
    final most = [stateBytes, tanBytes, tapeBytes, segBytes].reduce(math.max);
    if (most > limit) {
      throw ArgumentError(
        'A pass of $b elements needs a $most-byte buffer; the device allows '
        '$limit',
      );
    }
    _state = backend.storage(stateBytes);
    if (k > 0 && !kernel.reverse) _tangents = backend.storage(tanBytes);
    if (kernel.reverse) {
      _tape = backend.storage(tapeBytes);
      _seg = backend.storage(segBytes);
      _adj = backend.storage(nC * cap * 4);
      _tapeSegments = segments;
    }
    _job = backend.storage(
      (cap + kernel.initial.length + (k > 0 ? 2 : 1) * _dim * cap) * 4,
    );
    _params = backend.uniform(16);
    _capacity = cap;
  }

  /// Parameters per element (the model's dimension).
  int _dim = 0;

  Future<void> evaluate(GpuFitter fitter, Pass pass) async {
    fitter.prepareTime.start();
    final backend = fitter.backend;
    final model = pass.model;
    final k = model.dimension, b = pass.length, kd = kernel.directions;
    _dim = k;
    final data = fitter._data(pass.subjects, model);
    _ensure(backend, b, data.maxTime);

    // Job: each element's subject, the initial slots, then x and dx/dy.
    final nInit = kernel.initial.length;
    final xo = b + nInit;
    final job = Uint32List(xo + (kd > 0 ? 2 : 1) * k * b);
    final f = Float32List.view(job.buffer);
    final subjectOf = pass.subjectOf;
    for (var e = 0; e < b; e++) {
      final s = subjectOf[e];
      if (s < 0 || s >= pass.subjects.length) {
        throw RangeError.index(s, pass.subjects, 'subjectOf[$e]');
      }
      job[e] = s;
    }
    for (var s = 0; s < nInit; s++) {
      f[b + s] = kernel.initial[s];
    }
    final y = pass.y;
    for (var i = 0; i < k; i++) {
      final t = model.parameters[i].transform;
      for (var e = 0; e < b; e++) {
        final yi = y[e * k + i];
        f[xo + i * b + e] = t.toConstrained(yi);
        if (kd > 0) f[xo + (k + i) * b + e] = t.dToConstrained(yi);
      }
    }
    backend.writeUints(_job!, job);

    final nTime = data.maxTime;
    final chunk = fitter.trialsPerDispatch <= 0 || nTime == 0
        ? (nTime == 0 ? 1 : nTime)
        : fitter.trialsPerDispatch;
    final groups = (b + kernel.workgroupSize - 1) ~/ kernel.workgroupSize;
    final params = Uint32List(4);
    final bindings = [
      _state!,
      data.data,
      _params!,
      _job!,
      data.table,
      ?_tangents,
      ?_tape,
    ];
    for (var t0 = 0; t0 < (nTime == 0 ? 1 : nTime); t0 += chunk) {
      params
        ..[0] = b
        ..[1] = t0
        ..[2] = chunk;
      backend
        ..writeUints(_params!, params)
        ..dispatch(pipeline, bindings, groups);
    }
    final rp = reversePipeline;
    if (rp != null) {
      // The backward sweep, last segment first; a dispatch recomputes and
      // reverses `perDispatch` segments (each ~4 primal trials' work per
      // trial, so about as long as a forward dispatch).
      final nSeg = kernel.segments(nTime);
      final perDispatch = fitter.trialsPerDispatch <= 0
          ? nSeg
          : math.max(1, fitter.trialsPerDispatch ~/ (4 * kernel.checkpoint));
      final rb = [
        _state!,
        data.data,
        _params!,
        _job!,
        data.table,
        _tape!,
        _seg!,
        _adj!,
      ];
      for (var hi = nSeg; hi > 0; hi -= perDispatch) {
        params
          ..[0] = b
          ..[1] = hi
          ..[2] = perDispatch
          ..[3] = nSeg;
        backend
          ..writeUints(_params!, params)
          ..dispatch(rp, rb, groups);
      }
    }
    fitter
      ..prepareTime.stop()
      ..waitTime.start();
    final raw = await backend.read(
      _state!,
      offset: kernel.resultSlot * b * 4,
      bytes: (1 + kd) * b * 4,
    );
    fitter
      ..waitTime.stop()
      ..finishTime.start();
    final nll = Float64List(b);
    for (var e = 0; e < b; e++) {
      final v = raw[e];
      nll[e] = v >= 3.3e38 || v.isNaN ? double.infinity : v;
    }
    Float64List? grad;
    if (kd > 0) {
      grad = Float64List(b * k);
      for (var i = 0; i < k; i++) {
        for (var e = 0; e < b; e++) {
          grad[e * k + i] = raw[(1 + i) * b + e];
        }
      }
    }
    pass.negLogLikelihood?.setAll(0, nll);
    model.completeNegLogPosterior(
      y,
      nll,
      pass.out,
      jacobian: pass.jacobian,
      likelihoodGradient: grad,
      gradient: pass.gradient,
      includePrior: pass.includePrior,
    );
    fitter.finishTime.stop();
  }

  void release(GpuBackend backend) {
    for (final buf in [_state, _tangents, _job, _params, _tape, _seg, _adj]) {
      if (buf != null) backend.release(buf);
    }
    _state = _tangents = _job = _params = _tape = _seg = _adj = null;
    _capacity = 0;
    _tapeSegments = 0;
  }
}
