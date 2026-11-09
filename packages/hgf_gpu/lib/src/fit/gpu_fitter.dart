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
  GpuFitter(this.backend, {this.cpu, this.trialsPerDispatch = 0});

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

  final _models = <(Network, ResponseModel, String, bool), _Model?>{};
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
    final m = await _model(pass.model, pass.gradient != null);
    if (m == null) return _cpu(pass);
    gpuPasses++;
    gpuElements += pass.length;
    await m.evaluate(this, pass);
  }

  Future<_Model?> _model(HgfModel model, bool gradients) async {
    final sig = [
      for (final p in model.parameters) p.targets.join('|'),
    ].join(';');
    final key = (model.network, model.response, sig, gradients);
    if (_models.containsKey(key)) return _models[key];
    _Model? m;
    try {
      final kernel = FitKernel(model, gradients: gradients);
      m = _Model(kernel, await backend.pipeline(kernel.source));
    } on UnsupportedError catch (e) {
      fallbackReason = e.message;
    }
    return _models[key] = m;
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

/// A compiled model and its per-pass buffers (grown as needed).
final class _Model {
  _Model(this.kernel, this.pipeline);

  final FitKernel kernel;
  final GpuPipeline pipeline;
  GpuBuffer? _state, _tangents, _job, _params;
  int _capacity = 0;

  void _ensure(GpuBackend backend, int b) {
    if (b <= _capacity) return;
    release(backend);
    final cap = b < 64 ? 64 : b;
    final k = kernel.directions;
    final stateBytes = kernel.nSlots * cap * 4;
    final tanBytes = kernel.tangentSlots * k * cap * 4;
    final limit = backend.maxStorageBufferBytes;
    if (stateBytes > limit || tanBytes > limit) {
      throw ArgumentError(
        'A pass of $b elements needs ${stateBytes > tanBytes ? stateBytes : tanBytes} '
        'bytes of state; the device allows $limit',
      );
    }
    _state = backend.storage(stateBytes);
    if (k > 0) _tangents = backend.storage(tanBytes);
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
    _ensure(backend, b);
    final data = fitter._data(pass.subjects, model);

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
    for (final buf in [_state, _tangents, _job, _params]) {
      if (buf != null) backend.release(buf);
    }
    _state = _tangents = _job = _params = null;
    _capacity = 0;
  }
}
