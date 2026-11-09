import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';

import '../fit/gpu_fitter.dart';
import 'wgpu_backend.dart';

/// A [PassRunner] that answers passes with a [GpuFitter] on wgpu-native in
/// a background isolate.
///
/// wgpu-native completes every call before it returns, so a [GpuFitter] on
/// the calling isolate holds it for the whole job; a UI wouldn't draw.
/// Here the drivers (optimisers, samplers) stay on the calling isolate and
/// only the GPU passes cross over: the model and the subjects once, then
/// each pass's points and results. Precise passes (and empty ones) run on
/// [cpu], or inline without one, as with [GpuFitter].
///
/// The results are those of a [GpuFitter] on the same backend.
final class WgpuIsolateFitter implements PassRunner {
  WgpuIsolateFitter._(
    this._isolate,
    this._send,
    this._replies,
    this.description,
  ) {
    _replies.handler = _onReply;
    _send.send(('replies', _replies.sendPort));
  }

  /// Spawns the isolate and opens [library]'s high-performance adapter
  /// there (see [WgpuBackend.create]). Throws a [StateError] with the reason
  /// if there is no library or no adapter.
  static Future<WgpuIsolateFitter> spawn({String? library}) async {
    final ready = Completer<Object?>();
    final setup = RawReceivePort((Object? m) => ready.complete(m));
    final Isolate isolate;
    try {
      isolate = await Isolate.spawn(
        _main,
        (setup.sendPort, library),
        debugName: 'wgpu fitter',
        errorsAreFatal: false,
      );
    } catch (_) {
      setup.close();
      rethrow;
    }
    final reply = await ready.future;
    setup.close();
    switch (reply) {
      case (SendPort send, String description):
        return WgpuIsolateFitter._(
          isolate,
          send,
          RawReceivePort(),
          description,
        );
      case String reason:
        isolate.kill();
        throw StateError(reason);
      default:
        isolate.kill();
        throw StateError('wgpu isolate: unexpected $reply');
    }
  }

  final Isolate _isolate;
  final SendPort _send;
  final RawReceivePort _replies;

  /// The adapter, as [WgpuBackend.description] gives it.
  final String description;

  /// Where precise passes run (null: on the calling isolate).
  FitWorkers? cpu;

  /// Passes answered on the GPU and on the CPU (precise ones, and those of
  /// models without a kernel).
  int gpuPasses = 0, cpuPasses = 0;

  /// Why the last model that fell back to the CPU has no GPU kernel.
  String? fallbackReason;

  @override
  int get streams => 1;

  final _pending = <int, (Pass, Completer<void>)>{};
  int _nextRequest = 0;
  bool _closed = false;

  // Registered with the isolate: the latest model and subject list. A new
  // model replaces the old one there (its kernels are released).
  final _ids = Expando<int>();
  int _nextId = 0;
  Object? _model, _subjects;

  // Bumped by [cancel]; each run stops once it differs from its own.
  int _generation = 0;

  /// Stops every running job: each one's future fails with [FitCancelled]
  /// before its next pass.
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
          await evaluatePass(step);
        case Concurrent(:final streams):
          for (final s in streams) {
            await _drive(s, generation);
          }
      }
    }
  }

  /// Answers one [pass].
  Future<void> evaluatePass(Pass pass) async {
    if (_closed) throw StateError('WgpuIsolateFitter closed');
    if (pass.precise || pass.length == 0) {
      cpuPasses++;
      final c = cpu;
      if (c != null) return c.evaluatePass(pass);
      pass.run();
      return;
    }
    final modelId = _register(pass.model, 'model');
    final subjectsId = _register(pass.subjects, 'subjects');
    final id = _nextRequest++;
    final done = Completer<void>();
    _pending[id] = (pass, done);
    _send.send((
      'pass',
      id,
      modelId,
      subjectsId,
      pass.subjectOf,
      pass.y,
      pass.jacobian,
      pass.includePrior,
      pass.negLogLikelihood != null,
      pass.gradient != null,
    ));
    return done.future;
  }

  int _register(Object o, String kind) {
    final known = _ids[o];
    if (known != null && identical(o, kind == 'model' ? _model : _subjects)) {
      return known;
    }
    final id = _ids[o] = _nextId++;
    if (kind == 'model') {
      _model = o;
    } else {
      _subjects = o;
    }
    _send.send((kind, id, o));
    return id;
  }

  void _onReply(Object? m) {
    switch (m) {
      case (
        int id,
        bool onGpu,
        String? reason,
        Float64List out,
        Float64List? nll,
        Float64List? gradient,
      ):
        final (pass, done) = _pending.remove(id)!;
        onGpu ? gpuPasses++ : cpuPasses++;
        if (reason != null) fallbackReason = reason;
        pass.absorb(0, pass.length, out, nll, gradient);
        done.complete();
      case (int id, String error):
        _pending.remove(id)!.$2.completeError(StateError(error));
    }
  }

  /// Stops the isolate (and releases its GPU resources). Passes still
  /// running fail.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    _send.send(('close',));
    _replies.close();
    for (final (_, done) in _pending.values) {
      done.completeError(StateError('WgpuIsolateFitter closed'));
    }
    _pending.clear();
    // The isolate exits on 'close'; this only bounds a stuck one.
    Timer(const Duration(seconds: 2), _isolate.kill);
  }
}

void _main((SendPort, String?) args) {
  final (setup, library) = args;
  final WgpuBackend? backend;
  try {
    backend = WgpuBackend.create(library: library);
  } on ArgumentError catch (e) {
    setup.send('wgpu-native isn\'t installed (${e.message})');
    return;
  } catch (e) {
    setup.send('$e');
    return;
  }
  if (backend == null) {
    setup.send('wgpu-native found no GPU adapter');
    return;
  }
  final requests = RawReceivePort();
  setup.send((requests.sendPort, backend.description));
  SendPort? replies;
  GpuFitter? fitter;
  final models = <int, HgfModel>{};
  final subjects = <int, List<Subject>>{};
  requests.handler = (Object? m) async {
    switch (m) {
      case ('replies', SendPort p):
        replies = p;
      case ('model', int id, HgfModel model):
        // One model at a time: release the last one's kernels.
        fitter?.dispose();
        fitter = GpuFitter(backend!);
        models
          ..clear()
          ..[id] = model;
      case ('subjects', int id, List<Subject> list):
        subjects
          ..clear()
          ..[id] = list;
      case (
        'pass',
        int id,
        int modelId,
        int subjectsId,
        Int32List subjectOf,
        Float64List y,
        bool jacobian,
        List<bool>? includePrior,
        bool nll,
        bool gradient,
      ):
        final f = fitter!;
        final pass = Pass(
          models[modelId]!,
          subjects[subjectsId]!,
          subjectOf,
          y,
          jacobian: jacobian,
          includePrior: includePrior,
          negLogLikelihood: nll,
          gradient: gradient,
        );
        try {
          final before = f.gpuPasses;
          await f.evaluatePass(pass);
          final onGpu = f.gpuPasses > before;
          replies!.send((
            id,
            onGpu,
            onGpu ? null : f.fallbackReason,
            pass.out,
            pass.negLogLikelihood,
            pass.gradient,
          ));
        } catch (e) {
          replies!.send((id, '$e'));
        }
      case ('close',):
        fitter?.dispose();
        requests.close();
        Isolate.exit();
    }
  };
}
