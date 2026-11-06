import 'dart:async';
import 'dart:typed_data';

import 'model.dart';
import 'pass.dart';
import 'worker_transport.dart';
// `dart.library.isolate` is also defined on dart2wasm, where isolates throw
// at run time; `dart.library.io` is only defined where they work.
import 'worker_transport_stub.dart'
    if (dart.library.io) 'worker_transport_io.dart'
    if (dart.library.js_interop) 'worker_transport_web.dart'
    as platform;

export 'worker_transport.dart' show FitWorkerScript;

/// A pool of workers that the fitting functions split their batched passes
/// across: `workers.fitMap(...)`, `workers.laplace(...)`,
/// `workers.sampleNuts(...)`, `workers.sampleHierarchical(...)`,
/// `workers.fitEmpiricalBayes(...)` give the same results as the functions
/// of the same names, bit for bit.
///
/// The workers are isolates on the Dart VM and Web Workers on the web
/// (given a compiled worker script, see [FitWorkerScript]). Without
/// workers (`count: 0`, or no script on the web) every pass runs on the
/// calling isolate, so code written against `FitWorkers` runs everywhere.
///
/// Two kinds of parallelism, combined:
/// - drivers whose tasks are independent (per-subject optimisers and NUTS
///   chains, hierarchical NUTS chains) split them into [streams]
///   [Concurrent] streams, so a task that needs more passes than the rest
///   doesn't hold them all at every pass's barrier;
/// - each pass is divided into contiguous slices of about equal cost, as
///   many as the stream's share of the workers, none smaller than
///   [minSliceTrials] filtered trials (a round trip costs ~5 µs between
///   isolates; a slice of 2048 trials of a binary 2-level HGF ~350 µs).
///
/// Slices of every stream go to one queue; idle workers take them one at
/// a time. The calling isolate only routes: it resumes the drivers, which
/// would stall every stream if it were busy filtering. So the default
/// [spawn] is one worker per processor.
///
/// Spawn one pool and reuse it: the workers live until [close], and an
/// idle worker only waits for messages. Pools are not cheap to multiply:
/// the Dart VM limits how many isolates of a group can run at once, and
/// one pool already uses every core.
///
/// Models and subjects are copied to each worker the first time a pass
/// needs them, and forgotten there when they are garbage collected here.
/// Isolates take any model whose closures can be sent (custom response and
/// coupling functions are, unless they capture something that isn't, like
/// a port). Web Workers take models that `FitCodec` can serialise: no
/// `ResponseModel.fn`, and custom types need decoders in the worker.
final class FitWorkers {
  FitWorkers._(this._transport, this.minSliceTrials)
    : count = _transport.count,
      _busy = List.filled(_transport.count, false) {
    _transport
      ..onResult = _onResult
      ..onFailure = _onFailure
      ..onExit = _onExit;
  }

  /// Spawns [count] workers (default: one per processor). With [count] 0
  /// nothing is spawned and passes run on the calling isolate.
  ///
  /// On the web, workers need [script]; without it the pool has none. On
  /// the VM, [script] is ignored.
  static Future<FitWorkers> spawn({
    int? count,
    int minSliceTrials = defaultMinSliceTrials,
    FitWorkerScript? script,
  }) async {
    if (count != null && count < 0) throw ArgumentError.value(count, 'count');
    return FitWorkers._(
      await platform.spawnTransport(count, script),
      minSliceTrials,
    );
  }

  /// Default for [minSliceTrials].
  static const defaultMinSliceTrials = 2048;

  final WorkerTransport _transport;

  /// Number of workers.
  final int count;

  /// The smallest slice of a pass, in filtered trials, worth sending to
  /// a worker.
  int minSliceTrials;

  /// How many streams a driver should split independent tasks into: one
  /// per worker.
  int get streams => count == 0 ? 1 : count;

  bool _closed = false;

  int _nextId = 0;
  // Ids are wrapped: on dart2wasm an `Expando<int>` that is the program's
  // only Expando loses its values (docs/upstream D7).
  final _modelIds = Expando<_Id>('model');
  final _familyIds = Expando<_Id>('family');
  final _subjectIds = Expando<_Id>('subject');
  late final _finalizer = Finalizer<_Id>((id) => _drop(id.value));

  /// Slices waiting for a worker.
  final _queue = <_Slice>[];
  var _queueHead = 0;

  /// Whether each worker has a slice.
  final List<bool> _busy;

  /// Sent slices by message id.
  final _sent = <int, _Slice>{};

  /// Streams currently running (to size the slices).
  var _active = 0;

  /// Slices evaluated by the workers so far.
  int workerSlices = 0;

  /// Passes evaluated on the calling isolate (no workers) so far.
  int localSlices = 0;

  /// Without workers, passes run on the calling isolate one after the
  /// other, and a long job would block it (a UI wouldn't draw). With
  /// [yieldInterval] set, a job gives way to the event loop (a timer
  /// event, so frames and input get their turn) whenever it has run this
  /// long. Results don't change. Ignored when there are workers: the
  /// calling isolate only routes then.
  Duration? yieldInterval;
  final _sinceYield = Stopwatch();

  // Bumped by [cancel]; each run stops once it differs from its own.
  int _generation = 0;

  /// Stops every job running on this pool: each one's future fails with
  /// [FitCancelled] before its next pass (passes already sent finish on
  /// the workers, and their results are dropped). The pool stays usable.
  void cancel() => _generation++;

  /// Runs [program], answering each pass on this pool.
  Future<T> run<T>(Program<T> program) async {
    if (_closed) throw StateError('FitWorkers closed');
    final result = Out<T>();
    _active++;
    try {
      await _drive(program(result), _generation);
    } finally {
      _active--;
    }
    return result.value;
  }

  Future<void> _drive(Iterable<Step> steps, int generation) async {
    for (final step in steps) {
      if (generation != _generation) throw const FitCancelled();
      switch (step) {
        case Pass():
          await _submit(step);
          if (count == 0) await _maybeYield();
        case Concurrent(:final streams):
          _active--;
          try {
            await Future.wait([
              for (final s in streams) _stream(s, generation),
            ]);
          } finally {
            _active++;
          }
      }
    }
  }

  Future<void> _stream(Iterable<Step> steps, int generation) async {
    _active++;
    try {
      await _drive(steps, generation);
    } finally {
      _active--;
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

  /// Stops the workers. Passes still running fail.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    _transport.close();
    _failAll(StateError('FitWorkers closed'));
  }

  Future<void> _submit(Pass pass) {
    if (_closed) throw StateError('FitWorkers closed');
    if (count == 0) {
      localSlices++;
      pass.run();
      return Future.value();
    }
    // As many slices as this stream's share of the workers.
    final share = (count / (_active < 1 ? 1 : _active)).ceil();
    final bounds = pass.partition(share, minSliceTrials);
    final pending = _PendingPass(pass, bounds);
    try {
      pending
        ..modelId = _register(pass.model)
        ..subjectIds = _registerSubjects(pass.subjects);
    } catch (e, st) {
      // E.g. a model that can't be sent to the workers.
      pending
        ..fail(e, st)
        ..finish();
      return pending.done.future;
    }
    for (var r = 0; r < bounds.length - 1; r++) {
      _queue.add(_Slice(pending, r));
    }
    _pump();
    return pending.done.future;
  }

  /// Hands queued slices to idle workers.
  void _pump() {
    while (_queueHead < _queue.length) {
      final w = _busy.indexOf(false);
      if (w < 0) return;
      _send(w, _queue[_queueHead++]);
    }
    _queue.clear();
    _queueHead = 0;
  }

  void _send(int worker, _Slice slice) {
    final p = slice.pending, pass = p.pass, subjectIds = p.subjectIds;
    final start = p.bounds[slice.range], end = p.bounds[slice.range + 1];
    final k = pass.model.dimension;
    // The slice's subjects, renumbered locally.
    final local = <int, int>{};
    final ids = <int>[];
    final subjectOf = Int32List(end - start);
    for (var e = start; e < end; e++) {
      final s = pass.subjectOf[e];
      subjectOf[e - start] = local.putIfAbsent(s, () {
        ids.add(subjectIds[s]);
        return ids.length - 1;
      });
    }
    final id = _nextId++;
    _transport.evaluate(worker, (
      id: id,
      modelId: p.modelId,
      subjectIds: Int32List.fromList(ids),
      subjectOf: subjectOf,
      y: pass.y.sublist(start * k, end * k),
      jacobian: pass.jacobian,
      includePrior: pass.includePrior,
      negLogLikelihood: pass.negLogLikelihood != null,
      gradient: pass.gradient != null,
    ));
    _sent[id] = slice;
    _busy[worker] = true;
    workerSlices++;
  }

  int _register(HgfModel model) {
    final known = _modelIds[model];
    if (known != null) return known.value;
    final family = modelFamily(model);
    var familyId = _familyIds[family]?.value;
    if (familyId == null) {
      familyId = _nextId++;
      _transport.registerFamily(familyId, model);
      final boxed = _Id(familyId);
      _familyIds[family] = boxed;
      _finalizer.attach(family, boxed);
    }
    final id = _nextId++;
    _transport.registerModel(id, familyId, model.parameters);
    final boxed = _Id(id);
    _modelIds[model] = boxed;
    _finalizer.attach(model, boxed);
    return id;
  }

  /// Registry ids of [subjects], registering the new ones in one message
  /// (so series shared between subjects stay shared on the workers).
  List<int> _registerSubjects(List<Subject> subjects) {
    final ids = List<int>.filled(subjects.length, 0);
    final newIds = <int>[];
    final newSubjects = <Subject>[];
    for (var s = 0; s < subjects.length; s++) {
      final subject = subjects[s];
      final id = _subjectIds[subject]?.value;
      if (id == null) {
        newIds.add(_nextId++);
        newSubjects.add(subject);
        ids[s] = newIds.last;
      } else {
        ids[s] = id;
      }
    }
    if (newIds.isNotEmpty) {
      _transport.registerSubjects(newIds, newSubjects);
      for (var i = 0; i < newIds.length; i++) {
        final boxed = _Id(newIds[i]);
        _subjectIds[newSubjects[i]] = boxed;
        _finalizer.attach(newSubjects[i], boxed);
      }
    }
    return ids;
  }

  void _drop(int id) {
    if (!_closed) _transport.drop(id);
  }

  void _onResult(
    int id,
    int worker,
    Float64List out,
    Float64List? negLogLikelihood,
    Float64List? gradient,
  ) {
    final slice = _sent.remove(id);
    _busy[worker] = false;
    if (slice != null) {
      final p = slice.pending, b = p.bounds, r = slice.range;
      p.pass.absorb(b[r], b[r + 1], out, negLogLikelihood, gradient);
      p.arrived();
    }
    _pump();
  }

  void _onFailure(int id, int worker, String error, String stack) {
    final slice = _sent.remove(id);
    _busy[worker] = false;
    if (slice != null) {
      slice.pending
        ..fail(_WorkerError(error, stack), StackTrace.empty)
        ..arrived();
    }
    _pump();
  }

  void _onExit(Object? reason) {
    if (_closed) return;
    // A worker died (an uncaught error, or it was killed): the pool can't
    // answer passes any more.
    _closed = true;
    _failAll(
      StateError(
        'A FitWorkers worker exited${reason == null ? '' : ': $reason'}',
      ),
    );
    _transport.close();
  }

  void _failAll(Object error) {
    final pending = {
      for (final s in _sent.values) s.pending,
      for (var i = _queueHead; i < _queue.length; i++) _queue[i].pending,
    };
    _sent.clear();
    _queue.clear();
    _queueHead = 0;
    for (final p in pending) {
      p.fail(error, StackTrace.current);
      p.finish();
    }
  }
}

/// What a job on a [FitWorkers] pool throws after [FitWorkers.cancel].
final class FitCancelled implements Exception {
  const FitCancelled();

  @override
  String toString() => 'FitCancelled: the job was cancelled';
}

/// An error raised in a worker, with the worker's stack trace (as the
/// VM's `RemoteError`, which isn't available on the web).
final class _WorkerError extends Error {
  _WorkerError(this.message, this.remoteStack);

  final String message;
  final String remoteStack;

  @override
  StackTrace get stackTrace => StackTrace.fromString(remoteStack);

  @override
  String toString() => message;
}

/// A registry id, boxed (see `_modelIds`).
final class _Id {
  const _Id(this.value);

  final int value;
}

final class _Slice {
  _Slice(this.pending, this.range);

  final _PendingPass pending;
  final int range;
}

final class _PendingPass {
  _PendingPass(this.pass, this.bounds) : _left = bounds.length - 1;

  final Pass pass;
  final List<int> bounds;
  late final int modelId;
  late final List<int> subjectIds;
  final done = Completer<void>();
  int _left;
  (Object, StackTrace)? _error;

  void fail(Object error, StackTrace stack) => _error ??= (error, stack);

  /// Marks one slice as answered; completes once all have been.
  void arrived() {
    if (--_left == 0) finish();
  }

  void finish() {
    if (done.isCompleted) return;
    if (_error case (final e, final st)) {
      done.completeError(e, st);
    } else {
      done.complete();
    }
  }
}
