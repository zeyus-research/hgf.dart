import 'dart:async';
import 'dart:io' show Platform;
import 'dart:isolate';
import 'dart:typed_data';

import 'package:hgf/hgf.dart' show Network;

import 'model.dart';
import 'response_model.dart';
import 'parameter.dart';
import 'pass.dart';

/// A pool of worker isolates that the fitting functions split their
/// batched passes across: `workers.fitMap(...)`, `workers.laplace(...)`,
/// `workers.sampleNuts(...)`, `workers.sampleHierarchical(...)`,
/// `workers.fitEmpiricalBayes(...)` give the same results as the
/// functions of the same names, bit for bit.
///
/// Two kinds of parallelism, combined:
/// - drivers whose tasks are independent (per-subject optimisers and NUTS
///   chains, hierarchical NUTS chains) split them into [streams]
///   [Concurrent] streams, so a task that needs more passes than the rest
///   doesn't hold them all at every pass's barrier;
/// - each pass is divided into contiguous slices of about equal cost, as
///   many as the stream's share of the workers, none smaller than
///   [minSliceTrials] filtered trials (a round trip costs ~5 µs; a slice of
///   2048 trials of a binary 2-level HGF ~350 µs).
///
/// Slices of every stream go to one queue; idle workers take them one at
/// a time. The calling isolate only routes: it resumes the drivers, which
/// would stall every stream if it were busy filtering. So the default
/// [spawn] is one worker per processor.
///
/// Spawn one pool and reuse it: the isolates live until [close], and an
/// idle worker only waits on its port. Pools are not cheap to multiply:
/// the Dart VM limits how many isolates of a group can run at once, and
/// one pool already uses every core.
///
/// Models and subjects are copied to each worker the first time a pass
/// needs them, and forgotten there when they are garbage collected here.
/// Models must be sendable to an isolate (custom response and coupling
/// functions are, unless they capture something that isn't, like a port).
///
/// On the web, [spawn] gives a pool with no workers that runs everything
/// on the calling isolate.
final class FitWorkers {
  FitWorkers._(this._ports, this._replies, this._exits, this.minSliceTrials)
    : count = _ports.length,
      _busy = List.filled(_ports.length, false) {
    _replies.handler = _onReply;
    _exits.handler = _onExit;
  }

  /// Spawns [count] worker isolates (default: one per processor). With
  /// [count] 0 nothing is spawned and passes run on the calling isolate.
  static Future<FitWorkers> spawn({
    int? count,
    int minSliceTrials = defaultMinSliceTrials,
  }) async {
    final n = count ?? Platform.numberOfProcessors;
    if (n < 0) throw ArgumentError.value(count, 'count');
    final replies = RawReceivePort(null, 'FitWorkers replies');
    final exits = RawReceivePort(null, 'FitWorkers exits');
    final ports = await Future.wait([
      for (var i = 0; i < n; i++) _spawnOne(i, replies.sendPort, exits),
    ]);
    return FitWorkers._(ports, replies, exits, minSliceTrials);
  }

  /// Default for [minSliceTrials].
  static const defaultMinSliceTrials = 2048;

  static Future<SendPort> _spawnOne(
    int index,
    SendPort replies,
    RawReceivePort exits,
  ) async {
    final hello = ReceivePort();
    await Isolate.spawn(
      _workerMain,
      (hello.sendPort, replies),
      debugName: 'hgf_fit worker $index',
      onExit: exits.sendPort,
      onError: exits.sendPort,
    );
    final port = await hello.first as SendPort;
    return port;
  }

  /// Number of worker isolates.
  final int count;

  /// The smallest slice of a pass, in filtered trials, worth sending to
  /// a worker.
  int minSliceTrials;

  /// How many streams a driver should split independent tasks into: one
  /// per worker.
  int get streams => count == 0 ? 1 : count;

  final List<SendPort> _ports;
  final RawReceivePort _replies;
  final RawReceivePort _exits;
  bool _closed = false;

  int _nextId = 0;
  final _modelIds = Expando<int>('model');
  final _familyIds = Expando<int>('family');
  final _subjectIds = Expando<int>('subject');
  late final _finalizer = Finalizer<int>(_drop);

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

  /// Runs [program], answering each pass on this pool.
  Future<T> run<T>(Program<T> program) async {
    if (_closed) throw StateError('FitWorkers closed');
    final result = Out<T>();
    _active++;
    try {
      await _drive(program(result));
    } finally {
      _active--;
    }
    return result.value;
  }

  Future<void> _drive(Iterable<Step> steps) async {
    for (final step in steps) {
      switch (step) {
        case Pass():
          await _submit(step);
        case Concurrent(:final streams):
          _active--;
          try {
            await Future.wait([for (final s in streams) _stream(s)]);
          } finally {
            _active++;
          }
      }
    }
  }

  Future<void> _stream(Iterable<Step> steps) async {
    _active++;
    try {
      await _drive(steps);
    } finally {
      _active--;
    }
  }

  /// Stops the workers. Passes still running fail.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    for (final p in _ports) {
      p.send(null);
    }
    _failAll(StateError('FitWorkers closed'));
    _replies.close();
    _exits.close();
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
      // E.g. a model holding something that can't be sent to an isolate.
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
    _ports[worker].send([
      _evaluate,
      id,
      worker,
      p.modelId,
      Int32List.fromList(ids),
      subjectOf,
      pass.y.sublist(start * k, end * k),
      pass.jacobian,
      pass.includePrior,
      pass.negLogLikelihood != null,
      pass.gradient != null,
    ]);
    _sent[id] = slice;
    _busy[worker] = true;
    workerSlices++;
  }

  int _register(HgfModel model) {
    final known = _modelIds[model];
    if (known != null) return known;
    final family = modelFamily(model);
    var familyId = _familyIds[family];
    if (familyId == null) {
      familyId = _nextId++;
      _broadcast([
        _registerFamily,
        familyId,
        model.network,
        model.response,
        model.parameters,
        model.blockSize,
      ]);
      _familyIds[family] = familyId;
      _finalizer.attach(family, familyId);
    }
    final id = _nextId++;
    _broadcast([_registerModel, id, familyId, model.parameters]);
    _modelIds[model] = id;
    _finalizer.attach(model, id);
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
      final id = _subjectIds[subject];
      if (id == null) {
        newIds.add(_nextId++);
        newSubjects.add(subject);
        ids[s] = newIds.last;
      } else {
        ids[s] = id;
      }
    }
    if (newIds.isNotEmpty) {
      _broadcast([_registerSubjectsTag, newIds, newSubjects]);
      for (var i = 0; i < newIds.length; i++) {
        _subjectIds[newSubjects[i]] = newIds[i];
        _finalizer.attach(newSubjects[i], newIds[i]);
      }
    }
    return ids;
  }

  void _drop(int id) {
    if (!_closed) _broadcast([_dropTag, id]);
  }

  void _broadcast(List<Object?> message) {
    for (final p in _ports) {
      p.send(message);
    }
  }

  void _onReply(Object? message) {
    final m = message as List<Object?>;
    final slice = _sent.remove(m[0] as int);
    _busy[m[1] as int] = false;
    if (slice != null) {
      final p = slice.pending;
      if (m[2] is String) {
        p.fail(RemoteError(m[2] as String, m[3] as String), StackTrace.empty);
      } else {
        final b = p.bounds, r = slice.range;
        p.pass.absorb(
          b[r],
          b[r + 1],
          m[2] as Float64List,
          m[3] as Float64List?,
          m[4] as Float64List?,
        );
      }
      p.arrived();
    }
    _pump();
  }

  void _onExit(Object? message) {
    if (_closed) return;
    // A worker died (an uncaught error, or it was killed): the pool can't
    // answer passes any more.
    _closed = true;
    _failAll(
      StateError(
        'A FitWorkers isolate exited${message == null ? '' : ': $message'}',
      ),
    );
    for (final p in _ports) {
      p.send(null);
    }
    _replies.close();
    _exits.close();
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

const _registerFamily = 0;
const _registerModel = 1;
const _registerSubjectsTag = 2;
const _dropTag = 3;
const _evaluate = 4;

void _workerMain((SendPort, SendPort) ports) {
  final (hello, replies) = ports;
  final families = <int, HgfModel>{};
  final models = <int, HgfModel>{};
  final subjects = <int, Subject>{};
  final port = RawReceivePort(null, 'hgf_fit worker');
  port.handler = (Object? message) {
    if (message == null) {
      port.close();
      return;
    }
    final m = message as List<Object?>;
    switch (m[0]) {
      case _registerFamily:
        families[m[1] as int] = HgfModel(
          network: m[2] as Network,
          response: m[3] as ResponseModel,
          parameters: (m[4] as List).cast<Parameter>(),
          blockSize: m[5] as int,
        );
      case _registerModel:
        models[m[1] as int] = modelWithParameters(
          families[m[2] as int]!,
          (m[3] as List).cast<Parameter>(),
        );
      case _registerSubjectsTag:
        final ids = (m[1] as List).cast<int>(), list = m[2] as List;
        for (var i = 0; i < ids.length; i++) {
          subjects[ids[i]] = list[i] as Subject;
        }
      case _dropTag:
        final id = m[1] as int;
        families.remove(id);
        models.remove(id);
        subjects.remove(id);
      case _evaluate:
        final passId = m[1] as int, range = m[2] as int;
        try {
          final ids = m[4] as Int32List;
          final pass = Pass(
            models[m[3] as int]!,
            [for (final id in ids) subjects[id]!],
            m[5] as Int32List,
            m[6] as Float64List,
            jacobian: m[7] as bool,
            includePrior: (m[8] as List?)?.cast<bool>(),
            negLogLikelihood: m[9] as bool,
            gradient: m[10] as bool,
          )..run();
          replies.send([
            passId,
            range,
            pass.out,
            pass.negLogLikelihood,
            pass.gradient,
          ]);
        } catch (e, st) {
          replies.send([passId, range, e.toString(), st.toString()]);
        }
    }
  };
  hello.send(port.sendPort);
}
