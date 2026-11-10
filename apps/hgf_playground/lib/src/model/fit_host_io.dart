import 'dart:async';
import 'dart:isolate';

import 'package:hgf_fit/hgf_fit.dart';

import 'fit_engine.dart';
import 'fit_jobs.dart';

FitEngine? spawnFitHost() => _HostEngine();

/// The UI side: forwards jobs to the host isolate (spawned on first use)
/// and completes them from its replies.
final class _HostEngine implements FitEngine {
  Future<SendPort>? _host;
  RawReceivePort? _replies;
  Isolate? _isolate;
  int _next = 0;
  final _pending = <int, (Completer<Object?>, void Function(JobEvent)?)>{};
  bool _closed = false;
  final _exited = Completer<void>();

  Future<SendPort> _start() => _host ??= () async {
    final ready = Completer<SendPort>();
    final replies = _replies = RawReceivePort(null, 'fit host replies');
    replies.handler = (Object? m) {
      switch (m) {
        case SendPort port:
          ready.complete(port);
        case ('event', int id, JobEvent e):
          _pending[id]?.$2?.call(e);
        case ('done', int id, Object? result):
          _pending.remove(id)?.$1.complete(result);
        case ('cancelled', int id):
          _pending.remove(id)?.$1.completeError(const FitCancelled());
        case ('error', int id, String error, String stack):
          _pending
              .remove(id)
              ?.$1
              .completeError(_HostError(error), StackTrace.fromString(stack));
        case null:
          // onExit.
          _exited.complete();
          _failAll('the fit host stopped');
        case [final error, final stack]:
          // An uncaught error: the host goes on, but its job is lost.
          _failAll('the fit host failed: $error\n$stack');
      }
    };
    _isolate = await Isolate.spawn(
      _hostMain,
      replies.sendPort,
      debugName: 'fit host',
      onExit: replies.sendPort,
      onError: replies.sendPort,
    );
    return ready.future;
  }();

  void _failAll(String why) {
    for (final p in _pending.values) {
      p.$1.completeError(StateError(why));
    }
    _pending.clear();
  }

  Future<T> _request<T>(
    Object? Function(int id) message, [
    void Function(JobEvent)? onEvent,
  ]) async {
    if (_closed) throw StateError('FitEngine closed');
    final port = await _start();
    final id = _next++;
    final done = Completer<Object?>();
    _pending[id] = (done, onEvent);
    port.send(message(id));
    return (await done.future) as T;
  }

  @override
  Future<GpuInfo> probeGpu() => _request((id) => ('probe', id));

  @override
  Future<R> run<R>(FitJobSpec<R> job, void Function(JobEvent) onEvent) =>
      _request((id) => ('run', id, job), onEvent);

  @override
  void cancel() {
    // Not started: nothing runs.
    _host?.then((port) => port.send(('cancel',)));
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    final host = _host;
    if (host == null) return;
    (await host).send(('close',));
    await _exited.future.timeout(
      const Duration(seconds: 5),
      onTimeout: () => _isolate?.kill(),
    );
    _replies?.close();
    _failAll('FitEngine closed');
  }
}

/// An error from the host, as text.
final class _HostError implements Exception {
  _HostError(this.message);
  final String message;
  @override
  String toString() => message;
}

/// The host isolate: a [LocalFitEngine] (its own pool and GPU) answering
/// the UI's requests.
void _hostMain(SendPort ui) {
  final engine = LocalFitEngine();
  final requests = RawReceivePort(null, 'fit host requests');
  requests.handler = (Object? m) async {
    switch (m) {
      case ('run', int id, FitJobSpec<Object?> job):
        try {
          final result = await engine.run(
            job,
            (e) => ui.send(('event', id, e)),
          );
          ui.send(('done', id, result));
        } on FitCancelled {
          ui.send(('cancelled', id));
        } catch (e, st) {
          ui.send(('error', id, '$e', '$st'));
        }
      case ('probe', int id):
        try {
          ui.send(('done', id, await engine.probeGpu()));
        } catch (e, st) {
          ui.send(('error', id, '$e', '$st'));
        }
      case ('cancel',):
        engine.cancel();
      case ('close',):
        await engine.close();
        requests.close();
        Isolate.exit();
    }
  };
  ui.send(requests.sendPort);
}
