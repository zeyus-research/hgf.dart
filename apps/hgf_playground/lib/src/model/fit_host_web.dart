import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';

import 'fit_engine.dart';
import 'fit_jobs.dart';
import 'fit_wire.dart';
import 'web/bindings.dart';

/// Fits in the fit host worker (`worker/fit_host.dart`, built into `web/`
/// as `fit_host.wasm` / `fit_host.js` by `tool/build_worker.sh`): the
/// drivers, the pool (nested Web Workers) and WebGPU all off the page's
/// thread. If the worker doesn't start, jobs run on the page's thread as
/// before (a [LocalFitEngine]).
FitEngine? spawnFitHost() => _WebHostEngine();

const _isWasm = bool.fromEnvironment('dart.tool.dart2wasm');

/// Starts the host on first use, then forwards to it (or to the fallback).
final class _WebHostEngine implements FitEngine {
  Future<FitEngine>? _engine;
  FitEngine? _started;
  bool _closed = false;

  Future<FitEngine> _start() => _engine ??= () async {
    FitEngine e;
    try {
      e = await _WorkerEngine.spawn();
    } catch (error) {
      // ignore: avoid_print
      print('No fit host worker ($error); fitting from the page.');
      e = LocalFitEngine();
    }
    if (_closed) {
      await e.close();
      throw StateError('FitEngine closed');
    }
    return _started = e;
  }();

  @override
  Future<GpuInfo> probeGpu() async => (await _start()).probeGpu();

  @override
  Future<R> run<R>(FitJobSpec<R> job, void Function(JobEvent) onEvent) async {
    if (_closed) throw StateError('FitEngine closed');
    return (await _start()).run(job, onEvent);
  }

  // Not started yet: nothing runs.
  @override
  void cancel() => _started?.cancel();

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _started?.close();
  }
}

/// The page's side of the host worker: requests out, replies in.
final class _WorkerEngine implements FitEngine {
  _WorkerEngine._(this._worker) {
    _worker
      ..onmessage = _onMessage.toJS
      ..onerror = (ErrorEvent e) {
        e.preventDefault();
        _failAll('the fit host worker failed: ${e.message.dartify()}');
      }.toJS;
  }

  static Future<_WorkerEngine> spawn() async {
    final base = baseUrl;
    if (base == null || globalScope.workerConstructor == null) {
      throw StateError('no Web Workers here');
    }
    final url = Uri.parse(
      base,
    ).resolve(_isWasm ? 'fit_host.wasm' : 'fit_host.js').toString();
    String? bootstrap;
    if (_isWasm) {
      bootstrap = URL.createObjectURL(
        Blob(
          [
            _wasmBootstrap(
              url.replaceFirst(RegExp(r'\.wasm$'), '.mjs'),
              url,
            ).toJS,
          ].toJS,
          BlobOptions(type: 'text/javascript'),
        ),
      );
    }
    final worker = JSWorker(
      bootstrap ?? url,
      WorkerOptions(type: _isWasm ? 'module' : 'classic', name: 'fit host'),
    );
    try {
      await _ready(worker, url);
    } catch (_) {
      worker.terminate();
      rethrow;
    } finally {
      if (bootstrap != null) URL.revokeObjectURL(bootstrap);
    }
    // Its pool's scripts resolve against the page, not its blob: URL.
    worker.postMessage(['init'.toJS, base.toJS].toJS);
    return _WorkerEngine._(worker);
  }

  final JSWorker _worker;
  int _next = 0;
  final _pending = <int, _Pending<Object?>>{};
  bool _closed = false;

  Future<T> _request<T>(
    _Pending<T> pending,
    JSArray<JSAny?> Function(int id) message,
  ) {
    if (_closed) throw StateError('FitEngine closed');
    final id = _next++;
    _pending[id] = pending;
    _worker.postMessage(message(id));
    return pending.done.future;
  }

  @override
  Future<GpuInfo> probeGpu() => _request(
    _Pending<GpuInfo>((json, _) => decodeGpuInfo(json)),
    (id) => ['probe'.toJS, id.toJS].toJS,
  );

  @override
  Future<R> run<R>(FitJobSpec<R> job, void Function(JobEvent) onEvent) {
    // Encode first: an unserialisable model throws before anything is sent.
    final m = encodeJob(job);
    return _request(
      _Pending<R>((json, arrays) => decodeResult(job, json, arrays), onEvent),
      (id) => ['run'.toJS, id.toJS, m.json.toJS, arraysToJS(m.arrays)].toJS,
    );
  }

  @override
  void cancel() {
    if (!_closed) _worker.postMessage(['cancel'.toJS].toJS);
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    // The worker closes its pool's workers (nested workers stop with their
    // parent too) and the GPU device, then itself.
    _worker.postMessage(['close'.toJS].toJS);
    _failAll('FitEngine closed');
  }

  void _failAll(String why) {
    final pending = [..._pending.values];
    _pending.clear();
    for (final p in pending) {
      p.done.completeError(StateError(why));
    }
  }

  void _onMessage(MessageEvent e) {
    final m = (e.data! as JSArray<JSAny?>).toDart;
    final id = integer(m[1]);
    final p = _pending[id];
    if (p == null) return;
    switch (str(m[0])) {
      case 'event':
        p.onEvent?.call(decodeEvent(str(m[2])));
      case 'done':
        _pending.remove(id);
        try {
          p.complete(str(m[2]), arraysFromJS(m[3]));
        } catch (error, st) {
          p.done.completeError(error, st);
        }
      case 'cancelled':
        _pending.remove(id);
        p.done.completeError(const FitCancelled());
      case 'error':
        _pending.remove(id);
        p.done.completeError(
          _HostError(str(m[2])),
          StackTrace.fromString(str(m[3])),
        );
    }
  }
}

final class _Pending<T> {
  _Pending(this._decode, [this.onEvent]);

  final T Function(String json, List<Float64List> arrays) _decode;
  final void Function(JobEvent)? onEvent;
  final done = Completer<T>();

  void complete(String json, List<Float64List> arrays) =>
      done.complete(_decode(json, arrays));
}

/// An error thrown in the host, as text.
final class _HostError implements Exception {
  _HostError(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Waits for the worker's ready message (see `runFitHostWorker`).
Future<void> _ready(JSWorker worker, String url) {
  final done = Completer<void>();
  worker
    ..onmessage = (MessageEvent e) {
      final data = e.data;
      if (done.isCompleted || data == null || !data.isA<JSString>()) return;
      final s = str(data);
      if (s == fitHostReady) {
        done.complete();
      } else if (s.startsWith(fitHostFailed)) {
        done.completeError(
          StateError(
            '$url failed to start: ${s.substring(fitHostFailed.length)}',
          ),
        );
      }
    }.toJS
    ..onerror = (ErrorEvent e) {
      e.preventDefault();
      if (!done.isCompleted) {
        done.completeError(
          StateError('$url failed to load: ${e.message.dartify()}'),
        );
      }
    }.toJS;
  return done.future;
}

/// A module worker that loads a `dart compile wasm` output and runs its
/// `main`.
String _wasmBootstrap(String mjs, String wasm) =>
    '''
try {
  const loader = await import(${jsonEncode(mjs)});
  let compiled;
  try {
    compiled = await loader.compileStreaming(fetch(${jsonEncode(wasm)}));
  } catch (_) {
    const response = await fetch(${jsonEncode(wasm)});
    if (!response.ok) throw new Error(`\${response.status} for ${jsonEncode(wasm)}`);
    compiled = await loader.compile(await response.arrayBuffer());
  }
  const app = await compiled.instantiate({});
  app.invokeMain();
} catch (e) {
  postMessage(${jsonEncode(fitHostFailed)} + String(e && e.stack || e));
}
''';
