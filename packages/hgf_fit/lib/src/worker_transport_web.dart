import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

import 'codec.dart';
import 'model.dart';
import 'parameter.dart';
import 'web_interop.dart';
import 'worker_transport.dart';
import 'worker_transport_stub.dart';

/// Logical processors (`navigator.hardwareConcurrency`, else 4).
int get processorCount => globalScope.navigator?.hardwareConcurrency ?? 4;

/// [count] Web Workers running [script] (default: one per logical
/// processor). Without a script, or where there are no Web Workers
/// (Node), there are no workers.
Future<WorkerTransport> spawnTransport(
  int? count,
  FitWorkerScript? script,
) async {
  final global = globalScope;
  if (script == null || global.workerConstructor == null) return NoWorkers();
  final n = count ?? processorCount;
  if (n == 0) return NoWorkers();
  final base = global.document?.baseURI ?? global.location?.href;
  final url = base == null ? script.url : URL(script.url, base).href;
  String? bootstrap;
  if (script.wasm) {
    final mjs = url.replaceFirstMapped(
      RegExp(r'\.wasm($|[?#])'),
      (m) => '.mjs${m[1]}',
    );
    bootstrap = URL.createObjectURL(
      Blob(
        [_wasmBootstrap(mjs, url).toJS].toJS,
        BlobOptions(type: 'text/javascript'),
      ),
    );
  }
  final workers = <JSWorker>[];
  try {
    for (var i = 0; i < n; i++) {
      workers.add(
        JSWorker(
          bootstrap ?? url,
          WorkerOptions(
            type: script.wasm ? 'module' : 'classic',
            name: 'hgf_fit worker $i',
          ),
        ),
      );
    }
    await Future.wait([for (final w in workers) _ready(w, script.url)]);
  } catch (_) {
    for (final w in workers) {
      w.terminate();
    }
    rethrow;
  } finally {
    if (bootstrap != null) URL.revokeObjectURL(bootstrap);
  }
  return _WebTransport(workers, script.codec);
}

/// A module worker that loads a `dart compile wasm` output and runs its
/// `main` (which calls `runFitWorker`).
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
  postMessage(${jsonEncode(failedPrefix)} + String(e && e.stack || e));
}
''';

/// Waits for [worker]'s ready message.
Future<void> _ready(JSWorker worker, String url) {
  final done = Completer<void>();
  worker
    ..onmessage = (MessageEvent e) {
      final data = e.data;
      if (done.isCompleted || data == null || !data.isA<JSString>()) return;
      final s = stringFromJS(data);
      if (s == readyMessage) {
        done.complete();
      } else if (s.startsWith(failedPrefix)) {
        done.completeError(
          StateError(
            'The FitWorkers script $url failed to start: '
            '${s.substring(failedPrefix.length)}',
          ),
        );
      }
    }.toJS
    ..onerror = (ErrorEvent e) {
      e.preventDefault();
      if (done.isCompleted) return;
      done.completeError(
        StateError(
          'The FitWorkers script $url failed to load'
          '${e.message == null ? '' : ': ${e.message.dartify()}'}',
        ),
      );
    }.toJS;
  return done.future;
}

/// Web Workers; models travel as `FitCodec` JSON, data as typed arrays.
final class _WebTransport extends WorkerTransport {
  _WebTransport(this._workers, this._codec) {
    for (final w in _workers) {
      w
        ..onmessage = _onMessage.toJS
        ..onerror = (ErrorEvent e) {
          e.preventDefault();
          onExit(e.message.dartify());
        }.toJS;
    }
  }

  final List<JSWorker> _workers;
  final FitCodec _codec;
  bool _closed = false;

  @override
  int get count => _workers.length;

  void _broadcast(JSAny message) {
    for (final w in _workers) {
      w.postMessage(message);
    }
  }

  @override
  void registerFamily(int id, HgfModel model) {
    // Encode first: an unserialisable model throws before anything is sent.
    final json = jsonEncode(_codec.encodeModel(model));
    _broadcast([tagFamily.toJS, id.toJS, json.toJS].toJS);
  }

  @override
  void registerModel(int id, int familyId, List<Parameter> parameters) {
    final json = jsonEncode(_codec.encodeParameters(parameters));
    _broadcast([tagModel.toJS, id.toJS, familyId.toJS, json.toJS].toJS);
  }

  @override
  void registerSubjects(List<int> ids, List<Subject> subjects) {
    // Each distinct series once, so shared series stay shared.
    final index = LinkedHashMap<Float64List, int>.identity();
    final arrays = <JSAny>[];
    int ref(Float64List? v) {
      if (v == null) return -1;
      return index.putIfAbsent(v, () {
        arrays.add(f64ToJS(v));
        return arrays.length - 1;
      });
    }

    final refs = Int32List(subjects.length * 4);
    for (var s = 0; s < subjects.length; s++) {
      final subject = subjects[s];
      refs[s * 4] = ref(subject.inputs);
      refs[s * 4 + 1] = ref(subject.responses);
      refs[s * 4 + 2] = ref(subject.observed);
      refs[s * 4 + 3] = ref(subject.timeSteps);
    }
    _broadcast(
      [
        tagSubjects.toJS,
        i32ToJS(Int32List.fromList(ids)),
        arrays.toJS,
        i32ToJS(refs),
      ].toJS,
    );
  }

  @override
  void drop(int id) => _broadcast([tagDrop.toJS, id.toJS].toJS);

  @override
  void evaluate(int worker, EvaluateRequest r) => _workers[worker].postMessage(
    [
      tagEvaluate.toJS,
      r.id.toJS,
      worker.toJS,
      r.modelId.toJS,
      i32ToJS(r.subjectIds),
      i32ToJS(r.subjectOf),
      f64ToJS(r.y),
      r.jacobian.toJS,
      r.includePrior == null
          ? null
          : [for (final b in r.includePrior!) b.toJS].toJS,
      r.negLogLikelihood.toJS,
      r.gradient.toJS,
    ].toJS,
  );

  @override
  void close() {
    if (_closed) return;
    _closed = true;
    for (final w in _workers) {
      w.terminate();
    }
  }

  void _onMessage(MessageEvent e) {
    if (_closed) return;
    final m = (e.data as JSArray<JSAny?>).toDart;
    final id = intFromJS(m[0]), worker = intFromJS(m[1]);
    if (m[2].isA<JSString>()) {
      onFailure(id, worker, stringFromJS(m[2]), stringFromJS(m[3]));
    } else {
      onResult(
        id,
        worker,
        f64FromJS(m[2]),
        f64OrNullFromJS(m[3]),
        f64OrNullFromJS(m[4]),
      );
    }
  }
}
