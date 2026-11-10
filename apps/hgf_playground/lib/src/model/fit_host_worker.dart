import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';

import 'fit_engine.dart';
import 'fit_wire.dart';
import 'web/bindings.dart';

/// Runs this program as the playground's fit host worker (see
/// `worker/fit_host.dart` and `fit_host_web.dart`): a [LocalFitEngine]
/// whose pool is nested Web Workers and whose GPU is this worker's WebGPU,
/// answering the page's requests. Only stages, progress and results go
/// back to the page.
void runFitHostWorker() {
  final scope = globalScope;
  String? base;
  LocalFitEngine? engine;
  LocalFitEngine local() => engine ??= LocalFitEngine(
    spawnWorkers: (count) => spawnPlaygroundWorkers(count, base: base),
  );

  void reply(List<JSAny?> m) => scope.postMessage(m.toJS);

  Future<void> answer(int id, Future<Packed> Function() work) async {
    try {
      final r = await work();
      reply(['done'.toJS, id.toJS, r.json.toJS, arraysToJS(r.arrays)]);
    } on FitCancelled {
      reply(['cancelled'.toJS, id.toJS]);
    } catch (e, st) {
      reply(['error'.toJS, id.toJS, '$e'.toJS, '$st'.toJS]);
    }
  }

  scope.onmessage = (MessageEvent e) {
    final m = (e.data! as JSArray<JSAny?>).toDart;
    switch (str(m[0])) {
      case 'init':
        base = str(m[1]);
      case 'probe':
        unawaited(
          answer(integer(m[1]), () async {
            final g = await local().probeGpu();
            return (json: encodeGpuInfo(g), arrays: const <Float64List>[]);
          }),
        );
      case 'run':
        final id = integer(m[1]);
        unawaited(
          answer(id, () async {
            final job = decodeJob(str(m[2]), arraysFromJS(m[3]));
            final result = await local().run(
              job,
              (event) =>
                  reply(['event'.toJS, id.toJS, encodeEvent(event).toJS]),
            );
            return encodeResult(result);
          }),
        );
      case 'cancel':
        engine?.cancel();
      case 'close':
        unawaited(() async {
          await engine?.close();
          scope.close();
        }());
    }
  }.toJS;
  scope.postMessage(fitHostReady.toJS);
}
