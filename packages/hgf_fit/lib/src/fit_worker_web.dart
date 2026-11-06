import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

import 'codec.dart';
import 'model.dart';
import 'web_interop.dart';
import 'worker_host.dart';

/// Runs this program as a `FitWorkers` Web Worker: answers the slices of
/// passes the page sends. Call it from the worker's `main`.
///
/// [codec] decodes the models; it needs the decoders, couplings and
/// response functions of any custom types the models use (the page's
/// `FitWorkerScript.codec` encodes them).
void runFitWorker({FitCodec codec = const FitCodec()}) {
  final scope = globalScope;
  final host = WorkerHost();
  scope.onmessage = (MessageEvent e) {
    final m = (e.data as JSArray<JSAny?>).toDart;
    switch (intFromJS(m[0])) {
      case tagFamily:
        final json = stringFromJS(m[2]);
        host.addFamily(
          intFromJS(m[1]),
          () => codec.decodeModel(jsonDecode(json) as Map<String, Object?>),
        );
      case tagModel:
        final json = stringFromJS(m[3]);
        host.addModel(
          intFromJS(m[1]),
          intFromJS(m[2]),
          () => codec.decodeParameters(jsonDecode(json) as List<Object?>),
        );
      case tagSubjects:
        final ids = i32FromJS(m[1]);
        final arrays = [
          for (final a in (m[2] as JSArray<JSAny?>).toDart) f64FromJS(a),
        ];
        final refs = i32FromJS(m[3]);
        Float64List? at(int i) => i < 0 ? null : arrays[i];
        host.addSubjects(ids, [
          for (var s = 0; s < ids.length; s++)
            Subject(
              inputs: at(refs[s * 4])!,
              responses: at(refs[s * 4 + 1]),
              observed: at(refs[s * 4 + 2]),
              timeSteps: at(refs[s * 4 + 3]),
            ),
        ]);
      case tagDrop:
        host.drop(intFromJS(m[1]));
      case tagEvaluate:
        final id = m[1], worker = m[2];
        try {
          final includePrior = m[8];
          final pass = host.evaluate(
            modelId: intFromJS(m[3]),
            subjectIds: i32FromJS(m[4]),
            subjectOf: i32FromJS(m[5]),
            y: f64FromJS(m[6]),
            jacobian: boolFromJS(m[7]),
            includePrior: includePrior == null
                ? null
                : [
                    for (final b in (includePrior as JSArray<JSAny?>).toDart)
                      boolFromJS(b),
                  ],
            negLogLikelihood: boolFromJS(m[9]),
            gradient: boolFromJS(m[10]),
          );
          scope.postMessage(
            [
              id,
              worker,
              f64ToJS(pass.out),
              pass.negLogLikelihood == null
                  ? null
                  : f64ToJS(pass.negLogLikelihood!),
              pass.gradient == null ? null : f64ToJS(pass.gradient!),
            ].toJS,
          );
        } catch (e, st) {
          scope.postMessage(
            [id, worker, e.toString().toJS, st.toString().toJS].toJS,
          );
        }
    }
  }.toJS;
  scope.postMessage(readyMessage.toJS);
}
