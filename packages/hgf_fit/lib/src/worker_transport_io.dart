import 'dart:async';
import 'dart:io' show Platform;
import 'dart:isolate';
import 'dart:typed_data';

import 'package:hgf/hgf.dart' show Network;

import 'model.dart';
import 'parameter.dart';
import 'response_model.dart';
import 'worker_host.dart';
import 'worker_transport.dart';
import 'worker_transport_stub.dart';

/// Logical processors of this machine.
int get processorCount => Platform.numberOfProcessors;

/// [count] isolates (default: one per processor); [script] is for the web.
Future<WorkerTransport> spawnTransport(
  int? count,
  FitWorkerScript? script,
) async {
  final n = count ?? processorCount;
  if (n == 0) return NoWorkers();
  final replies = RawReceivePort(null, 'FitWorkers replies');
  final exits = RawReceivePort(null, 'FitWorkers exits');
  try {
    final ports = await Future.wait([
      for (var i = 0; i < n; i++) _spawnOne(i, replies.sendPort, exits),
    ]);
    return _IsolateTransport(ports, replies, exits);
  } catch (_) {
    replies.close();
    exits.close();
    rethrow;
  }
}

Future<SendPort> _spawnOne(
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
  return await hello.first as SendPort;
}

/// Worker isolates; models and subjects are sent as Dart objects (copied
/// by the VM, closures included).
final class _IsolateTransport extends WorkerTransport {
  _IsolateTransport(this._ports, this._replies, this._exits) {
    _replies.handler = _onReply;
    _exits.handler = (Object? message) => onExit(message);
  }

  final List<SendPort> _ports;
  final RawReceivePort _replies;
  final RawReceivePort _exits;
  bool _closed = false;

  @override
  int get count => _ports.length;

  void _broadcast(List<Object?> message) {
    for (final p in _ports) {
      p.send(message);
    }
  }

  @override
  void registerFamily(int id, HgfModel model) => _broadcast([
    _registerFamily,
    id,
    model.network,
    model.response,
    model.parameters,
    model.blockSize,
  ]);

  @override
  void registerModel(int id, int familyId, List<Parameter> parameters) =>
      _broadcast([_registerModel, id, familyId, parameters]);

  @override
  void registerSubjects(List<int> ids, List<Subject> subjects) =>
      _broadcast([_registerSubjects, ids, subjects]);

  @override
  void drop(int id) => _broadcast([_drop, id]);

  @override
  void evaluate(int worker, EvaluateRequest r) => _ports[worker].send([
    _evaluate,
    r.id,
    worker,
    r.modelId,
    r.subjectIds,
    r.subjectOf,
    r.y,
    r.jacobian,
    r.includePrior,
    r.negLogLikelihood,
    r.gradient,
  ]);

  @override
  void close() {
    if (_closed) return;
    _closed = true;
    for (final p in _ports) {
      p.send(null);
    }
    _replies.close();
    _exits.close();
  }

  void _onReply(Object? message) {
    final m = message as List<Object?>;
    final id = m[0] as int, worker = m[1] as int;
    if (m[2] is String) {
      onFailure(id, worker, m[2] as String, m[3] as String);
    } else {
      onResult(
        id,
        worker,
        m[2] as Float64List,
        m[3] as Float64List?,
        m[4] as Float64List?,
      );
    }
  }
}

const _registerFamily = 0;
const _registerModel = 1;
const _registerSubjects = 2;
const _drop = 3;
const _evaluate = 4;

void _workerMain((SendPort, SendPort) ports) {
  final (hello, replies) = ports;
  final host = WorkerHost();
  final port = RawReceivePort(null, 'hgf_fit worker');
  port.handler = (Object? message) {
    if (message == null) {
      port.close();
      return;
    }
    final m = message as List<Object?>;
    switch (m[0]) {
      case _registerFamily:
        host.addFamily(
          m[1] as int,
          () => HgfModel(
            network: m[2] as Network,
            response: m[3] as ResponseModel,
            parameters: (m[4] as List).cast<Parameter>(),
            blockSize: m[5] as int,
          ),
        );
      case _registerModel:
        host.addModel(
          m[1] as int,
          m[2] as int,
          () => (m[3] as List).cast<Parameter>(),
        );
      case _registerSubjects:
        host.addSubjects(
          (m[1] as List).cast<int>(),
          (m[2] as List).cast<Subject>(),
        );
      case _drop:
        host.drop(m[1] as int);
      case _evaluate:
        final id = m[1] as int, worker = m[2] as int;
        try {
          final pass = host.evaluate(
            modelId: m[3] as int,
            subjectIds: m[4] as Int32List,
            subjectOf: m[5] as Int32List,
            y: m[6] as Float64List,
            jacobian: m[7] as bool,
            includePrior: (m[8] as List?)?.cast<bool>(),
            negLogLikelihood: m[9] as bool,
            gradient: m[10] as bool,
          );
          replies.send([
            id,
            worker,
            pass.out,
            pass.negLogLikelihood,
            pass.gradient,
          ]);
        } catch (e, st) {
          replies.send([id, worker, e.toString(), st.toString()]);
        }
    }
  };
  hello.send(port.sendPort);
}
