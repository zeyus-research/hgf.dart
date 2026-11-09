import 'worker_transport.dart';

/// Logical processors: unknown here, so 1.
int get processorCount => 1;

/// No workers on this platform.
Future<WorkerTransport> spawnTransport(
  int? count,
  FitWorkerScript? script,
) async => NoWorkers();

/// A transport without workers: `FitWorkers` runs every pass on the
/// calling isolate. Package-internal.
final class NoWorkers extends WorkerTransport {
  @override
  int get count => 0;

  Never _none() => throw StateError('No workers');

  @override
  void registerFamily(int id, _) => _none();

  @override
  void registerModel(int id, int familyId, _) => _none();

  @override
  void registerSubjects(_, _) => _none();

  @override
  void drop(int id) {}

  @override
  void evaluate(int worker, _) => _none();

  @override
  void close() {}
}
