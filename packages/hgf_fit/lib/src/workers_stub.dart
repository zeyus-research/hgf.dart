import 'pass.dart';

/// A pool of worker isolates that the fitting functions split their
/// batched passes across (see the `dart:io` platforms' documentation).
///
/// This platform has no isolates: [spawn] gives a pool with no workers,
/// which runs every pass on the calling isolate. Code written against
/// `FitWorkers` works unchanged.
final class FitWorkers {
  FitWorkers._(this.minSliceTrials);

  /// A pool without workers; [count] is ignored.
  static Future<FitWorkers> spawn({
    int? count,
    int minSliceTrials = defaultMinSliceTrials,
  }) async => FitWorkers._(minSliceTrials);

  static const defaultMinSliceTrials = 2048;

  /// Number of worker isolates: always 0 here.
  int get count => 0;

  /// Unused here.
  int minSliceTrials;

  /// How many streams a driver should split independent tasks into.
  int get streams => 1;

  /// Slices evaluated by workers so far: always 0 here.
  int get workerSlices => 0;

  /// Passes evaluated on the calling isolate so far.
  int localSlices = 0;

  bool _closed = false;

  /// Runs [program], answering each pass on the calling isolate.
  Future<T> run<T>(Program<T> program) async {
    if (_closed) throw StateError('FitWorkers closed');
    final result = Out<T>();
    _runSteps(program(result));
    return result.value;
  }

  void _runSteps(Iterable<Step> steps) {
    for (final step in steps) {
      switch (step) {
        case Pass():
          localSlices++;
          step.run();
        case Concurrent(:final streams):
          for (final s in streams) {
            _runSteps(s);
          }
      }
    }
  }

  Future<void> close() async => _closed = true;
}
