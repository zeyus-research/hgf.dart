import 'package:flutter/foundation.dart';
import 'package:hgf_fit/hgf_fit.dart';

import 'fit_config.dart';
import 'fit_jobs.dart';
import 'gpu/open_gpu.dart';

/// The GPU as the settings show it: its adapter, or why there is none.
typedef GpuInfo = ({String? adapter, String? reason});

/// Runs fit jobs: on this isolate ([LocalFitEngine]), or in a host isolate
/// that owns the pool and the GPU, so that only progress and results reach
/// the UI (`spawnFitHost`, desktop).
abstract interface class FitEngine {
  /// Opens the GPU if that hasn't been tried yet.
  Future<GpuInfo> probeGpu();

  /// Runs [job], passing its events to [onEvent]; throws [FitCancelled] if
  /// [cancel] stopped it.
  Future<R> run<R>(FitJobSpec<R> job, void Function(JobEvent) onEvent);

  /// Stops the running job.
  void cancel();

  Future<void> close();
}

/// Jobs on this isolate: a [FitWorkers] pool spawned on first use
/// (isolates on desktop, Web Workers on the web, or none: then passes run
/// here, in slices between frames) and the GPU, opened on first use.
class LocalFitEngine implements FitEngine {
  LocalFitEngine({
    Future<FitWorkers> Function(int? count)? spawnWorkers,
    Future<GpuOpened> Function()? openGpu,
  }) : _spawnWorkers = spawnWorkers ?? spawnPlaygroundWorkers,
       _openGpu = openGpu ?? _openPlatformGpu;

  final Future<FitWorkers> Function(int? count) _spawnWorkers;
  final Future<GpuOpened> Function() _openGpu;

  FitWorkers? _workers;
  int? _workersAsked;
  String _where = '';
  Future<FitWorkers>? _spawning;

  Future<GpuOpened>? _gpuOpening;
  GpuRunner? _gpuRunner;

  // Bumped by [cancel]: a job stops at its next check once it differs from
  // the one it started with.
  int _generation = 0;
  bool _closed = false;

  @override
  Future<GpuInfo> probeGpu() async {
    final g = await (_gpuOpening ??= _openGpu());
    return (adapter: g.device?.description, reason: g.reason);
  }

  @override
  Future<R> run<R>(FitJobSpec<R> job, void Function(JobEvent) onEvent) {
    if (_closed) throw StateError('FitEngine closed');
    return job.execute(_LocalContext(this, _generation, onEvent));
  }

  @override
  void cancel() {
    _generation++;
    _workers?.cancel();
    _gpuRunner?.cancel();
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    cancel();
    final gpu = _gpuOpening;
    if (gpu != null) (await gpu).device?.close();
    await (await _spawning)?.close();
  }

  /// The worker pool, (re)spawned when [FitConfig.workers] changed.
  Future<FitWorkers> _pool(FitConfig config) async {
    final asked = config.workers == 0 ? null : config.workers;
    final current = await _spawning;
    if (current != null && asked == _workersAsked) return current;
    final spawning = _spawning = () async {
      if (current != null) {
        _workers = null;
        _gpuRunner = null;
        await current.close();
      }
      return _spawnWorkers(asked);
    }();
    _workersAsked = asked;
    final pool = _workers = await spawning;
    _where = pool.count == 0
        ? 'the main thread'
        : '${pool.count} ${kIsWeb ? (kIsWasm ? 'Wasm workers' : 'JS workers') : 'isolates'}';
    return pool;
  }
}

final class _LocalContext implements JobContext {
  _LocalContext(this.engine, this.generation, this.onEvent);

  final LocalFitEngine engine;
  final int generation;
  final void Function(JobEvent) onEvent;

  @override
  void check() {
    if (generation != engine._generation) throw const FitCancelled();
  }

  @override
  void report(JobEvent event) {
    if (generation == engine._generation) onEvent(event);
  }

  @override
  Future<JobRunner> runner(FitConfig config, {required bool gpu}) async {
    report(const JobStage('Starting workers…'));
    final pool = await engine._pool(config);
    check();
    final where = engine._where;
    if (!gpu) return JobRunner(pool, where, where);
    report(const JobStage('Opening the GPU…'));
    final g = await (engine._gpuOpening ??= engine._openGpu());
    check();
    final device = g.device;
    if (device == null) {
      return JobRunner(pool, where, where)
        ..deviceNote = config.device == FitDevice.gpu
            ? 'No GPU (${g.reason}): ran on the CPU.'
            : null;
    }
    final f = engine._gpuRunner = device.runner(pool);
    return JobRunner(f, 'the GPU', where, gpu: f);
  }
}

Future<GpuOpened> _openPlatformGpu() => openGpu();

/// The pool the playground fits on (default: one worker per logical
/// processor): Web Workers running the compiled `fit_worker` next to the
/// page (Wasm or JS, like the page), isolates on desktop. If the workers
/// can't start (e.g. the worker wasn't built), the fits run on the calling
/// thread, a slice per frame.
Future<FitWorkers> spawnPlaygroundWorkers(int? count) async {
  try {
    return await FitWorkers.spawn(
      count: count,
      // A single subject's passes are small: split none of them.
      minSliceTrials: 1 << 30,
      script: kIsWasm
          ? const FitWorkerScript.wasm('fit_worker.wasm')
          : const FitWorkerScript.js('fit_worker.js'),
    );
  } catch (e) {
    debugPrint('No fit workers ($e); fitting on the main thread.');
    return await FitWorkers.spawn(count: 0)
      ..yieldInterval = const Duration(milliseconds: 12);
  }
}
