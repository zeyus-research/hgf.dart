import 'fit_engine.dart';
import 'fit_host_stub.dart' if (dart.library.io) 'fit_host_io.dart' as host;

/// A [FitEngine] whose jobs run in a background isolate that owns the
/// worker pool and the GPU (desktop), or null where there are no isolates
/// (the web: jobs run on the page's thread, their passes on Web Workers).
///
/// The UI isolate then only receives stages, progress and results: the
/// drivers (optimisers, samplers), the simulation of agents and the replies
/// of the pool's workers no longer wait for frames, or hold them up.
FitEngine? spawnFitHost() => host.spawnFitHost();
