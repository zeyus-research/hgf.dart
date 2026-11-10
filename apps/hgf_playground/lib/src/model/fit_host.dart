import 'fit_engine.dart';
import 'fit_host_stub.dart'
    if (dart.library.io) 'fit_host_io.dart'
    if (dart.library.js_interop) 'fit_host_web.dart'
    as host;

/// A [FitEngine] whose jobs run in a host that owns the worker pool and the
/// GPU: a background isolate on desktop, a Web Worker on the web (its pool
/// nested workers; if it doesn't start, jobs run on the page's thread).
///
/// The UI then only receives stages, progress and results: the drivers
/// (optimisers, samplers), the simulation of agents and the replies of the
/// pool's workers no longer wait for frames, or hold them up.
FitEngine? spawnFitHost() => host.spawnFitHost();
