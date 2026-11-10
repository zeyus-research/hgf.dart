import 'package:meta/meta.dart' show immutable;

/// Where the fits run.
enum FitDevice {
  /// Whichever is faster for the job: the CPU, except NUTS with many
  /// chains ([FitConfig.autoGpuChains]) and large parameter recoveries
  /// ([FitConfig.autoGpuMapRuns]).
  auto,

  /// The CPU (f64), on the worker pool.
  cpu,

  /// The GPU (f32 likelihood and gradients; precise passes such as the
  /// Laplace Hessian still run on the CPU).
  gpu,
}

/// How the playground fits: the device, the worker pool and the samplers'
/// settings. Kept between sessions by `Settings`.
@immutable
class FitConfig {
  const FitConfig({
    this.device = FitDevice.auto,
    this.workers = 0,
    this.restarts = 3,
    this.chains = 4,
    this.warmup = 500,
    this.draws = 500,
    this.targetAccept = 0.8,
    this.maxTreeDepth = 10,
  });

  final FitDevice device;

  /// CPU workers (isolates or Web Workers); 0: one per logical processor.
  final int workers;

  /// MAP: extra L-BFGS starts drawn from the priors.
  final int restarts;

  /// NUTS chains, and each chain's warm-up and kept iterations.
  final int chains, warmup, draws;

  /// NUTS: the step-size adaptation's target acceptance rate.
  final double targetAccept;

  /// NUTS: the deepest trajectory tree (2^depth leapfrog steps).
  final int maxTreeDepth;

  /// With [FitDevice.auto], NUTS runs on the GPU from this many chains on.
  /// One series is too small a batch for the GPU: each leapfrog round
  /// costs a round trip (~0.4 ms native, 0.3–1 ms in browsers), which a
  /// few chains on CPU workers beat. Measured on one subject, 500 + 500
  /// iterations (RX 7600 XT, i7-7700, 4 workers):
  /// - native (AOT isolates): 4 chains 2.5 s GPU vs 1.3 s CPU, 64 chains
  ///   4.5 vs 6.5 s, 256 chains 13 vs 21 s;
  /// - playground, Wasm + WebGPU (binary 3-level, 400 trials): 4 chains
  ///   30 vs 12 s, 64 chains 37 vs 89 s.
  static const int autoGpuChains = 32;

  /// With [FitDevice.auto], a parameter recovery runs on the GPU from this
  /// many L-BFGS runs on (agents × starts). Measured MAP, one start per
  /// agent, binary 2-level, 320 trials (`fit_gpu --only map`, RX 7600 XT,
  /// 8 AOT isolates): 100 agents 22 ms CPU vs 25 ms GPU, 200: 37 vs 31,
  /// 1000: 180 vs 51. In browsers each GPU round trip costs up to ~1 ms more
  /// (~30 per fit), and Wasm workers are ~1.6× slower than AOT, so the
  /// crossover there is about the same.
  static const int autoGpuMapRuns = 256;

  static const chainChoices = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  static const iterationChoices = [100, 200, 300, 500, 1000, 2000, 4000];
  static const restartChoices = [0, 1, 2, 3, 5, 10];
  static const targetAcceptChoices = [0.65, 0.8, 0.9, 0.95, 0.99];
  static const treeDepthChoices = [6, 7, 8, 9, 10, 11, 12];

  FitConfig copyWith({
    FitDevice? device,
    int? workers,
    int? restarts,
    int? chains,
    int? warmup,
    int? draws,
    double? targetAccept,
    int? maxTreeDepth,
  }) => FitConfig(
    device: device ?? this.device,
    workers: workers ?? this.workers,
    restarts: restarts ?? this.restarts,
    chains: chains ?? this.chains,
    warmup: warmup ?? this.warmup,
    draws: draws ?? this.draws,
    targetAccept: targetAccept ?? this.targetAccept,
    maxTreeDepth: maxTreeDepth ?? this.maxTreeDepth,
  );

  @override
  bool operator ==(Object other) =>
      other is FitConfig &&
      other.device == device &&
      other.workers == workers &&
      other.restarts == restarts &&
      other.chains == chains &&
      other.warmup == warmup &&
      other.draws == draws &&
      other.targetAccept == targetAccept &&
      other.maxTreeDepth == maxTreeDepth;

  @override
  int get hashCode => Object.hash(
    device,
    workers,
    restarts,
    chains,
    warmup,
    draws,
    targetAccept,
    maxTreeDepth,
  );
}
