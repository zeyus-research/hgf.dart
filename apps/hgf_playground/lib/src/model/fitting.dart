import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import 'fit_config.dart';
import 'gpu/open_gpu.dart';
import 'playground.dart';
import 'presets.dart';

/// What a fit explains.
enum FitTarget {
  /// The observations themselves: the parameters under which they are
  /// least surprising (the perceptual model on its own).
  observations,

  /// An agent's responses, read out from its beliefs by a response model.
  responses,
}

/// How a fit summarises the posterior.
enum FitMethod {
  /// The mode (L-BFGS from a few starts) and the Laplace approximation
  /// around it.
  map,

  /// Draws from the full posterior (NUTS, [FitConfig.chains] chains),
  /// started at the mode.
  nuts,
}

/// One parameter the fit can estimate.
class FreeParam {
  FreeParam({
    required this.id,
    required this.symbol,
    required this.help,
    required this.target,
    required this.prior,
    required this.priorText,
    this.defaultFree = false,
  });

  /// Stable id (also the parameter's name in the `HgfModel`).
  final String id;

  /// Short label, e.g. `ω x₂`.
  final String symbol;
  final String help;

  /// What it sets; null for the response model's parameter.
  final ParameterTarget? target;
  final Prior prior;

  /// The prior in words, e.g. `N(−4, 2²)`.
  final String priorText;
  final bool defaultFree;

  bool get isResponse => target == null;
}

/// One parameter's estimate.
class ParamEstimate {
  ParamEstimate({
    required this.param,
    required this.estimate,
    required this.lower,
    required this.upper,
    required this.truth,
    required this.density,
    this.draws,
    this.rhat,
    this.ess,
  });

  final FreeParam param;

  /// MAP estimate, or posterior mean (NUTS).
  final double estimate;

  /// A 95 % interval (Laplace) or the 2.5–97.5 % quantiles (NUTS).
  final double lower, upper;

  /// The value that generated simulated responses, if known.
  final double? truth;

  /// Approximate posterior density on the natural scale (Laplace), as a
  /// function for plotting; null if the Hessian wasn't positive definite.
  final double Function(double x)? density;

  /// Pooled posterior draws (NUTS).
  final Float64List? draws;
  final double? rhat, ess;
}

/// A finished fit.
class FitOutcome {
  FitOutcome({
    required this.method,
    required this.target,
    required this.estimates,
    required this.millis,
    required this.where,
    this.logEvidence,
    this.converged = true,
    this.divergences,
    this.chains,
    this.note,
    this.deviceNote,
  });

  final FitMethod method;
  final FitTarget target;
  final List<ParamEstimate> estimates;
  final int millis;

  /// Where it ran ("4 Wasm workers", "the main thread", …).
  final String where;
  final double? logEvidence;
  final bool converged;
  final int? divergences;

  /// NUTS chains that ran.
  final int? chains;

  /// A warning to show with the results.
  final String? note;

  /// Why the fit didn't run where [FitConfig.device] asked.
  final String? deviceNote;
}

/// Parameter estimation for the playground: which parameters are free, the
/// responses (simulated from an agent, or loaded), and the running or last
/// fit. Fits run on a [FitWorkers] pool spawned on first use (isolates on
/// desktop, Web Workers on the web), or, if none can be spawned, on the
/// calling thread in slices between frames; or on the GPU ([config]).
class Fitting extends ChangeNotifier {
  Fitting(
    this.pg, {
    Future<FitWorkers> Function(int? count)? spawnWorkers,
    Future<GpuOpened> Function()? openGpu,
    FitConfig config = const FitConfig(),
  }) : _spawnWorkers = spawnWorkers ?? spawnPlaygroundWorkers,
       _openGpu = openGpu ?? _openPlatformGpu,
       _config = config {
    _sync();
    pg.addListener(_onPlayground);
  }

  final Playground pg;
  final Future<FitWorkers> Function(int? count) _spawnWorkers;
  final Future<GpuOpened> Function() _openGpu;

  /// The device, pool and sampler settings the next fit uses.
  FitConfig get config => _config;
  FitConfig _config;

  void configure(FitConfig c) {
    if (c == _config) return;
    _config = c;
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------

  FitTarget target = FitTarget.observations;
  FitMethod method = FitMethod.map;

  /// The parameters on offer for the current network.
  List<FreeParam> candidates = const [];

  /// Ids of the free ones.
  final Set<String> free = {};

  /// Why this network can't be fitted (null if it can).
  String? unsupported;

  /// The simulated agent's response parameter: the inverse temperature β
  /// (binary) or the response noise sd (continuous).
  double agentBeta = 4, agentNoise = 0.1;

  /// The responses being fitted (NaN: none on that trial), and where they
  /// came from.
  Float64List? responses;
  String? responsesSource;

  /// The values that generated [responses], by [FreeParam.id] (simulated
  /// responses only).
  Map<String, double>? truth;

  bool get binary => pg.preset.input == InputKind.binary;

  void setTarget(FitTarget t) {
    if (t == target) return;
    target = t;
    _cancel();
    result = null;
    _defaultFree();
    notifyListeners();
  }

  void setMethod(FitMethod m) {
    if (m == method) return;
    method = m;
    notifyListeners();
  }

  void toggle(String id, bool on) {
    on ? free.add(id) : free.remove(id);
    notifyListeners();
  }

  void setAgent({double? beta, double? noise}) {
    agentBeta = beta ?? agentBeta;
    agentNoise = noise ?? agentNoise;
    notifyListeners();
  }

  /// The parameters offered under the current [target].
  List<FreeParam> get offered => [
    for (final c in candidates)
      if (!c.isResponse || target == FitTarget.responses) c,
  ];

  // ---------------------------------------------------------------------------
  // Responses
  // ---------------------------------------------------------------------------

  /// Draw responses from an agent with the current slider values and the
  /// agent's response parameter; remembers those values as the truth.
  void simulateAgent({int? seed}) {
    _catchUp();
    if (unsupported != null) return;
    final net = Network.fromJson(pg.network.toJson());
    final response = _responseModel(agent: true);
    final rp = candidates.firstWhere((c) => c.isResponse);
    final model = HgfModel(
      network: net,
      response: response,
      parameters: [
        Parameter.response(
          rp.id,
          responseName: response.parameters.keys.single,
          prior: rp.prior,
        ),
      ],
    );
    final value = response.parameters.values.single;
    final rng = math.Random(seed ?? math.Random().nextInt(1 << 30));
    final sim = model
        .simulate(
          [_subject(includeResponses: false)],
          [
            [value],
          ],
          rng,
        )
        .single;
    // No response where there was no observation to respond to.
    final obs = pg.observed;
    if (obs != null) {
      for (var t = 0; t < sim.length; t++) {
        if (obs[t] == 0) sim[t] = double.nan;
      }
    }
    truth = {
      for (final c in candidates)
        c.id: c.isResponse ? value : _current(c.target!),
    };
    _setResponses(sim, 'simulated agent');
  }

  /// Use loaded [values] (one per trial, NaN for none) as the responses.
  void loadResponses(Float64List values, String name) {
    _catchUp();
    if (values.length != pg.observations.length) {
      throw FormatException(
        '${values.length} responses for ${pg.observations.length} trials',
      );
    }
    if (binary && values.any((v) => v.isFinite && v != 0 && v != 1)) {
      throw const FormatException('Binary responses must be 0 or 1');
    }
    truth = null;
    _setResponses(values, name);
  }

  void clearResponses() {
    responses = null;
    responsesSource = null;
    truth = null;
    if (target == FitTarget.responses) result = null;
    revision++;
    notifyListeners();
  }

  void _setResponses(Float64List values, String source) {
    _cancel();
    responses = values;
    responsesSource = source;
    result = null;
    revision++;
    if (target != FitTarget.responses) {
      target = FitTarget.responses;
      _defaultFree();
    }
    notifyListeners();
  }

  /// Bumped when [responses] change (for the plots).
  int revision = 0;

  // ---------------------------------------------------------------------------
  // Running
  // ---------------------------------------------------------------------------

  FitOutcome? result;
  String? error;

  bool get running => _job != null;

  /// NUTS progress while [running] (null for MAP).
  SamplingProgress? progress;

  /// What the running job is doing.
  String? stage;

  Object? _job;
  FitWorkers? _workers;
  int? _workersAsked;
  String _where = '';

  // The GPU, opened on first use (or by [probeGpu]).
  Future<GpuOpened>? _gpuOpening;
  GpuRunner? _gpuRunner;

  /// The GPU, once probed: the device (null if there is none) and why not.
  GpuOpened? gpu;

  /// Whether the next [method] fit runs on the GPU (if there is one).
  bool get wantsGpu => switch (_config.device) {
    FitDevice.cpu => false,
    FitDevice.gpu => true,
    FitDevice.auto =>
      method == FitMethod.nuts && _config.chains >= FitConfig.autoGpuChains,
  };

  /// Opens the GPU if that hasn't been tried yet; [gpu] holds the outcome.
  Future<GpuOpened> probeGpu() => _gpuOpening ??= () async {
    final g = await _openGpu();
    gpu = g;
    notifyListeners();
    return g;
  }();

  bool get canFit =>
      unsupported == null &&
      free.any((id) => offered.any((c) => c.id == id)) &&
      (target == FitTarget.observations || responses != null);

  Future<void> fit() async {
    if (_catchUp()) notifyListeners();
    if (!canFit || running) return;
    final job = Object();
    _job = job;
    error = null;
    result = null;
    progress = null;
    stage = 'Starting workers…';
    notifyListeners();
    final sw = Stopwatch()..start();
    final config = _config;
    try {
      final pool = await _pool();
      if (_job != job) return;
      String? deviceNote;
      PassRunner workers = pool;
      var where = _where;
      GpuRunner? gpuFitter;
      if (wantsGpu) {
        stage = 'Opening the GPU…';
        notifyListeners();
        final g = await probeGpu();
        if (_job != job) return;
        final device = g.device;
        if (device == null) {
          if (config.device == FitDevice.gpu) {
            deviceNote = 'No GPU (${g.reason}): ran on the CPU.';
          }
        } else {
          gpuFitter = _gpuRunner = device.runner(pool);
          workers = gpuFitter;
          where = 'the GPU';
        }
      }
      final gpuPasses0 = gpuFitter?.gpuPasses ?? 0;
      String whereNow() {
        final f = gpuFitter;
        if (f == null || f.gpuPasses > gpuPasses0) return where;
        // Every pass fell back: the kernel doesn't cover this model.
        deviceNote ??= f.fallbackReason == null
            ? null
            : 'No GPU kernel for this model (${f.fallbackReason}): ran on '
                  'the CPU.';
        return _where;
      }

      final params = [
        for (final c in offered)
          if (free.contains(c.id)) c,
      ];
      final model = _model(params);
      final subject = _subject(includeResponses: target == FitTarget.responses);
      stage = 'Finding the mode (MAP)…';
      notifyListeners();
      final map = await workers.fitMap(
        model,
        [subject],
        restarts: config.restarts,
        random: math.Random(1),
      );
      if (_job != job) return;
      final best = map.results.single;
      if (method == FitMethod.map) {
        stage = 'Laplace approximation…';
        notifyListeners();
        final lap = (await workers.laplace(model, map)).results.single;
        if (_job != job) return;
        final (lo, hi) = lap.interval();
        final se = lap.unconstrainedStandardErrors;
        result = FitOutcome(
          method: FitMethod.map,
          target: target,
          millis: sw.elapsedMilliseconds,
          where: whereNow(),
          logEvidence: lap.logEvidence,
          converged: best.converged,
          deviceNote: deviceNote,
          note: !best.negLogPosterior.isFinite
              ? 'No finite fit: the beliefs diverge everywhere tried.'
              : !lap.positiveDefinite
              ? 'The posterior is not curved at the mode (a ridge or a '
                    'bound): no intervals. Free fewer parameters?'
              : !best.converged
              ? 'The optimiser stopped before converging: '
                    '${best.optimization.message}'
              : null,
          estimates: [
            for (var i = 0; i < params.length; i++)
              ParamEstimate(
                param: params[i],
                estimate: best.x[i],
                lower: lo[i],
                upper: hi[i],
                truth: truth?[params[i].id],
                density: lap.positiveDefinite
                    ? _laplaceDensity(
                        model.parameters[i].transform,
                        best.y[i],
                        se[i],
                      )
                    : null,
              ),
          ],
        );
      } else {
        final p = progress = SamplingProgress();
        stage =
            'Sampling (NUTS, ${config.chains} '
            '${config.chains == 1 ? 'chain' : 'chains'}'
            '${gpuFitter != null ? ' on the GPU' : ''})…';
        notifyListeners();
        final timer = Timer.periodic(
          const Duration(milliseconds: 100),
          (_) => notifyListeners(),
        );
        final NutsFit nuts;
        try {
          nuts = await workers.sampleNuts(
            model,
            [subject],
            sampler: Nuts(
              targetAccept: config.targetAccept,
              maxTreeDepth: config.maxTreeDepth,
            ),
            chains: config.chains,
            warmup: config.warmup,
            draws: config.draws,
            start: [
              if (best.negLogPosterior.isFinite) best.x else model.start(),
            ],
            jitter: 0.5,
            progress: p,
          );
        } finally {
          timer.cancel();
        }
        if (_job != job) return;
        final draws = nuts.results.single;
        final summary = draws.summary();
        final rhatMax = summary.fold(0.0, (m, s) => math.max(m, s.rhat));
        final essMin = summary.fold(
          double.infinity,
          (m, s) => math.min(m, s.essBulk),
        );
        result = FitOutcome(
          method: FitMethod.nuts,
          target: target,
          millis: sw.elapsedMilliseconds,
          where: whereNow(),
          chains: draws.nChains,
          deviceNote: deviceNote,
          divergences: draws.divergences,
          // draws.converged() would compute the summary again (~0.2 s
          // at 64 chains × 500 draws).
          converged:
              draws.divergences == 0 &&
              summary.every(
                (s) => s.rhat <= 1.01 && s.essBulk >= 400 && s.essTail >= 400,
              ),
          note: draws.nChains == 0
              ? 'No chain found a finite starting point.'
              : draws.divergences > 0
              ? '${draws.divergences} divergent transitions: the posterior '
                    'has regions the sampler can\'t explore reliably.'
              : !(rhatMax <= 1.01 && essMin >= 400)
              ? 'Not converged yet (R̂ ${rhatMax.toStringAsFixed(3)}, '
                    'ESS ${essMin.toStringAsFixed(0)}): take the '
                    'estimates with care.'
              : null,
          estimates: [
            for (var i = 0; i < params.length; i++)
              () {
                final all = draws.parameter(i).expand((c) => c).toList()
                  ..sort();
                final sorted = Float64List.fromList(all);
                return ParamEstimate(
                  param: params[i],
                  estimate: summary[i].mean,
                  lower: quantileOf(sorted, 0.025),
                  upper: quantileOf(sorted, 0.975),
                  truth: truth?[params[i].id],
                  density: null,
                  draws: sorted,
                  rhat: summary[i].rhat,
                  ess: summary[i].essBulk,
                );
              }(),
          ],
        );
      }
    } on FitCancelled {
      // A newer job or a data change took over.
    } catch (e) {
      if (_job == job) error = '$e';
    } finally {
      if (_job == job) {
        _job = null;
        stage = null;
        progress = null;
        notifyListeners();
      }
    }
  }

  /// Stop the running fit.
  void cancel() {
    if (!running) return;
    _cancel();
    notifyListeners();
  }

  void _cancel() {
    if (_job == null) return;
    _job = null;
    stage = null;
    progress = null;
    _workers?.cancel();
    _gpuRunner?.cancel();
  }

  /// Set the playground's sliders to the estimates.
  void apply() {
    final r = result;
    if (r == null) return;
    for (final e in r.estimates) {
      switch (e.param.target) {
        case NodeTarget(:final node, :final field):
          pg.setParam(node, field, e.estimate);
        case VolatilityCouplingTarget(:final parent, :final child):
          pg.setCoupling(_edge(parent, child, EdgeKind.volatility), e.estimate);
        case ValueCouplingTarget(:final parent, :final child):
          pg.setCoupling(_edge(parent, child, EdgeKind.value), e.estimate);
        case ResponseTarget() || null:
          break; // the response model isn't part of the playground
      }
    }
  }

  /// The worker pool, (re)spawned when [FitConfig.workers] changed. Only
  /// called between jobs.
  Future<FitWorkers> _pool() async {
    final asked = _config.workers == 0 ? null : _config.workers;
    final w = _workers;
    if (w != null && asked == _workersAsked) return w;
    if (w != null) {
      _workers = null;
      _gpuRunner = null;
      await w.close();
    }
    final pool = await _spawnWorkers(asked);
    _workers = pool;
    _workersAsked = asked;
    _where = pool.count == 0
        ? 'the main thread'
        : '${pool.count} ${kIsWeb ? (kIsWasm ? 'Wasm workers' : 'JS workers') : 'isolates'}';
    return pool;
  }

  @override
  void dispose() {
    pg.removeListener(_onPlayground);
    _cancel();
    gpu?.device?.close();
    _workers?.close();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Keeping up with the playground
  // ---------------------------------------------------------------------------

  // What the candidates, responses and result were made for.
  String? _presetId;
  VolatilityUpdates? _updates;
  Float64List? _data;

  void _onPlayground() {
    if (_catchUp()) notifyListeners();
  }

  /// [_sync] if the playground moved on; true if it did. The entry points
  /// call this first too: a coalescing playground swaps in new data at
  /// once but notifies only at the next frame.
  bool _catchUp() {
    if (_presetId == pg.preset.id &&
        _updates == pg.volatilityUpdates &&
        identical(_data, pg.observations)) {
      return false;
    }
    _sync();
    return true;
  }

  /// Catch up with the playground: new candidates for a new preset; drop
  /// the responses when the data changed, and the result on any change.
  void _sync() {
    if (_data != null) _cancel();
    result = null;
    if (!identical(_data, pg.observations) && responses != null) {
      responses = null;
      responsesSource = null;
      truth = null;
      revision++;
    }
    _data = pg.observations;
    _updates = pg.volatilityUpdates;
    if (_presetId != pg.preset.id) {
      _presetId = pg.preset.id;
      _buildCandidates();
      _defaultFree();
    }
  }

  void _buildCandidates() {
    final net = pg.network;
    final kinds = {for (var i = 0; i < net.nNodes; i++) net.kindOf(i)};
    final vector = kinds.where(
      (k) =>
          k == NodeKind.efState ||
          k == NodeKind.categoricalState ||
          k == NodeKind.dpState,
    );
    if (vector.isNotEmpty) {
      unsupported =
          'Fitting runs on the batched engine, which doesn\'t support '
          '${switch (vector.first) {
            NodeKind.categoricalState => 'categorical',
            NodeKind.dpState => 'Dirichlet process',
            _ => 'exponential-family',
          }} nodes yet.';
      candidates = const [];
      return;
    }
    unsupported = null;
    // The preset's own values centre the priors, so they don't move with
    // the sliders (or with the truth of a simulated agent).
    final base = pg.preset.build()..build();
    final inputs = net.inputIdxs.toSet();
    final out = <FreeParam>[];
    for (var i = 0; i < net.nNodes; i++) {
      if (net.kindOf(i) != NodeKind.continuousState) continue;
      final label = defaultNodeLabel(i);
      if (inputs.contains(i)) {
        final p0 = base.getInitial(i, NodeField.precision);
        out.add(
          FreeParam(
            id: 'pi_u',
            symbol: 'π_u $label',
            help: 'Sensory precision: how reliable the observations are.',
            target: NodeTarget(i, NodeField.precision),
            prior: LogNormal(math.log(p0), 2),
            priorText: _logNormalText(math.log(p0), 2),
          ),
        );
        continue;
      }
      final w = base.getInitial(i, NodeField.tonicVolatility);
      out.add(
        FreeParam(
          id: 'omega_$i',
          symbol: 'ω $label',
          help: 'Tonic volatility of $label.',
          target: NodeTarget(i, NodeField.tonicVolatility),
          prior: TruncatedNormal(w, 2, lower: -16, upper: 4),
          priorText: 'N(${_n(w)}, 2²) on [−16, 4]',
          defaultFree: true,
        ),
      );
      final m = base.getInitial(i, NodeField.mean);
      out.add(
        FreeParam(
          id: 'mu0_$i',
          symbol: 'μ₀ $label',
          help: 'Initial mean of $label.',
          target: NodeTarget(i, NodeField.mean),
          prior: Normal(m, 2),
          priorText: 'N(${_n(m)}, 2²)',
        ),
      );
      for (final c in net.volatilityChildrenOf(i)) {
        out.add(
          FreeParam(
            id: 'kappa_${i}_$c',
            symbol: 'κ $label→${defaultNodeLabel(c)}',
            help: 'Volatility coupling from $label to ${defaultNodeLabel(c)}.',
            target: VolatilityCouplingTarget(i, c),
            prior: const LogNormal(0, 0.5),
            priorText: _logNormalText(0, 0.5),
          ),
        );
      }
    }
    out.add(
      binary
          ? FreeParam(
              id: 'beta',
              symbol: 'β',
              help:
                  'Inverse temperature: how deterministically the agent '
                  'picks the outcome it expects (0: at random).',
              target: null,
              prior: const LogNormal(0.7, 1),
              priorText: _logNormalText(0.7, 1),
              defaultFree: true,
            )
          : FreeParam(
              id: 'response_precision',
              symbol: 'π_r',
              help:
                  'Response precision: 1 / the variance of the agent\'s '
                  'responses around its prediction.',
              target: null,
              prior: const LogNormal(4.6, 2),
              priorText: _logNormalText(4.6, 2),
              defaultFree: true,
            ),
    );
    candidates = out;
  }

  void _defaultFree() {
    free
      ..clear()
      ..addAll([
        for (final c in offered)
          if (c.defaultFree) c.id,
      ]);
  }

  // ---------------------------------------------------------------------------
  // Model
  // ---------------------------------------------------------------------------

  ResponseModel _responseModel({bool agent = false}) {
    if (target == FitTarget.observations && !agent) {
      return ObservationSurprise(
        binary
            ? const FirstLevelBinarySurprise()
            : const TotalGaussianSurprise(),
      );
    }
    return binary
        ? BinarySoftmaxReadout(inverseTemperature: agentBeta)
        : GaussianReadout(precision: 1 / (agentNoise * agentNoise));
  }

  HgfModel _model(List<FreeParam> params) {
    final response = _responseModel();
    return HgfModel(
      // A copy: the fit must not touch the playground's network.
      network: Network.fromJson(pg.network.toJson()),
      response: response,
      parameters: [
        for (final p in params)
          p.isResponse
              ? Parameter.response(
                  p.id,
                  responseName: response.parameters.keys.single,
                  prior: p.prior,
                )
              : Parameter(p.id, targets: [p.target!], prior: p.prior),
      ],
    );
  }

  // Categorical inputs (one-hot) never get here: see [unsupported].
  Subject _subject({required bool includeResponses}) {
    return Subject(
      inputs: pg.observations,
      observed: pg.observed,
      responses: includeResponses ? responses : null,
    );
  }

  double _current(ParameterTarget t) => switch (t) {
    NodeTarget(:final node, :final field) => pg.network.getInitial(node, field),
    VolatilityCouplingTarget(:final parent, :final child) => _edge(
      parent,
      child,
      EdgeKind.volatility,
    ).strength,
    ValueCouplingTarget(:final parent, :final child) => _edge(
      parent,
      child,
      EdgeKind.value,
    ).strength,
    ResponseTarget() => double.nan,
  };

  GraphEdge _edge(int parent, int child, EdgeKind kind) =>
      pg.layout.edges.firstWhere(
        (e) => e.parent == parent && e.child == child && e.kind == kind,
      );
}

/// The Laplace posterior `y ~ N(ŷ, s²)` as a density on the natural scale.
double Function(double) _laplaceDensity(Transform t, double yHat, double s) {
  final c = 1 / (s * math.sqrt(2 * math.pi));
  return (x) {
    final y = t.toUnconstrained(x);
    if (!y.isFinite) return 0;
    final z = (y - yHat) / s;
    return c * math.exp(-0.5 * z * z - t.logAbsJacobian(y));
  };
}

/// A log-normal prior in words: its median and central 95 %.
String _logNormalText(double mu, double sigma) =>
    'log-normal, median ${_n(math.exp(mu))} '
    '(95 %: ${_n(math.exp(mu - 1.96 * sigma))}–'
    '${_n(math.exp(mu + 1.96 * sigma))})';

String _n(double v) => v.abs() >= 1e4 || (v != 0 && v.abs() < 1e-2)
    ? v.toStringAsExponential(1)
    : v == v.roundToDouble()
    ? v.toStringAsFixed(0)
    : v.toStringAsFixed(2);

Future<GpuOpened> _openPlatformGpu() => openGpu();

/// The pool the playground fits on (default: one worker per logical
/// processor): Web Workers running the compiled
/// `fit_worker` next to the page (Wasm or JS, like the page), isolates on
/// desktop. If the workers can't start (e.g. the worker wasn't built), the
/// fits run on the main thread, a slice per frame.
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
