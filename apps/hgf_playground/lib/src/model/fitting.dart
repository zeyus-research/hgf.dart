import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import 'fit_config.dart';
import 'fit_engine.dart';
import 'fit_jobs.dart';
import 'gpu/open_gpu.dart';
import 'playground.dart';
import 'presets.dart';

export 'fit_engine.dart' show FitEngine, GpuInfo, spawnPlaygroundWorkers;
export 'fit_jobs.dart';

/// Parameter estimation for the playground: which parameters are free, the
/// responses (simulated from an agent, or loaded), the running or last fit,
/// and parameter recovery. Jobs run on a [FitEngine]: by default a
/// [LocalFitEngine] on this isolate (a [FitWorkers] pool spawned on first
/// use, isolates on desktop or Web Workers on the web, and the GPU), or the
/// given [engine], such as a host isolate (`spawnFitHost`).
class Fitting extends ChangeNotifier {
  Fitting(
    this.pg, {
    FitEngine? engine,
    Future<FitWorkers> Function(int? count)? spawnWorkers,
    Future<GpuOpened> Function()? openGpu,
    FitConfig config = const FitConfig(),
  }) : _engine =
           engine ??
           LocalFitEngine(spawnWorkers: spawnWorkers, openGpu: openGpu),
       _config = config {
    _sync();
    pg.addListener(_onPlayground);
  }

  final Playground pg;
  final FitEngine _engine;

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

  bool get categorical => pg.preset.input == InputKind.categorical;

  /// Whether the agent's responses are choices (binary or categorical,
  /// with an inverse temperature β) rather than continuous reports.
  bool get choices => binary || categorical;

  /// Whether the network clusters its input with a Dirichlet process node.
  bool get dirichlet {
    final net = pg.network;
    for (final i in net.inputIdxs) {
      if (net.kindOf(i) == NodeKind.dpState) return true;
    }
    return false;
  }

  /// Why only the observations can be fitted (null if responses can too):
  /// no response model reads a Dirichlet process node's beliefs.
  String? get responsesUnavailable => dirichlet
      ? 'An agent\'s responses can\'t be fitted here: no response model '
            'reads a Dirichlet process node\'s clusters.'
      : null;

  /// Why NUTS is not offered (null if it is). A dp node's likelihood jumps
  /// where a cluster assignment flips, so it has no gradients, and each of
  /// its trials weighs pyhgf's 20 000 cluster proposals (~1 ms): NUTS on
  /// finite differences would take hours.
  String? get nutsUnavailable => dirichlet
      ? 'NUTS needs gradients; a Dirichlet process\'s likelihood jumps '
            'wherever an observation changes cluster, and each trial costs '
            '~1 ms, so only MAP (Nelder–Mead) is offered.'
      : null;

  void setTarget(FitTarget t) {
    if (t == target ||
        (t == FitTarget.responses && responsesUnavailable != null)) {
      return;
    }
    target = t;
    _cancel();
    result = null;
    _defaultFree();
    notifyListeners();
  }

  void setMethod(FitMethod m) {
    if (m == method || (m == FitMethod.nuts && nutsUnavailable != null)) {
      return;
    }
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
    final k = pg.nCategories;
    if (categorical &&
        values.any(
          (v) => v.isFinite && (v != v.roundToDouble() || v < 0 || v >= k),
        )) {
      throw FormatException(
        'Categorical responses must be category indices, 0 to ${k - 1}',
      );
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

  /// What is running (null if nothing).
  FitJob? runningJob;

  /// The last job started (whose [error] it is).
  FitJob? lastJob;

  /// NUTS progress while [running] (null for MAP).
  JobProgress? progress;

  /// What the running job is doing.
  String? stage;

  Object? _job;

  /// The GPU, once probed: its adapter, or why there is none.
  GpuInfo? gpu;
  Future<GpuInfo>? _gpuProbe;

  /// Whether the next [method] fit runs on the GPU (if there is one).
  bool get wantsGpu => switch (_config.device) {
    FitDevice.cpu => false,
    FitDevice.gpu => true,
    FitDevice.auto =>
      method == FitMethod.nuts && _config.chains >= FitConfig.autoGpuChains,
  };

  /// Opens the GPU if that hasn't been tried yet; [gpu] holds the outcome.
  Future<GpuInfo> probeGpu() => _gpuProbe ??= () async {
    GpuInfo g;
    try {
      g = await _engine.probeGpu();
    } catch (e) {
      g = (adapter: null, reason: '$e');
    }
    gpu = g;
    notifyListeners();
    return g;
  }();

  bool get canFit =>
      unsupported == null &&
      (method != FitMethod.nuts || nutsUnavailable == null) &&
      free.any((id) => offered.any((c) => c.id == id)) &&
      (target == FitTarget.observations || responses != null);

  Future<void> fit() async {
    if (_catchUp()) notifyListeners();
    if (!canFit || running) return;
    final params = [
      for (final c in offered)
        if (free.contains(c.id)) c,
    ];
    final outcome = await _run(
      FitJob.fit,
      ParameterFitJob(
        config: _config,
        gpu: wantsGpu,
        method: method,
        target: target,
        params: params,
        model: _model(params),
        subject: _subject(includeResponses: target == FitTarget.responses),
        truth: truth,
      ),
      () => result = null,
    );
    if (outcome != null) {
      result = outcome;
      notifyListeners();
    }
  }

  /// Runs [job] as [kind] on the engine; its outcome, or null if it failed
  /// or a newer job or a data change took over.
  Future<R?> _run<R>(
    FitJob kind,
    FitJobSpec<R> job,
    void Function() clear,
  ) async {
    final token = Object();
    _job = token;
    runningJob = lastJob = kind;
    error = null;
    clear();
    progress = null;
    stage = 'Starting…';
    notifyListeners();
    // The GPU's adapter for the settings, if this job opens it.
    if (job.gpu) unawaited(probeGpu());
    try {
      final r = await _engine.run(job, (e) {
        if (_job != token) return;
        switch (e) {
          case JobStage(:final text):
            stage = text;
          case JobProgress p:
            progress = p;
        }
        notifyListeners();
      });
      return _job == token ? r : null;
    } on FitCancelled {
      // A newer job or a data change took over.
      return null;
    } catch (e) {
      if (_job == token) error = '$e';
      return null;
    } finally {
      if (_job == token) {
        _job = null;
        runningJob = null;
        stage = null;
        progress = null;
        notifyListeners();
      }
    }
  }

  /// Stop the running fit or recovery.
  void cancel() {
    if (!running) return;
    _cancel();
    notifyListeners();
  }

  void _cancel() {
    if (_job == null) return;
    _job = null;
    runningJob = null;
    stage = null;
    progress = null;
    _engine.cancel();
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

  @override
  void dispose() {
    pg.removeListener(_onPlayground);
    _cancel();
    _engine.close();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Parameter recovery
  // ---------------------------------------------------------------------------

  /// How many agents a recovery simulates and fits.
  int recoveryAgents = 200;

  static const recoveryAgentChoices = [50, 200, 1000, 2000];

  /// The last recovery, until the network or the data change.
  RecoveryOutcome? recovery;

  void setRecoveryAgents(int n) {
    if (n == recoveryAgents) return;
    recoveryAgents = n;
    notifyListeners();
  }

  /// Whether the next recovery runs on the GPU (if there is one).
  bool get recoveryWantsGpu => switch (_config.device) {
    FitDevice.cpu => false,
    FitDevice.gpu => true,
    FitDevice.auto =>
      recoveryAgents * (_config.restarts + 1) >= FitConfig.autoGpuMapRuns,
  };

  /// A recovery fits agents' responses: it needs the response target and
  /// a free parameter, but no responses of its own.
  bool get canRecover =>
      unsupported == null &&
      responsesUnavailable == null &&
      target == FitTarget.responses &&
      free.any((id) => offered.any((c) => c.id == id));

  /// Simulates [recoveryAgents] agents on the playground's observations,
  /// each with its free parameters drawn from their priors (the others at
  /// the sliders' values), fits every agent's responses by MAP in one
  /// batched job, and keeps truth and estimate per agent in [recovery]
  /// (see [RecoveryJob]).
  Future<void> recover({int? seed}) async {
    if (_catchUp()) notifyListeners();
    if (!canRecover || running) return;
    final params = [
      for (final c in offered)
        if (free.contains(c.id)) c,
    ];
    final outcome = await _run(
      FitJob.recovery,
      RecoveryJob(
        config: _config,
        gpu: recoveryWantsGpu,
        params: params,
        model: _model(params),
        base: _subject(includeResponses: false),
        agents: recoveryAgents,
        seed: seed ?? math.Random().nextInt(1 << 30),
      ),
      () => recovery = null,
    );
    if (outcome != null) {
      recovery = outcome;
      notifyListeners();
    }
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
    recovery = null;
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
    if (kinds.contains(NodeKind.dpState)) {
      _buildDpCandidates();
      return;
    }
    if (kinds.contains(NodeKind.efState)) {
      // pyhgf scores no observation at a fixed-rate ef-state node (its
      // node surprise is undefined), so there is nothing to fit to.
      unsupported =
          'A fixed-rate exponential-family node defines no surprise for its '
          'observations, so there is nothing to fit its learning rate to.';
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
      choices
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

  /// α and π_s of each dp-state node, fitted to the observations' surprise
  /// under the node's predictive mixture (`DpSurprise`, an hgf.dart
  /// extension: pyhgf defines none).
  void _buildDpCandidates() {
    unsupported = null;
    target = FitTarget.observations;
    method = FitMethod.map;
    final net = pg.network;
    final base = pg.preset.build()..build();
    final out = <FreeParam>[];
    for (var i = 0; i < net.nNodes; i++) {
      if (net.kindOf(i) != NodeKind.dpState) continue;
      final label = defaultNodeLabel(i);
      final a = math.log(base.getInitial(i, NodeField.alpha));
      out.add(
        FreeParam(
          id: 'alpha_$i',
          symbol: 'α $label',
          help: 'Concentration of $label: how readily a new cluster opens.',
          target: NodeTarget(i, NodeField.alpha),
          prior: LogNormal(a, 2),
          priorText: _logNormalText(a, 2),
          defaultFree: true,
        ),
      );
      final sp = math.log(base.getInitial(i, NodeField.sensoryPrecision));
      out.add(
        FreeParam(
          id: 'pi_s_$i',
          symbol: 'π_s $label',
          help:
              'Sensory precision of $label: how tightly a new cluster is '
              'fitted to the observation that opens it.',
          target: NodeTarget(i, NodeField.sensoryPrecision),
          prior: LogNormal(sp, 1),
          priorText: _logNormalText(sp, 1),
        ),
      );
    }
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
            : categorical
            ? const CategoricalSurprise()
            : dirichlet
            ? const DpSurprise()
            : const TotalGaussianSurprise(),
      );
    }
    return binary
        ? BinarySoftmaxReadout(inverseTemperature: agentBeta)
        : categorical
        ? CategoricalSoftmaxReadout(
            node: pg.network.inputIdxs.single,
            inverseTemperature: agentBeta,
          )
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

  Subject _subject({required bool includeResponses}) {
    return Subject(
      // One-hot rows for a categorical input.
      inputs: pg.networkInput,
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
