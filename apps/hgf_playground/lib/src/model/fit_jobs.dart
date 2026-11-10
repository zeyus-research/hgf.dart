import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';

import 'fit_config.dart';
import 'gpu/open_gpu.dart';

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

/// What `Fitting` is running.
enum FitJob { fit, recovery }

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

  /// Positive parameters (log-normal priors) are plotted on a log scale.
  bool get logScale => prior is LogNormal;

  /// [x] on the plotting scale.
  double scale(double x) => logScale ? math.log(x) : x;
}

/// One parameter's estimate.
class ParamEstimate {
  ParamEstimate({
    required this.param,
    required this.estimate,
    required this.lower,
    required this.upper,
    required this.truth,
    this.laplace,
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

  /// The Laplace approximation of the marginal posterior; null for NUTS,
  /// or if the Hessian wasn't positive definite.
  final LaplaceMarginal? laplace;

  /// Approximate posterior density on the natural scale (Laplace), for
  /// plotting.
  double Function(double x)? get density => laplace?.density;

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

/// A parameter recovery: per free parameter, the agents' true values and
/// their MAP estimates.
class RecoveryOutcome {
  RecoveryOutcome({
    required this.params,
    required this.truth,
    required this.estimate,
    required this.ok,
    required this.dropped,
    required this.redrawn,
    required this.unconverged,
    required this.millis,
    required this.where,
    this.deviceNote,
  }) : stats = [
         for (var i = 0; i < params.length; i++)
           RecoveryStats.of(params[i], truth[i], estimate[i], ok),
       ];

  final List<FreeParam> params;

  /// `truth[i][j]`, `estimate[i][j]`: parameter `i` of agent `j`.
  final List<Float64List> truth, estimate;

  /// Agents whose fit found a finite posterior.
  final List<bool> ok;

  /// Agents left out because they still diverged after redrawing, and
  /// draws replaced because they diverged.
  final int dropped, redrawn;

  /// Fits whose optimiser stopped before converging (kept).
  final int unconverged;
  final int millis;
  final String where;
  final String? deviceNote;

  /// Per parameter, on its plotting scale.
  final List<RecoveryStats> stats;

  int get agents => ok.length;
  int get failed => ok.where((o) => !o).length;
}

/// How well one parameter was recovered, on its plotting scale (log for
/// positive parameters).
class RecoveryStats {
  RecoveryStats(this.r, this.rmse, this.n);

  factory RecoveryStats.of(
    FreeParam p,
    Float64List truth,
    Float64List estimate,
    List<bool> ok,
  ) {
    var n = 0;
    var st = 0.0, se = 0.0, stt = 0.0, see = 0.0, ste = 0.0, sd = 0.0;
    for (var j = 0; j < truth.length; j++) {
      if (!ok[j]) continue;
      final t = p.scale(truth[j]), e = p.scale(estimate[j]);
      if (!t.isFinite || !e.isFinite) continue;
      n++;
      st += t;
      se += e;
      stt += t * t;
      see += e * e;
      ste += t * e;
      sd += (e - t) * (e - t);
    }
    if (n < 2) return RecoveryStats(double.nan, double.nan, n);
    final ct = stt - st * st / n, ce = see - se * se / n;
    final cte = ste - st * se / n;
    final r = ct > 0 && ce > 0 ? cte / math.sqrt(ct * ce) : double.nan;
    return RecoveryStats(r, math.sqrt(sd / n), n);
  }

  /// Pearson correlation of truth and estimate.
  final double r;

  /// Root mean squared error.
  final double rmse;

  /// Agents counted.
  final int n;
}

// -----------------------------------------------------------------------------
// Jobs
// -----------------------------------------------------------------------------

/// Where a job's passes run, and how to describe that afterwards.
class JobRunner {
  JobRunner(this.workers, this._where, this._cpuWhere, {this.gpu})
    : _gpuPasses0 = gpu?.gpuPasses ?? 0;

  final PassRunner workers;
  final GpuRunner? gpu;
  final String _where, _cpuWhere;
  final int _gpuPasses0;

  /// Why the job didn't run where the config asked (set by [where] if the
  /// GPU had no kernel for the model).
  String? deviceNote;

  /// Where the job ran: the CPU pool if every GPU pass fell back.
  String where() {
    final f = gpu;
    if (f == null || f.gpuPasses > _gpuPasses0) return _where;
    deviceNote ??= f.fallbackReason == null
        ? null
        : 'No GPU kernel for this model (${f.fallbackReason}): ran on '
              'the CPU.';
    return _cpuWhere;
  }
}

/// What a running job reports.
sealed class JobEvent {
  const JobEvent();
}

/// What the job is doing now.
final class JobStage extends JobEvent {
  const JobStage(this.text);
  final String text;
}

/// Sampling progress: transitions done of [total] (0 before the start).
final class JobProgress extends JobEvent {
  const JobProgress(this.completed, this.total);
  final int completed, total;

  double get fraction => total == 0 ? 0 : completed / total;
}

/// What a job runs against: an engine's pool and GPU, and its listener.
abstract interface class JobContext {
  /// Where the passes run: the GPU if [gpu] and there is one, else the CPU
  /// pool (with [FitConfig.workers] workers).
  Future<JobRunner> runner(FitConfig config, {required bool gpu});

  void report(JobEvent event);

  /// Throws [FitCancelled] if the job was cancelled.
  void check();
}

/// A unit of fitting work with all its inputs. Sendable to a host isolate
/// (isolates spawned from this program share its code), or encoded for the
/// web's host worker (`fit_wire.dart`); [execute] runs it against the
/// host's pool and GPU.
sealed class FitJobSpec<R> {
  const FitJobSpec({required this.config, required this.gpu});

  final FitConfig config;

  /// Whether the passes should run on the GPU (if there is one).
  final bool gpu;

  Future<R> execute(JobContext ctx);
}

/// One subject's fit: MAP + Laplace, or NUTS from the mode.
final class ParameterFitJob extends FitJobSpec<FitOutcome> {
  const ParameterFitJob({
    required super.config,
    required super.gpu,
    required this.method,
    required this.target,
    required this.params,
    required this.model,
    required this.subject,
    this.truth,
  });

  final FitMethod method;
  final FitTarget target;
  final List<FreeParam> params;
  final HgfModel model;
  final Subject subject;
  final Map<String, double>? truth;

  @override
  Future<FitOutcome> execute(JobContext ctx) async {
    final sw = Stopwatch()..start();
    final runner = await ctx.runner(config, gpu: gpu);
    final workers = runner.workers;
    ctx.report(const JobStage('Finding the mode (MAP)…'));
    final map = await workers.fitMap(
      model,
      [subject],
      restarts: config.restarts,
      random: math.Random(1),
    );
    ctx.check();
    final best = map.results.single;
    if (method == FitMethod.map) {
      ctx.report(const JobStage('Laplace approximation…'));
      final lap = (await workers.laplace(model, map)).results.single;
      ctx.check();
      final (lo, hi) = lap.interval();
      final se = lap.unconstrainedStandardErrors;
      return FitOutcome(
        method: FitMethod.map,
        target: target,
        millis: sw.elapsedMilliseconds,
        where: runner.where(),
        logEvidence: lap.logEvidence,
        converged: best.converged,
        deviceNote: runner.deviceNote,
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
              laplace: lap.positiveDefinite
                  ? LaplaceMarginal(
                      model.parameters[i].transform,
                      best.y[i],
                      se[i],
                    )
                  : null,
            ),
        ],
      );
    }
    final p = SamplingProgress();
    ctx.report(
      JobStage(
        'Sampling (NUTS, ${config.chains} '
        '${config.chains == 1 ? 'chain' : 'chains'}'
        '${runner.gpu != null ? ' on the GPU' : ''})…',
      ),
    );
    final timer = Timer.periodic(
      const Duration(milliseconds: 100),
      (_) => ctx.report(JobProgress(p.completed, p.total)),
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
        start: [if (best.negLogPosterior.isFinite) best.x else model.start()],
        jitter: 0.5,
        progress: p,
      );
    } finally {
      timer.cancel();
    }
    ctx.check();
    final draws = nuts.results.single;
    final summary = draws.summary();
    final rhatMax = summary.fold(0.0, (m, s) => math.max(m, s.rhat));
    final essMin = summary.fold(
      double.infinity,
      (m, s) => math.min(m, s.essBulk),
    );
    return FitOutcome(
      method: FitMethod.nuts,
      target: target,
      millis: sw.elapsedMilliseconds,
      where: runner.where(),
      chains: draws.nChains,
      deviceNote: runner.deviceNote,
      divergences: draws.divergences,
      // draws.converged() would compute the summary again (~0.2 s at 64
      // chains × 500 draws).
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
            final all = draws.parameter(i).expand((c) => c).toList()..sort();
            final sorted = Float64List.fromList(all);
            return ParamEstimate(
              param: params[i],
              estimate: summary[i].mean,
              lower: quantileOf(sorted, 0.025),
              upper: quantileOf(sorted, 0.975),
              truth: truth?[params[i].id],
              draws: sorted,
              rhat: summary[i].rhat,
              ess: summary[i].essBulk,
            );
          }(),
      ],
    );
  }
}

/// Simulate [agents] agents on [base]'s inputs, each with the free
/// parameters drawn from their priors, and fit them all by MAP.
///
/// Agents whose beliefs diverge at their drawn values (a runaway
/// volatility, say) are redrawn, a few times at most: their responses say
/// nothing about the parameters.
final class RecoveryJob extends FitJobSpec<RecoveryOutcome> {
  const RecoveryJob({
    required super.config,
    required super.gpu,
    required this.params,
    required this.model,
    required this.base,
    required this.agents,
    required this.seed,
  });

  final List<FreeParam> params;
  final HgfModel model;

  /// The inputs (and which were observed) every agent sees.
  final Subject base;
  final int agents;
  final int seed;

  @override
  Future<RecoveryOutcome> execute(JobContext ctx) async {
    final sw = Stopwatch()..start();
    final n = agents;
    final runner = await ctx.runner(config, gpu: gpu);
    final k = model.dimension;
    final rng = math.Random(seed);
    final obs = base.observed;

    final truth = List<Float64List>.generate(n, (_) => Float64List(k));
    final responses = List<Float64List?>.filled(n, null);
    // Agents still to draw; after each round, those that diverged.
    var pending = [for (var s = 0; s < n; s++) s];
    var redrawn = 0;
    for (var round = 0; round < 4 && pending.isNotEmpty; round++) {
      ctx.report(
        JobStage(
          round == 0
              ? 'Simulating $n agents…'
              : 'Redrawing ${pending.length} diverging '
                    '${pending.length == 1 ? 'agent' : 'agents'}…',
        ),
      );
      if (round > 0) redrawn += pending.length;
      // In chunks, with the event loop in between (for cancellation, and
      // frames when this runs on the UI thread).
      const chunk = 64;
      for (var i = 0; i < pending.length; i += chunk) {
        final ids = pending.sublist(i, math.min(i + chunk, pending.length));
        final xs = [for (final _ in ids) model.samplePrior(rng)];
        final sims = model.simulate([for (final _ in ids) base], xs, rng);
        for (var j = 0; j < ids.length; j++) {
          truth[ids[j]] = xs[j];
          final r = sims[j];
          if (obs != null) {
            for (var t = 0; t < r.length; t++) {
              if (obs[t] == 0) r[t] = double.nan;
            }
          }
          responses[ids[j]] = r;
        }
        await Future<void>.delayed(Duration.zero);
        ctx.check();
      }
      final finite = await _finiteAt(
        runner,
        [for (final s in pending) _agent(responses[s]!, s)],
        [for (final s in pending) truth[s]],
      );
      ctx.check();
      pending = [
        for (var j = 0; j < pending.length; j++)
          if (!finite[j]) pending[j],
      ];
    }
    final dropped = pending.toSet();
    final kept = [
      for (var s = 0; s < n; s++)
        if (!dropped.contains(s)) s,
    ];
    final subjects = [for (final s in kept) _agent(responses[s]!, s)];
    final runs = subjects.length * (config.restarts + 1);
    ctx.report(
      JobStage(
        'Fitting ${subjects.length} agents (MAP, $runs L-BFGS '
        '${runs == 1 ? 'run' : 'runs'}'
        '${runner.gpu != null ? ' on the GPU' : ''})…',
      ),
    );
    final map = await runner.workers.fitMap(
      model,
      subjects,
      restarts: config.restarts,
      random: math.Random(1),
    );
    ctx.check();
    final m = subjects.length;
    final t = [for (var i = 0; i < k; i++) Float64List(m)];
    final e = [for (var i = 0; i < k; i++) Float64List(m)];
    final ok = List<bool>.filled(m, false);
    var unconverged = 0;
    for (var j = 0; j < m; j++) {
      final r = map.results[j];
      ok[j] = r.negLogPosterior.isFinite;
      if (ok[j] && !r.converged) unconverged++;
      for (var i = 0; i < k; i++) {
        t[i][j] = truth[kept[j]][i];
        e[i][j] = r.x[i];
      }
    }
    return RecoveryOutcome(
      params: params,
      truth: t,
      estimate: e,
      ok: ok,
      dropped: dropped.length,
      redrawn: redrawn,
      unconverged: unconverged,
      millis: sw.elapsedMilliseconds,
      where: runner.where(),
      deviceNote: runner.deviceNote,
    );
  }

  Subject _agent(Float64List responses, int id) => Subject(
    inputs: base.inputs,
    observed: base.observed,
    responses: responses,
    id: id,
  );

  /// Whether each subject's negative log posterior is finite at its
  /// constrained parameters [xs], in batched passes on [runner] (split
  /// over the pool's workers).
  Future<List<bool>> _finiteAt(
    JobRunner runner,
    List<Subject> subjects,
    List<Float64List> xs,
  ) async {
    final objective = BatchedObjective(
      model,
      subjects,
      gradients: Gradients.forward,
    );
    final n = subjects.length;
    final parts = runner.gpu != null ? 1 : runner.workers.streams;
    final size = math.max(1, (n / parts).ceil());
    final chunks = await Future.wait([
      for (var a = 0; a < n; a += size)
        () {
          final b = math.min(a + size, n);
          return runner.workers.evaluate(
            objective,
            [for (var s = a; s < b; s++) s],
            [for (var s = a; s < b; s++) model.toUnconstrained(xs[s])],
            List.filled(b - a, false),
          );
        }(),
    ]);
    return [
      for (final (values, _) in chunks)
        for (final v in values) v.isFinite,
    ];
  }
}

/// One parameter's Laplace posterior, `y ~ N(mode, sd²)` on the
/// unconstrained scale `y = transform.toUnconstrained(x)`.
class LaplaceMarginal {
  const LaplaceMarginal(this.transform, this.mode, this.sd);

  final Transform transform;
  final double mode, sd;

  /// The density on the natural scale.
  double density(double x) {
    final y = transform.toUnconstrained(x);
    if (!y.isFinite) return 0;
    final z = (y - mode) / sd;
    return math.exp(-0.5 * z * z - transform.logAbsJacobian(y)) /
        (sd * math.sqrt(2 * math.pi));
  }
}
