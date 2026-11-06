import 'dart:math' as math;
import 'dart:typed_data';

import '../linalg.dart';
import '../optim/optimizer.dart';
import '../special.dart';

/// The inverse metric (the posterior covariance estimate) HMC moves with.
enum MetricKind {
  /// The identity: no adaptation.
  unit,

  /// Per-parameter variances, adapted in warmup (Stan's default).
  diagonal,

  /// A full covariance, adapted in warmup; for correlated parameters.
  dense,
}

/// The No-U-Turn Sampler, as in Stan: multinomial sampling along
/// trajectories doubled until the generalised no-U-turn criterion (with the
/// extra checks across subtrees) stops them, a dual-averaging step size and
/// a windowed metric adaptation in warmup (75 / 25·2ⁿ / 50 iterations).
///
/// Like the optimisers, a chain is a generator that yields evaluation
/// requests, one point with its gradient per leapfrog step, so a driver
/// (`sampleNuts`) advances many chains (and subjects) in lock step, one
/// batched pass per step.
///
/// The gradients may also be finite differences: the leapfrog integrator is
/// volume preserving and reversible for any deterministic force field, and
/// the multinomial weights use the exact density, so the chain still targets
/// the exact posterior; gradient error only costs efficiency.
final class Nuts {
  const Nuts({
    this.targetAccept = 0.8,
    this.maxTreeDepth = 10,
    this.metric = MetricKind.diagonal,
    this.maxEnergyError = 1000,
    this.initialStepSize = 1,
    this.maxInitAttempts = 100,
  });

  /// The mean acceptance statistic the step size is tuned to (`δ`).
  final double targetAccept;

  /// Trajectories stop after `2^maxTreeDepth` leapfrog steps.
  final int maxTreeDepth;

  final MetricKind metric;

  /// An energy error above this marks the transition divergent.
  final double maxEnergyError;

  /// The step size the warmup heuristic starts from.
  final double initialStepSize;

  /// Starting points tried (from the `init` callback) until one has a
  /// finite density and gradient.
  final int maxInitAttempts;

  /// One chain: [warmup] adaptation iterations, then [draws] kept ones,
  /// from the first finite point [init] returns (unconstrained scale).
  /// Requests ask for the objective `U` (the negative log density) and its
  /// gradient at one point. The draws and statistics go to [out].
  Iterable<EvalRequest> chain({
    required Float64List Function(int attempt) init,
    required int warmup,
    required int draws,
    required math.Random random,
    required ChainOutput out,
  }) sync* {
    if (warmup < 0) throw ArgumentError.value(warmup, 'warmup');
    if (draws < 0) throw ArgumentError.value(draws, 'draws');

    // A finite starting point.
    _State? z;
    for (var a = 0; a < maxInitAttempts && z == null; a++) {
      final q = Float64List.fromList(init(a));
      final req = EvalRequest([q], gradient: true);
      yield req;
      final u = req.values[0], g = req.gradients[0];
      if (u.isFinite && g.every((v) => v.isFinite)) {
        z = _State(q.length)
          ..q.setAll(0, q)
          ..g.setAll(0, g)
          ..u = u;
      }
    }
    if (z == null) {
      out._fail('no finite starting point in $maxInitAttempts attempts');
      return;
    }
    final k = z.q.length;
    out._start(k, draws);
    final c = _Ctx(
      this,
      z,
      _Metric.unit(k, dense: metric == MetricKind.dense),
      random,
    );

    yield* c.initStepSize();
    final da = _DualAveraging(targetAccept)..restart(c.eps);
    final window = _Windows(warmup, adapt: metric != MetricKind.unit);
    final est = _CovarianceEstimator(k, dense: metric == MetricKind.dense);

    for (var it = 0; it < warmup + draws; it++) {
      yield* c.transition();
      out.iterations++;
      if (it < warmup) {
        c.eps = da.learn(c.acceptStat);
        if (window.inWindow) est.add(c.z.q);
        if (window.endOfWindow) {
          c.metric = est.regularised();
          est.restart();
          yield* c.initStepSize();
          da.restart(c.eps);
        }
        window.advance();
        if (it == warmup - 1) c.eps = da.finalStepSize;
        out.warmupDivergences += c.divergent ? 1 : 0;
      } else {
        out._record(it - warmup, c);
      }
    }
    out
      ..stepSize = c.eps
      ..inverseMetric = Float64List.fromList(c.metric.inv);
  }
}

/// What one chain produced: draws on the unconstrained scale and the
/// sampler statistics of each kept iteration.
final class ChainOutput {
  /// Draws, row-major `draws x k`, unconstrained scale.
  Float64List y = Float64List(0);

  /// The log density (`−U`, Jacobian included) at each draw.
  Float64List logDensity = Float64List(0);

  /// The mean Metropolis acceptance probability over each trajectory.
  Float64List acceptStat = Float64List(0);

  /// The Hamiltonian at each draw.
  Float64List energy = Float64List(0);

  /// Tree depth and leapfrog steps of each transition.
  Int32List treeDepth = Int32List(0);
  Int32List leapfrogSteps = Int32List(0);

  /// Whether each transition diverged.
  Uint8List divergent = Uint8List(0);

  /// Divergent transitions during warmup (not counted in [divergent]).
  int warmupDivergences = 0;

  /// The adapted step size.
  double stepSize = double.nan;

  /// The adapted inverse metric: `k` variances, or `k x k` row-major.
  Float64List inverseMetric = Float64List(0);

  /// Set when the chain could not start.
  String? error;

  /// Transitions done so far, warmup included (for progress displays).
  int iterations = 0;

  int _k = 0;

  void _fail(String message) => error = message;

  void _start(int k, int draws) {
    _k = k;
    y = Float64List(draws * k);
    logDensity = Float64List(draws);
    acceptStat = Float64List(draws);
    energy = Float64List(draws);
    treeDepth = Int32List(draws);
    leapfrogSteps = Int32List(draws);
    divergent = Uint8List(draws);
  }

  void _record(int d, _Ctx c) {
    y.setRange(d * _k, (d + 1) * _k, c.z.q);
    logDensity[d] = -c.z.u;
    acceptStat[d] = c.acceptStat;
    energy[d] = c.energy;
    treeDepth[d] = c.depth;
    leapfrogSteps[d] = c.nLeapfrog;
    divergent[d] = c.divergent ? 1 : 0;
  }
}

/// A point in phase space with the potential and its gradient at `q`.
final class _State {
  _State(int k) : q = Float64List(k), p = Float64List(k), g = Float64List(k);

  final Float64List q, p, g;
  double u = double.infinity;

  void copyFrom(_State o) {
    q.setAll(0, o.q);
    p.setAll(0, o.p);
    g.setAll(0, o.g);
    u = o.u;
  }

  _State copy() => _State(q.length)..copyFrom(this);
}

final class _Metric {
  _Metric(this.k, this.inv, {required this.dense})
    : chol = dense ? cholesky(inv, k) : null {
    if (dense && chol == null) {
      throw StateError('inverse metric is not positive definite');
    }
  }

  factory _Metric.unit(int k, {required bool dense}) {
    final inv = Float64List(dense ? k * k : k);
    for (var i = 0; i < k; i++) {
      inv[dense ? i * k + i : i] = 1;
    }
    return _Metric(k, inv, dense: dense);
  }

  final int k;
  final bool dense;

  /// `M⁻¹`: variances, or the covariance row-major.
  final Float64List inv;

  /// Cholesky factor of [inv] (dense only).
  final Float64List? chol;

  /// `out = M⁻¹ p`.
  void velocity(Float64List p, Float64List out) {
    if (!dense) {
      for (var i = 0; i < k; i++) {
        out[i] = inv[i] * p[i];
      }
      return;
    }
    for (var i = 0; i < k; i++) {
      var s = 0.0;
      for (var j = 0; j < k; j++) {
        s += inv[i * k + j] * p[j];
      }
      out[i] = s;
    }
  }

  double kinetic(Float64List p, Float64List scratch) {
    velocity(p, scratch);
    var s = 0.0;
    for (var i = 0; i < k; i++) {
      s += p[i] * scratch[i];
    }
    return s / 2;
  }

  /// `p ~ N(0, M)`.
  void sampleMomentum(math.Random rng, Float64List p) {
    if (!dense) {
      for (var i = 0; i < k; i++) {
        p[i] = standardNormal(rng) / math.sqrt(inv[i]);
      }
      return;
    }
    // M⁻¹ = L Lᵀ, so p = L⁻ᵀ z has covariance (L Lᵀ)⁻¹ = M.
    final l = chol!;
    for (var i = 0; i < k; i++) {
      p[i] = standardNormal(rng);
    }
    for (var i = k - 1; i >= 0; i--) {
      var s = p[i];
      for (var j = i + 1; j < k; j++) {
        s -= l[j * k + i] * p[j];
      }
      p[i] = s / l[i * k + i];
    }
  }
}

/// Mutable state shared by one chain's transitions.
final class _Ctx {
  _Ctx(this.nuts, this.z, this.metric, this.rng)
    : k = z.q.length,
      eps = nuts.initialStepSize,
      _scratch = Float64List(z.q.length);

  final Nuts nuts;
  final int k;
  final math.Random rng;
  final Float64List _scratch;

  /// The chain's current point.
  _State z;
  _Metric metric;
  double eps;

  // Per transition.
  double h0 = 0;
  int nLeapfrog = 0;
  double sumMetroProb = 0;
  bool divergent = false;
  int depth = 0;
  double energy = 0;
  double get acceptStat => nLeapfrog == 0 ? 0 : sumMetroProb / nLeapfrog;

  // Set by each _buildTree call.
  bool _valid = true;

  /// The integrator's moving state during a transition.
  late _State _w;

  double hamiltonian(_State s) => s.u + metric.kinetic(s.p, _scratch);

  /// One leapfrog step of [s] with step [e]; one request.
  Iterable<EvalRequest> leapfrog(_State s, double e) sync* {
    for (var i = 0; i < k; i++) {
      s.p[i] -= e / 2 * s.g[i];
    }
    metric.velocity(s.p, _scratch);
    for (var i = 0; i < k; i++) {
      s.q[i] += e * _scratch[i];
    }
    final req = EvalRequest([Float64List.fromList(s.q)], gradient: true);
    yield req;
    s.u = req.values[0];
    s.g.setAll(0, req.gradients[0]);
    for (var i = 0; i < k; i++) {
      s.p[i] -= e / 2 * s.g[i];
    }
  }

  /// Stan's heuristic: double or halve the step size until one leapfrog
  /// step's acceptance probability crosses 0.8.
  Iterable<EvalRequest> initStepSize() sync* {
    final log08 = math.log(0.8);
    var direction = 0;
    for (var tries = 0; tries < 100; tries++) {
      final w = z.copy();
      metric.sampleMomentum(rng, w.p);
      final h0 = hamiltonian(w);
      yield* leapfrog(w, eps);
      var h = hamiltonian(w);
      if (h.isNaN) h = double.infinity;
      final dH = h0 - h;
      if (direction == 0) direction = dH > log08 ? 1 : -1;
      if (direction == 1 && !(dH > log08)) return;
      if (direction == -1 && !(dH < log08)) return;
      final next = direction == 1 ? 2 * eps : eps / 2;
      if (!(next > 1e-12 && next < 1e7)) return;
      eps = next;
    }
  }

  /// One NUTS transition from [z]; leaves the new point in [z].
  Iterable<EvalRequest> transition() sync* {
    _w = z.copy();
    metric.sampleMomentum(rng, _w.p);
    final zFwd = _w.copy(), zBck = _w.copy();
    final zSample = _w.copy(), zPropose = _w.copy();

    Float64List v() => Float64List(k);
    final pSharp0 = v();
    metric.velocity(_w.p, pSharp0);
    final pSharpFwdFwd = Float64List.fromList(pSharp0);
    final pSharpFwdBck = Float64List.fromList(pSharp0);
    final pSharpBckFwd = Float64List.fromList(pSharp0);
    final pSharpBckBck = Float64List.fromList(pSharp0);
    final pFwdFwd = Float64List.fromList(_w.p);
    final pFwdBck = Float64List.fromList(_w.p);
    final pBckFwd = Float64List.fromList(_w.p);
    final pBckBck = Float64List.fromList(_w.p);
    final rho = Float64List.fromList(_w.p);
    var logSumWeight = 0.0;

    h0 = hamiltonian(_w);
    nLeapfrog = 0;
    sumMetroProb = 0;
    divergent = false;
    depth = 0;

    final rhoFwd = v(), rhoBck = v(), ext = v();
    while (depth < nuts.maxTreeDepth) {
      rhoFwd.fillRange(0, k, 0);
      rhoBck.fillRange(0, k, 0);
      final lswSubtree = _Box(double.negativeInfinity);
      bool valid;
      if (rng.nextDouble() > 0.5) {
        _w.copyFrom(zFwd);
        rhoBck.setAll(0, rho);
        pBckFwd.setAll(0, pFwdBck);
        pSharpBckFwd.setAll(0, pSharpFwdBck);
        yield* _buildTree(
          depth,
          zPropose,
          pSharpFwdBck,
          pSharpFwdFwd,
          rhoFwd,
          pFwdBck,
          pFwdFwd,
          lswSubtree,
          1,
        );
        valid = _valid;
        zFwd.copyFrom(_w);
      } else {
        _w.copyFrom(zBck);
        rhoFwd.setAll(0, rho);
        pFwdBck.setAll(0, pBckFwd);
        pSharpFwdBck.setAll(0, pSharpBckFwd);
        yield* _buildTree(
          depth,
          zPropose,
          pSharpBckFwd,
          pSharpBckBck,
          rhoBck,
          pBckFwd,
          pBckBck,
          lswSubtree,
          -1,
        );
        valid = _valid;
        zBck.copyFrom(_w);
      }
      if (!valid) break;
      depth++;

      // Biased progressive sampling at the top level.
      if (lswSubtree.v > logSumWeight) {
        zSample.copyFrom(zPropose);
      } else if (rng.nextDouble() < math.exp(lswSubtree.v - logSumWeight)) {
        zSample.copyFrom(zPropose);
      }
      logSumWeight = _logSumExp(logSumWeight, lswSubtree.v);

      for (var i = 0; i < k; i++) {
        rho[i] = rhoBck[i] + rhoFwd[i];
      }
      var persist = _criterion(pSharpBckBck, pSharpFwdFwd, rho);
      for (var i = 0; i < k; i++) {
        ext[i] = rhoBck[i] + pFwdBck[i];
      }
      persist = persist && _criterion(pSharpBckBck, pSharpFwdBck, ext);
      for (var i = 0; i < k; i++) {
        ext[i] = rhoFwd[i] + pBckFwd[i];
      }
      persist = persist && _criterion(pSharpBckFwd, pSharpFwdFwd, ext);
      if (!persist) break;
    }
    z = zSample;
    energy = hamiltonian(z);
  }

  /// Builds a subtree of `2^depth` leapfrog steps from [_w] in direction
  /// [sign], as Stan's `build_tree`; sets [_valid].
  Iterable<EvalRequest> _buildTree(
    int depth,
    _State zPropose,
    Float64List pSharpBeg,
    Float64List pSharpEnd,
    Float64List rho,
    Float64List pBeg,
    Float64List pEnd,
    _Box logSumWeight,
    int sign,
  ) sync* {
    if (depth == 0) {
      yield* leapfrog(_w, sign * eps);
      nLeapfrog++;
      var h = hamiltonian(_w);
      if (h.isNaN) h = double.infinity;
      if (h - h0 > nuts.maxEnergyError) divergent = true;
      logSumWeight.v = _logSumExp(logSumWeight.v, h0 - h);
      sumMetroProb += h0 - h > 0 ? 1 : math.exp(h0 - h);
      zPropose.copyFrom(_w);
      metric.velocity(_w.p, pSharpBeg);
      pSharpEnd.setAll(0, pSharpBeg);
      for (var i = 0; i < k; i++) {
        rho[i] += _w.p[i];
      }
      pBeg.setAll(0, _w.p);
      pEnd.setAll(0, _w.p);
      _valid = !divergent;
      return;
    }

    // The first half.
    final lswInit = _Box(double.negativeInfinity);
    final pInitEnd = Float64List(k), pSharpInitEnd = Float64List(k);
    final rhoInit = Float64List(k);
    yield* _buildTree(
      depth - 1,
      zPropose,
      pSharpBeg,
      pSharpInitEnd,
      rhoInit,
      pBeg,
      pInitEnd,
      lswInit,
      sign,
    );
    if (!_valid) return;

    // The second half.
    final zProposeFinal = _w.copy();
    final lswFinal = _Box(double.negativeInfinity);
    final pFinalBeg = Float64List(k), pSharpFinalBeg = Float64List(k);
    final rhoFinal = Float64List(k);
    yield* _buildTree(
      depth - 1,
      zProposeFinal,
      pSharpFinalBeg,
      pSharpEnd,
      rhoFinal,
      pFinalBeg,
      pEnd,
      lswFinal,
      sign,
    );
    if (!_valid) return;

    // Multinomial sample from the two halves.
    final lswSubtree = _logSumExp(lswInit.v, lswFinal.v);
    logSumWeight.v = _logSumExp(logSumWeight.v, lswSubtree);
    if (lswFinal.v > lswSubtree) {
      zPropose.copyFrom(zProposeFinal);
    } else if (rng.nextDouble() < math.exp(lswFinal.v - lswSubtree)) {
      zPropose.copyFrom(zProposeFinal);
    }

    final rhoSubtree = Float64List(k);
    for (var i = 0; i < k; i++) {
      rhoSubtree[i] = rhoInit[i] + rhoFinal[i];
      rho[i] += rhoSubtree[i];
    }
    var persist = _criterion(pSharpBeg, pSharpEnd, rhoSubtree);
    final ext = Float64List(k);
    for (var i = 0; i < k; i++) {
      ext[i] = rhoInit[i] + pFinalBeg[i];
    }
    persist = persist && _criterion(pSharpBeg, pSharpFinalBeg, ext);
    for (var i = 0; i < k; i++) {
      ext[i] = rhoFinal[i] + pInitEnd[i];
    }
    persist = persist && _criterion(pSharpInitEnd, pSharpEnd, ext);
    _valid = persist;
  }

  static bool _criterion(
    Float64List pSharpMinus,
    Float64List pSharpPlus,
    Float64List rho,
  ) {
    var a = 0.0, b = 0.0;
    for (var i = 0; i < rho.length; i++) {
      a += pSharpPlus[i] * rho[i];
      b += pSharpMinus[i] * rho[i];
    }
    return a > 0 && b > 0;
  }
}

final class _Box {
  _Box(this.v);
  double v;
}

double _logSumExp(double a, double b) {
  if (a == double.negativeInfinity) return b;
  if (b == double.negativeInfinity) return a;
  final m = math.max(a, b);
  return m + math.log(math.exp(a - m) + math.exp(b - m));
}

/// Nesterov dual averaging of `log ε` (Hoffman & Gelman 2014, Stan's
/// constants).
final class _DualAveraging {
  _DualAveraging(this.delta);

  final double delta;
  static const double _gamma = 0.05, _t0 = 10, _kappa = 0.75;

  double _mu = 0, _sBar = 0, _xBar = 0;
  int _counter = 0;

  void restart(double eps) {
    _mu = math.log(10 * eps);
    _counter = 0;
    _sBar = 0;
    _xBar = 0;
  }

  /// The next step size after a transition with acceptance statistic [a].
  double learn(double a) {
    _counter++;
    final stat = a > 1 ? 1.0 : a;
    final eta = 1 / (_counter + _t0);
    _sBar = (1 - eta) * _sBar + eta * (delta - stat);
    final x = _mu - _sBar * math.sqrt(_counter.toDouble()) / _gamma;
    final xEta = math.pow(_counter, -_kappa).toDouble();
    _xBar = (1 - xEta) * _xBar + xEta * x;
    return math.exp(x);
  }

  double get finalStepSize => math.exp(_xBar);
}

/// Stan's warmup windows for the metric: an initial buffer of 75 iterations
/// (step size only), windows of 25, 50, 100, … (the last one stretched to
/// the end), then a terminal buffer of 50. Scaled down for short warmups.
final class _Windows {
  _Windows(this.warmup, {required this.adapt}) {
    if (warmup < 20) {
      _initBuffer = warmup;
      _termBuffer = 0;
      _windowSize = 0;
      _active = false;
      return;
    }
    if (_initBuffer + _termBuffer + _windowSize > warmup) {
      _initBuffer = (0.15 * warmup).floor();
      _termBuffer = (0.1 * warmup).floor();
      _windowSize = warmup - (_initBuffer + _termBuffer);
    }
    _nextEnd = _initBuffer + _windowSize - 1;
    _active = adapt;
  }

  final int warmup;
  final bool adapt;
  int _initBuffer = 75, _termBuffer = 50, _windowSize = 25;
  int _counter = 0, _nextEnd = 0;
  bool _active = false;

  bool get inWindow =>
      _active &&
      _counter >= _initBuffer &&
      _counter < warmup - _termBuffer &&
      _counter != warmup;

  bool get endOfWindow => _active && _counter == _nextEnd && _counter != warmup;

  void advance() {
    if (endOfWindow) _computeNext();
    _counter++;
  }

  void _computeNext() {
    final last = warmup - _termBuffer - 1;
    if (_nextEnd == last) return;
    _windowSize *= 2;
    _nextEnd = _counter + _windowSize;
    if (_nextEnd != last &&
        _nextEnd + 2 * _windowSize >= warmup - _termBuffer) {
      _nextEnd = last;
    }
  }
}

/// Welford's running mean and (co)variance of the warmup draws.
final class _CovarianceEstimator {
  _CovarianceEstimator(this.k, {required this.dense})
    : _mean = Float64List(k),
      _m2 = Float64List(dense ? k * k : k);

  final int k;
  final bool dense;
  final Float64List _mean, _m2;
  final Float64List _delta = Float64List(64);
  int _n = 0;

  void restart() {
    _n = 0;
    _mean.fillRange(0, k, 0);
    _m2.fillRange(0, _m2.length, 0);
  }

  void add(Float64List q) {
    _n++;
    final d = k <= 64 ? _delta : Float64List(k);
    for (var i = 0; i < k; i++) {
      d[i] = q[i] - _mean[i];
      _mean[i] += d[i] / _n;
    }
    if (!dense) {
      for (var i = 0; i < k; i++) {
        _m2[i] += d[i] * (q[i] - _mean[i]);
      }
      return;
    }
    for (var i = 0; i < k; i++) {
      for (var j = 0; j < k; j++) {
        _m2[i * k + j] += d[i] * (q[j] - _mean[j]);
      }
    }
  }

  /// The sample (co)variance shrunk towards `1e-3 · I` as Stan does:
  /// `n/(n+5) Σ + 1e-3 · 5/(n+5) I`.
  _Metric regularised() {
    final n = _n.toDouble();
    final inv = Float64List(_m2.length);
    final w = n / (n + 5), r = 1e-3 * 5 / (n + 5);
    for (var i = 0; i < _m2.length; i++) {
      inv[i] = w * _m2[i] / (n - 1);
    }
    for (var i = 0; i < k; i++) {
      inv[dense ? i * k + i : i] += r;
    }
    return _Metric(k, inv, dense: dense);
  }
}
