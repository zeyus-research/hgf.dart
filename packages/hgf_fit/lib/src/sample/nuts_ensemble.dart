part of 'nuts.dart';

/// Many NUTS chains advanced together, each to its next leapfrog step, with
/// every chain's state in one flat buffer and Stan's recursive tree
/// building run as an explicit stack.
///
/// It runs the same algorithm as [Nuts.chain], in the same order and with
/// the same random draws, so each chain's draws are bit-identical to that
/// generator's. What it removes is the per-chain cost of the generators
/// (a suspended `sync*` stack per chain and tree level) and of the vectors
/// allocated per transition and subtree, which grew with the number of
/// chains and dominated GPU sampling at thousands of chains.
///
/// A driver alternates [request] (how many points are pending, in
/// [points], each belonging to subject [pendingSubjects]) and [answer]
/// (their objective `U` and gradient), until [request] returns 0.
final class NutsEnsemble {
  NutsEnsemble(
    this.nuts,
    this.dimension, {
    required this.warmup,
    required this.draws,
  }) : _layout = _Layout(dimension, nuts.maxTreeDepth) {
    if (warmup < 0) throw ArgumentError.value(warmup, 'warmup');
    if (draws < 0) throw ArgumentError.value(draws, 'draws');
    if (dimension < 1) throw ArgumentError.value(dimension, 'dimension');
  }

  final Nuts nuts;

  /// Parameters per point (`k`).
  final int dimension;

  final int warmup, draws;
  final _Layout _layout;
  final List<_Chain> _chains = [];
  List<_Chain> _pending = const [];
  bool _started = false;

  /// The pending points from the last [request], row-major `n x k`.
  Float64List points = Float64List(0);

  /// Which subject each pending point belongs to.
  Int32List pendingSubjects = Int32List(0);

  /// Number of chains.
  int get length => _chains.length;

  /// Adds a chain for [subject] that starts from the first point [init]
  /// returns with a finite density and gradient (unconstrained scale),
  /// draws its randomness from [random] and writes to [out].
  void add({
    required Float64List Function(int attempt) init,
    required math.Random random,
    required ChainOutput out,
    int subject = 0,
  }) {
    if (_started) throw StateError('The ensemble has started');
    _chains.add(_Chain(this, init, random, out, subject));
  }

  /// Advances the chains to their next evaluation and returns how many
  /// points are pending (0 once every chain has finished).
  int request() {
    if (!_started) {
      _started = true;
      _pending = [
        for (final c in _chains)
          if (c.advance()) c,
      ];
    }
    final n = _pending.length, k = dimension;
    points = Float64List(n * k);
    pendingSubjects = Int32List(n);
    for (var j = 0; j < n; j++) {
      final c = _pending[j];
      points.setRange(j * k, (j + 1) * k, c.b, c.pending);
      pendingSubjects[j] = c.subject;
    }
    return n;
  }

  /// The objective `U` at each pending point (`+∞` where it isn't finite)
  /// and the gradients, row-major `n x k`.
  void answer(Float64List values, Float64List gradients) {
    final k = dimension;
    var live = 0;
    for (var j = 0; j < _pending.length; j++) {
      final c = _pending[j];
      final s = c.pending;
      c.b[s + 3 * k] = values[j];
      c.b.setRange(s + 2 * k, s + 3 * k, gradients, j * k);
      if (c.advance()) _pending[live++] = c;
    }
    _pending.length = live;
  }
}

/// Offsets into a chain's buffer.
final class _Layout {
  _Layout(this.k, int maxTreeDepth) {
    var o = 0;
    int state() => (o += 3 * k + 1) - (3 * k + 1);
    int vec() => (o += k) - k;
    z = state();
    w = state();
    zFwd = state();
    zBck = state();
    zSample = state();
    zPropose = state();
    scratch = vec();
    pSharpFwdFwd = vec();
    pSharpFwdBck = vec();
    pSharpBckFwd = vec();
    pSharpBckBck = vec();
    pFwdFwd = vec();
    pFwdBck = vec();
    pBckFwd = vec();
    pBckBck = vec();
    rho = vec();
    rhoFwd = vec();
    rhoBck = vec();
    rhoSubtree = vec();
    ext = vec();
    lswSubtree = o++;
    // Tree level d ≥ 1 (the locals of a subtree of depth d); level 0 is a
    // leaf, which has none.
    levels = maxTreeDepth < 1 ? 1 : maxTreeDepth;
    levelBase = o;
    levelSize = 3 * k + 1 + 6 * k + 2;
    o += levels * levelSize;
    size = o;
  }

  final int k;
  late final int z, w, zFwd, zBck, zSample, zPropose;
  late final int scratch;
  late final int pSharpFwdFwd, pSharpFwdBck, pSharpBckFwd, pSharpBckBck;
  late final int pFwdFwd, pFwdBck, pBckFwd, pBckBck;
  late final int rho, rhoFwd, rhoBck, rhoSubtree, ext, lswSubtree;
  late final int levels, levelBase, levelSize, size;

  int level(int d) => levelBase + d * levelSize;

  // Within a level: zProposeFinal (a state), then the vectors, then the two
  // log weights.
  int zProposeFinal(int lvl) => lvl;
  int pInitEnd(int lvl) => lvl + 3 * k + 1;
  int pSharpInitEnd(int lvl) => lvl + 4 * k + 1;
  int rhoInit(int lvl) => lvl + 5 * k + 1;
  int pFinalBeg(int lvl) => lvl + 6 * k + 1;
  int pSharpFinalBeg(int lvl) => lvl + 7 * k + 1;
  int rhoFinal(int lvl) => lvl + 8 * k + 1;
  int lswInit(int lvl) => lvl + 9 * k + 1;
  int lswFinal(int lvl) => lvl + 9 * k + 2;
}

// Where a chain resumes.
const int _pcInit = 0,
    _pcInitAnswer = 1,
    _pcStepSizeBegin = 2,
    _pcStepSizeTry = 3,
    _pcStepSizeAnswer = 4,
    _pcAdaptStart = 5,
    _pcIterate = 6,
    _pcTransition = 7,
    _pcSubtree = 8,
    _pcTree = 9,
    _pcSubtreeEnd = 10,
    _pcTransitionEnd = 11,
    _pcWindowStepSize = 12,
    _pcWarmupTail = 13,
    _pcDone = 14;

final double _log08 = math.log(0.8);

// A tree frame: 7 buffer offsets (zPropose, pSharpBeg, pSharpEnd, rho,
// pBeg, pEnd, logSumWeight), then the frame's state.
const int _frame = 8;

/// One chain of a [NutsEnsemble]: [Nuts.chain] as a state machine.
final class _Chain {
  _Chain(this.ens, this.init, this.rng, this.out, this.subject)
    : nuts = ens.nuts,
      k = ens.dimension,
      l = ens._layout,
      b = Float64List(ens._layout.size),
      frames = Int32List(ens._layout.levels * _frame),
      eps = ens.nuts.initialStepSize,
      metric = _Metric.unit(
        ens.dimension,
        dense: ens.nuts.metric == MetricKind.dense,
      );

  final NutsEnsemble ens;
  final Nuts nuts;
  final int k, subject;
  final _Layout l;
  final Float64List b;
  final Int32List frames;
  final Float64List Function(int attempt) init;
  final math.Random rng;
  final ChainOutput out;

  _Metric metric;
  double eps;
  late _DualAveraging da;
  late _Windows window;
  late _CovarianceEstimator est;

  int pc = _pcInit;

  /// The state whose `q` waits for evaluation.
  int pending = 0;

  int attempt = 0, it = 0;

  // Step-size search.
  int ssReturn = 0, ssDirection = 0, ssTries = 0;
  double ssH0 = 0;

  // Per transition.
  double h0 = 0, sumMetroProb = 0, energy = 0, logSumWeight = 0;
  int nLeapfrog = 0, depth = 0, sign = 1, top = 0;
  bool divergent = false, valid = true;

  double get acceptStat => nLeapfrog == 0 ? 0 : sumMetroProb / nLeapfrog;

  /// Runs until the chain needs a point evaluated (true; its `q` is at
  /// [pending]) or has finished (false).
  bool advance() {
    while (true) {
      switch (pc) {
        case _pcInit:
          if (attempt >= nuts.maxInitAttempts) {
            out._fail(
              'no finite starting point in ${nuts.maxInitAttempts} attempts',
            );
            pc = _pcDone;
            return false;
          }
          final q = init(attempt);
          if (q.length != k) {
            throw ArgumentError('Starting points must have $k entries');
          }
          b.setRange(l.z, l.z + k, q);
          pending = l.z;
          pc = _pcInitAnswer;
          return true;
        case _pcInitAnswer:
          final z = l.z;
          var finite = b[z + 3 * k].isFinite;
          for (var i = 0; i < k && finite; i++) {
            finite = b[z + 2 * k + i].isFinite;
          }
          if (!finite) {
            attempt++;
            pc = _pcInit;
            continue;
          }
          // The generator starts from a fresh state: p = 0.
          b.fillRange(z + k, z + 2 * k, 0);
          out._start(k, ens.draws);
          ssReturn = _pcAdaptStart;
          pc = _pcStepSizeBegin;
        case _pcStepSizeBegin:
          ssDirection = 0;
          ssTries = 0;
          pc = _pcStepSizeTry;
        case _pcStepSizeTry:
          if (ssTries >= 100) {
            pc = ssReturn;
            continue;
          }
          _copyState(l.z, l.w);
          _sampleMomentum(l.w + k);
          ssH0 = _hamiltonian(l.w);
          _leapfrogStart(l.w, eps);
          pending = l.w;
          pc = _pcStepSizeAnswer;
          return true;
        case _pcStepSizeAnswer:
          _leapfrogEnd(l.w, eps);
          var h = _hamiltonian(l.w);
          if (h.isNaN) h = double.infinity;
          final dH = ssH0 - h;
          if (ssDirection == 0) ssDirection = dH > _log08 ? 1 : -1;
          if ((ssDirection == 1 && !(dH > _log08)) ||
              (ssDirection == -1 && !(dH < _log08))) {
            pc = ssReturn;
            continue;
          }
          final next = ssDirection == 1 ? 2 * eps : eps / 2;
          if (!(next > 1e-12 && next < 1e7)) {
            pc = ssReturn;
            continue;
          }
          eps = next;
          ssTries++;
          pc = _pcStepSizeTry;
        case _pcAdaptStart:
          da = _DualAveraging(nuts.targetAccept)..restart(eps);
          window = _Windows(ens.warmup, adapt: nuts.metric != MetricKind.unit);
          est = _CovarianceEstimator(k, dense: nuts.metric == MetricKind.dense);
          it = 0;
          pc = _pcIterate;
        case _pcIterate:
          if (it >= ens.warmup + ens.draws) {
            out
              ..stepSize = eps
              ..inverseMetric = Float64List.fromList(metric.inv);
            pc = _pcDone;
            return false;
          }
          pc = _pcTransition;
        case _pcTransition:
          _beginTransition();
          pc = _pcSubtree;
        case _pcSubtree:
          if (depth >= nuts.maxTreeDepth) {
            pc = _pcTransitionEnd;
            continue;
          }
          _beginSubtree();
          pc = _pcTree;
        case _pcTree:
          if (_runTree()) return true;
        case _pcSubtreeEnd:
          pc = _endSubtree() ? _pcSubtree : _pcTransitionEnd;
        case _pcTransitionEnd:
          _copyState(l.zSample, l.z);
          energy = _hamiltonian(l.z);
          out.iterations++;
          if (it >= ens.warmup) {
            _record(it - ens.warmup);
            it++;
            pc = _pcIterate;
            continue;
          }
          eps = da.learn(acceptStat);
          if (window.inWindow) {
            est.add(Float64List.sublistView(b, l.z, l.z + k));
          }
          if (window.endOfWindow) {
            metric = est.regularised();
            est.restart();
            ssReturn = _pcWindowStepSize;
            pc = _pcStepSizeBegin;
            continue;
          }
          pc = _pcWarmupTail;
        case _pcWindowStepSize:
          da.restart(eps);
          pc = _pcWarmupTail;
        case _pcWarmupTail:
          window.advance();
          if (it == ens.warmup - 1) eps = da.finalStepSize;
          out.warmupDivergences += divergent ? 1 : 0;
          it++;
          pc = _pcIterate;
        case _pcDone:
          return false;
      }
    }
  }

  void _beginTransition() {
    final w = l.w;
    _copyState(l.z, w);
    _sampleMomentum(w + k);
    _copyState(w, l.zFwd);
    _copyState(w, l.zBck);
    _copyState(w, l.zSample);
    _copyState(w, l.zPropose);
    _velocity(w + k, l.pSharpFwdFwd);
    _copy(l.pSharpFwdFwd, l.pSharpFwdBck);
    _copy(l.pSharpFwdFwd, l.pSharpBckFwd);
    _copy(l.pSharpFwdFwd, l.pSharpBckBck);
    _copy(w + k, l.pFwdFwd);
    _copy(w + k, l.pFwdBck);
    _copy(w + k, l.pBckFwd);
    _copy(w + k, l.pBckBck);
    _copy(w + k, l.rho);
    logSumWeight = 0;
    h0 = _hamiltonian(w);
    nLeapfrog = 0;
    sumMetroProb = 0;
    divergent = false;
    depth = 0;
  }

  void _beginSubtree() {
    b.fillRange(l.rhoFwd, l.rhoFwd + k, 0);
    b.fillRange(l.rhoBck, l.rhoBck + k, 0);
    b[l.lswSubtree] = double.negativeInfinity;
    if (rng.nextDouble() > 0.5) {
      sign = 1;
      _copyState(l.zFwd, l.w);
      _copy(l.rho, l.rhoBck);
      _copy(l.pFwdBck, l.pBckFwd);
      _copy(l.pSharpFwdBck, l.pSharpBckFwd);
      _push(
        depth,
        l.zPropose,
        l.pSharpFwdBck,
        l.pSharpFwdFwd,
        l.rhoFwd,
        l.pFwdBck,
        l.pFwdFwd,
        l.lswSubtree,
      );
    } else {
      sign = -1;
      _copyState(l.zBck, l.w);
      _copy(l.rho, l.rhoFwd);
      _copy(l.pBckFwd, l.pFwdBck);
      _copy(l.pSharpBckFwd, l.pSharpFwdBck);
      _push(
        depth,
        l.zPropose,
        l.pSharpBckFwd,
        l.pSharpBckBck,
        l.rhoBck,
        l.pBckFwd,
        l.pBckBck,
        l.lswSubtree,
      );
    }
  }

  /// After the subtree: whether the trajectory goes on.
  bool _endSubtree() {
    _copyState(l.w, sign == 1 ? l.zFwd : l.zBck);
    if (!valid) return false;
    depth++;

    // Biased progressive sampling at the top level.
    final lsw = b[l.lswSubtree];
    if (lsw > logSumWeight) {
      _copyState(l.zPropose, l.zSample);
    } else if (rng.nextDouble() < math.exp(lsw - logSumWeight)) {
      _copyState(l.zPropose, l.zSample);
    }
    logSumWeight = _logSumExp(logSumWeight, lsw);

    for (var i = 0; i < k; i++) {
      b[l.rho + i] = b[l.rhoBck + i] + b[l.rhoFwd + i];
    }
    var persist = _criterion(l.pSharpBckBck, l.pSharpFwdFwd, l.rho);
    for (var i = 0; i < k; i++) {
      b[l.ext + i] = b[l.rhoBck + i] + b[l.pFwdBck + i];
    }
    persist = persist && _criterion(l.pSharpBckBck, l.pSharpFwdBck, l.ext);
    for (var i = 0; i < k; i++) {
      b[l.ext + i] = b[l.rhoFwd + i] + b[l.pBckFwd + i];
    }
    persist = persist && _criterion(l.pSharpBckFwd, l.pSharpFwdFwd, l.ext);
    return persist;
  }

  void _push(
    int level,
    int zPropose,
    int pSharpBeg,
    int pSharpEnd,
    int rho,
    int pBeg,
    int pEnd,
    int lsw,
  ) {
    top = level;
    final f = level * _frame;
    frames[f] = zPropose;
    frames[f + 1] = pSharpBeg;
    frames[f + 2] = pSharpEnd;
    frames[f + 3] = rho;
    frames[f + 4] = pBeg;
    frames[f + 5] = pEnd;
    frames[f + 6] = lsw;
    frames[f + 7] = 0;
  }

  /// Runs the subtree's frames until a leaf needs its point evaluated
  /// (true) or the subtree is done (false; [valid] says how it ended).
  bool _runTree() {
    final e = sign * eps;
    while (true) {
      final f = top * _frame;
      final st = frames[f + 7];
      if (top == 0) {
        if (st == 0) {
          _leapfrogStart(l.w, e);
          pending = l.w;
          frames[f + 7] = 1;
          return true;
        }
        _leapfrogEnd(l.w, e);
        _leaf(f);
      } else {
        final lvl = l.level(top);
        if (st == 0) {
          b[l.lswInit(lvl)] = double.negativeInfinity;
          final rhoInit = l.rhoInit(lvl);
          b.fillRange(rhoInit, rhoInit + k, 0);
          frames[f + 7] = 1;
          _push(
            top - 1,
            frames[f],
            frames[f + 1],
            l.pSharpInitEnd(lvl),
            rhoInit,
            frames[f + 4],
            l.pInitEnd(lvl),
            l.lswInit(lvl),
          );
          continue;
        }
        if (st == 1 && valid) {
          _copyState(l.w, l.zProposeFinal(lvl));
          b[l.lswFinal(lvl)] = double.negativeInfinity;
          final rhoFinal = l.rhoFinal(lvl);
          b.fillRange(rhoFinal, rhoFinal + k, 0);
          frames[f + 7] = 2;
          _push(
            top - 1,
            l.zProposeFinal(lvl),
            l.pSharpFinalBeg(lvl),
            frames[f + 2],
            rhoFinal,
            l.pFinalBeg(lvl),
            frames[f + 5],
            l.lswFinal(lvl),
          );
          continue;
        }
        if (st == 2 && valid) _merge(f, lvl);
      }
      // Return from this frame.
      if (top == depth) {
        pc = _pcSubtreeEnd;
        return false;
      }
      top++;
    }
  }

  /// A leaf after its leapfrog step.
  void _leaf(int f) {
    final w = l.w;
    nLeapfrog++;
    var h = _hamiltonian(w);
    if (h.isNaN) h = double.infinity;
    if (h - h0 > nuts.maxEnergyError) divergent = true;
    final lsw = frames[f + 6];
    b[lsw] = _logSumExp(b[lsw], h0 - h);
    sumMetroProb += h0 - h > 0 ? 1 : math.exp(h0 - h);
    _copyState(w, frames[f]);
    final pSharpBeg = frames[f + 1];
    _velocity(w + k, pSharpBeg);
    _copy(pSharpBeg, frames[f + 2]);
    final rho = frames[f + 3];
    for (var i = 0; i < k; i++) {
      b[rho + i] += b[w + k + i];
    }
    _copy(w + k, frames[f + 4]);
    _copy(w + k, frames[f + 5]);
    valid = !divergent;
  }

  /// Joins a subtree's two halves (both valid).
  void _merge(int f, int lvl) {
    final lswInit = b[l.lswInit(lvl)], lswFinal = b[l.lswFinal(lvl)];
    final lswSubtree = _logSumExp(lswInit, lswFinal);
    final lsw = frames[f + 6];
    b[lsw] = _logSumExp(b[lsw], lswSubtree);
    if (lswFinal > lswSubtree) {
      _copyState(l.zProposeFinal(lvl), frames[f]);
    } else if (rng.nextDouble() < math.exp(lswFinal - lswSubtree)) {
      _copyState(l.zProposeFinal(lvl), frames[f]);
    }

    final rho = frames[f + 3], rhoSub = l.rhoSubtree;
    final rhoInit = l.rhoInit(lvl), rhoFinal = l.rhoFinal(lvl);
    for (var i = 0; i < k; i++) {
      b[rhoSub + i] = b[rhoInit + i] + b[rhoFinal + i];
      b[rho + i] += b[rhoSub + i];
    }
    final pSharpBeg = frames[f + 1], pSharpEnd = frames[f + 2];
    var persist = _criterion(pSharpBeg, pSharpEnd, rhoSub);
    final ext = l.ext, pFinalBeg = l.pFinalBeg(lvl);
    for (var i = 0; i < k; i++) {
      b[ext + i] = b[rhoInit + i] + b[pFinalBeg + i];
    }
    persist = persist && _criterion(pSharpBeg, l.pSharpFinalBeg(lvl), ext);
    final pInitEnd = l.pInitEnd(lvl);
    for (var i = 0; i < k; i++) {
      b[ext + i] = b[rhoFinal + i] + b[pInitEnd + i];
    }
    persist = persist && _criterion(l.pSharpInitEnd(lvl), pSharpEnd, ext);
    valid = persist;
  }

  void _record(int d) {
    final z = l.z;
    out.y.setRange(d * k, (d + 1) * k, b, z);
    out.logDensity[d] = -b[z + 3 * k];
    out.acceptStat[d] = acceptStat;
    out.energy[d] = energy;
    out.treeDepth[d] = depth;
    out.leapfrogSteps[d] = nLeapfrog;
    out.divergent[d] = divergent ? 1 : 0;
  }

  // Vector helpers on buffer offsets.

  void _copy(int from, int to) => b.setRange(to, to + k, b, from);

  void _copyState(int from, int to) => b.setRange(to, to + 3 * k + 1, b, from);

  /// The first half of a leapfrog step of the state at [s] with step [e]:
  /// a half kick and the drift, leaving the new `q` to evaluate.
  void _leapfrogStart(int s, double e) {
    final p = s + k, g = s + 2 * k;
    for (var i = 0; i < k; i++) {
      b[p + i] -= e / 2 * b[g + i];
    }
    _velocity(p, l.scratch);
    for (var i = 0; i < k; i++) {
      b[s + i] += e * b[l.scratch + i];
    }
  }

  /// The second half kick, with the new gradient in place.
  void _leapfrogEnd(int s, double e) {
    final p = s + k, g = s + 2 * k;
    for (var i = 0; i < k; i++) {
      b[p + i] -= e / 2 * b[g + i];
    }
  }

  double _hamiltonian(int s) {
    final p = s + k, v = l.scratch;
    _velocity(p, v);
    var sum = 0.0;
    for (var i = 0; i < k; i++) {
      sum += b[p + i] * b[v + i];
    }
    return b[s + 3 * k] + sum / 2;
  }

  /// `b[out..] = M⁻¹ b[p..]`.
  void _velocity(int p, int out) {
    final inv = metric.inv;
    if (!metric.dense) {
      for (var i = 0; i < k; i++) {
        b[out + i] = inv[i] * b[p + i];
      }
      return;
    }
    for (var i = 0; i < k; i++) {
      var s = 0.0;
      for (var j = 0; j < k; j++) {
        s += inv[i * k + j] * b[p + j];
      }
      b[out + i] = s;
    }
  }

  void _sampleMomentum(int p) {
    final inv = metric.inv;
    if (!metric.dense) {
      for (var i = 0; i < k; i++) {
        b[p + i] = standardNormal(rng) / math.sqrt(inv[i]);
      }
      return;
    }
    final c = metric.chol!;
    for (var i = 0; i < k; i++) {
      b[p + i] = standardNormal(rng);
    }
    for (var i = k - 1; i >= 0; i--) {
      var s = b[p + i];
      for (var j = i + 1; j < k; j++) {
        s -= c[j * k + i] * b[p + j];
      }
      b[p + i] = s / c[i * k + i];
    }
  }

  bool _criterion(int pSharpMinus, int pSharpPlus, int rho) {
    var a = 0.0, c = 0.0;
    for (var i = 0; i < k; i++) {
      a += b[pSharpPlus + i] * b[rho + i];
      c += b[pSharpMinus + i] * b[rho + i];
    }
    return a > 0 && c > 0;
  }
}
