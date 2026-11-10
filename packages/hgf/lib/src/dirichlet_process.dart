/// The cluster proposal of pyhgf's Dirichlet process node (`dp-state`): a
/// port of `get_candidate` and `likely_cluster_proposal` in
/// `pyhgf/updates/prediction_error/dirichlet.py`.
///
/// On every trial the node weighs [dpProposalCount] candidate clusters,
/// drawn once from the base distribution `μ ~ N(0, 10²)`, `σ ~ |N(0, 3²)|`
/// with JAX's `random.key(42)` (pyhgf draws them again each trial from the
/// same key, so they never change). Each proposal's weight combines how
/// little the existing clusters already explain it (specificity and
/// isolation), a preference for wide clusters (spread), and the likelihood
/// of the new observation under it. The candidate is the weighted average of
/// the [dpCandidatesKept] best proposals.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'jax_random.dart';
import 'math.dart' show fastExp;

/// The number of cluster proposals pyhgf samples (`n_samples`).
const int dpProposalCount = 20000;

/// How many of the best proposals form the candidate.
const int dpCandidatesKept = 1000;

/// The fixed proposals (from `random.key(42)`, as pyhgf draws them) and
/// the parts of their weights that depend on nothing else.
final class DpProposals {
  DpProposals._(this.mu, this.sigma)
    : _sigma2 = Float64List(mu.length),
      _logNorm = Float64List(mu.length),
      _selfDensity = Float64List(mu.length),
      _spread = Float64List(mu.length) {
    final n = mu.length;
    for (var j = 0; j < n; j++) {
      final s2 = sigma[j] * sigma[j];
      _sigma2[j] = s2;
      _logNorm[j] = math.log(_twoPi * s2);
      // pdf(μ_j; μ_j, σ_j): the proposal's density at its own mean.
      _selfDensity[j] = math.exp((_logNorm[j] + 0.0 / s2) * -0.5);
      // Cluster spread: pdf(1/σ²; 0, 5), wide clusters preferred.
      _spread[j] = jaxNormPdf(1.0 / s2, 0.0, 5.0);
    }
    _minMaxScale(_spread);
  }

  static DpProposals? _instance;

  /// The proposals, generated on first use (about 10 ms).
  static DpProposals get instance => _instance ??= _generate();

  static DpProposals _generate() {
    final (key, use) = jaxSplit(jaxKey(42));
    final mu = jaxNormal(use, dpProposalCount);
    for (var j = 0; j < mu.length; j++) {
      mu[j] = 10.0 * mu[j] + 0.0;
    }
    final sigma = jaxNormal(jaxSplit(key).$2, dpProposalCount);
    for (var j = 0; j < sigma.length; j++) {
      sigma[j] = (sigma[j] * 3.0).abs();
    }
    return DpProposals._(mu, sigma);
  }

  /// Proposal means and standard deviations.
  final Float64List mu, sigma;

  final Float64List _sigma2, _logNorm, _selfDensity, _spread;
}

const double _twoPi = 2.0 * math.pi;

/// `jax.scipy.stats.norm.pdf(x, loc, scale)`, with JAX's order of
/// operations: `exp((log(2π σ²) + (x − loc)² / σ²) / −2)`. (Here and below,
/// `/ −2` is written `· −0.5`: the same exact scaling, without a division.)
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
double jaxNormPdf(double x, double loc, double scale) {
  final s2 = scale * scale;
  final d = x - loc;
  return fastExp((math.log(_twoPi * s2) + d * d / s2) * -0.5);
}

/// `v -= v.min(); v /= v.max()` (NaN propagates, as in JAX).
void _minMaxScale(Float64List v) {
  var lo = double.infinity;
  for (final x in v) {
    if (x.isNaN || x < lo) lo = x;
    if (lo.isNaN) break;
  }
  var hi = double.negativeInfinity;
  for (var j = 0; j < v.length; j++) {
    final x = v[j] - lo;
    v[j] = x;
    if (hi.isNaN) continue;
    if (x.isNaN || x > hi) hi = x;
  }
  for (var j = 0; j < v.length; j++) {
    v[j] /= hi;
  }
}

/// Scratch for [dpCandidate], reused across trials (one per network), with
/// room for [clusters] clusters (all dp-state nodes together) in each of
/// [slots] row caches.
///
/// Each cache holds each cluster's two rows of proposal terms, keyed on the
/// cluster's expected mean and standard deviation: on most trials only the
/// cluster that learned changes, so the others' 2 × 20000 densities are
/// reused. (The sums over clusters are redone in the same order, so the
/// result is the same as without the cache, whichever slot is used.) A
/// batch gives each element it runs at once its own slot; the working
/// buffers are shared. 320 KB per cluster and slot.
final class DpScratch {
  DpScratch(this.clusters, {this.slots = 1})
    : _pre = Float64List(dpProposalCount),
      _iso = Float64List(dpProposalCount),
      _w = Float64List(dpProposalCount),
      _ll = Float64List(dpProposalCount),
      _idx = Int32List(dpProposalCount),
      _sample = Float64List(dpProposalCount ~/ _sampleStride),
      _rowPre = Float64List(slots * clusters * dpProposalCount),
      _rowIso = Float64List(slots * clusters * dpProposalCount),
      _rowMean = Float64List(slots * clusters)
        ..fillRange(0, slots * clusters, double.nan),
      _rowSigma = Float64List(slots * clusters)
        ..fillRange(0, slots * clusters, double.nan) {
    if (clusters < 1 || slots < 1) {
      throw ArgumentError('DpScratch needs at least one cluster and slot');
    }
  }

  /// Bytes of row cache per cluster and slot.
  static const int bytesPerRow = 2 * dpProposalCount * 8;

  /// How many clusters' rows fit in each slot.
  final int clusters;

  /// How many independent row caches there are.
  final int slots;

  final Float64List _pre, _iso, _w, _ll, _sample;
  final Int32List _idx;

  /// The log of the base measure's predictive density at the value of the
  /// last [dpCandidate] call: `log(Σⱼ N(x; μⱼ, σⱼ) / n)` over the
  /// proposals, a Monte Carlo estimate of `∫ N(x; θ) G₀(dθ)` (what
  /// [dpSurprise] weighs a new cluster by).
  double logBaseDensity = double.nan;

  // Per row r = slot * clusters + cluster: the cluster's density at each
  // proposal's mean, and the proposal's isolation factor from it; computed
  // for (_rowMean[r], _rowSigma[r]) (NaN: not yet).
  final Float64List _rowPre, _rowIso, _rowMean, _rowSigma;
}

/// Fill [s]'s cached row [i] (`slot * clusters + cluster`) at mean [m],
/// standard deviation [sd].
@pragma('vm:unsafe:no-bounds-checks')
void _clusterRows(DpProposals p, DpScratch s, int i, double m, double sd) {
  final mu = p.mu, sigma2 = p._sigma2, logNorm = p._logNorm;
  final n = mu.length, base = i * n;
  final rowPre = s._rowPre, rowIso = s._rowIso;
  final s2 = sd * sd, ln = math.log(_twoPi * s2);
  // The cluster's density at its own mean.
  final own = fastExp((ln + 0.0 / s2) * -0.5);
  for (var j = 0; j < n; j++) {
    final d = mu[j] - m;
    rowPre[base + j] = fastExp((ln + d * d / s2) * -0.5);
    final dj = m - mu[j];
    rowIso[base + j] =
        own / (own + fastExp((logNorm[j] + dj * dj / sigma2[j]) * -0.5));
  }
  s._rowMean[i] = m;
  s._rowSigma[i] = sd;
}

/// The new-cluster candidate `(mean, sigma)` for [value] given the existing
/// clusters' expected [means] and [sigmas] (all of them, active or not, as
/// pyhgf does), read from `[start, start + count)`; [s] must have room for
/// `start + count` clusters. The rows are cached in [s]'s [slot].
///
/// The steps are JAX's, in its order of operations, with two shortcuts that
/// give the same result: the min-max scalings use `max(v − min v) =
/// max v − min v` (rounding is monotonic), and a NaN anywhere in the
/// weights returns NaN at once (JAX's argsort puts NaNs last, so they are
/// among the kept proposals and the average is NaN).
@pragma('vm:unsafe:no-bounds-checks')
(double, double) dpCandidate(
  double value,
  double sensoryPrecision,
  Float64List means,
  Float64List sigmas,
  int start,
  int count,
  DpScratch s, {
  int slot = 0,
}) {
  if (start < 0 ||
      count < 1 ||
      slot < 0 ||
      slot >= s.slots ||
      start + count > s.clusters ||
      start + count > means.length ||
      start + count > sigmas.length) {
    throw RangeError('Clusters [$start, ${start + count}) out of range');
  }
  final p = DpProposals.instance;
  final mu = p.mu, sigma = p.sigma, sigma2 = p._sigma2, logNorm = p._logNorm;
  final self = p._selfDensity, spread = p._spread;
  final pre = s._pre, iso = s._iso, w = s._w, ll = s._ll;
  final rowPre = s._rowPre, rowIso = s._rowIso;
  // Every per-proposal list has this length (checked once here, so the
  // loops can skip bounds checks).
  const n = dpProposalCount;
  for (final v in [mu, sigma, sigma2, logNorm, self, spread, pre, iso, w, ll]) {
    if (v.length != n) throw StateError('Bad proposal buffers');
  }
  if (s._idx.length != n ||
      rowPre.length < s.slots * s.clusters * n ||
      s._rowIso.length != rowPre.length ||
      s._rowMean.length < s.slots * s.clusters ||
      s._rowSigma.length != s._rowMean.length) {
    throw StateError('Bad proposal buffers');
  }

  // 1. Specificity: the proposal's density at its own mean against the sum
  // of the densities the existing clusters give its mean. 2. Isolation: per
  // existing cluster, its density at its own mean against that plus the
  // proposal's density there, multiplied over the clusters.
  final row0 = slot * s.clusters;
  for (var i = start; i < start + count; i++) {
    final m = means[i], sd = sigmas[i], r = row0 + i;
    if (m != s._rowMean[r] || sd != s._rowSigma[r]) {
      _clusterRows(p, s, r, m, sd);
    }
    final base = r * n;
    if (i == start) {
      for (var j = 0; j < n; j++) {
        pre[j] = 0.0 + rowPre[base + j];
        iso[j] = 1.0 * rowIso[base + j];
      }
    } else {
      for (var j = 0; j < n; j++) {
        pre[j] += rowPre[base + j];
        iso[j] *= rowIso[base + j];
      }
    }
  }
  var rLo = double.infinity, rHi = double.negativeInfinity;
  var iLo = double.infinity, iHi = double.negativeInfinity;
  var nan = false;
  for (var j = 0; j < n; j++) {
    final r = self[j] / (self[j] + pre[j]);
    pre[j] = r;
    final q = iso[j];
    if (r < rLo) rLo = r;
    if (r > rHi) rHi = r;
    if (q < iLo) iLo = q;
    if (q > iHi) iHi = q;
    if (r != r || q != q) nan = true;
  }
  final rScale = rHi - rLo, iScale = iHi - iLo;

  // 3. The new observation's likelihood under each proposal, normalised,
  // to the power of the sensory precision.
  var sum = 0.0;
  for (var j = 0; j < n; j++) {
    w[j] = (pre[j] - rLo) / rScale * ((iso[j] - iLo) / iScale) * spread[j];
    final d = value - mu[j];
    final l = fastExp((logNorm[j] + d * d / sigma2[j]) * -0.5);
    ll[j] = l;
    sum += l;
  }
  s.logBaseDensity = sum > 0.0
      ? math.log(sum / n)
      : _logMeanDensity(value, mu, sigma2, logNorm);
  if (nan) return (double.nan, double.nan);
  if (sensoryPrecision == 1.0) {
    for (var j = 0; j < n; j++) {
      final x = w[j] * (ll[j] / sum);
      w[j] = x;
      if (x != x) nan = true;
    }
  } else {
    for (var j = 0; j < n; j++) {
      final x = w[j] * math.pow(ll[j] / sum, sensoryPrecision);
      w[j] = x;
      if (x != x) nan = true;
    }
  }
  if (nan) return (double.nan, double.nan);

  // 4. The weighted average of the best proposals.
  final idx = s._idx;
  final kept = _largest(w, idx, s._sample, dpCandidatesKept);
  var wSum = 0.0, muSum = 0.0, sigmaSum = 0.0;
  for (var k = kept - dpCandidatesKept; k < kept; k++) {
    final j = idx[k], wj = w[j];
    wSum += wj;
    muSum += mu[j] * wj;
    sigmaSum += sigma[j] * wj;
  }
  return (muSum / wSum, sigmaSum / wSum);
}

/// Put the indexes of the [k] largest of [w] (no NaNs), ties going to the
/// larger index as in JAX's stable `argsort(w)[-k:]`, at
/// `idx[kept - k .. kept)`, and return `kept`.
///
/// A cutoff from a stride sample of [w] (about 1.6 k values pass it) first
/// narrows the candidates; when fewer than [k] pass, all of [w] are used.
/// Either way every value at or above the k-th largest is a candidate, so
/// the selection is exact.
@pragma('vm:unsafe:no-bounds-checks')
int _largest(Float64List w, Int32List idx, Float64List sample, int k) {
  final n = w.length;
  if (idx.length != n || sample.length != n ~/ _sampleStride) {
    throw StateError('Bad selection buffers');
  }
  for (var t = 0; t < sample.length; t++) {
    sample[t] = w[t * _sampleStride];
  }
  sample.sort();
  // The sample's k/n quantile from the top is near the k-th largest; take
  // a lower one, so that the cutoff is below it with near certainty.
  final rank = sample.length - (sample.length * k * 8) ~/ (n * 5) - 1;
  final cutoff = sample[rank < 0 ? 0 : rank];
  var c = 0;
  for (var j = 0; j < n; j++) {
    if (w[j] >= cutoff) idx[c++] = j;
  }
  if (c < k) {
    for (var j = 0; j < n; j++) {
      idx[j] = j;
    }
    c = n;
  }
  _selectLargest(idx, c, w, k);
  return c;
}

/// Every how many weights [_largest] samples one for its cutoff.
const int _sampleStride = 20;

/// Whether proposal [a] sorts after [b] in JAX's stable ascending argsort
/// of [w] (no NaNs): by weight, then by index.
@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
bool _after(Float64List w, int a, int b) {
  final x = w[a], y = w[b];
  return x > y || (x == y && a > b);
}

/// Partition `idx[0 .. length)` so that its last [k] entries are the
/// largest (in any order): quickselect.
@pragma('vm:unsafe:no-bounds-checks')
void _selectLargest(Int32List idx, int length, Float64List w, int k) {
  final target = length - k;
  var lo = 0, hi = length - 1;
  while (lo < hi) {
    // Median of three as the pivot, moved to hi.
    final mid = lo + ((hi - lo) >> 1);
    if (_after(w, idx[lo], idx[mid])) _swap(idx, lo, mid);
    if (_after(w, idx[mid], idx[hi])) _swap(idx, mid, hi);
    if (_after(w, idx[lo], idx[mid])) _swap(idx, lo, mid);
    _swap(idx, mid, hi);
    final pivot = idx[hi];
    var store = lo;
    for (var j = lo; j < hi; j++) {
      if (_after(w, pivot, idx[j])) _swap(idx, j, store++);
    }
    _swap(idx, store, hi);
    if (store == target) return;
    if (store < target) {
      lo = store + 1;
    } else {
      hi = store - 1;
    }
  }
}

@pragma('vm:prefer-inline')
@pragma('wasm:prefer-inline')
void _swap(Int32List a, int i, int j) {
  final t = a[i];
  a[i] = a[j];
  a[j] = t;
}

/// The surprise of [value] under a dp-state node's predictive distribution
/// (the Chinese-restaurant process mixture), before the value is assigned:
///
/// `−log(Σᵢ nᵢ/(α+N) · N(x; μᵢ, σᵢ) + α/(α+N) · p₀(x))`
///
/// over the open clusters `i` (`active[i] = 1`), where
/// `p₀(x) = ∫ N(x; θ) G₀(dθ)` is the base measure's predictive density
/// ([DpScratch.logBaseDensity], passed as [logBaseDensity]). Cluster `i` is
/// read at `start + i * stride` of [means], [sigmas], [counts] and [active]
/// (`stride` is 1 for a `Network` and the batch width for a
/// `BatchNetwork`).
///
/// **An hgf.dart extension:** pyhgf defines no surprise for a dp-state
/// node. The new-cluster term uses the base measure, not the candidate
/// cluster the node would open: the candidate is fitted to `x` itself, so
/// weighing `x` by it rewards opening a cluster per observation (on three
/// well-separated clusters, α = 10⁴ would score better than α = 0.1).
/// Computed in log space (log-sum-exp), so it stays finite far from every
/// cluster, where the densities underflow. NaN if any term is NaN.
double dpSurprise(
  double value,
  double alpha,
  double nTotal,
  Float64List means,
  Float64List sigmas,
  Float64List counts,
  Float64List active,
  int start,
  int stride,
  int count,
  double logBaseDensity,
) {
  // Log weights and densities; the normaliser log(α + N) is added at the
  // end. Terms with zero weight (closed or empty clusters) are left out.
  final base = math.log(alpha) + logBaseDensity;
  if (base.isNaN) return double.nan;
  var hi = base;
  // Two passes over the (few) clusters: the largest term, then the sum.
  for (var i = 0, q = start; i < count; i++, q += stride) {
    final w = counts[q] * active[q];
    if (w.isNaN) return double.nan;
    if (w == 0.0) continue;
    final l = math.log(w) + _normLogPdf(value, means[q], sigmas[q]);
    if (l.isNaN) return double.nan;
    if (l > hi) hi = l;
  }
  if (hi == double.negativeInfinity) return double.infinity;
  var sum = math.exp(base - hi);
  for (var i = 0, q = start; i < count; i++, q += stride) {
    final w = counts[q] * active[q];
    if (w == 0.0) continue;
    sum += math.exp(math.log(w) + _normLogPdf(value, means[q], sigmas[q]) - hi);
  }
  return math.log(alpha + nTotal) - hi - math.log(sum);
}

/// `log N(x; loc, scale)`.
double _normLogPdf(double x, double loc, double scale) {
  final s2 = scale * scale, d = x - loc;
  return (math.log(_twoPi * s2) + d * d / s2) * -0.5;
}

/// `log(Σⱼ N(x; μⱼ, σⱼ) / n)` in log space, for when every density
/// underflows.
double _logMeanDensity(
  double x,
  Float64List mu,
  Float64List sigma2,
  Float64List logNorm,
) {
  final n = mu.length;
  var hi = double.negativeInfinity;
  for (var j = 0; j < n; j++) {
    final d = x - mu[j];
    final l = (logNorm[j] + d * d / sigma2[j]) * -0.5;
    if (l.isNaN) return double.nan;
    if (l > hi) hi = l;
  }
  if (hi == double.negativeInfinity) return hi;
  var sum = 0.0;
  for (var j = 0; j < n; j++) {
    final d = x - mu[j];
    sum += math.exp((logNorm[j] + d * d / sigma2[j]) * -0.5 - hi);
  }
  return hi + math.log(sum / n);
}
