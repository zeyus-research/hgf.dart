import 'dart:math' as math;
import 'dart:typed_data';

import '../special.dart';

// MCMC convergence diagnostics after Vehtari, Gelman, Simpson, Carpenter &
// Bürkner (2021), "Rank-normalization, folding, and localization: an
// improved R̂", computed exactly as ArviZ (arviz_stats) does. Each takes
// the draws of one scalar as one list per chain, all of equal length.

/// Rank-normalised split R̂: the larger of the bulk R̂ (rank-normalised
/// split chains) and the tail R̂ (the same, folded around the median).
/// Values above 1.01 suggest the chains have not mixed. NaN with fewer than
/// two chains or four draws.
double rhat(List<Float64List> chains) {
  if (chains.length < 2 || _nDraws(chains) < 4) return double.nan;
  final split = _split(chains);
  final bulk = _rhat(_zScale(split));
  final med = _median(split);
  final folded = [
    for (final c in split)
      Float64List.fromList([for (final v in c) (v - med).abs()]),
  ];
  return math.max(bulk, _rhat(_zScale(folded)));
}

/// The bulk effective sample size: the ESS of the rank-normalised split
/// chains.
double essBulk(List<Float64List> chains) {
  if (_nDraws(chains) < 4) return double.nan;
  return _ess(_zScale(_split(chains)));
}

/// The tail effective sample size: the smaller ESS of the indicators
/// `x ≤ q₀.₀₅` and `x ≤ q₀.₉₅` over split chains.
double essTail(List<Float64List> chains, [double prob = 0.05]) {
  if (_nDraws(chains) < 4) return double.nan;
  final lo = math.min(prob, 1 - prob), hi = math.max(prob, 1 - prob);
  return math.min(_essQuantile(chains, lo), _essQuantile(chains, hi));
}

/// The effective sample size for the mean (split chains, no ranks).
double essMean(List<Float64List> chains) {
  if (_nDraws(chains) < 4) return double.nan;
  return _ess(_split(chains));
}

/// The Monte Carlo standard error of the mean: `sd / √essMean`.
double mcseMean(List<Float64List> chains) {
  final all = _flatten(chains);
  return _sd(all) / math.sqrt(essMean(chains));
}

/// The [p] quantile of all draws, linearly interpolated (NumPy's default).
double quantileOf(Float64List sorted, double p) {
  final n = sorted.length;
  if (n == 0) return double.nan;
  final h = (n - 1) * p;
  final lo = h.floor(), hi = math.min(lo + 1, n - 1);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

int _nDraws(List<Float64List> chains) =>
    chains.isEmpty ? 0 : chains.map((c) => c.length).reduce(math.min);

Float64List _flatten(List<Float64List> chains) {
  final n = chains.fold(0, (s, c) => s + c.length);
  final out = Float64List(n);
  var o = 0;
  for (final c in chains) {
    out.setRange(o, o + c.length, c);
    o += c.length;
  }
  return out;
}

double _sd(Float64List a) {
  var m = 0.0;
  for (final v in a) {
    m += v;
  }
  m /= a.length;
  var s = 0.0;
  for (final v in a) {
    s += (v - m) * (v - m);
  }
  return math.sqrt(s / (a.length - 1));
}

double _median(List<Float64List> chains) {
  final all = _flatten(chains)..sort();
  return quantileOf(all, 0.5);
}

/// Each chain's first and last halves as separate chains (the middle draw
/// of an odd-length chain is dropped).
List<Float64List> _split(List<Float64List> chains) {
  final half = _nDraws(chains) ~/ 2;
  return [
    for (final c in chains) Float64List.sublistView(c, 0, half),
    for (final c in chains) Float64List.sublistView(c, c.length - half),
  ];
}

/// Normal scores of the pooled ranks (ties averaged):
/// `Φ⁻¹((r − 3/8) / (S + 1/4))`.
List<Float64List> _zScale(List<Float64List> chains) {
  final all = _flatten(chains);
  final s = all.length;
  final idx = List<int>.generate(s, (i) => i)
    ..sort((a, b) => all[a].compareTo(all[b]));
  final z = Float64List(s);
  var i = 0;
  while (i < s) {
    var j = i;
    while (j + 1 < s && all[idx[j + 1]] == all[idx[i]]) {
      j++;
    }
    final rank = (i + j) / 2 + 1; // average of ranks i+1 .. j+1
    final v = normalQuantile((rank - 0.375) / (s + 0.25));
    for (var m = i; m <= j; m++) {
      z[idx[m]] = v;
    }
    i = j + 1;
  }
  final out = <Float64List>[];
  var o = 0;
  for (final c in chains) {
    out.add(Float64List.sublistView(z, o, o + c.length));
    o += c.length;
  }
  return out;
}

double _rhat(List<Float64List> chains) {
  final m = chains.length, n = chains.first.length;
  final means = Float64List(m);
  var w = 0.0;
  for (var c = 0; c < m; c++) {
    final x = chains[c];
    var mean = 0.0;
    for (final v in x) {
      mean += v;
    }
    mean /= n;
    means[c] = mean;
    var s = 0.0;
    for (final v in x) {
      s += (v - mean) * (v - mean);
    }
    w += s / (n - 1);
  }
  w /= m;
  final b = n * _variance(means);
  return math.sqrt((b / w + n - 1) / n);
}

/// Sample variance (`ddof = 1`).
double _variance(Float64List a) {
  final sd = _sd(a);
  return sd * sd;
}

double _essQuantile(List<Float64List> chains, double p) {
  final q = quantileOf(_flatten(chains)..sort(), p);
  final ind = [
    for (final c in chains)
      Float64List.fromList([for (final v in c) v <= q ? 1.0 : 0.0]),
  ];
  return _ess(_split(ind));
}

/// The multi-chain ESS with Geyer's initial monotone sequence (Stan's and
/// ArviZ's estimator), from biased per-chain autocovariances.
double _ess(List<Float64List> chains) {
  final m = chains.length, n = chains.first.length;
  var lo = double.infinity, hi = double.negativeInfinity;
  for (final c in chains) {
    for (final v in c) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (hi - lo < 1e-15) return (m * n).toDouble();
  if (lo.isNaN || hi.isNaN) return double.nan;

  final centred = <Float64List>[];
  final means = Float64List(m);
  for (var c = 0; c < m; c++) {
    final x = chains[c];
    var mean = 0.0;
    for (final v in x) {
      mean += v;
    }
    mean /= n;
    means[c] = mean;
    centred.add(Float64List.fromList([for (final v in x) v - mean]));
  }
  // Mean (over chains) autocovariance at lag t, computed on demand.
  double acov(int t) {
    var s = 0.0;
    for (final x in centred) {
      var a = 0.0;
      for (var i = 0; i + t < n; i++) {
        a += x[i] * x[i + t];
      }
      s += a / n;
    }
    return s / m;
  }

  final meanVar = acov(0) * n / (n - 1);
  var varPlus = meanVar * (n - 1) / n;
  if (m > 1) varPlus += _variance(means);

  final rho = Float64List(n);
  var rhoEven = 1.0;
  rho[0] = rhoEven;
  var rhoOdd = 1 - (meanVar - acov(1)) / varPlus;
  rho[1] = rhoOdd;

  // Geyer's initial positive sequence.
  var t = 1;
  while (t < n - 3 && rhoEven + rhoOdd > 0) {
    rhoEven = 1 - (meanVar - acov(t + 1)) / varPlus;
    rhoOdd = 1 - (meanVar - acov(t + 2)) / varPlus;
    if (rhoEven + rhoOdd >= 0) {
      rho[t + 1] = rhoEven;
      rho[t + 2] = rhoOdd;
    }
    t += 2;
  }
  final maxT = t - 2;
  if (rhoEven > 0) rho[maxT + 1] = rhoEven;

  // Geyer's initial monotone sequence.
  t = 1;
  while (t <= maxT - 2) {
    if (rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]) {
      rho[t + 1] = (rho[t - 1] + rho[t]) / 2;
      rho[t + 2] = rho[t + 1];
    }
    t += 2;
  }
  final total = (m * n).toDouble();
  var sum = 0.0;
  for (var i = 0; i <= maxT; i++) {
    sum += rho[i];
  }
  var tau = -1 + 2 * sum + (maxT + 1 < n ? rho[maxT + 1] : 0.0);
  tau = math.max(tau, 1 / (math.log(total) / math.ln10));
  for (final v in rho) {
    if (v.isNaN) return double.nan;
  }
  return total / tau;
}
