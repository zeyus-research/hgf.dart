import 'dart:math' as math;
import 'dart:typed_data';

/// The lower Cholesky factor `L` (row-major `k x k`, `A = L Lᵀ`) of the
/// symmetric matrix [a], or null if [a] isn't (numerically) positive
/// definite or has non-finite entries.
Float64List? cholesky(Float64List a, int k) {
  final l = Float64List(k * k);
  for (var i = 0; i < k; i++) {
    for (var j = 0; j <= i; j++) {
      var s = a[i * k + j];
      for (var m = 0; m < j; m++) {
        s -= l[i * k + m] * l[j * k + m];
      }
      if (i == j) {
        if (!(s > 0) || !s.isFinite) return null;
        l[i * k + i] = math.sqrt(s);
      } else {
        l[i * k + j] = s / l[j * k + j];
      }
    }
  }
  return l;
}

/// `log det A` from its Cholesky factor [l].
double logDetFromCholesky(Float64List l, int k) {
  var s = 0.0;
  for (var i = 0; i < k; i++) {
    s += math.log(l[i * k + i]);
  }
  return 2 * s;
}

/// `A⁻¹` from the Cholesky factor [l] of `A`.
Float64List inverseFromCholesky(Float64List l, int k) {
  // L⁻¹ by forward substitution, then A⁻¹ = L⁻ᵀ L⁻¹.
  final li = Float64List(k * k);
  for (var j = 0; j < k; j++) {
    li[j * k + j] = 1 / l[j * k + j];
    for (var i = j + 1; i < k; i++) {
      var s = 0.0;
      for (var m = j; m < i; m++) {
        s -= l[i * k + m] * li[m * k + j];
      }
      li[i * k + j] = s / l[i * k + i];
    }
  }
  final out = Float64List(k * k);
  for (var i = 0; i < k; i++) {
    for (var j = 0; j <= i; j++) {
      var s = 0.0;
      for (var m = i; m < k; m++) {
        s += li[m * k + i] * li[m * k + j];
      }
      out[i * k + j] = s;
      out[j * k + i] = s;
    }
  }
  return out;
}
