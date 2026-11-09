import 'dart:math' as math;
import 'dart:typed_data';

/// A weight initialisation strategy (pyhgf's `weight_initialisation`).
///
/// The draws use `dart:math`'s `Random`, so they follow the same
/// distributions as pyhgf's but not the same values.
sealed class WeightInit {
  const WeightInit();

  /// `n_children × n_parents` values, row-major.
  Float64List draw(int nParents, int nChildren, math.Random rng);
}

/// Uniform on ±√(6 / (fan-in + fan-out)).
final class XavierInit extends WeightInit {
  const XavierInit();

  @override
  Float64List draw(int nParents, int nChildren, math.Random rng) {
    final limit = math.sqrt(6.0 / (nParents + nChildren));
    return Float64List.fromList([
      for (var k = 0; k < nParents * nChildren; k++)
        -limit + 2 * limit * rng.nextDouble(),
    ]);
  }
}

/// Normal with standard deviation √(2 / fan-in).
final class HeInit extends WeightInit {
  const HeInit();

  @override
  Float64List draw(int nParents, int nChildren, math.Random rng) {
    final std = math.sqrt(2.0 / nParents);
    return Float64List.fromList([
      for (var k = 0; k < nParents * nChildren; k++) std * _normal(rng),
    ]);
  }
}

/// A (semi-)orthogonal matrix times [gain]: orthonormal columns when there
/// are at least as many children as parents, orthonormal rows otherwise.
final class OrthogonalInit extends WeightInit {
  const OrthogonalInit({this.gain = 1.0});

  final double gain;

  @override
  Float64List draw(int nParents, int nChildren, math.Random rng) {
    final tall = nChildren >= nParents;
    final rows = tall ? nChildren : nParents;
    final cols = tall ? nParents : nChildren;
    final q = _orthonormalColumns(rows, cols, rng);
    final out = Float64List(nParents * nChildren);
    for (var i = 0; i < nChildren; i++) {
      for (var j = 0; j < nParents; j++) {
        out[i * nParents + j] =
            gain * (tall ? q[i * cols + j] : q[j * cols + i]);
      }
    }
    return out;
  }
}

/// A fraction `1 − sparsity` of the weights drawn from N(0, std²), the rest
/// zero.
final class SparseInit extends WeightInit {
  const SparseInit({this.sparsity = 0.9, this.std = 0.01});

  final double sparsity;
  final double std;

  @override
  Float64List draw(int nParents, int nChildren, math.Random rng) {
    final size = nParents * nChildren;
    final out = Float64List(size);
    final nonZero = math.max(1, ((1.0 - sparsity) * size).round());
    final indices = List<int>.generate(size, (i) => i)..shuffle(rng);
    for (var k = 0; k < nonZero; k++) {
      out[indices[k]] = std * _normal(rng);
    }
    return out;
  }
}

double _normal(math.Random rng) {
  // Box–Muller.
  final u1 = 1.0 - rng.nextDouble();
  final u2 = rng.nextDouble();
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2);
}

/// A `rows × cols` (rows ≥ cols) matrix with orthonormal columns: the Q of
/// a Gaussian matrix's QR decomposition, signs fixed so that the
/// distribution is Haar.
Float64List _orthonormalColumns(int rows, int cols, math.Random rng) {
  final a = Float64List.fromList([
    for (var k = 0; k < rows * cols; k++) _normal(rng),
  ]);
  // Modified Gram–Schmidt, column by column.
  for (var j = 0; j < cols; j++) {
    for (var p = 0; p < j; p++) {
      var dot = 0.0;
      for (var i = 0; i < rows; i++) {
        dot += a[i * cols + p] * a[i * cols + j];
      }
      for (var i = 0; i < rows; i++) {
        a[i * cols + j] -= dot * a[i * cols + p];
      }
    }
    var norm = 0.0;
    for (var i = 0; i < rows; i++) {
      norm += a[i * cols + j] * a[i * cols + j];
    }
    norm = math.sqrt(norm);
    for (var i = 0; i < rows; i++) {
      a[i * cols + j] /= norm;
    }
  }
  return a;
}
