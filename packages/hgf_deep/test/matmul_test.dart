import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf_deep/src/kernels.dart';
import 'package:test/test.dart';

/// A zeroed view that starts 8 bytes into its buffer: not
/// 16-byte aligned, so [matMul] takes its scalar path.
Float64List _misaligned(int n) =>
    Float64List.sublistView(Float64List(n + 1), 1);

void main() {
  // The SIMD tile (native targets) and the scalar tile must agree bit for
  // bit, for both orientations, every row and sample remainder.
  test('matMul: SIMD and scalar tiles give the same bits', () {
    final rng = math.Random(7);
    for (final rows in [1, 2, 3, 4, 5, 7, 8, 13]) {
      for (final depth in [1, 3, 8, 17]) {
        for (final b in [2, 4, 6, 8, 10, 12, 30]) {
          for (final transposed in [false, true]) {
            final stride = (transposed ? rows : depth) + 1;
            final w = Float64List((transposed ? depth : rows) * stride);
            for (var k = 0; k < w.length; k++) {
              w[k] = rng.nextDouble() * 2 - 1;
            }
            final (rs, cs) = transposed ? (1, stride) : (stride, 1);
            final a = Float64List(depth * b);
            final aMis = _misaligned(depth * b);
            for (var k = 0; k < a.length; k++) {
              aMis[k] = a[k] = rng.nextDouble() * 2 - 1;
            }
            final fast = Float64List(rows * b);
            final slow = _misaligned(rows * b);
            matMul(fast, w, rs, cs, rows, depth, a, b);
            matMul(slow, w, rs, cs, rows, depth, aMis, b);
            for (var k = 0; k < fast.length; k++) {
              if (fast[k] != slow[k]) {
                fail(
                  'rows $rows depth $depth b $b transposed $transposed: '
                  'element $k ${fast[k]} != ${slow[k]}',
                );
              }
            }
            // Spot check against the definition.
            final r = rows - 1, s = b - 1;
            var e = 0.0;
            for (var k = 0; k < depth; k++) {
              e += w[r * rs + k * cs] * a[k * b + s];
            }
            expect(fast[r * b + s], e);
          }
        }
      }
    }
  });
}
