/// The parts of JAX's random number generation that pyhgf's results depend
/// on: the threefry-2x32 PRNG (`jax_threefry_partitionable`, the default
/// since JAX 0.5), `random.split`, and float64 `random.normal`, which needs
/// XLA's float64 `erf_inv`.
///
/// pyhgf's Dirichlet process node draws its cluster proposals from
/// `random.key(42)` on every trial (always the same key, so always the same
/// draws). Reproducing them here is what makes the node's results comparable
/// with pyhgf's.
///
/// The draws match JAX to about 1 ulp, not bit for bit: XLA's CPU backend
/// evaluates the `erf_inv` polynomial with fused multiply-adds and has its
/// own `log1p`, and Dart has neither. Checked against JAX in
/// `test/dp_state_test.dart`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

const int _m32 = 0xFFFFFFFF;

/// A threefry key: two unsigned 32-bit words (`jax.random.key_data`).
typedef JaxKey = (int, int);

/// `jax.random.key(seed)` for a seed below 2³² (JAX puts the high word of
/// a 64-bit seed first).
JaxKey jaxKey(int seed) {
  RangeError.checkValueInInterval(seed, 0, _m32, 'seed');
  return (0, seed);
}

@pragma('vm:prefer-inline')
int _rotl(int v, int r) => ((v << r) & _m32) | (v >> (32 - r));

/// Threefry-2x32 with 20 rounds: the hash of the counter pair `(x0, x1)`
/// under [key], as JAX's `threefry2x32_p`. All values are unsigned 32-bit.
JaxKey threefry2x32(JaxKey key, int x0, int x1) {
  final (k0, k1) = key;
  final ks = [k0, k1, k0 ^ k1 ^ 0x1BD11BDA];
  const rotations = [
    [13, 15, 26, 6],
    [17, 29, 16, 24],
  ];
  var a = (x0 + k0) & _m32, b = (x1 + k1) & _m32;
  for (var i = 0; i < 5; i++) {
    for (final r in rotations[i & 1]) {
      a = (a + b) & _m32;
      b = _rotl(b, r) ^ a;
    }
    a = (a + ks[(i + 1) % 3]) & _m32;
    b = (b + ks[(i + 2) % 3] + i + 1) & _m32;
  }
  return (a, b);
}

/// `jax.random.split(key)`: two new keys (`[0]` is the one pyhgf carries
/// on, `[1]` the one it draws with).
(JaxKey, JaxKey) jaxSplit(JaxKey key) =>
    (threefry2x32(key, 0, 0), threefry2x32(key, 0, 1));

/// `jax.random.normal(key, (n,), dtype=float64)`: uniform draws on
/// `(-1, 1)` from 64 random bits each, through `√2 · erf⁻¹`.
Float64List jaxNormal(JaxKey key, int n) {
  // JAX's lower bound: the largest double above -1.
  const lo = -0.9999999999999999;
  const sqrt2 = 1.4142135623730951;
  final out = Float64List(n);
  final bits = ByteData(8);
  for (var i = 0; i < n; i++) {
    // Partitionable mode: element i hashes the 64-bit counter i, and its 64
    // bits are the two output words, high first.
    final (hi, lo32) = threefry2x32(key, 0, i);
    // The top 52 bits become the mantissa of a double in [1, 2). Built from
    // 32-bit halves, so it also works where ints are JS numbers.
    bits
      ..setUint32(4, 0x3FF00000 | (hi >> 12), Endian.little)
      ..setUint32(0, ((hi & 0xFFF) << 20) | (lo32 >> 12), Endian.little);
    final f = bits.getFloat64(0, Endian.little) - 1.0;
    final u = math.max(lo, f * (1.0 - lo) + lo);
    out[i] = sqrt2 * erfInv(u);
  }
  return out;
}

/// `log(1 + x)`, accurate near 0 (Goldberg's correction; Dart has no
/// `log1p`).
double log1p(double x) {
  final u = 1.0 + x;
  if (u == 1.0) return x;
  return math.log(u) * (x / (u - 1.0));
}

/// The inverse error function in float64, as XLA computes it (Giles,
/// "Approximating the erfinv function", the double-precision version):
/// three polynomials in `w = -log(1 - x²)`. ±∞ at ±1, NaN outside.
double erfInv(double x) {
  var w = -log1p(-x * x);
  final List<double> c;
  if (w < 6.25) {
    c = _erfInvW6;
    w -= 3.125;
  } else if (w < 16.0) {
    c = _erfInvW16;
    w = math.sqrt(w) - 3.25;
  } else {
    c = _erfInvWInf;
    w = math.sqrt(w) - 5.0;
  }
  var p = c[0];
  for (var i = 1; i < c.length; i++) {
    p = c[i] + p * w;
  }
  return x.abs() == 1.0 ? x * double.infinity : p * x;
}

const List<double> _erfInvW6 = [
  -3.6444120640178196996e-21,
  -1.685059138182016589e-19,
  1.2858480715256400167e-18,
  1.115787767802518096e-17,
  -1.333171662854620906e-16,
  2.0972767875968561637e-17,
  6.6376381343583238325e-15,
  -4.0545662729752068639e-14,
  -8.1519341976054721522e-14,
  2.6335093153082322977e-12,
  -1.2975133253453532498e-11,
  -5.4154120542946279317e-11,
  1.051212273321532285e-09,
  -4.1126339803469836976e-09,
  -2.9070369957882005086e-08,
  4.2347877827932403518e-07,
  -1.3654692000834678645e-06,
  -1.3882523362786468719e-05,
  0.0001867342080340571352,
  -0.00074070253416626697512,
  -0.0060336708714301490533,
  0.24015818242558961693,
  1.6536545626831027356,
];

const List<double> _erfInvW16 = [
  2.2137376921775787049e-09,
  9.0756561938885390979e-08,
  -2.7517406297064545428e-07,
  1.8239629214389227755e-08,
  1.5027403968909827627e-06,
  -4.013867526981545969e-06,
  2.9234449089955446044e-06,
  1.2475304481671778723e-05,
  -4.7318229009055733981e-05,
  6.8284851459573175448e-05,
  2.4031110387097893999e-05,
  -0.0003550375203628474796,
  0.00095328937973738049703,
  -0.0016882755560235047313,
  0.0024914420961078508066,
  -0.0037512085075692412107,
  0.005370914553590063617,
  1.0052589676941592334,
  3.0838856104922207635,
];

const List<double> _erfInvWInf = [
  -2.7109920616438573243e-11,
  -2.5556418169965252055e-10,
  1.5076572693500548083e-09,
  -3.7894654401267369937e-09,
  7.6157012080783393804e-09,
  -1.4960026627149240478e-08,
  2.9147953450901080826e-08,
  -6.7711997758452339498e-08,
  2.2900482228026654717e-07,
  -9.9298272942317002539e-07,
  4.5260625972231537039e-06,
  -1.9681778105531670567e-05,
  7.5995277030017761139e-05,
  -0.00021503011930044477347,
  -0.00013871931833623122026,
  1.0103004648645343977,
  4.8499064014085844221,
];
