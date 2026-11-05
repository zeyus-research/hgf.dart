/// Micro-probes of runtime costs that drive engine design decisions. Re-run
/// on new Dart releases (especially for dart2wasm SIMD support):
///
///     dart compile exe benchmark/bin/runtime_probe.dart -o probe && ./probe
///     dart compile wasm -O4 benchmark/bin/runtime_probe.dart -o probe.wasm
///     node ../../tool/web/run_wasm.mjs probe.wasm
///
/// Results on 2026-11-05 (i7-7700, Dart 3.13.7, Node 26). "Chained" calls
/// each wait on the previous result (latency, as in Halley iterations);
/// the others are independent (throughput).
///
/// | probe                       | AOT    | Wasm -O4 |
/// |-----------------------------|--------|----------|
/// | math.exp, 5M calls          | 29 ms  | 56 ms    |  <- a JS import on wasm
/// | math.exp chained, 5M        | 50 ms  | 125 ms   |
/// | tableExp, 5M calls          | 41 ms  | 41 ms    |  (math.dart; used on wasm)
/// | tableExp chained, 5M        | 92 ms  | 87 ms    |
/// | math.log, 5M calls          | 37 ms  | 51 ms    |
/// | math.log chained, 5M        | 75 ms  | 135 ms   |
/// | pure-Dart exp, 5M calls     | 168 ms | 200 ms   |  (degree-11 Horner loop)
/// | scalar f32 mul-add, 4M      | 9.6 ms | 8.0 ms   |
/// | Float32x4 mul-add, 4M       | 3.7 ms | 108 ms   |  <- emulated on wasm
/// | scalar f64 mul-add, 4M      | 8.0 ms | 6.6 ms   |
/// | Float64x2 mul-add, 4M       | 6.5 ms | 103 ms   |  <- emulated on wasm
library;

import 'dart:math' as math;
import 'dart:typed_data';

// ignore: implementation_imports
import 'package:hgf/src/math.dart' show tableExp;

final _bits = ByteData(8);

/// 1/k! for k = 11..0 (Horner order).
const _taylor = [
  1 / 39916800, 1 / 3628800, 1 / 362880, 1 / 40320, 1 / 5040, 1 / 720, //
  1 / 120, 1 / 24, 1 / 6, 0.5, 1.0, 1.0,
];

/// exp via Cody-Waite reduction + degree-11 polynomial (~1e-14 rel. error).
double dartExp(double x) {
  const ln2hi = 6.93147180369123816490e-01, ln2lo = 1.90821492927058770002e-10;
  const invLn2 = 1.44269504088896338700e+00;
  final k = (x * invLn2).roundToDouble();
  final r = (x - k * ln2hi) - k * ln2lo;
  var p = 0.0;
  for (final c in _taylor) {
    p = p * r + c;
  }
  _bits.setInt64(0, (k.toInt() + 1023) << 52);
  return p * _bits.getFloat64(0);
}

void _time(String name, double Function() f) {
  f();
  final sw = Stopwatch()..start();
  final v = f();
  print(
    '${name.padRight(28)} ${(sw.elapsedMicroseconds / 1000).toStringAsFixed(1).padLeft(7)} ms'
    '  (checksum ${v.toStringAsFixed(3)})',
  );
}

void main() {
  const n = 5000000;
  final xs = Float64List(n);
  for (var i = 0; i < n; i++) {
    xs[i] = -5 + 10 * (i / n);
  }
  _time('math.exp x5M', () {
    var s = 0.0;
    for (var i = 0; i < n; i++) {
      s += math.exp(xs[i]);
    }
    return s;
  });
  _time('math.log x5M', () {
    var s = 0.0;
    for (var i = 0; i < n; i++) {
      s += math.log(xs[i] + 6);
    }
    return s;
  });
  _time('pure-Dart exp x5M', () {
    var s = 0.0;
    for (var i = 0; i < n; i++) {
      s += dartExp(xs[i]);
    }
    return s;
  });
  _time('tableExp x5M', () {
    var s = 0.0;
    for (var i = 0; i < n; i++) {
      s += tableExp(xs[i]);
    }
    return s;
  });
  // Chained: each call waits on the previous result (x = exp(-x) converges
  // to the omega constant), as in Halley iterations. Latency, not throughput.
  _time('math.exp chained x5M', () {
    var x = 0.5;
    for (var i = 0; i < n; i++) {
      x = math.exp(-x);
    }
    return x;
  });
  _time('tableExp chained x5M', () {
    var x = 0.5;
    for (var i = 0; i < n; i++) {
      x = tableExp(-x);
    }
    return x;
  });
  _time('math.log chained x5M', () {
    var x = 2.0;
    for (var i = 0; i < n; i++) {
      x = 1.0 + math.log(x + 1.0);
    }
    return x;
  });

  const m = 4000000;
  final a = Float32List(m), b = Float32List(m), out = Float32List(m);
  for (var i = 0; i < m; i++) {
    a[i] = i * 1e-6;
    b[i] = 1 - i * 1e-7;
  }
  final a4 = Float32x4List.view(a.buffer),
      b4 = Float32x4List.view(b.buffer),
      out4 = Float32x4List.view(out.buffer);
  _time('scalar f32 mul-add 4M', () {
    for (var i = 0; i < m; i++) {
      out[i] = a[i] * b[i] + a[i];
    }
    return out[m - 1].toDouble();
  });
  _time('Float32x4 mul-add 4M', () {
    for (var i = 0; i < m ~/ 4; i++) {
      out4[i] = a4[i] * b4[i] + a4[i];
    }
    return out[m - 1].toDouble();
  });
  final a2 = Float64List(m), b2 = Float64List(m), o2 = Float64List(m);
  for (var i = 0; i < m; i++) {
    a2[i] = a[i];
    b2[i] = b[i];
  }
  final a2v = Float64x2List.view(a2.buffer),
      b2v = Float64x2List.view(b2.buffer),
      o2v = Float64x2List.view(o2.buffer);
  _time('scalar f64 mul-add 4M', () {
    for (var i = 0; i < m; i++) {
      o2[i] = a2[i] * b2[i] + a2[i];
    }
    return o2[m - 1];
  });
  _time('Float64x2 mul-add 4M', () {
    for (var i = 0; i < m ~/ 2; i++) {
      o2v[i] = a2v[i] * b2v[i] + a2v[i];
    }
    return o2[m - 1];
  });
}
