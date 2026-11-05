// dart2wasm: a typed-data view anywhere in the program slows down reads of
// lists that are never views.
//
// `Engine`'s columns only ever hold `Float64List(n)`. With
// `-Dview=true`, an unrelated function also receives a
// `Float64List.sublistView`; `Engine.kernel` then loads `_offsetInElements`
// for its lists (see run.sh, which counts the loads in the disassembly and
// times both builds). Views are hard to avoid in real apps (`sublistView`,
// `buffer.asFloat64List`, packages); hgf.dart measured 8-10% on its engine,
// whose per-node steps are short loops like this one.
import 'dart:typed_data';

const bool withView = bool.fromEnvironment('view');

final class Engine {
  Engine(int n) : a = Float64List(n), b = Float64List(n);

  // Not final, as in an engine that re-allocates on rebuild.
  Float64List a, b;

  @pragma('wasm:never-inline')
  double kernel() {
    var s = 0.0;
    final a = this.a, b = this.b;
    for (var i = 0; i < a.length; i++) {
      a[i] = a[i] * b[i] + 1e-3;
      s += a[i];
    }
    return s;
  }
}

@pragma('wasm:never-inline')
double unrelated(Float64List x) => x[0];

void main(List<String> args) {
  final e = Engine(8);
  for (var i = 0; i < 8; i++) {
    e.b[i] = 0.5;
  }
  var sink = unrelated(Float64List(4));
  if (withView) sink += unrelated(Float64List.sublistView(Float64List(8), 2));
  const calls = 20000000;
  final ns = <double>[];
  for (var k = 0; k < 7; k++) {
    final sw = Stopwatch()..start();
    for (var c = 0; c < calls; c++) {
      sink += e.kernel();
    }
    ns.add(sw.elapsedMicroseconds * 1e3 / calls);
  }
  ns.sort();
  print(
    'view in program: $withView  ${ns[3].toStringAsFixed(2)} ns/call '
    '(median of 7)  [$sink]',
  );
}
