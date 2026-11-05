// dart2wasm code generation, read from the disassembly (see run.sh):
//
// 1. `1 / 6` with int literals is not constant-folded: the function calls
//    the int-to-double conversion at run time instead of using 0.1666...
//    Writing `1.0 / 6.0` (or the literal) avoids it.
// 2. A top-level `final` lookup table is read through a lazy-initialising
//    getter on every access, even when its initialiser has no side effects
//    (hgf.dart: ~1.5% of the eHGF profile for its exp tables).
import 'dart:typed_data';

@pragma('wasm:never-inline')
double intDivision(double x) => x * (1 / 6) + x * x * (1 / 24);

@pragma('wasm:never-inline')
double doubleDivision(double x) => x * (1.0 / 6.0) + x * x * (1.0 / 24.0);

final Float64List table = Float64List.fromList([1.0, 2.0, 4.0, 8.0]);

@pragma('wasm:never-inline')
double tableLookup(int i) => table[i & 3];

void main(List<String> args) {
  final x = args.length + 0.5;
  print(intDivision(x) + doubleDivision(x) + tableLookup(args.length));
}
