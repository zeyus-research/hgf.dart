// D7: an Expando whose only instantiation in the program is Expando<int>
// can't read back what it stored on dart2wasm ("The Wasm reference is not
// a Dart object"). dart2js and the VM print "ok: 3"; so does dart2wasm once
// the program also instantiates an Expando of a class type, e.g.
// `Expando<Object>()`. Run by run.sh.
void main() {
  final k = Object(), e = Expando<int>();
  e[k] = 3;
  try {
    print('ok: ${e[k]}');
  } catch (err) {
    print('fail: $err');
  }
}
