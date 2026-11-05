import 'dart:js_interop';

@JS('process.getBuiltinModule')
external _Fs _getBuiltinModule(String name);

extension type _Fs(JSObject _) implements JSObject {
  external String readFileSync(String path, String encoding);
  external JSArray<JSString> readdirSync(String path);
}

/// Read a text file relative to the working directory (Node, JS or Wasm).
String readText(String path) =>
    _getBuiltinModule('fs').readFileSync(path, 'utf8');

/// Names of the entries in directory [path] (Node, JS or Wasm).
List<String> listDir(String path) => [
  for (final s in _getBuiltinModule('fs').readdirSync(path).toDart) s.toDart,
];
