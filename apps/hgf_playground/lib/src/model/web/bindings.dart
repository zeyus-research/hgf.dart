// The few browser APIs the fit host worker and its page side use (the app
// doesn't depend on package:web).
import 'dart:js_interop';
import 'dart:typed_data';

@JS('Worker')
extension type JSWorker._(JSObject _) implements JSObject {
  external factory JSWorker(String url, [WorkerOptions? options]);
  external void postMessage(JSAny? message);
  external void terminate();
  external set onmessage(JSFunction? handler);
  external set onerror(JSFunction? handler);
}

extension type WorkerOptions._(JSObject _) implements JSObject {
  external factory WorkerOptions({String? type, String? name});
}

extension type MessageEvent._(JSObject _) implements JSObject {
  external JSAny? get data;
}

extension type ErrorEvent._(JSObject _) implements JSObject {
  external JSAny? get message;
  external void preventDefault();
}

@JS('globalThis')
external GlobalScope get globalScope;

extension type GlobalScope._(JSObject _) implements JSObject {
  @JS('Worker')
  external JSAny? get workerConstructor;
  external Document? get document;
  external Location? get location;

  // Inside a worker.
  external set onmessage(JSFunction? handler);
  external void postMessage(JSAny? message);
  external void close();
}

extension type Document._(JSObject _) implements JSObject {
  external String? get baseURI;
}

extension type Location._(JSObject _) implements JSObject {
  external String get href;
}

@JS('Blob')
extension type Blob._(JSObject _) implements JSObject {
  external factory Blob(JSArray<JSString> parts, BlobOptions options);
}

extension type BlobOptions._(JSObject _) implements JSObject {
  external factory BlobOptions({String type});
}

@JS('URL')
extension type URL._(JSObject _) implements JSObject {
  external static String createObjectURL(Blob blob);
  external static void revokeObjectURL(String url);
}

/// The page's base URL (in a worker, its script's URL).
String? get baseUrl =>
    globalScope.document?.baseURI ?? globalScope.location?.href;

/// Message parts: arrays cross as JS typed arrays, and come back as
/// Dart-native lists (the filters' hot loops read them; a JS-backed list is
/// slow to read on dart2wasm).
JSArray<JSFloat64Array> arraysToJS(List<Float64List> arrays) =>
    [for (final a in arrays) a.toJS].toJS;

List<Float64List> arraysFromJS(JSAny? arrays) => [
  for (final a in (arrays! as JSArray<JSFloat64Array>).toDart)
    Float64List.fromList(a.toDart),
];

String str(JSAny? v) => (v! as JSString).toDart;
int integer(JSAny? v) => (v! as JSNumber).toDartInt;
