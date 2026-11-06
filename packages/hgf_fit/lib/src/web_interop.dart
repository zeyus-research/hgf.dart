// Minimal browser bindings shared by the Web Worker transport and the
// worker entry point (no dependency on package:web).
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

/// The global scope (a window, or a worker's `self`).
@JS('globalThis')
external GlobalScope get globalScope;

extension type GlobalScope._(JSObject _) implements JSObject {
  @JS('Worker')
  external JSAny? get workerConstructor;
  external Navigator? get navigator;
  external Document? get document;
  external Location? get location;

  // Inside a worker.
  external set onmessage(JSFunction? handler);
  external void postMessage(JSAny? message);
  external void close();
}

extension type Navigator._(JSObject _) implements JSObject {
  external int? get hardwareConcurrency;
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
  external factory URL(String url, [String base]);
  external String get href;
  external static String createObjectURL(Blob blob);
  external static void revokeObjectURL(String url);
}

// Conversions. JS typed arrays are copied into Dart-native lists: the
// filter's hot loops read them, and on dart2wasm a JS-backed list is slow
// to read.

JSAny f64ToJS(Float64List v) => v.toJS;

JSAny i32ToJS(Int32List v) => v.toJS;

Float64List f64FromJS(JSAny? v) =>
    Float64List.fromList((v as JSFloat64Array).toDart);

Float64List? f64OrNullFromJS(JSAny? v) => v == null ? null : f64FromJS(v);

Int32List i32FromJS(JSAny? v) => Int32List.fromList((v as JSInt32Array).toDart);

int intFromJS(JSAny? v) => (v as JSNumber).toDartInt;

bool boolFromJS(JSAny? v) => (v as JSBoolean).toDart;

String stringFromJS(JSAny? v) => (v as JSString).toDart;

// Protocol between `FitWorkers` and `runFitWorker` (main → worker message
// tags; a worker announces itself with [readyMessage], or a failed Wasm
// start with [failedPrefix] and the error).
const tagFamily = 0;
const tagModel = 1;
const tagSubjects = 2;
const tagDrop = 3;
const tagEvaluate = 4;
const readyMessage = 'hgf_fit:ready';
const failedPrefix = 'hgf_fit:failed:';
