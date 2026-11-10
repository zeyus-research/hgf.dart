import 'dart:js_interop';

@JS('fetch')
external JSPromise<JSAny?> _fetch(String url, JSObject init);

void postReport(String url, String body) {
  _fetch(url, {'method': 'POST', 'body': body}.jsify()! as JSObject);
}
