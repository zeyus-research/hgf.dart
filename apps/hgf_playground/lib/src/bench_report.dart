import 'bench_report_stub.dart'
    if (dart.library.js_interop) 'bench_report_web.dart'
    as impl;

/// POSTs [body] to [url] (web only): `&report=/path` hands bench results to
/// a harness that can't read the console (headless Firefox).
void postReport(String url, String body) => impl.postReport(url, body);
