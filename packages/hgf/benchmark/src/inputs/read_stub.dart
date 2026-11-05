/// Fallback for platforms with neither `dart:io` nor JS interop.
String readText(String path) => throw UnsupportedError('No file access');

/// Fallback for platforms with neither `dart:io` nor JS interop.
List<String> listDir(String path) => throw UnsupportedError('No file access');
