import 'dart:io';

/// Read a text file relative to the working directory (VM / AOT).
String readText(String path) => File(path).readAsStringSync();

/// Names of the entries in directory [path] (VM / AOT).
List<String> listDir(String path) => [
  for (final e in Directory(path).listSync())
    e.uri.pathSegments.lastWhere((s) => s.isNotEmpty),
];
