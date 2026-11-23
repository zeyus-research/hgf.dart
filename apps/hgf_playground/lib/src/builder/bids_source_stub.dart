import 'bids_source.dart';

/// No folders without dart:io.
Future<BidsSource> bidsFolder(Uri uri) async =>
    throw UnsupportedError('Pick the files instead of a folder');
