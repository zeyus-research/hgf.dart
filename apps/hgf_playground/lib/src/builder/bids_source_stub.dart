import 'bids_source.dart';

/// No folders without dart:io.
BidsSource bidsFolder(String path) =>
    throw UnsupportedError('Pick the files instead of a folder');
