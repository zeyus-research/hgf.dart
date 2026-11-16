/// Model documents: one JSON format (`hgf.document/1`) for a whole model as
/// the playground's builder edits it, with nodes by id, named inputs and
/// outputs, the response model, free parameters, the data layout and the
/// fit settings. It compiles to a `Network` and an `HgfModel`, reads
/// subjects from CSV by column name, and feeds the code generators.
library;

export 'src/document/network_document.dart';
export 'src/document/model_document.dart';
