/// The expression IR of an HGF trial, shared by the GPU code generator
/// (`package:hgf_gpu`) and the CPU's reverse-mode evaluator: [Ex] trees
/// with forward-mode tangents ([Tangents]) and adjoints ([Adjoints]), and
/// [TrialCompiler], which writes one trial of a `Network` in it.
library;

export 'src/ir/eval.dart';
export 'src/ir/ir.dart';
export 'src/ir/response.dart';
export 'src/ir/trial.dart';
