/// Deep HGF networks: layers of volatile, binary and categorical nodes
/// connected by learned weights, trained by belief propagation.
///
/// A port of pyhgf's vectorised `DeepNetwork`, checked numerically against
/// it (see `test/parity_test.dart`).
library;

export 'package:hgf/hgf.dart' show Coupling, CouplingFn, CustomCoupling;

export 'src/config.dart';
export 'src/deep_network.dart';
export 'src/init.dart';
export 'src/kernels.dart' show DeepSettings, Layer;
export 'src/matrix.dart';
export 'src/optimiser.dart';
export 'src/state.dart';
export 'src/trace.dart' hide TraceRecorder;
