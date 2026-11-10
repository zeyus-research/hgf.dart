/// Hierarchical Gaussian Filter (HGF) networks in pure Dart.
///
/// A port of the nodalised engine of [pyhgf](https://github.com/ComputationalPsychiatry/pyhgf),
/// checked numerically against it (see `test/parity_test.dart`).
library;

export 'src/coupling.dart' show Coupling, CouplingFn, CustomCoupling;
export 'src/math.dart'
    show
        binarySurprise,
        binarySurpriseFinitePrecision,
        binarySurpriseDerivative,
        digamma,
        dirichletKullbackLeibler,
        erfc,
        gammaSign,
        gaussianDensity,
        gaussianSurprise,
        lambertW0,
        lambertW0Log,
        logGamma,
        logAddExp,
        sigmoid,
        sigmoidInverseTemperature;
export 'src/network.dart'
    show BatchNetwork, Network, NetworkState, SurprisePartials;
export 'src/response.dart';
export 'src/spec.dart' show NetworkSpec;
export 'src/trajectories.dart' show Trajectories;
export 'src/types.dart';
