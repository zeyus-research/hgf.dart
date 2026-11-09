import 'open_gpu.dart';

Future<GpuOpened> openGpu() async =>
    (device: null, reason: 'No GPU support on this platform.');
