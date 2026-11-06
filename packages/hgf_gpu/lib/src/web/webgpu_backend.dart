import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import '../batch.dart';

// The subset of WebGPU that GpuBatch needs (package:web 1.1 only has the
// flag constants).

@JS('navigator.gpu')
external _Gpu? get _navigatorGpu;

extension type _Gpu._(JSObject _) implements JSObject {
  external JSPromise<_Adapter?> requestAdapter([JSObject? options]);
}

extension type _Adapter._(JSObject _) implements JSObject {
  external JSPromise<_Device> requestDevice([JSObject? descriptor]);
  external _Limits get limits;
  external _AdapterInfo get info;
}

extension type _AdapterInfo._(JSObject _) implements JSObject {
  external String get vendor;
  external String get architecture;
  external String get device;
  external String get description;
}

extension type _Limits._(JSObject _) implements JSObject {
  external double get maxStorageBufferBindingSize;
  external double get maxBufferSize;
}

extension type _Device._(JSObject _) implements JSObject {
  external _Queue get queue;
  external _Limits get limits;
  external _Buffer createBuffer(JSObject descriptor);
  external _ShaderModule createShaderModule(JSObject descriptor);
  external JSPromise<_Pipeline> createComputePipelineAsync(JSObject descriptor);
  external JSObject createBindGroup(JSObject descriptor);
  external _Encoder createCommandEncoder();
}

extension type _Queue._(JSObject _) implements JSObject {
  external void writeBuffer(
    _Buffer buffer,
    int offset,
    JSObject data, [
    int dataOffset,
    int size,
  ]);
  external void submit(JSArray<JSObject> buffers);
  external JSPromise<JSAny?> onSubmittedWorkDone();
}

extension type _Buffer._(JSObject _) implements JSObject {
  external JSPromise<JSAny?> mapAsync(int mode, [int offset, int size]);
  external JSArrayBuffer getMappedRange([int offset, int size]);
  external void unmap();
  external void destroy();
}

extension type _ShaderModule._(JSObject _) implements JSObject {
  external JSPromise<_CompilationInfo> getCompilationInfo();
}

extension type _CompilationInfo._(JSObject _) implements JSObject {
  external JSArray<_CompilationMessage> get messages;
}

extension type _CompilationMessage._(JSObject _) implements JSObject {
  external String get message;
  external String get type;
  external int get lineNum;
  external int get linePos;
}

extension type _Pipeline._(JSObject _) implements JSObject {
  external JSObject getBindGroupLayout(int index);
}

extension type _Encoder._(JSObject _) implements JSObject {
  external _Pass beginComputePass([JSObject? descriptor]);
  external void copyBufferToBuffer(
    _Buffer source,
    int sourceOffset,
    _Buffer destination,
    int destinationOffset,
    int size,
  );
  external JSObject finish();
}

extension type _Pass._(JSObject _) implements JSObject {
  external void setPipeline(_Pipeline pipeline);
  external void setBindGroup(int index, JSObject group);
  external void dispatchWorkgroups(int x, [int y, int z]);
  external void end();
}

const _mapRead = 1, _copySrc = 4, _copyDst = 8, _uniform = 64, _storage = 128;

final class _WebBuffer implements GpuBuffer {
  _WebBuffer(this.buffer, this.bytes);
  final _Buffer buffer;
  @override
  final int bytes;
}

final class _WebPipeline implements GpuPipeline {
  _WebPipeline(this.pipeline);
  final _Pipeline pipeline;
}

/// [GpuBackend] on the browser's WebGPU (`navigator.gpu`), from dart2wasm or
/// dart2js.
final class WebGpuBackend implements GpuBackend {
  WebGpuBackend._(this._device, this.description, this.maxStorageBufferBytes);

  /// The default adapter's device with its largest buffer limits, or null if
  /// the browser has no WebGPU (or no adapter).
  static Future<WebGpuBackend?> create({bool highPerformance = true}) async {
    final gpu = _navigatorGpu;
    if (gpu == null) return null;
    final adapter = await gpu
        .requestAdapter(
          {
                'powerPreference': highPerformance
                    ? 'high-performance'
                    : 'low-power',
              }.jsify()!
              as JSObject,
        )
        .toDart;
    if (adapter == null) return null;
    final limits = adapter.limits;
    final device = await adapter
        .requestDevice(
          {
                'requiredLimits': {
                  'maxStorageBufferBindingSize':
                      limits.maxStorageBufferBindingSize,
                  'maxBufferSize': limits.maxBufferSize,
                },
              }.jsify()!
              as JSObject,
        )
        .toDart;
    final info = adapter.info;
    final description = [
      info.vendor,
      info.architecture,
      info.device,
      info.description,
    ].where((s) => s.isNotEmpty).join(' / ');
    final maxBytes = device.limits.maxStorageBufferBindingSize;
    final maxBuffer = device.limits.maxBufferSize;
    return WebGpuBackend._(
      device,
      'WebGPU: $description',
      (maxBytes < maxBuffer ? maxBytes : maxBuffer).toInt(),
    );
  }

  final _Device _device;

  @override
  final String description;

  @override
  final int maxStorageBufferBytes;

  @override
  Future<GpuPipeline> pipeline(String wgsl) async {
    final module = _device.createShaderModule(
      {'code': wgsl}.jsify()! as JSObject,
    );
    final info = await module.getCompilationInfo().toDart;
    final errors = [
      for (final m in info.messages.toDart)
        if (m.type == 'error') '${m.lineNum}:${m.linePos}: ${m.message}',
    ];
    if (errors.isNotEmpty) {
      throw StateError('WGSL compilation failed:\n${errors.join('\n')}');
    }
    final descriptor = JSObject()
      ..setProperty('layout'.toJS, 'auto'.toJS)
      ..setProperty(
        'compute'.toJS,
        JSObject()
          ..setProperty('module'.toJS, module)
          ..setProperty('entryPoint'.toJS, 'main'.toJS),
      );
    return _WebPipeline(
      await _device.createComputePipelineAsync(descriptor).toDart,
    );
  }

  _WebBuffer _create(int bytes, int usage) => _WebBuffer(
    _device.createBuffer({'size': bytes, 'usage': usage}.jsify()! as JSObject),
    bytes,
  );

  @override
  GpuBuffer storage(int bytes) =>
      _create(_align4(bytes), _storage | _copySrc | _copyDst);

  @override
  GpuBuffer uniform(int bytes) => _create(_align4(bytes), _uniform | _copyDst);

  static int _align4(int n) => (n + 3) & ~3;

  @override
  Float32List hostFloats(int length) =>
      JSFloat32Array.withLength(length).toDart;

  @override
  void write(
    GpuBuffer buffer,
    Float32List data, {
    int offset = 0,
    int start = 0,
    int? length,
  }) {
    // toJS is free for a hostFloats array; for other lists dart2wasm copies
    // element by element.
    _device.queue.writeBuffer(
      (buffer as _WebBuffer).buffer,
      offset,
      data.toJS,
      start,
      length ?? data.length - start,
    );
  }

  @override
  void copy(
    GpuBuffer source,
    int sourceOffset,
    GpuBuffer destination,
    int destinationOffset,
    int bytes,
  ) {
    final encoder = _device.createCommandEncoder()
      ..copyBufferToBuffer(
        (source as _WebBuffer).buffer,
        sourceOffset,
        (destination as _WebBuffer).buffer,
        destinationOffset,
        bytes,
      );
    _device.queue.submit([encoder.finish()].toJS);
  }

  @override
  void dispatch(GpuPipeline pipeline, List<GpuBuffer> bindings, int groups) {
    final p = (pipeline as _WebPipeline).pipeline;
    final entries = [
      for (var i = 0; i < bindings.length; i++)
        JSObject()
          ..setProperty('binding'.toJS, i.toJS)
          ..setProperty(
            'resource'.toJS,
            JSObject()
              ..setProperty('buffer'.toJS, (bindings[i] as _WebBuffer).buffer),
          ),
    ].toJS;
    final group = _device.createBindGroup(
      JSObject()
        ..setProperty('layout'.toJS, p.getBindGroupLayout(0))
        ..setProperty('entries'.toJS, entries),
    );
    final encoder = _device.createCommandEncoder();
    final pass = encoder.beginComputePass()
      ..setPipeline(p)
      ..setBindGroup(0, group);
    // Large batches: split over the y dimension (65535 groups per axis).
    if (groups <= 65535) {
      pass.dispatchWorkgroups(groups);
    } else {
      throw ArgumentError('More than 65535 workgroups');
    }
    pass.end();
    _device.queue.submit([encoder.finish()].toJS);
  }

  @override
  Future<Float32List> read(
    GpuBuffer buffer, {
    int offset = 0,
    required int bytes,
  }) async {
    final staging = _create(_align4(bytes), _mapRead | _copyDst);
    final encoder = _device.createCommandEncoder()
      ..copyBufferToBuffer(
        (buffer as _WebBuffer).buffer,
        offset,
        staging.buffer,
        0,
        staging.bytes,
      );
    _device.queue.submit([encoder.finish()].toJS);
    await staging.buffer.mapAsync(_mapRead).toDart;
    // A JS-side copy (slice) before unmap detaches the mapped range.
    final mapped = JSFloat32Array(
      staging.buffer.getMappedRange(),
      0,
      bytes ~/ 4,
    );
    final out = (mapped.callMethod<JSFloat32Array>('slice'.toJS)).toDart;
    staging.buffer
      ..unmap()
      ..destroy();
    return out;
  }

  @override
  Future<void> idle() => _device.queue.onSubmittedWorkDone().toDart;

  @override
  void release(GpuBuffer buffer) => (buffer as _WebBuffer).buffer.destroy();
}
