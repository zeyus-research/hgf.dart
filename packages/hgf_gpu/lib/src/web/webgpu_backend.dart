import 'dart:async';
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
  external String? get vendor;
  external String? get architecture;
  external String? get device;
  external String? get description;
  external bool? get isFallbackAdapter;
}

@JS('navigator.userAgent')
external String? get _userAgent;

@JS('MessageChannel')
extension type _MessageChannel._(JSObject _) implements JSObject {
  external factory _MessageChannel();
  external _MessagePort get port1;
  external _MessagePort get port2;
}

extension type _MessagePort._(JSObject _) implements JSObject {
  external set onmessage(JSFunction? handler);
  external void postMessage(JSAny? message);
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
  _WebBuffer(this.buffer, this.bytes) : id = _nextId++;
  static int _nextId = 0;

  /// Identity for the bind-group cache.
  final int id;
  final _Buffer buffer;
  @override
  final int bytes;
}

final class _WebPipeline implements GpuPipeline {
  _WebPipeline(this.pipeline) : layout = pipeline.getBindGroupLayout(0);
  final _Pipeline pipeline;
  final JSObject layout;

  /// Bind groups by the ids of their buffers.
  final groups = <String, (List<int>, JSObject)>{};
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
    final maxBytes = device.limits.maxStorageBufferBindingSize;
    final maxBuffer = device.limits.maxBufferSize;
    return WebGpuBackend._(
      device,
      _describe(adapter),
      (maxBytes < maxBuffer ? maxBytes : maxBuffer).toInt(),
    );
  }

  /// `WebGPU: vendor / architecture / …`. Firefox leaves every field of
  /// `GPUAdapterInfo` empty (fingerprinting protection), so the adapter
  /// may have no name at all.
  static String _describe(_Adapter adapter) {
    final info = adapter.info;
    final parts = [
      info.vendor,
      info.architecture,
      info.device,
      info.description,
    ].whereType<String>().where((s) => s.isNotEmpty).toList();
    final fallback = info.isFallbackAdapter ?? false;
    if (parts.isEmpty) {
      final ua = _userAgent ?? '';
      final browser = ua.contains('Firefox/')
          ? 'Firefox'
          : ua.contains('Safari/') && !ua.contains('Chrome/')
          ? 'Safari'
          : 'the browser';
      parts.add(
        fallback
            ? 'a software adapter'
            : 'a hardware adapter ($browser doesn\'t name it)',
      );
    } else if (fallback) {
      parts.add('software');
    }
    return 'WebGPU: ${parts.join(' / ')}';
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
    _flush();
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
  void writeUints(GpuBuffer buffer, Uint32List data, {int offset = 0}) {
    _flush();
    _device.queue.writeBuffer(
      (buffer as _WebBuffer).buffer,
      offset,
      data.toJS,
      0,
      data.length,
    );
  }

  // Dispatches and copies are recorded into one pending encoder (one
  // compute pass for consecutive dispatches) and submitted together, before
  // the next queue write or read: queue writes are ordered against
  // submissions, not against commands still being recorded.
  _Encoder? _encoder;
  _Pass? _pass;
  int _pending = 0;

  /// Commands per submission at most, so one submission stays short.
  static const _maxPending = 2048;

  _Encoder _recording() => _encoder ??= _device.createCommandEncoder();

  void _endPass() {
    _pass?.end();
    _pass = null;
  }

  void _flush() {
    final encoder = _encoder;
    if (encoder == null) return;
    _endPass();
    _device.queue.submit([encoder.finish()].toJS);
    _encoder = null;
    _pending = 0;
  }

  @override
  void flush() => _flush();

  void _counted() {
    if (++_pending >= _maxPending) _flush();
  }

  @override
  void copy(
    GpuBuffer source,
    int sourceOffset,
    GpuBuffer destination,
    int destinationOffset,
    int bytes,
  ) {
    final encoder = _recording();
    _endPass();
    encoder.copyBufferToBuffer(
      (source as _WebBuffer).buffer,
      sourceOffset,
      (destination as _WebBuffer).buffer,
      destinationOffset,
      bytes,
    );
    _counted();
  }

  /// Pipelines with cached bind groups (to drop those of released buffers).
  final _pipelines = <_WebPipeline>{};

  JSObject _bindGroup(_WebPipeline p, List<GpuBuffer?> bindings) {
    final ids = [for (final b in bindings) (b as _WebBuffer?)?.id ?? -1];
    final key = ids.join(',');
    final cached = p.groups[key];
    if (cached != null) return cached.$2;
    final entries = [
      for (var i = 0; i < bindings.length; i++)
        if (bindings[i] != null)
          JSObject()
            ..setProperty('binding'.toJS, i.toJS)
            ..setProperty(
              'resource'.toJS,
              JSObject()..setProperty(
                'buffer'.toJS,
                (bindings[i] as _WebBuffer).buffer,
              ),
            ),
    ].toJS;
    final group = _device.createBindGroup(
      JSObject()
        ..setProperty('layout'.toJS, p.layout)
        ..setProperty('entries'.toJS, entries),
    );
    p.groups[key] = (ids, group);
    _pipelines.add(p);
    return group;
  }

  @override
  void dispatch(
    GpuPipeline pipeline,
    List<GpuBuffer?> bindings,
    int groups, [
    int groupsY = 1,
  ]) {
    if (groups > 65535 || groupsY > 65535) {
      throw ArgumentError('More than 65535 workgroups along one axis');
    }
    final p = pipeline as _WebPipeline;
    final group = _bindGroup(p, bindings);
    final pass = _pass ??= _recording().beginComputePass();
    pass
      ..setPipeline(p.pipeline)
      ..setBindGroup(0, group)
      ..dispatchWorkgroups(groups, groupsY);
    _counted();
  }

  @override
  Future<Float32List> read(
    GpuBuffer buffer, {
    int offset = 0,
    required int bytes,
  }) async {
    final staging = _create(_align4(bytes), _mapRead | _copyDst);
    _endPass();
    _recording().copyBufferToBuffer(
      (buffer as _WebBuffer).buffer,
      offset,
      staging.buffer,
      0,
      staging.bytes,
    );
    _flush();
    await _poked(staging.buffer.mapAsync(_mapRead).toDart);
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
  Future<void> idle() {
    _flush();
    return _poked(_device.queue.onSubmittedWorkDone().toDart);
  }

  /// Whether [read] and [idle] keep the queue polled while they wait (see
  /// [_poked]). On by default; off only to measure the difference.
  bool pokeQueue = true;

  /// [done], with the GPU queue polled until it completes.
  ///
  /// Browsers resolve `mapAsync` and `onSubmittedWorkDone` only when they
  /// next poll the device. Firefox polls on a 100 ms timer (`POLL_TIME_MS`
  /// in `WebGPUParent.cpp`, bug 1870699) and otherwise only on a submit,
  /// so every readback took 100 ms; Chromium's fence polling took
  /// ~2.5 ms in a worker. A submit, even an empty one, polls: so while
  /// waiting, this yields to the event loop (a task through a
  /// `MessageChannel`, unclamped unlike `setTimeout`) and submits nothing
  /// every 8th yield. Measured round trip (4 KB write, dispatch, readback;
  /// RX 7600 XT): Firefox 159 100 → 0.25 ms, Chromium 156 1.8–2.5 →
  /// 0.25–0.35 ms; poking at every yield floods the GPU process (3–9 ms).
  /// After [_spinFor] it polls once per timer tick (clamped to 4 ms)
  /// instead of keeping the thread busy; spinning 4 ms instead of 20 made
  /// 64k-element passes 10–30 % slower.
  Future<void> _poked(Future<JSAny?> done) async {
    if (!pokeQueue) {
      await done;
      return;
    }
    var settled = false;
    Object? error;
    StackTrace? stack;
    done.then(
      (_) => settled = true,
      onError: (Object e, StackTrace st) {
        error = e;
        stack = st;
        settled = true;
      },
    );
    final queue = _device.queue;
    final sw = Stopwatch()..start();
    var yields = 0;
    while (!settled) {
      if (sw.elapsed < _spinFor) {
        await _nextTask();
        if (!settled && ++yields % 8 == 0) queue.submit(_noCommands);
      } else {
        await Future<void>.delayed(Duration.zero);
        if (!settled) queue.submit(_noCommands);
      }
    }
    if (error case final e?) Error.throwWithStackTrace(e, stack!);
  }

  static const _spinFor = Duration(milliseconds: 20);
  static final _noCommands = <JSObject>[].toJS;

  // One channel for every yield: a message to port2 wakes port1.
  static final _MessageChannel _channel = () {
    final c = _MessageChannel();
    c.port1.onmessage = ((JSAny? _) {
      final w = _wake;
      _wake = null;
      w?.complete();
    }).toJS;
    return c;
  }();
  static Completer<void>? _wake;

  /// Completes in a new event-loop task (after rendering, if due).
  static Future<void> _nextTask() {
    final w = _wake;
    if (w != null) return w.future;
    final c = _wake = Completer<void>();
    _channel.port2.postMessage(null);
    return c.future;
  }

  @override
  void release(GpuBuffer buffer) {
    _flush();
    final b = buffer as _WebBuffer;
    for (final p in _pipelines) {
      p.groups.removeWhere((_, v) => v.$1.contains(b.id));
    }
    b.buffer.destroy();
  }
}
