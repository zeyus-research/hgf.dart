/// Reverse-mode gradients on the CPU: the trial's IR (the program the GPU
/// kernels are generated from, [TrialCompiler.compileSsa]) and its adjoint
/// program ([Adjoints]), evaluated in `f64` over a chunk of elements at a
/// time: one typed-data loop per operation.
///
/// Dart compiled ahead of time cannot load generated code, so instead of
/// generating Dart, the program is compiled to flat register code once per
/// model and interpreted; the interpretation is paid once per operation
/// and chunk, not per element. A gradient costs a fixed multiple of the
/// likelihood whatever the number of parameters, where the forward-mode
/// tangents of `BatchNetwork` cost about `1 + 0.6·k` likelihoods.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart' as hgf;
import 'package:hgf/hgf.dart' show CouplingFn, Network, sigmoid;

import '../model.dart' show Subject;
import 'ir.dart';
import 'trial.dart';

// Opcodes. Each instruction is five ints: op, destination, then up to
// three operand registers.
const _copy = 0, _add = 1, _sub = 2, _mul = 3, _div = 4, _neg = 5;
const _sel = 6; // d = c ? t : f, operands (c, f, t)
const _exp = 7, _log = 8, _tanh = 9, _sigm = 10, _dsigm = 11, _d2sigm = 12;
const _d3sigm = 13, _dtanh = 14, _d2tanh = 15, _d3tanh = 16, _gelu = 17;
const _dgelu = 18, _d2gelu = 19, _d3gelu = 20, _max = 21, _min = 22;
const _clamp = 23, _abs = 24, _sign = 25, _lambertW = 26, _logAddExp = 27;
const _gt = 28, _ge = 29, _lt = 30, _le = 31, _eq = 32, _ne = 33;
const _and = 34, _or = 35, _not = 36, _finite = 37;
const _acc = 38; // d += a
const _accSel = 39; // d += c ? a : 0, operands (a, c)
const _zero = 40;

const _calls = <String, int>{
  'exp': _exp,
  'log': _log,
  'tanh': _tanh,
  'sigm': _sigm,
  'dsigm': _dsigm,
  'd2sigm': _d2sigm,
  'd3sigm': _d3sigm,
  'dtanh': _dtanh,
  'd2tanh': _d2tanh,
  'd3tanh': _d3tanh,
  'gelu': _gelu,
  'dgelu': _dgelu,
  'd2gelu': _d2gelu,
  'd3gelu': _d3gelu,
  'max': _max,
  'min': _min,
  'clamp': _clamp,
  'abs': _abs,
  'sign': _sign,
  'lambertW0': _lambertW,
  'logAddExp': _logAddExp,
};

const _cmps = <String, int>{
  '>': _gt,
  '>=': _ge,
  '<': _lt,
  '<=': _le,
  '==': _eq,
  '!=': _ne,
};

/// Source reads as tokens the compiler resolves to per-trial registers.
final class _TokenSource implements TrialSource {
  const _TokenSource();

  @override
  String input(int c) => '@in$c';

  @override
  String observed(int c) => '@ob$c';

  @override
  String responsePresent() => '@rp';

  @override
  String response() => '@r';
}

/// Compiles IR trees to register code, numbering values: an operation on
/// the same operands is computed once.
final class _Compiler {
  final code = <int>[];
  final reg = <String, int>{};
  final _numbered = <String, int>{};
  final consts = <(int, double)>[];
  final _constOf = <double, int>{};
  final _seen = Map<Object, int>.identity();
  int nRegs = 0;

  int fresh() => nRegs++;

  int constant(double v) => _constOf.putIfAbsent(v, () {
    final r = fresh();
    consts.add((r, v));
    return r;
  });

  void emit(int op, int d, [int a = 0, int b = 0, int c = 0]) =>
      code.addAll([op, d, a, b, c]);

  int _op(int op, [int a = 0, int b = 0, int c = 0]) {
    final key = '$op,$a,$b,$c';
    final hit = _numbered[key];
    if (hit != null) return hit;
    final d = fresh();
    emit(op, d, a, b, c);
    return _numbered[key] = d;
  }

  int variable(String name) {
    final r = reg[name];
    if (r == null) throw StateError('Unknown variable $name');
    return r;
  }

  int ex(Ex e) {
    final cached = _seen[e];
    if (cached != null) return cached;
    final r = switch (e) {
      Lit(:final text) => switch (e.f64) {
        final v? => constant(v),
        null => variable(text),
      },
      Ref(:final name) => variable(name),
      TZero() => constant(0),
      TRef() => throw StateError('Tangent in a reverse program'),
      Neg(:final a) => _op(_neg, ex(a)),
      Bin(:final op, :final a, :final b) => _op(
        switch (op) {
          '+' => _add,
          '-' => _sub,
          '*' => _mul,
          '/' => _div,
          _ => throw StateError(op),
        },
        ex(a),
        ex(b),
      ),
      Sel(:final f, :final t, :final c) => _op(_sel, cond(c), ex(f), ex(t)),
      Call(:final fn, :final args) => _op(
        _calls[fn] ?? (throw StateError('No CPU kernel for $fn')),
        ex(args[0]),
        args.length > 1 ? ex(args[1]) : 0,
        args.length > 2 ? ex(args[2]) : 0,
      ),
    };
    // Accumulators change: only immutable values are cached by identity.
    _seen[e] = r;
    return r;
  }

  int cond(Cond c) {
    final cached = _seen[c];
    if (cached != null) return cached;
    final r = switch (c) {
      CondAtom(:final name) => switch (name) {
        'true' => constant(1),
        'false' => constant(0),
        _ => variable(name),
      },
      CondCmp(:final op, :final a, :final b) => _op(_cmps[op]!, ex(a), ex(b)),
      CondLogic(:final op, :final a, :final b) => _op(
        op == '&&' ? _and : _or,
        cond(a),
        cond(b),
      ),
      CondNot(:final a) => _op(_not, cond(a)),
      CondFinite(:final a) => _op(_finite, ex(a)),
    };
    _seen[c] = r;
    return r;
  }

  /// `let name = value`: a bare variable or atom is copied, so that every
  /// statement owns a register (a final version is copied back into the
  /// carried state).
  void let(String name, Object value) {
    final r = value is Cond ? cond(value) : ex(value as Ex);
    if (value is Ref || value is Lit || value is CondAtom) {
      final d = fresh();
      emit(_copy, d, r);
      reg[name] = d;
    } else {
      reg[name] = r;
    }
  }
}

/// The likelihood of a network under a response model and its gradient
/// with respect to the initial value of every slot, by reverse mode on the
/// CPU in `f64` (see the library comment).
///
/// Slots are [TrialCompiler]'s: [nSlots] per element, in its layout
/// (`TrialCompiler.slotOfVar`, [initialSlots]).
final class IrReverse {
  factory IrReverse(Network network, GpuResponse response, {int chunk = 64}) {
    final tc = TrialCompiler(network, response);
    final ssa = tc.compileSsa(const _TokenSource());
    final nIn = network.inputIdxs.length;
    final c = _Compiler();
    // Per-trial inputs.
    final dt = c.reg['dt'] = c.fresh();
    final logT = c.reg['logT'] = c.fresh();
    final inputs = [for (var q = 0; q < nIn; q++) c.reg['@in$q'] = c.fresh()];
    final observed = [for (var q = 0; q < nIn; q++) c.reg['@ob$q'] = c.fresh()];
    final present = c.reg['@rp'] = c.fresh();
    final resp = c.reg['@r'] = c.fresh();
    // The carried slots, in slot order.
    final carried = [
      for (final v in tc.slotOfVar.keys)
        if (ssa.carried.contains(v)) v,
    ];
    final carriedReg = [for (final v in carried) c.reg[v] = c.fresh()];
    // Primal.
    for (final (name, value) in ssa.steps) {
      c.let(name, value);
    }
    final primal = c.code.length;
    final objective = c.variable(ssa.objective);
    final finals = [
      for (final v in carried)
        switch (ssa.finalVersion[v]) {
          final f? => c.variable(f),
          null => -1,
        },
    ];
    // Adjoints: one accumulator per statement; persistent ones for the
    // carried slots.
    final names = {for (final (n, _) in ssa.statements) n};
    for (final n in names) {
      c.reg['${n}_b'] = c.fresh();
    }
    final carriedAdj = [for (final v in carried) c.reg['${v}_b'] = c.fresh()];
    for (final n in names) {
      c.emit(_zero, c.variable('${n}_b'));
    }
    c.emit(_copy, c.variable('${ssa.objective}_b'), c.constant(1));
    for (var j = 0; j < carried.length; j++) {
      final f = ssa.finalVersion[carried[j]];
      if (f == null) continue;
      c
        ..emit(_acc, c.variable('${f}_b'), carriedAdj[j])
        ..emit(_zero, carriedAdj[j]);
    }
    final adj = Adjoints((n) => names.contains(n) || ssa.carried.contains(n));
    for (final (n, e) in ssa.statements.reversed) {
      adj.statement(n, e);
    }
    for (final st in adj.steps) {
      switch (st) {
        case AdjointLet(:final name, :final value):
          c.reg[name] = c.ex(value);
        case AdjointAdd(:final target, :final g, :final guard):
          final d = c.variable('${target}_b');
          final gr = c.ex(g);
          if (guard == null) {
            c.emit(_acc, d, gr);
          } else {
            c.emit(_accSel, d, gr, c.cond(guard));
          }
      }
    }
    return IrReverse._(
      chunk: chunk,
      nSlots: tc.nSlots,
      initialSlots: tc.initialSlots(),
      slotOfVar: tc.slotOfVar,
      nIn: nIn,
      usesResponses: response.usesResponses,
      code: Int32List.fromList(c.code),
      primalLength: primal,
      nRegs: c.nRegs,
      consts: c.consts,
      dtReg: dt,
      logTReg: logT,
      inputRegs: Int32List.fromList(inputs),
      observedRegs: Int32List.fromList(observed),
      presentReg: present,
      responseReg: resp,
      carriedSlots: Int32List.fromList([
        for (final v in carried) tc.slotOfVar[v]!,
      ]),
      carriedRegs: Int32List.fromList(carriedReg),
      finalRegs: Int32List.fromList(finals),
      carriedAdjRegs: Int32List.fromList(carriedAdj),
      objectiveReg: objective,
    );
  }

  IrReverse._({
    required this.chunk,
    required this.nSlots,
    required this.initialSlots,
    required this.slotOfVar,
    required this.nIn,
    required this.usesResponses,
    required Int32List code,
    required int primalLength,
    required this.nRegs,
    required List<(int, double)> consts,
    required int dtReg,
    required int logTReg,
    required Int32List inputRegs,
    required Int32List observedRegs,
    required int presentReg,
    required int responseReg,
    required Int32List carriedSlots,
    required Int32List carriedRegs,
    required Int32List finalRegs,
    required Int32List carriedAdjRegs,
    required int objectiveReg,
  }) : _code = code,
       _primal = primalLength,
       _consts = consts,
       _dt = dtReg,
       _logT = logTReg,
       _in = inputRegs,
       _ob = observedRegs,
       _rp = presentReg,
       _r = responseReg,
       _carriedSlots = carriedSlots,
       _carried = carriedRegs,
       _finals = finalRegs,
       _carriedAdj = carriedAdjRegs,
       _objective = objectiveReg;

  /// Elements evaluated together.
  final int chunk;

  /// Slots per element ([TrialCompiler.nSlots]).
  final int nSlots;

  /// The network's initial slots.
  final List<double> initialSlots;
  final Map<String, int> slotOfVar;
  final int nIn;
  final bool usesResponses;

  /// Registers per element.
  final int nRegs;

  /// Instructions (primal and adjoint).
  int get instructions => _code.length ~/ 5;

  final Int32List _code;
  final int _primal;
  final List<(int, double)> _consts;
  final int _dt, _logT, _rp, _r, _objective;
  final Int32List _in, _ob, _carriedSlots, _carried, _finals, _carriedAdj;

  /// Slots whose initial value the likelihood depends on (the others'
  /// adjoints are zero).
  List<int> get carriedSlots => _carriedSlots;

  /// The negative log-likelihood of every element into [nll], and with
  /// [adjoints] its derivative with respect to each initial slot
  /// (`adjoints[e * nSlots + s]`).
  ///
  /// Element `e` filters `subjects[subjectOf[e]]` from initial slots
  /// `slots[e * nSlots + s]`. A non-finite likelihood becomes `+∞` (and its
  /// adjoints are then meaningless).
  void evaluate(
    List<Subject> subjects,
    Int32List subjectOf,
    Float64List slots,
    Float64List nll, {
    Float64List? adjoints,
  }) {
    final nE = subjectOf.length;
    if (slots.length != nE * nSlots || nll.length != nE) {
      throw ArgumentError('slots or nll disagree with subjectOf');
    }
    if (adjoints != null && adjoints.length != nE * nSlots) {
      throw ArgumentError('adjoints must have ${nE * nSlots} entries');
    }
    final w = chunk;
    final regs = Float64List(nRegs * w);
    for (final (r, v) in _consts) {
      regs.fillRange(r * w, r * w + w, v);
    }
    final nC = _carried.length;
    var tape = Float64List(0);
    final saved = Float64List(nC * w);
    final total = Float64List(w);
    final len = Int32List(w);
    final subj = List<Subject?>.filled(w, null);
    for (var e0 = 0; e0 < nE; e0 += w) {
      final m = math.min(w, nE - e0);
      var maxT = 0;
      for (var i = 0; i < w; i++) {
        if (i < m) {
          final s = subjects[subjectOf[e0 + i]];
          subj[i] = s;
          len[i] = nIn == 0 ? 0 : s.inputs.length ~/ nIn;
        } else {
          subj[i] = null;
          len[i] = 0;
        }
        maxT = math.max(maxT, len[i]);
        total[i] = 0;
      }
      // Initial carried state (padding lanes copy lane 0's).
      for (var j = 0; j < nC; j++) {
        final base = _carried[j] * w, s = _carriedSlots[j];
        for (var i = 0; i < w; i++) {
          final e = e0 + (i < m ? i : 0);
          regs[base + i] = slots[e * nSlots + s];
        }
      }
      final need = maxT * nC * w;
      if (adjoints != null && tape.length < need) tape = Float64List(need);
      // Forward sweep.
      for (var t = 0; t < maxT; t++) {
        _setTrial(regs, w, t, subj, len);
        if (adjoints != null) {
          for (var j = 0; j < nC; j++) {
            tape.setRange(
              (t * nC + j) * w,
              (t * nC + j + 1) * w,
              regs,
              _carried[j] * w,
            );
          }
        }
        _run(regs, w, 0, _primal);
        final ob = _objective * w;
        for (var i = 0; i < w; i++) {
          if (t < len[i]) total[i] += regs[ob + i];
        }
        for (var j = 0; j < nC; j++) {
          final f = _finals[j];
          if (f < 0) continue;
          final src = f * w, dst = _carried[j] * w;
          for (var i = 0; i < w; i++) {
            if (t < len[i]) regs[dst + i] = regs[src + i];
          }
        }
      }
      for (var i = 0; i < m; i++) {
        final v = total[i];
        nll[e0 + i] = v.isFinite ? v : double.infinity;
      }
      if (adjoints == null) continue;
      // Reverse sweep: each trial's primal is recomputed from the tape.
      for (var j = 0; j < nC; j++) {
        final b = _carriedAdj[j] * w;
        regs.fillRange(b, b + w, 0.0);
      }
      for (var t = maxT - 1; t >= 0; t--) {
        _setTrial(regs, w, t, subj, len);
        for (var j = 0; j < nC; j++) {
          regs.setRange(
            _carried[j] * w,
            _carried[j] * w + w,
            tape,
            (t * nC + j) * w,
          );
          saved.setRange(j * w, j * w + w, regs, _carriedAdj[j] * w);
        }
        _run(regs, w, 0, _code.length);
        // Lanes past the end of their series keep their adjoints.
        for (var j = 0; j < nC; j++) {
          final b = _carriedAdj[j] * w;
          for (var i = 0; i < w; i++) {
            if (t >= len[i]) regs[b + i] = saved[j * w + i];
          }
        }
      }
      for (var i = 0; i < m; i++) {
        final e = e0 + i;
        adjoints.fillRange(e * nSlots, (e + 1) * nSlots, 0.0);
        for (var j = 0; j < nC; j++) {
          adjoints[e * nSlots + _carriedSlots[j]] =
              regs[_carriedAdj[j] * w + i];
        }
      }
    }
  }

  /// Forward-over-reverse: [evaluate]'s likelihood and adjoints, and the
  /// derivative of the adjoints along [direction] (a perturbation of the
  /// initial slots, `[e * nSlots + s]`), which is the Hessian of the
  /// likelihood with respect to the initial slots times [direction], into
  /// [hvp]. The reverse program runs in dual numbers (a value and a tangent
  /// per register), so this is exact, at about twice the cost of
  /// [evaluate] with adjoints.
  void hessianVectorProducts(
    List<Subject> subjects,
    Int32List subjectOf,
    Float64List slots,
    Float64List direction,
    Float64List nll,
    Float64List adjoints,
    Float64List hvp,
  ) {
    final nE = subjectOf.length;
    for (final a in [slots, direction, adjoints, hvp]) {
      if (a.length != nE * nSlots) {
        throw ArgumentError('Slot arrays must have ${nE * nSlots} entries');
      }
    }
    if (nll.length != nE) throw ArgumentError('nll must have $nE entries');
    final w = chunk;
    final regs = Float64List(nRegs * w), dregs = Float64List(nRegs * w);
    for (final (r, v) in _consts) {
      regs.fillRange(r * w, r * w + w, v);
    }
    final nC = _carried.length;
    var tape = Float64List(0), dtape = Float64List(0);
    final saved = Float64List(nC * w), dsaved = Float64List(nC * w);
    final total = Float64List(w);
    final len = Int32List(w);
    final subj = List<Subject?>.filled(w, null);
    for (var e0 = 0; e0 < nE; e0 += w) {
      final m = math.min(w, nE - e0);
      var maxT = 0;
      for (var i = 0; i < w; i++) {
        if (i < m) {
          final s = subjects[subjectOf[e0 + i]];
          subj[i] = s;
          len[i] = nIn == 0 ? 0 : s.inputs.length ~/ nIn;
        } else {
          subj[i] = null;
          len[i] = 0;
        }
        maxT = math.max(maxT, len[i]);
        total[i] = 0;
      }
      // Tangents are zero except on the carried slots, seeded here.
      dregs.fillRange(0, dregs.length, 0.0);
      for (var j = 0; j < nC; j++) {
        final base = _carried[j] * w, s = _carriedSlots[j];
        for (var i = 0; i < w; i++) {
          final e = e0 + (i < m ? i : 0);
          regs[base + i] = slots[e * nSlots + s];
          dregs[base + i] = i < m ? direction[e * nSlots + s] : 0.0;
        }
      }
      final need = maxT * nC * w;
      if (tape.length < need) {
        tape = Float64List(need);
        dtape = Float64List(need);
      }
      for (var t = 0; t < maxT; t++) {
        _setTrial(regs, w, t, subj, len);
        for (var j = 0; j < nC; j++) {
          final at = (t * nC + j) * w, c = _carried[j] * w;
          tape.setRange(at, at + w, regs, c);
          dtape.setRange(at, at + w, dregs, c);
        }
        _runDual(regs, dregs, w, 0, _primal);
        final ob = _objective * w;
        for (var i = 0; i < w; i++) {
          if (t < len[i]) total[i] += regs[ob + i];
        }
        for (var j = 0; j < nC; j++) {
          final f = _finals[j];
          if (f < 0) continue;
          final src = f * w, dst = _carried[j] * w;
          for (var i = 0; i < w; i++) {
            if (t < len[i]) {
              regs[dst + i] = regs[src + i];
              dregs[dst + i] = dregs[src + i];
            }
          }
        }
      }
      for (var i = 0; i < m; i++) {
        final v = total[i];
        nll[e0 + i] = v.isFinite ? v : double.infinity;
      }
      for (var j = 0; j < nC; j++) {
        final b = _carriedAdj[j] * w;
        regs.fillRange(b, b + w, 0.0);
        dregs.fillRange(b, b + w, 0.0);
      }
      for (var t = maxT - 1; t >= 0; t--) {
        _setTrial(regs, w, t, subj, len);
        for (var j = 0; j < nC; j++) {
          final at = (t * nC + j) * w, c = _carried[j] * w;
          regs.setRange(c, c + w, tape, at);
          dregs.setRange(c, c + w, dtape, at);
          final a = _carriedAdj[j] * w;
          saved.setRange(j * w, j * w + w, regs, a);
          dsaved.setRange(j * w, j * w + w, dregs, a);
        }
        _runDual(regs, dregs, w, 0, _code.length);
        for (var j = 0; j < nC; j++) {
          final b = _carriedAdj[j] * w;
          for (var i = 0; i < w; i++) {
            if (t >= len[i]) {
              regs[b + i] = saved[j * w + i];
              dregs[b + i] = dsaved[j * w + i];
            }
          }
        }
      }
      for (var i = 0; i < m; i++) {
        final e = e0 + i;
        adjoints.fillRange(e * nSlots, (e + 1) * nSlots, 0.0);
        hvp.fillRange(e * nSlots, (e + 1) * nSlots, 0.0);
        for (var j = 0; j < nC; j++) {
          final a = _carriedAdj[j] * w + i;
          adjoints[e * nSlots + _carriedSlots[j]] = regs[a];
          hvp[e * nSlots + _carriedSlots[j]] = dregs[a];
        }
      }
    }
  }

  /// [_run] in dual numbers: [r] values, [dr] tangents.
  @pragma('vm:unsafe:no-bounds-checks')
  void _runDual(Float64List r, Float64List dr, int w, int from, int to) {
    final code = _code;
    for (var pc = from; pc < to; pc += 5) {
      final op = code[pc];
      final d = code[pc + 1] * w;
      final a = code[pc + 2] * w;
      final b = code[pc + 3] * w;
      final c = code[pc + 4] * w;
      // Unary functions: value f(x), tangent f′(x)·dx.
      void unary(double Function(double) f, double Function(double) df) {
        for (var i = 0; i < w; i++) {
          final x = r[a + i];
          r[d + i] = f(x);
          dr[d + i] = df(x) * dr[a + i];
        }
      }

      switch (op) {
        case _copy:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i];
            dr[d + i] = dr[a + i];
          }
        case _add:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] + r[b + i];
            dr[d + i] = dr[a + i] + dr[b + i];
          }
        case _sub:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] - r[b + i];
            dr[d + i] = dr[a + i] - dr[b + i];
          }
        case _mul:
          for (var i = 0; i < w; i++) {
            final x = r[a + i], y = r[b + i];
            r[d + i] = x * y;
            dr[d + i] = dr[a + i] * y + x * dr[b + i];
          }
        case _div:
          for (var i = 0; i < w; i++) {
            final y = r[b + i], q = r[a + i] / y;
            r[d + i] = q;
            dr[d + i] = (dr[a + i] - q * dr[b + i]) / y;
          }
        case _neg:
          for (var i = 0; i < w; i++) {
            r[d + i] = -r[a + i];
            dr[d + i] = -dr[a + i];
          }
        case _sel:
          for (var i = 0; i < w; i++) {
            final on = r[a + i] != 0.0;
            r[d + i] = on ? r[c + i] : r[b + i];
            dr[d + i] = on ? dr[c + i] : dr[b + i];
          }
        case _exp:
          for (var i = 0; i < w; i++) {
            final v = math.exp(r[a + i]);
            r[d + i] = v;
            dr[d + i] = v * dr[a + i];
          }
        case _log:
          for (var i = 0; i < w; i++) {
            final x = r[a + i];
            r[d + i] = math.log(x);
            dr[d + i] = dr[a + i] / x;
          }
        case _tanh:
          unary(CouplingFn.tanh.f, CouplingFn.tanh.df);
        case _sigm:
          unary(sigmoid, CouplingFn.sigmoid.df);
        case _dsigm:
          unary(CouplingFn.sigmoid.df, CouplingFn.sigmoid.d2f);
        case _d2sigm:
          unary(CouplingFn.sigmoid.d2f, CouplingFn.sigmoid.d3f);
        case _d3sigm:
          unary(CouplingFn.sigmoid.d3f, _d4sigm);
        case _dtanh:
          unary(CouplingFn.tanh.df, CouplingFn.tanh.d2f);
        case _d2tanh:
          unary(CouplingFn.tanh.d2f, CouplingFn.tanh.d3f);
        case _d3tanh:
          unary(CouplingFn.tanh.d3f, _d4tanh);
        case _gelu:
          unary(CouplingFn.gelu.f, CouplingFn.gelu.df);
        case _dgelu:
          unary(CouplingFn.gelu.df, CouplingFn.gelu.d2f);
        case _d2gelu:
          unary(CouplingFn.gelu.d2f, CouplingFn.gelu.d3f);
        case _d3gelu:
          unary(CouplingFn.gelu.d3f, _d4gelu);
        case _max:
          for (var i = 0; i < w; i++) {
            final x = r[a + i], y = r[b + i];
            r[d + i] = math.max(x, y);
            dr[d + i] = x >= y ? dr[a + i] : dr[b + i];
          }
        case _min:
          for (var i = 0; i < w; i++) {
            final x = r[a + i], y = r[b + i];
            r[d + i] = math.min(x, y);
            dr[d + i] = x <= y ? dr[a + i] : dr[b + i];
          }
        case _clamp:
          for (var i = 0; i < w; i++) {
            final x = r[a + i], lo = r[b + i], hi = r[c + i];
            r[d + i] = math.min(math.max(x, lo), hi);
            dr[d + i] = x > lo && x < hi ? dr[a + i] : 0.0;
          }
        case _abs:
          for (var i = 0; i < w; i++) {
            final x = r[a + i];
            r[d + i] = x.abs();
            dr[d + i] = x.sign * dr[a + i];
          }
        case _sign:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i].sign;
            dr[d + i] = 0.0;
          }
        case _lambertW:
          for (var i = 0; i < w; i++) {
            final z = r[a + i], v = hgf.lambertW0(z);
            r[d + i] = v;
            dr[d + i] = v / (z * (1 + v)) * dr[a + i];
          }
        case _logAddExp:
          for (var i = 0; i < w; i++) {
            final x = r[a + i], y = r[b + i];
            r[d + i] = hgf.logAddExp(x, y);
            dr[d + i] = sigmoid(x - y) * dr[a + i] + sigmoid(y - x) * dr[b + i];
          }
        case _acc:
          for (var i = 0; i < w; i++) {
            r[d + i] += r[a + i];
            dr[d + i] += dr[a + i];
          }
        case _accSel:
          for (var i = 0; i < w; i++) {
            if (r[b + i] != 0.0) {
              r[d + i] += r[a + i];
              dr[d + i] += dr[a + i];
            }
          }
        case _zero:
          for (var i = 0; i < w; i++) {
            r[d + i] = 0.0;
            dr[d + i] = 0.0;
          }
        default:
          // Comparisons and logic: values only, no tangent.
          _run(r, w, pc, pc + 5);
          for (var i = 0; i < w; i++) {
            dr[d + i] = 0.0;
          }
      }
    }
  }

  /// The per-trial registers of trial [t] (lanes past their series read
  /// benign values; their results are discarded).
  void _setTrial(
    Float64List regs,
    int w,
    int t,
    List<Subject?> subj,
    Int32List len,
  ) {
    for (var i = 0; i < w; i++) {
      final s = subj[i];
      final on = s != null && t < len[i];
      final ts = on ? s.timeSteps : null;
      final dt = ts == null ? 1.0 : ts[t];
      regs[_dt * w + i] = dt;
      regs[_logT * w + i] = dt == 1.0 ? 0.0 : math.log(dt);
      for (var q = 0; q < nIn; q++) {
        regs[_in[q] * w + i] = on ? s.inputs[t * nIn + q] : 0.0;
        final o = on ? s.observed : null;
        regs[_ob[q] * w + i] = o == null ? (on ? 1.0 : 0.0) : o[t * nIn + q];
      }
      if (usesResponses) {
        final r = on ? s.responses![t] : double.nan;
        regs[_rp * w + i] = r.isNaN ? 0.0 : 1.0;
        regs[_r * w + i] = r.isNaN ? 0.0 : r;
      }
    }
  }

  @pragma('vm:unsafe:no-bounds-checks')
  void _run(Float64List r, int w, int from, int to) {
    final code = _code;
    for (var pc = from; pc < to; pc += 5) {
      final op = code[pc];
      final d = code[pc + 1] * w;
      final a = code[pc + 2] * w;
      final b = code[pc + 3] * w;
      final c = code[pc + 4] * w;
      switch (op) {
        case _copy:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i];
          }
        case _add:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] + r[b + i];
          }
        case _sub:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] - r[b + i];
          }
        case _mul:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] * r[b + i];
          }
        case _div:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] / r[b + i];
          }
        case _neg:
          for (var i = 0; i < w; i++) {
            r[d + i] = -r[a + i];
          }
        case _sel:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] != 0.0 ? r[c + i] : r[b + i];
          }
        case _exp:
          for (var i = 0; i < w; i++) {
            r[d + i] = math.exp(r[a + i]);
          }
        case _log:
          for (var i = 0; i < w; i++) {
            r[d + i] = math.log(r[a + i]);
          }
        case _tanh:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.tanh.f(r[a + i]);
          }
        case _sigm:
          for (var i = 0; i < w; i++) {
            r[d + i] = sigmoid(r[a + i]);
          }
        case _dsigm:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.sigmoid.df(r[a + i]);
          }
        case _d2sigm:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.sigmoid.d2f(r[a + i]);
          }
        case _d3sigm:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.sigmoid.d3f(r[a + i]);
          }
        case _dtanh:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.tanh.df(r[a + i]);
          }
        case _d2tanh:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.tanh.d2f(r[a + i]);
          }
        case _d3tanh:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.tanh.d3f(r[a + i]);
          }
        case _gelu:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.gelu.f(r[a + i]);
          }
        case _dgelu:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.gelu.df(r[a + i]);
          }
        case _d2gelu:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.gelu.d2f(r[a + i]);
          }
        case _d3gelu:
          for (var i = 0; i < w; i++) {
            r[d + i] = CouplingFn.gelu.d3f(r[a + i]);
          }
        case _max:
          for (var i = 0; i < w; i++) {
            r[d + i] = math.max(r[a + i], r[b + i]);
          }
        case _min:
          for (var i = 0; i < w; i++) {
            r[d + i] = math.min(r[a + i], r[b + i]);
          }
        case _clamp:
          for (var i = 0; i < w; i++) {
            r[d + i] = math.min(math.max(r[a + i], r[b + i]), r[c + i]);
          }
        case _abs:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i].abs();
          }
        case _sign:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i].sign;
          }
        case _lambertW:
          for (var i = 0; i < w; i++) {
            r[d + i] = hgf.lambertW0(r[a + i]);
          }
        case _logAddExp:
          for (var i = 0; i < w; i++) {
            r[d + i] = hgf.logAddExp(r[a + i], r[b + i]);
          }
        case _gt:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] > r[b + i] ? 1.0 : 0.0;
          }
        case _ge:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] >= r[b + i] ? 1.0 : 0.0;
          }
        case _lt:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] < r[b + i] ? 1.0 : 0.0;
          }
        case _le:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] <= r[b + i] ? 1.0 : 0.0;
          }
        case _eq:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] == r[b + i] ? 1.0 : 0.0;
          }
        case _ne:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] != r[b + i] ? 1.0 : 0.0;
          }
        case _and:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] != 0.0 && r[b + i] != 0.0 ? 1.0 : 0.0;
          }
        case _or:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] != 0.0 || r[b + i] != 0.0 ? 1.0 : 0.0;
          }
        case _not:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i] != 0.0 ? 0.0 : 1.0;
          }
        case _finite:
          for (var i = 0; i < w; i++) {
            r[d + i] = r[a + i].isFinite ? 1.0 : 0.0;
          }
        case _acc:
          for (var i = 0; i < w; i++) {
            r[d + i] += r[a + i];
          }
        case _accSel:
          for (var i = 0; i < w; i++) {
            if (r[b + i] != 0.0) r[d + i] += r[a + i];
          }
        case _zero:
          for (var i = 0; i < w; i++) {
            r[d + i] = 0.0;
          }
      }
    }
  }
}

/// The fourth derivative of the sigmoid: σ′(1 − 2σ)(1 − 12σ′), σ′ = σ(1 − σ).
double _d4sigm(double x) {
  final s = sigmoid(x), q = s * (1 - s);
  return q * (1 - 2 * s) * (1 - 12 * q);
}

/// The fourth derivative of tanh: 8t(1 − t²)(2 − 3t²).
double _d4tanh(double x) {
  final t = CouplingFn.tanh.f(x);
  return 8 * t * (1 - t * t) * (2 - 3 * t * t);
}

/// The fourth derivative of GELU: φ(x)(−x⁴ + 7x² − 4).
double _d4gelu(double x) {
  final pdf = math.exp(-(x * x) / 2) / math.sqrt(2 * math.pi);
  return pdf * (-x * x * x * x + 7 * x * x - 4);
}
