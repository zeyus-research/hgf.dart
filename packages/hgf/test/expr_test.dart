import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:test/test.dart';

void main() {
  group('Expr', () {
    test('prints canonically and parses back', () {
      const cases = {
        '1 + 2 * x': '1 + 2 * x',
        '(1 + 2) * x': '(1 + 2) * x',
        'a - (b - c)': 'a - (b - c)',
        'a - b - c': 'a - b - c',
        'a / (b * c)': 'a / (b * c)',
        '-x^2': '-x^2',
        '(-x)^2': '(-x)^2',
        'x ** 2 ** 3': 'x^2^3',
        '(x^2)^3': '(x^2)^3',
        'x^-1': 'x^-1',
        '2 * tanh(x) - 0.5': '2 * tanh(x) - 0.5',
        '  clip( x ,0, 1 )': 'clip(x, 0, 1)',
        'a.mean > 0': 'a.mean > 0',
        '(a < b) == 1': '(a < b) == 1',
        '`press left` + 1': '`press left` + 1',
        "press == 'left'": "press == 'left'",
        'cat.2.expected_mean': 'cat.2.expected_mean',
        '1e-3 * x': '0.001 * x',
        '.5': '0.5',
        'a - -b': 'a - -b',
      };
      for (final MapEntry(key: src, value: printed) in cases.entries) {
        final e = Expr.parse(src);
        expect(e.toString(), printed, reason: src);
        expect(Expr.parse(e.toString()), e, reason: src);
      }
    });

    test('evaluates', () {
      double ev(String s, [Map<String, double> env = const {}]) =>
          Expr.parse(s).eval(env);
      expect(ev('1 + 2 * 3'), 7);
      expect(ev('2^3^2'), 512);
      expect(ev('-2^2'), -4);
      expect(ev('(x > 1) + (x <= 1) * 10', {'x': 2}), 1);
      expect(ev('where(x - 2, 5, 6)', {'x': 2}), 6);
      expect(ev('clip(x, -1, 1)', {'x': 3}), 1);
      expect(ev('sigmoid(0) + tanh(0) + abs(-3) + sign(-3)'), 0.5 + 3 - 1);
      expect(ev('sqrt(4) + log(exp(2)) + min(1, 2) + max(1, 2)'), 2 + 2 + 3);
      expect(() => ev('lag(x)', {'x': 1}), throwsA(isA<UnsupportedError>()));
      expect(() => ev('y + 1', {'x': 1}), throwsArgumentError);
    });

    test('reports where it fails', () {
      void fails(String s, int offset) => expect(
        () => Expr.parse(s),
        throwsA(
          isA<FormatException>().having((e) => e.offset, 'offset', offset),
        ),
        reason: s,
      );
      fails('1 +', 3);
      fails('foo(x)', 0);
      fails('exp(1, 2)', 0);
      fails('(1 + 2', 6);
      fails('1 < 2 < 3', 6);
      fails('x \$ 2', 2);
      fails('', 0);
      fails("'open", 0);
    });

    test('derivatives match finite differences', () {
      const sources = [
        '2 * tanh(x) - 0.5 * x^2',
        'x / (1 + abs(x))',
        'sigmoid(3 * x) * exp(-x^2)',
        'log(1 + exp(x))',
        'sqrt(1 + x^2) / (2 + sin_free)',
        'x^3 - 2 * x^-2',
        '(1 + x^2)^0.7',
        '(2 + x)^(1 + x / 4)',
        'max(x, 0.2) * min(x, 0.5) + clip(x, -0.3, 0.9)',
        'where(x > 0, x^2, -x)',
      ];
      final xs = [-1.3, -0.4, 0.15, 0.35, 0.7, 1.9];
      for (final src in sources) {
        final e = Expr.parse(src);
        var d = e;
        for (var order = 1; order <= 3; order++) {
          final prev = d;
          d = d.derivative('x');
          final f = prev.compile(['x', 'sin_free']);
          final df = d.compile(['x', 'sin_free']);
          for (final x in xs) {
            const h = 1e-5;
            final fd = (f([x + h, 0.3]) - f([x - h, 0.3])) / (2 * h);
            final exact = df([x, 0.3]);
            expect(
              exact,
              closeTo(fd, 2e-5 * (1 + fd.abs())),
              reason: "$src, order $order at $x: ${d.toString()}",
            );
          }
          // The derivative prints and parses back.
          expect(Expr.parse(d.toString()), d, reason: d.toString());
        }
      }
    });

    test('simplifies derivatives', () {
      String d(String s) => Expr.parse(s).derivative('x').toString();
      expect(d('x'), '1');
      expect(d('3 * x + 2'), '3');
      expect(d('x^2'), '2 * x');
      expect(d('2 * tanh(x)'), '2 * (1 - tanh(x)^2)');
      expect(d('y * x'), 'y');
      expect(d('exp(y)'), '0');
      expect(d('x > 0'), '0');
    });
  });

  group('ExprCoupling', () {
    test('rejects other variables and data-only functions', () {
      expect(() => ExprCoupling('x + y'), throwsFormatException);
      expect(() => ExprCoupling('lag(x)'), throwsFormatException);
      expect(ExprCoupling(' tanh( x ) ').source, 'tanh(x)');
      expect(ExprCoupling('tanh(x)'), ExprCoupling('tanh(x)'));
    });

    test('parseCoupling reads names and expressions', () {
      expect(parseCoupling('tanh'), CouplingFn.tanh);
      expect(parseCoupling('identity'), CouplingFn.linear);
      expect(parseCoupling('2 * x'), ExprCoupling('2 * x'));
      expect(() => parseCoupling('tanhh'), throwsArgumentError);
      expect(couplingText(ExprCoupling('2*x')), '2 * x');
    });

    test('filters as the named coupling it spells out', () {
      final u = Float64List.fromList([
        for (var t = 0; t < 200; t++) math.sin(t / 7) + 0.3 * math.cos(t / 3),
      ]);
      for (final (fn, src) in [
        (CouplingFn.tanh, 'tanh(x)'),
        (CouplingFn.sigmoid, 'sigmoid(x)'),
        (CouplingFn.linear, 'x'),
      ]) {
        Network build(Coupling c) => Network()
          ..addNodes(precision: 1e2)
          ..addNodes(valueChildren: [0], coupling: c, tonicVolatility: -2)
          ..addNodes(valueChildren: [1], tonicVolatility: -4);
        final a = build(fn).inputSeries(u);
        final b = build(ExprCoupling(src)).inputSeries(u);
        for (final f in [NodeField.mean, NodeField.precision]) {
          final sa = a.series(1, f), sb = b.series(1, f);
          for (var t = 0; t < u.length; t++) {
            expect(sa[t].isFinite, isTrue);
            expect(sb[t], closeTo(sa[t], 1e-12 * (1 + sa[t].abs())));
          }
        }
      }
    });

    test('round-trips through Network.toJson and NetworkSpec', () {
      final net = Network()
        ..addNodes()
        ..addNodes(valueChildren: [0], coupling: ExprCoupling('2 * tanh(x)'));
      final copy = Network.fromJson(net.toJson());
      expect(copy.couplingOf(1), ExprCoupling('2 * tanh(x)'));
      final spec = NetworkSpec.fromJson({
        'nodes': [
          {'kind': 'continuous-state'},
          {
            'kind': 'continuous-state',
            'value_children': [0],
            'coupling_fn': 'x / (1 + abs(x))',
          },
        ],
      }).build();
      expect(spec.couplingOf(1), ExprCoupling('x / (1 + abs(x))'));
    });

    test('its derivatives agree with the named coupling', () {
      // The filter and g'', g''' agree with the named coupling's.
      final u = Float64List.fromList([
        for (var t = 0; t < 60; t++) math.sin(t / 5),
      ]);
      double total(Coupling c, double omega) {
        final net = Network()
          ..addNodes(precision: 1e2)
          ..addNodes(valueChildren: [0], coupling: c, tonicVolatility: omega);
        final traj = net.inputSeries(u);
        var s = 0.0;
        for (var t = 0; t < u.length; t++) {
          s += traj.series(1, NodeField.mean)[t];
        }
        return s;
      }

      for (final omega in [-4.0, -2.0]) {
        expect(
          total(ExprCoupling('tanh(x)'), omega),
          closeTo(total(CouplingFn.tanh, omega), 1e-10),
        );
      }
      final c = ExprCoupling('tanh(x)');
      for (final x in [-2.0, -0.3, 0.4, 1.7]) {
        expect(c.d3f(x), closeTo(CouplingFn.tanh.d3f(x), 1e-14));
        expect(c.d2f(x), closeTo(CouplingFn.tanh.d2f(x), 1e-15));
      }
    });
  });
}
