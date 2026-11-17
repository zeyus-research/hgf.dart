import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

/// A 2-level continuous HGF with a Gaussian readout of the input's
/// prediction, participants by column.
ModelDocument _doc() => ModelDocument(
  name: 'two-level',
  network: NetworkDocument(
    nodes: [
      ScalarNode('x', kind: NodeKind.continuousState),
      ScalarNode(
        'v',
        kind: NodeKind.continuousState,
        params: {NodeField.tonicVolatility: -4},
      ),
      ScalarNode('vol', kind: NodeKind.continuousState),
    ],
    edges: const [
      DocEdge(EdgeType.value, 'v', 'x'),
      DocEdge(EdgeType.volatility, 'vol', 'v'),
    ],
  ),
  inputs: [DocInput('obs', node: 'x')],
  response: DocResponse(
    'guess',
    readout: {
      'type': 'GaussianReadout',
      'node': 'x',
      'field': 'expected_mean',
      'precision': 4.0,
    },
  ),
  data: const DocData(participantColumn: 'id'),
);

void main() {
  group('free parameters', () {
    test('freeing targets suggests names and priors', () {
      final d = _doc()
          .freeTarget(nodeTarget('v', NodeField.tonicVolatility), -4)
          .freeTarget(couplingTarget(EdgeType.volatility, 'vol', 'v'), 1)
          .freeTarget(responseTarget('precision'), 4);
      expect(d.parameterNames, [
        'omega_v',
        'kappa_vol_v',
        'response_precision',
      ]);
      // Freeing it again changes nothing.
      expect(
        identical(
          d.freeTarget(nodeTarget('v', NodeField.tonicVolatility), 0),
          d,
        ),
        isTrue,
      );
      final c = d.compile();
      final m = c.model!;
      expect(m.parameters.map((p) => p.prior.toString()), [
        TruncatedNormal(-4, 2, lower: -16, upper: 4).toString(),
        const LogNormal(0, 0.5).toString(),
        LogNormal((math.log(4) * 1000).roundToDouble() / 1000, 2).toString(),
      ]);
      expect(checkModel(d), isEmpty);
      // The preview's values follow the document, not the priors.
      expect(c.currentValues(), [-4, 1, 4]);
      final moved = d.network.withParam('v', NodeField.tonicVolatility, -2);
      expect(d.withNetwork(moved).compile().currentValues(), [-2, 1, 4]);

      final fixed = d.fixTarget(nodeTarget('v', NodeField.tonicVolatility));
      expect(fixed.parameterNames, ['kappa_vol_v', 'response_precision']);
    });

    test('flags parameters that set nothing that exists', () {
      final d = _doc().withParameters([
        docParameter('a', [
          nodeTarget('gone', NodeField.mean),
        ], const Normal(0, 1)),
        docParameter('b', [
          couplingTarget(EdgeType.value, 'vol', 'v'),
        ], const Normal(0, 1)),
        docParameter('c', [responseTarget('beta')], const LogNormal(0, 1)),
        docParameter('c', [
          nodeTarget('v', NodeField.mean),
        ], const Normal(0, 1)),
      ]);
      final issues = checkModel(d).where((i) => i.isError).toList();
      expect(issues.every((i) => i.parts.contains(DocPart.fit)), isTrue);
      final text = issues.map((i) => i.message).join('\n');
      expect(text, contains('"gone", not a node'));
      expect(text, contains('which is not an edge'));
      expect(text, contains('the response does not have'));
      expect(text, contains('Two free parameters are named "c"'));
    });
  });

  group('group level', () {
    ModelDocument grouped() => _doc()
        .freeTarget(nodeTarget('v', NodeField.tonicVolatility), -4)
        .freeTarget(responseTarget('precision'), 4)
        .withRole('omega_v', ParameterRole.pooled);

    test('round-trips through JSON and compiles to a HierarchicalModel', () {
      final d = grouped();
      expect(d.group.roleOf('omega_v'), ParameterRole.pooled);
      expect(d.group.roleOf('response_precision'), ParameterRole.individual);
      final pop = d.group.pooled['omega_v']!.decode(const FitCodec());
      // Around the prior's typical value, on its (logit) interval scale.
      expect(pop.sd, isA<HalfNormal>());

      final back = ModelDocument.fromJson(
        jsonDecode(jsonEncode(d.toJson())) as Map<String, Object?>,
      );
      expect(jsonEncode(back.toJson()), jsonEncode(d.toJson()));
      final h = back.compile().hierarchical!;
      expect(h.pooled.keys, ['omega_v']);
      expect(h.shared, isEmpty);
      expect(h.dimension(3), 2 + 3 * 2);
      expect(_doc().compile().hierarchical, isNull);
    });

    test('follows renames and removals', () {
      var d = grouped().withRole('response_precision', ParameterRole.shared);
      d = d.replaceParameter('omega_v', {
        ...d.parameter('omega_v')!,
        'name': 'omega',
      });
      expect(d.group.pooled.keys, ['omega']);
      expect(d.withoutParameter('omega').group.pooled, isEmpty);
      // Removing the response drops its parameter from the group.
      expect(d.withResponse(null).group.shared, isEmpty);
      // So does removing the node a parameter sets.
      final net = d.network.withoutNode('vol').withoutNode('v');
      final g = d.withNetwork(net).group;
      expect(g.pooled, isEmpty);
      expect(g.shared, ['response_precision']);
      expect(
        d.withRole('omega', ParameterRole.individual).group.pooled,
        isEmpty,
      );
    });

    test('MAP cannot share; NUTS can', () {
      final d = grouped().withRole('response_precision', ParameterRole.shared);
      final issues = checkModel(d).where((i) => i.isError);
      expect(issues.single.message, contains('needs NUTS'));
      expect(issues.single.parts, [DocPart.fit]);
      final nuts = d.copyWith(fit: const DocFit(method: 'nuts'));
      expect(checkModel(nuts), isEmpty);
      expect(nuts.compile().hierarchical!.shared, {'response_precision'});

      final bad = d.withGroup(
        DocGroup(
          pooled: {
            'omega_v': DocPopulation(
              mean: const Normal(0, 1).toJson(),
              sd: const Normal(0, 1).toJson(),
            ),
            'nope': DocPopulation.around(const Normal(0, 1)),
          },
        ),
      );
      final text = checkModel(bad).map((i) => i.message).join('\n');
      expect(text, contains('"nope", which is not a free parameter'));
      expect(text, contains('positive values'));
    });

    test('empirical Bayes runs on simulated participants', () {
      final d = grouped();
      final c = d.compile();
      final rng = math.Random(3);
      final subjects = [
        for (var k = 0; k < 4; k++)
          Subject(
            inputs: Float64List.fromList([
              for (var t = 0; t < 80; t++) math.sin(t / 8) + rng.nextDouble(),
            ]),
            id: 'p$k',
          ),
      ];
      final sims = c.simulate(subjects, rng);
      final csv = c.toCsv(sims);
      expect(csv.header, ['id', 'obs', 'guess']);
      final back = c.subjects(csv);
      expect(back.map((s) => s.id), ['p0', 'p1', 'p2', 'p3']);
      final eb = fitEmpiricalBayes(c.hierarchical!, back, maxIterations: 5);
      expect(eb.subjects, hasLength(4));
      expect(eb.populations.keys, ['omega_v']);
      expect(eb.logEvidence.isFinite, isTrue);
    });
  });

  test('a fit to the observations reads no response column', () {
    final d = _doc()
        .copyWith(
          response: DocResponse(
            'surprise',
            readout: {
              'type': 'ObservationSurprise',
              'function': 'TotalGaussianSurprise',
            },
          ),
        )
        .freeTarget(nodeTarget('v', NodeField.tonicVolatility), -4);
    final c = d.compile();
    final s = c.subjects(CsvData.parse('id,obs\na,0.1\na,0.3\nb,1\n'));
    expect(s.map((x) => x.responses), [null, null]);
    expect(checkModel(d, columns: ['id', 'obs']), isEmpty);
    expect(c.model!.negLogPosteriorAt(s[0], [-4]).isFinite, isTrue);
  });
}
