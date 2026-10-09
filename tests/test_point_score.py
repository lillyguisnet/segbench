"""Run: python -m unittest discover -s tests -p 'test_point_score.py'."""
import itertools
import json
import unittest
from segbench.point_score import maximum_matching, parse_points, read_points, recover_points, reference_distances, score, score_at


class PointScoreTests(unittest.TestCase):
    def test_match_exhaustive_small_graphs(self):
        for bits in itertools.product((0, 1), repeat=9):
            edges = [[j for j in range(3) if bits[i*3+j]] for i in range(3)]
            best = 0
            for assignment in itertools.product((-1, 0, 1, 2), repeat=3):
                chosen = [j for j in assignment if j >= 0]
                if len(set(chosen)) == len(chosen) and all(j < 0 or j in edges[i] for i, j in enumerate(assignment)):
                    best = max(best, len(chosen))
            self.assertEqual(len(maximum_matching(edges)), best)

    def test_augment_not_greedy(self):
        self.assertEqual(len(maximum_matching([[0, 1], [0]])), 2)

    def test_exact_duplicate_and_outside(self):
        refs = [{'id':'a','x':200,'y':200}, {'id':'b','x':400,'y':200}]
        points = [{'point':[200,200]}, {'point':[400,200]}]
        s = score(points, refs, 1000, 1000)
        self.assertEqual(s['score'], 1)
        points += [{'point':[200,200]}, {'point':[-1,200]}]
        for level in score(points,refs,1000,1000)['tolerances'].values():
            self.assertEqual((level['tp'],level['fp'],level['fn']),(2,2,0))
            self.assertAlmostEqual(level['f1'], 2/3)
        self.assertEqual(score(list(reversed(points)), refs, 1000,1000)['score'], 2/3)

    def test_labels(self):
        refs = [{'id':'a','x':500,'y':500,'label':'dirty'}]
        s = score([{'point':[500,500],'label':'clean'}],refs,1000,1000,True)
        self.assertEqual(s['score'],0)
        self.assertEqual(s['geometry_only_f1']['medium'],1)
        self.assertEqual(s['tolerances']['medium']['fp'],1)
        self.assertEqual(s['tolerances']['medium']['fn'],1)

    def test_aspect_ratio_and_boundary(self):
        refs=[{'id':'a','x':1000,'y':250}]
        p=[{'point':[500,250]}]
        self.assertEqual(score_at(p,refs,2000,1000,[1])['tp'],1)
        # Closed radius, but no clipping an off-image prediction onto a border target.
        refs=[{'id':'a','x':0,'y':200}]
        self.assertEqual(score_at([{'point':[10,200]}],refs,1000,1000,[10])['tp'],1)
        self.assertEqual(score_at([{'point':[-1,200]}],refs,1000,1000,[10])['tp'],0)

    def test_empty_predictions(self):
        self.assertEqual(score([], [{'id':'a','x':2,'y':2}],10,10)['score'],0)

    def test_reference_duplicates(self):
        with self.assertRaises(ValueError):
            reference_distances([{'x':1,'y':1},{'x':1,'y':1}],10,10)

    def test_parsing(self):
        self.assertEqual(parse_points('```json\n{"objects":[]}\n```'),[])
        self.assertEqual(parse_points('{"objects":[{"point":[-1,500]}]}')[0]['point'],[-1,500])
        bad = ['{"objects":{}}','{"objects":[],"objects":[]}',
               '{"objects":[{"point":[true,5]}]}','{"objects":[{"point":[NaN,5]}]}',
               '{"objects":[{"point":[1e999,5]}]}',
               '{"objects":[{"point":[1,2],"point":[3,4]}]}']
        for text in bad:
            with self.assertRaises(ValueError):parse_points(text)
        with self.assertRaises(ValueError):parse_points('{"objects":[{"point":[1,2]}]}',True)
        self.assertEqual(parse_points('{"objects":[{"point":[1,2],"label":" CLEAN "}]}',True)[0]['label'],'clean')

    def test_recovery_only_when_strict_fails(self):
        ok = '{"objects":[{"label":"dirty","point":[1,2]},{"point":[3,4],"label":"clean"}]}'
        self.assertEqual(read_points(ok, True, 'lenient'), (parse_points(ok, True), 'strict'))
        with self.assertRaises(ValueError):
            read_points('{"objects":[{"point":[1,2]}, "point":[3,4]}]}', parser='strict')

    def test_recovery_cases_seen_in_pilot(self):
        missing_brace = '```json\n{"objects": [{"point": [757, 787]}, "point": [834, 694]}, {"point": [912, 800]}]}\n```'
        self.assertEqual([p['point'] for p in recover_points(missing_brace)], [[757, 787], [834, 694], [912, 800]])
        one_key = '{"objects": [{"point": [114, 377], [599, 293], [726, 122]}]}'
        self.assertEqual(len(recover_points(one_key)), 3)
        open_bracket = '{"objects": [{"point": [57, 57], "label": "x"}, {"point": [188, 30, "label": "x"}]}'
        self.assertEqual(recover_points(open_bracket)[-1]['point'], [188, 30])
        self.assertEqual(recover_points('[[1, 2, 3]]' + '[4, 5]'), [{'point': [4.0, 5.0]}])
        labelled = '{"objects": [{"point": [1, 2], "label": "Dirty"}, "point": [3, 4]}, {"point": [5, 6], "label": "clean"}'
        self.assertEqual([p['label'] for p in recover_points(labelled, True)], ['dirty', None, 'clean'])
        # As written: no swapping, rescaling, clipping or deduplication.
        self.assertEqual(recover_points('[2000, -5] [2000, -5]'), [{'point': [2000.0, -5.0]}] * 2)
        with self.assertRaises(ValueError):
            recover_points('I see no objects.')


if __name__ == '__main__':
    unittest.main()
