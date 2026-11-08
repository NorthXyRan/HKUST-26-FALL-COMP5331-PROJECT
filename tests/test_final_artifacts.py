"""Independent checks of the frozen package; skipped until a package exists."""
from collections import Counter, deque
import math
from itertools import groupby
import hashlib
import json
import subprocess
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'results/final_m2_m4'


@unittest.skipUnless((PACKAGE / 'manifest.json').is_file(), 'final package not yet generated')
class FrozenArtifactTests(unittest.TestCase):
    def test_all_saved_file_and_source_hashes(self):
        manifest = json.loads((PACKAGE / 'manifest.json').read_text())
        actual_files = {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        self.assertEqual(actual_files, set(manifest['files_sha256']))
        for name, digest in manifest['files_sha256'].items():
            self.assertEqual(hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest(), digest, name)
        # The saved experiment predates M5's modeling.py refactor. Validate the
        # immutable producing snapshot, not today's evolving source files.
        # This is the commit that first published this manifest and its outputs.
        producing_commit = '8cf638c122937aeceee5c77a641f69b75bf08276'
        for name, digest in manifest['source_sha256'].items():
            saved = subprocess.run(['git', 'show', f'{producing_commit}:{name}'],
                                   cwd=ROOT, capture_output=True, check=False)
            source = saved.stdout if saved.returncode == 0 else (ROOT/name).read_bytes()
            self.assertEqual(hashlib.sha256(source).hexdigest(), digest, name)

    def test_stream_constraints_timestamps_and_reference_alignment(self):
        config = json.loads((PACKAGE / 'protocol.json').read_text())
        original = pd.read_csv(ROOT/'data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt', sep=r'\s+', names=['t','u','v'])
        for rate in config['rates_percent_of_original']:
            for seed in config['seeds']:
                run = PACKAGE/f'{rate}p_seed_{seed}'
                schedule = pd.read_csv(run/'timestamps.csv')
                reference_sets = []
                for condition in ('random','activity_matched'):
                    frame = pd.read_csv(run/f'{condition}_stream.csv.gz')
                    audit = pd.read_csv(run/f'{condition}_audit.csv.gz')
                    anomalies = frame.loc[frame.label == 1]
                    self.assertEqual(len(anomalies), int(len(original)*rate/100))
                    self.assertEqual(Counter(anomalies.t), Counter(schedule.t))
                    pd.testing.assert_frame_equal(frame.loc[frame.label == 0, ['t','u','v']].sort_values(['t','u','v']).reset_index(drop=True),
                                                  original.sort_values(['t','u','v']).reset_index(drop=True))
                    reference_sets.append(audit[['t','reference_u','reference_v']])
                    seen, pairs = set(), set()
                    recent, activity = deque(), Counter()
                    audit_rows = {(int(r.t), frozenset((int(r.u), int(r.v)))): r for r in audit.itertuples(index=False)}
                    for t, group in groupby(frame.itertuples(index=False, name=None), key=lambda r: r[0]):
                        batch = list(group)
                        benign_pairs = {frozenset((u,v)) for _,u,v,label in batch if label == 0}
                        used = set()
                        for _,u,v,label in batch:
                            if label == 0:
                                continue
                            pair = frozenset((u,v))
                            self.assertNotEqual(u,v)
                            self.assertTrue({u,v} <= seen)
                            self.assertNotIn(pair, pairs)
                            self.assertNotIn(pair, benign_pairs)
                            self.assertNotIn(pair, used)
                            used.add(pair)
                            record = audit_rows[(t, pair)]
                            ref_u, ref_v = record.reference_u, record.reference_v
                            self.assertIn(frozenset((ref_u, ref_v)), benign_pairs)
                            f = lambda n: math.log1p(activity[n])
                            expected_error = min(abs(f(ref_u)-f(u)) + abs(f(ref_v)-f(v)),
                                                 abs(f(ref_u)-f(v)) + abs(f(ref_v)-f(u))) / 2
                            self.assertAlmostEqual(record.activity_error, expected_error)
                        for row in sorted(batch, key=lambda r: (r[1], r[2], r[3])):
                            _, a, b, _ = row
                            recent.append((a, b))
                            activity[a] += 1
                            activity[b] += 1
                            if len(recent) > 1000:
                                old_a, old_b = recent.popleft()
                                activity[old_a] -= 1
                                activity[old_b] -= 1
                        seen.update(u for _,u,_,_ in batch)
                        seen.update(v for _,_,v,_ in batch)
                        pairs.update(benign_pairs | used)
                pd.testing.assert_frame_equal(*reference_sets)

    def test_metrics_recompute_and_thresholds_are_shared(self):
        table = pd.read_csv(PACKAGE/'metrics_all_seeds.csv', float_precision='round_trip')
        self.assertEqual(len(table), 54)
        for row in table.itertuples(index=False):
            run = PACKAGE/f'{row.rate}p_seed_{row.seed}'
            prediction = pd.read_csv(run/f'{row.feature_set}_{row.setting}_predictions.csv.gz', float_precision='round_trip')
            self.assertAlmostEqual(average_precision_score(prediction.label, prediction.score), row.pr_auc)
            self.assertAlmostEqual(roc_auc_score(prediction.label, prediction.score), row.roc_auc)
            predicted = prediction.score >= row.threshold
            np.testing.assert_array_equal(prediction.prediction, predicted.astype(int))
            self.assertAlmostEqual(predicted[prediction.label == 0].mean(), row.actual_test_fpr)
            self.assertAlmostEqual(predicted[prediction.label == 1].mean(), row.recall_at_validation_1pct_fpr)
            self.assertLessEqual(row.actual_validation_fpr, .01)
        thresholds = table.pivot(index=['rate','seed','feature_set'], columns='setting', values='threshold')
        np.testing.assert_array_equal(thresholds.random_to_random, thresholds.random_to_activity_matched)

    def test_feature_metadata_and_split_membership(self):
        config = json.loads((PACKAGE/'protocol.json').read_text())
        for run in PACKAGE.glob('*p_seed_*'):
            for condition in ('random','activity_matched'):
                frame = pd.read_csv(run/f'{condition}_features.csv.gz')
                self.assertFalse(frame.isna().any().any())
                expected = np.where(frame.t < config['train_end_exclusive'], 'train',
                                    np.where(frame.t < config['validation_end_exclusive'], 'validation', 'test'))
                np.testing.assert_array_equal(frame.split, expected)
                for feature_set in ('H1000','combined_H'):
                    cols = json.loads((run/f'{feature_set}_columns.json').read_text())
                    self.assertTrue(set(cols) <= set(frame.columns))
                    self.assertFalse(set(cols) & {'t','u','v','split','event_id','H5_is_fraud'})
                    self.assertFalse(any('fraud' in col or 'label' in col for col in cols))


if __name__ == '__main__':
    unittest.main()
