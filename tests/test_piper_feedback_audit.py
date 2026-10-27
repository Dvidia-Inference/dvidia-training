"""Forensic audit detects numeric/identity tampering after rehashed receipts.

Uses an independently specified numeric lift/contact fixture, not native physics
or an implementation-shaped model stub. No new simulated result is implied.
"""
from hashlib import sha256
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from test_piper_feedback_score import fixture, GOAL

script = Path(__file__).resolve().parents[1]/'scripts/audit-piper-feedback.py'
spec = importlib.util.spec_from_file_location('feedback_forensic_audit', script)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FeedbackAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.folder = self.root/'attempt'
        self.folder.mkdir()
        self.episode, self.arrays, self.metadata = fixture()
        self.case = dict(id='fixture-case', start_xy=[.30, -.03], goal_xy=GOAL,
                         seed=7, condition='nominal', pulses=[])
        self.runtime = {'piper_feedback_score.py': 'a'*64}
        self.episode.update(schema='dvidia.piper-feedback-episode/v0.2', case_id=self.case['id'], seed=7,
            policy_kind='feedback', condition='nominal', pulses=[], base_model_sha256='b'*64,
            model_sha256=None, implementation_sha256=self.runtime, physical_robot_ready=False,
            asset_provenance={'fixture': True}, compiled_model_sha256='e'*64)
        self.episode['config']['object_position_m'] = [.30, -.03, .082]
        self.metadata['pulses'] = []
        self.episode['timing'] = dict(inference_samples_s=[.0001]*40, control_step_samples_s=[.003]*39+[.021],
            inference=audit._summary([.0001]*40), control_step=audit._summary([.003]*39+[.021]),
            missed_20ms_boundaries=1, wall_seconds=.2, process_peak_rss_bytes=1024)
        self.score = audit.score_feedback_episode(self.episode, self.arrays, self.metadata, GOAL)
        self.assertTrue(self.score['success'])
        self.bound = dict(case=self.case, variant='feedback', collect_labels=False, expert_probability=1.,
            protocol_sha256='c'*64, runtime_sha256=self.runtime, base_model_raw_sha256='d'*64,
            base_model_sha256='b'*64, model_raw_sha256=None, model_sha256=None)
        self.save()

    def tearDown(self):
        self.directory.cleanup()

    def save(self):
        stream = io.BytesIO()
        np.savez_compressed(stream, **self.arrays)
        values = {'started.json': audit._encoded(self.bound), 'episode.json.gz': gzip.compress(audit._encoded(self.episode), mtime=0),
            'native.npz': stream.getvalue(), 'native.json': audit._encoded(self.metadata), 'score.json': audit._encoded(self.score)}
        for name, raw in values.items():
            (self.folder/name).write_bytes(raw)
        (self.folder/'receipt.json').write_bytes(audit._encoded(dict(case_id=self.case['id'], variant='feedback',
            binding=self.bound, files={name: sha256(raw).hexdigest() for name, raw in values.items()})))

    def verify(self):
        return audit._audit_attempt(self.root, 'attempt', self.case, 'feedback', 'c'*64,
            self.runtime, 'd'*64, 'b'*64, require_full_time=False)

    def test_numeric_fixture_is_reconstructed_and_deadline_is_measured_from_samples(self):
        row, times, score = self.verify()
        self.assertTrue(row['success'])
        self.assertEqual(row['timing']['missed_20ms_boundaries'], 1)
        self.assertEqual(len(times['control_step']), 40)
        self.assertTrue(score['native_recomputed']['carry_complete'])

    def test_rehashed_native_contact_forgery_cannot_preserve_a_success_claim(self):
        self.arrays['grasp_contact_counts'][329:380] = 0.
        self.arrays['pad_normal_forces_n'][329:380] = 0.
        self.save()  # Attacker can update hashes; numeric native gates still disagree.
        with self.assertRaisesRegex(audit.AuditError, 'native reconstruction differs'):
            self.verify()

    def test_rehashed_old_protocol_relabeling_and_forged_deadline_counts_fail(self):
        self.bound['protocol_sha256'] = 'f'*64
        self.save()
        with self.assertRaisesRegex(audit.AuditError, 'supervision binding differs'):
            self.verify()
        self.bound['protocol_sha256'] = 'c'*64
        self.episode['timing']['missed_20ms_boundaries'] = 0
        self.save()
        with self.assertRaisesRegex(audit.AuditError, 'deadline count differs'):
            self.verify()

    def test_audit_refuses_incomplete_confirmation_without_creating_output(self):
        result = audit.main(['--run', str(self.root), '--output', str(self.root/'audit.json')])
        self.assertEqual(result, 1)
        self.assertFalse((self.root/'audit.json').exists())


if __name__ == '__main__':
    unittest.main()
