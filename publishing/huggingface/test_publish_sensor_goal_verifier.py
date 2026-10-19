"""Static release checks for the append-only v0.4 Space publisher."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import publish_sensor_goal_verifier as publisher


class SensorGoalPublisherTests(unittest.TestCase):
    def test_inventory_rejects_private_keys_and_token_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            (root/'good.json').write_text('{"schema":"public"}')
            self.assertEqual(set(publisher.inventory(root)),{'good.json'})
            (root/'bad.json').write_text('{"nested":{"api_key":"redacted"}}')
            with self.assertRaises(publisher.PublicationError): publisher.inventory(root)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'README.md').write_text('hf_'+'a'*40)
            with self.assertRaises(publisher.PublicationError): publisher.inventory(root)

    def test_replay_comparison_tolerates_only_machine_rounding(self):
        saved = {"status": "observed", "position": [0.42386426501580593, 0.0]}
        rounded = {"status": "observed", "position": [0.4238642650158061, 0.0]}
        self.assertTrue(publisher._same_replay(saved, rounded))
        self.assertFalse(publisher._same_replay(saved, {"status": "unknown", "position": rounded["position"]}))
        self.assertFalse(publisher._same_replay(saved, {"status": "observed", "position": [0.4239, 0.0]}))
    def test_root_updates_are_append_only_and_disclose_tradeoffs(self):
        parent={'index.html':b'<main><section aria-labelledby="local-goal-verifier-title">v03</section></main>',
            'README.md':b'Prior release', 'provenance.json':b'{"local_goal_verifier_v03":{"retained":true}}\n'}
        export={'cases.json':json.dumps({'cases':[{} for _ in range(272)]}).encode(),
            'publication-integrity.json':b'{"sha256":"sealed"}'}
        roots=publisher.updated_roots(parent,export,'a'*40)
        self.assertIn(b'69 of 112',roots['index.html'])
        self.assertIn(b'73/80',roots['README.md'])
        self.assertIn(b'local_goal_verifier_v03',roots['provenance.json'])
        self.assertIn(b'sensor_goal_v04',roots['provenance.json'])
        self.assertIn(b'image-goals/v04/index.html',roots['index.html'])

    def test_commit_batches_are_bounded_and_publish_pointers_last(self):
        payload = {f'image-goals/v04/{index:04}.json': b'x' * 1024 for index in range(130)}
        payload.update({name: b'root' for name in publisher.ROOT_FILES})
        batches = list(publisher._batches(payload))
        self.assertGreater(len(batches), 1)
        self.assertEqual([name for batch in batches for name in batch],
            [*sorted(name for name in payload if name not in publisher.ROOT_FILES), *sorted(publisher.ROOT_FILES)])
        self.assertTrue(all(len(batch) <= publisher.MAX_COMMIT_FILES for batch in batches))
        self.assertTrue(all(sum(map(lambda name: len(payload[name]), batch)) <= publisher.MAX_COMMIT_BYTES for batch in batches))
        self.assertEqual(publisher._remaining_batch(['a','b','c'], {'a','b'}), ['c'])
        self.assertEqual(publisher._remaining_batch(['a','b'], {'a','b'}), [])
        with self.assertRaises(publisher.PublicationError): publisher._remaining_batch(['a','b','c'], {'a','c'})

    def test_xet_attribute_additions_preserve_parent_and_match_uploaded_images(self):
        parent = b'*.bin filter=lfs diff=lfs merge=lfs -text\n'
        image = 'image-goals/v04/images/a.png'
        line = f'{image} filter=lfs diff=lfs merge=lfs -text\n'.encode()
        publisher._verify_lfs_attributes(parent, parent + line, [image], {image:{'lfs_sha256':'a'*64}})
        with self.assertRaises(publisher.PublicationError):
            publisher._verify_lfs_attributes(parent, parent + line + b'unexpected.png filter=lfs diff=lfs merge=lfs -text\n', [image], {image:{'lfs_sha256':'a'*64}})
        with self.assertRaises(publisher.PublicationError):
            publisher._verify_lfs_attributes(parent, b'*.bin filter=lfs diff=lfs merge=lfs -text\nchanged\n' + line, [image], {image:{'lfs_sha256':'a'*64}})


if __name__=='__main__': unittest.main()
