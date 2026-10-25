import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training import video_goal_install_benchmark as benchmark


class EvidenceReaderTests(unittest.TestCase):
    def test_report_reference_cannot_read_outside_the_sealed_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root/'episode.json.gz').write_bytes(b'test')
            inventory = {'episode.json.gz': {}}
            for value in ('../outside.json.gz', '/tmp/outside.json.gz', 'absent.json.gz',
                          './episode.json.gz', 'directory/../episode.json.gz', '', None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    benchmark._evidence_file(root, value, inventory)
            self.assertEqual(benchmark._evidence_file(root, 'episode.json.gz', inventory), root/'episode.json.gz')

    def test_episode_expansion_is_bounded_before_json_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'large.json.gz'
            path.write_bytes(gzip.compress(b' '*33))
            with patch.object(benchmark, 'MAX_EPISODE_BYTES', 32):
                with self.assertRaisesRegex(ValueError, 'byte limit'):
                    benchmark._read_episode(path)


if __name__ == '__main__': unittest.main()
