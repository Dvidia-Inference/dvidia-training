"""Bound prepared-source validation before metadata grouping or symlink reads."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dvidia_training.footage_pipeline import _canonical
from dvidia_training.improvement import (_prepared_episodes, _read,
                                        load_benchmark, MAX_PREPARED_EPISODES)
from test_improvement_binding import FIXTURE, authored_followup


def write_authored_bundle(root, dataset):
    """Write test contracts; the authored outcomes are never experiment evidence."""
    bundle = authored_followup(load_benchmark(FIXTURE))
    protocol, report = bundle['protocol'], bundle['report']
    raw = _canonical(dataset) + b'\n'
    source_hash = sha256(raw).hexdigest()
    (root / 'source').mkdir()
    (root / 'source' / 'dataset.json').write_bytes(raw)
    protocol['source_dataset_sha256'] = source_hash
    protocol['input_files_sha256']['source/dataset.json'] = source_hash
    experiment = protocol['experiment']
    experiment['plan']['source_dataset_sha256'] = source_hash
    experiment['plan_sha256'] = sha256(_canonical(experiment['plan']) + b'\n').hexdigest()
    report['experiment'] = deepcopy(experiment)
    raw = _canonical(protocol) + b'\n'
    protocol_hash = sha256(raw).hexdigest()
    (root / 'protocol.json').write_bytes(raw)
    (root / 'protocol.sha256').write_text(protocol_hash + '\n')
    models = {'protocol_sha256': protocol_hash,
              'candidates': [{key: value for key, value in row.items() if key != 'outcomes'}
                             for row in report['comparisons']]}
    raw = _canonical(models) + b'\n'
    (root / 'models-frozen.json').write_bytes(raw)
    report['protocol_sha256'] = protocol_hash
    report['models_frozen_sha256'] = sha256(raw).hexdigest()
    (root / 'report.json').write_bytes(_canonical(report) + b'\n')


class ImprovementReaderTests(unittest.TestCase):
    def test_reader_rejects_a_symlink_in_any_parent_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            actual = root / 'actual'
            actual.mkdir()
            (actual / 'dataset.json').write_text('{}')
            linked = root / 'linked'
            linked.symlink_to(actual, target_is_directory=True)
            with self.assertRaises(ValueError):
                _read(linked / 'dataset.json')

    def test_loader_rejects_symlinked_source_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            write_authored_bundle(root, {'episodes': []})
            (root / 'source').rename(root / 'actual-source')
            (root / 'source').symlink_to(root / 'actual-source', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'regular benchmark JSON'):
                load_benchmark(root)

    def test_oversized_or_nonobject_episode_lists_fail_before_grouping(self):
        for episodes in ([{}] * (MAX_PREPARED_EPISODES + 1), [None], [], {}, 'episodes'):
            with self.subTest(kind=type(episodes).__name__, size=len(episodes)):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    write_authored_bundle(root, {'episodes': episodes})
                    with patch('dvidia_training.footage_benchmark.explicit_subsets') as grouping:
                        with self.assertRaisesRegex(ValueError, 'prepared episode objects'):
                            load_benchmark(root)
                        grouping.assert_not_called()

    def test_prepared_metadata_cap_accepts_boundary_without_grouping(self):
        episodes = [{'id': f'episode-{i}', 'group_id': f'group-{i}',
                     'media_sha256': f'{i:064x}', 'split': 'train',
                     'source_recording_id': None, 'session_id': None, 'shoe_pair_id': None}
                    for i in range(MAX_PREPARED_EPISODES)]
        _prepared_episodes({'episodes': episodes})
        episodes.append(deepcopy(episodes[0]))
        with self.assertRaises(ValueError):
            _prepared_episodes({'episodes': episodes})

    def test_invalid_lineage_and_duplicate_ids_fail_before_grouping(self):
        row = {'id': 'episode-1', 'group_id': 'group-1', 'media_sha256': 'a' * 64,
               'split': 'train', 'source_recording_id': None, 'session_id': None, 'shoe_pair_id': None}
        for mutate in (lambda r: r.update(session_id=[]),
                       lambda r: r.update(media_sha256='not-a-digest'),
                       lambda r: r.update(split='confirmation')):
            bad = deepcopy(row)
            mutate(bad)
            with self.assertRaises(ValueError):
                _prepared_episodes({'episodes': [bad]})
        with self.assertRaisesRegex(ValueError, 'unique'):
            _prepared_episodes({'episodes': [row, deepcopy(row)]})


if __name__ == '__main__':
    unittest.main()
