"""Execution scheduling checks with mocked workers; no simulator is launched."""
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dvidia_training.piper_imitation_benchmark import read
from publishing import run_piper_grip_clock_v03 as driver


class Future:
    def __init__(self, events, result=None, error=None):
        self.events, self.value, self.error = events, result, error

    def result(self):
        self.events.append(('await', self.value))
        if self.error:
            raise self.error
        return self.value


class Pool:
    def __init__(self, events, fail=None):
        self.events, self.fail = events, fail

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, root, assets, case, variant):
        item = (case['id'], variant)
        self.events.append(('submit', item))
        error = RuntimeError('worker failed') if item == self.fail else None
        return Future(self.events, item, error)


class ExecutionDriverTests(unittest.TestCase):
    def test_bind_copies_exact_driver_and_declares_four_worker_roster(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'protocol.json').write_text('{}')
            with patch.object(driver.lifecycle, '_locked', return_value=nullcontext()), \
                 patch.object(driver.lifecycle, '_study'):
                driver.bind(root)
            self.assertEqual((root/'execution-driver.py').read_bytes(), Path(driver.__file__).read_bytes())
            plan = read(root/'execution-plan.json')
            self.assertEqual(plan['workers'], 4)
            self.assertEqual(plan['process_start_method'], 'spawn')
            self.assertEqual(plan['variants'], ['clock', 'blind_clock', 'fixed', 'teacher'])
            self.assertEqual(plan['attempts_per_controller'], 128)
            self.assertEqual(plan['scheduled_attempts'], 512)
            self.assertIn('shared host', plan['timing_scope'])

    @contextmanager
    def _patched_confirm(self, root, cases, events, *, fail=None, plan_matches=True):
        frozen = {'files': {'execution-plan.json': {}, 'execution-driver.py': {}}}
        protocol = {'cases': {'confirmation': cases}}
        expected_plan = {'matched': True}
        with ExitStack() as stack:
            stack.enter_context(patch.object(driver.lifecycle, '_locked', return_value=nullcontext()))
            stack.enter_context(patch.object(driver.lifecycle, '_frozen', return_value=(protocol, frozen)))
            stack.enter_context(patch.object(driver, 'declaration', return_value=expected_plan))
            stack.enter_context(patch.object(driver, 'read', return_value=expected_plan if plan_matches else {}))
            stack.enter_context(patch.object(driver, 'sha', return_value='same'))
            stack.enter_context(patch.object(driver, 'ProcessPoolExecutor', side_effect=lambda **kwargs: Pool(events, fail)))
            stack.enter_context(patch.object(driver, 'get_context', return_value='spawn-context'))
            stack.enter_context(patch.object(driver.lifecycle, '_report', return_value={'report': True}))
            yield

    def test_dispatches_four_variants_and_awaits_each_case_before_next(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'study'
            root.mkdir()
            assets = Path(temporary)/'assets'
            assets.mkdir()
            cases = [{'id': 'case-1'}, {'id': 'case-2'}]
            events = []
            with self._patched_confirm(root, cases, events):
                self.assertEqual(driver.confirm(root, assets), {'report': True})
            expected = list(driver.lifecycle.CONFIRM_VARIANTS)
            for index, case in enumerate(cases):
                chunk = events[index*8:(index+1)*8]
                self.assertEqual([entry for kind, entry in chunk if kind == 'submit'],
                                 [(case['id'], variant) for variant in expected])
                self.assertEqual([kind for kind, _ in chunk], ['submit']*4+['await']*4)

    def test_worker_failure_waits_for_pair_and_dispatches_remaining_cases_then_raises(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'study'
            root.mkdir()
            assets = Path(temporary)/'assets'
            assets.mkdir()
            cases = [{'id': f'case-{i}'} for i in range(3)]
            events = []
            with self._patched_confirm(root, cases, events, fail=('case-1', 'clock')):
                with self.assertRaisesRegex(RuntimeError, 'all confirmation cases were dispatched'):
                    driver.confirm(root, assets)
            self.assertEqual(len([x for x in events if x[0] == 'submit']), 12)
            self.assertEqual(len([x for x in events if x[0] == 'await']), 12)
            self.assertEqual([entry[1][0] for entry in events if entry[0] == 'submit'],
                             ['case-0']*4+['case-1']*4+['case-2']*4)

    def test_plan_mismatch_is_rejected_before_worker_pool_starts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'study'
            root.mkdir()
            assets = Path(temporary)/'assets'
            assets.mkdir()
            events = []
            with self._patched_confirm(root, [{'id': 'case-1'}], events, plan_matches=False):
                with self.assertRaisesRegex(ValueError, 'must be bound before freeze'):
                    driver.confirm(root, assets)
            self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main()
