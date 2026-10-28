"""No-network root exposure derivation, preservation and recovery checks."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import link_grip_clock_v03 as link
import publish_grip_clock_v03 as pub
from test_publish_grip_clock_v03 import API, Operation


def report():
    successes = {'clock': 7, 'blind_clock': 8, 'fixed': 6, 'teacher': 128}
    return dict(schema='dvidia.piper-grip-clock-study/v0.3', qualified_robot_skill=False,
        physical_robot_ready=False,
        variants={variant: dict(scheduled=128, valid=128, strict_successes=number)
                  for variant, number in successes.items()},
        rows=[dict(case_id=f'case-{index:03}', condition='nominal' if index < 64 else 'post_grip',
                   variant=variant, score=dict(valid=True, success=index < successes[variant]))
              for variant in link.VARIANTS for index in range(128)])


class RootLinkTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.parent = 'a'*40
        self.dataset = 'b'*40
        self.before = {'README.md': b'---\nsdk: static\nlicense: mit\n---\n\n# Existing DVIDIA lab\n\nPreserve all original license notices.\n',
                       'index.html': b'<!doctype html><html><main><h1>Existing lab</h1>'+link.OLD_SCOPE+b'\n<section id="old">Preserve the existing gallery.</section></main></html>'}
        self.contents = dict(self.before, **{'.gitattributes': pub.ATTRIBUTES,
            link.VIEWER: b'<html>Verified v03 viewer</html>',
            pub.NAMESPACE+'viewer/geometry.bin': b'preserve new geometry',
            'learned-placement/v02/viewer/index.html': b'preserve old viewer'})
        self.api = API(self.contents)
        self.report = self.home/'report.json'; self.report.write_bytes(pub.encoded(report()))
        self.output = self.home/'staged'

    def stage(self):
        return link.stage(self.parent, self.report, self.dataset, self.output,
                          api=self.api, downloader=self.api.download)

    def publish(self):
        return link.publish(self.output/link.PLAN, api=self.api, downloader=self.api.download, operation_cls=Operation)

    def test_stage_derives_exact_counts_and_pin_and_keeps_readme_prefix_and_other_html(self):
        with patch.object(pub, 'client', side_effect=AssertionError('stage cannot access credentials')):
            plan = self.stage(); checked, tree, payload = link.checked_plan(self.output/link.PLAN)
        self.assertEqual(plan, checked)
        self.assertEqual(set(payload), set(link.ALLOWED))
        self.assertTrue(payload['README.md'].startswith(self.before['README.md']))
        self.assertEqual(payload['index.html'].count(link.NEW_SCOPE), 1)
        self.assertNotIn(link.OLD_SCOPE, payload['index.html'])
        self.assertIn(b'Preserve the existing gallery.', payload['index.html'])
        self.assertLess(payload['index.html'].index(b'piper-grip-clock-v03'), payload['index.html'].index(b'id="old"'))
        for raw in payload.values():
            self.assertIn(b'learned grip clock 7/128; blind clock 8/128; fixed reference timing 6/128; authored grip-wait teacher 128/128', raw)
            self.assertIn(self.dataset.encode(), raw)
            self.assertIn(link.VIEWER.encode(), raw)
            self.assertNotIn(b'outperforms', raw)
        self.assertEqual(tree, self.api.repos['space']['trees'][self.parent])

    def test_publish_changes_only_root_docs_preserving_all_native_and_old_versions(self):
        self.stage(); completed = self.publish()
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(self.api.calls[0][2], list(link.ALLOWED))
        value = self.api.repos['space']
        for name, raw in self.contents.items():
            if name not in link.ALLOWED:
                self.assertEqual(value['contents'][value['head']][name], raw)
                self.assertEqual(value['trees'][value['head']][name], value['trees'][self.parent][name])
        self.assertTrue(completed['anonymous_readback'])
        self.assertFalse(completed['physical_robot_ready'])
        self.assertEqual(completed['dataset_revision'], self.dataset)
        self.assertEqual(completed['unchanged_git_lfs_files'], len(self.contents)-2)
        self.assertEqual(self.publish(), completed)

    def test_unknown_accepted_reply_reconciles_without_duplicate_commit(self):
        self.stage(); self.api.drop_after = True
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.assertTrue((self.output/link.JOURNAL).exists())
        completed = self.publish()
        self.assertEqual(len(self.api.calls), 1)
        self.assertTrue(completed['reconciled'])

    def test_unknown_before_acceptance_and_readback_failure_resume_safely(self):
        self.stage(); self.api.drop_before = True
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.api.fail_readback = True
        with self.assertRaises(RuntimeError):
            self.publish()
        completed = self.publish()
        self.assertEqual(len(self.api.calls), 2)  # one unaccepted request, one child
        self.assertTrue(completed['reconciled'])

    def test_foreign_parent_and_foreign_child_of_unknown_reply_are_refused(self):
        self.stage(); self.api.drop_after = True
        with self.assertRaises(pub.PublicationError):
            self.publish()
        self.api.foreign('space')
        before = len(self.api.calls)
        with self.assertRaisesRegex(pub.PublicationError, 'exact intended immediate child'):
            self.publish()
        self.assertEqual(len(self.api.calls), before)

    def test_unexpected_remote_change_is_not_accepted_as_success(self):
        self.stage(); self.api.extra_path = True
        with self.assertRaisesRegex(pub.PublicationError, 'unexpected Git/LFS identities'):
            self.publish()
        self.assertFalse((self.output/link.RECEIPT).exists())

    def test_report_denominator_count_roster_and_qualification_tampering_are_rejected(self):
        mutations = [lambda r: r['variants']['clock'].update(scheduled=127),
                     lambda r: r['variants']['clock'].update(strict_successes=8),
                     lambda r: r.update(physical_robot_ready=True),
                     lambda r: r.update(qualified_robot_skill=True),
                     lambda r: r['rows'].pop(),
                     lambda r: r['rows'][0].update(case_id='case-001'),
                     lambda r: r['rows'][0]['score'].update(success='yes')]
        for mutate in mutations:
            value = report(); mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(pub.PublicationError):
                link.counts(value)

    def test_old_scope_must_match_once_and_existing_v03_marker_is_refused(self):
        for before in [dict(self.before, **{'index.html': self.before['index.html'].replace(link.OLD_SCOPE, b'<p>different scope</p>')}),
                       dict(self.before, **{'index.html': self.before['index.html']+link.OLD_SCOPE}),
                       dict(self.before, **{'README.md': self.before['README.md']+link.MARKER})]:
            with self.subTest(before=before), self.assertRaises(pub.PublicationError):
                link.derive(before, report(), self.dataset)
        with self.assertRaisesRegex(pub.PublicationError, 'exact pinned dataset revision'):
            link.derive(self.before, report(), 'main')

    def test_missing_viewer_and_changed_parent_cannot_be_staged(self):
        value = self.api.repos['space']
        del value['trees'][self.parent][link.VIEWER]
        with self.assertRaisesRegex(pub.PublicationError, 'verified v0.3 viewer'):
            self.stage()
        self.assertFalse(self.output.exists())
        self.api = API(self.contents); self.api.foreign('space')
        with self.assertRaisesRegex(pub.PublicationError, 'current exact parent'):
            self.stage()

    def test_modified_staged_docs_plan_pin_and_implementation_are_rejected(self):
        self.stage()
        target = self.output/'payload'/'index.html'; raw = target.read_bytes(); target.write_bytes(raw+b'foreign')
        with self.assertRaises(pub.PublicationError):
            link.checked_plan(self.output/link.PLAN)
        target.write_bytes(raw)
        plan_file = self.output/link.PLAN; original = plan_file.read_bytes()
        value = json.loads(original); value['dataset_revision'] = 'c'*40; plan_file.write_bytes(pub.encoded(value))
        with self.assertRaisesRegex(pub.PublicationError, 'derive exactly'):
            link.checked_plan(plan_file)
        plan_file.write_bytes(original)
        implementation = self.output/'implementation'/'link_grip_clock_v03.py'
        implementation.write_bytes(implementation.read_bytes()+b'\n')
        with self.assertRaisesRegex(pub.PublicationError, 'snapshot changed'):
            link.checked_plan(plan_file)


if __name__ == '__main__':
    unittest.main()
