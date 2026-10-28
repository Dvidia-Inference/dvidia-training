import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import resume_grip_clock_http_v03 as driver

class FakeReader:
    def __init__(self, path, *, workers, window):
        assert workers == 12 and window == 16
        self.scope_receipts=[];self.peak_futures=0;self.closed=False
    def download(self, *args):raise AssertionError('No network in driver tests')
    def __enter__(self):return self
    def __exit__(self, *ignored):self.closed=True

class DriverTests(unittest.TestCase):
    def test_only_downloader_is_injected_and_completed_receipt_is_bound(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);plan=root/'publication-plan.json';plan.write_text('{}\n')
            result={'schema':'test','complete':True}
            def publish(path, **kwargs):
                self.assertEqual(set(kwargs), {'downloader'})
                self.assertIsInstance(kwargs['downloader'].__self__, FakeReader)
                self.assertEqual(path,plan.resolve())
                driver.pub.atomic(root/driver.pub.RECEIPT,result,exclusive=True)
                return result
            with patch.object(driver,'ReadAhead',FakeReader), patch.object(driver.pub,'publish',side_effect=publish):
                self.assertEqual(driver.run(plan),result)
            record=json.loads((root/'http-transport-attempt-complete.json').read_text())
            self.assertTrue(record['original_credential_and_parent_guards'])
            self.assertEqual(record['anonymous_get_workers'],12)
            self.assertTrue(record['fresh_final_readback'])
            self.assertEqual(record['publication_receipt'],driver.pub.receipt((root/driver.pub.RECEIPT).read_bytes()))
    def test_failure_retains_started_record_and_never_invents_completion(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);plan=root/'publication-plan.json';plan.write_text('{}\n')
            with patch.object(driver,'ReadAhead',FakeReader), patch.object(driver.pub,'publish',side_effect=driver.PublicationError('stopped')):
                with self.assertRaises(driver.PublicationError):driver.run(plan)
            self.assertTrue((root/'http-transport-attempt.json').exists())
            self.assertFalse((root/'http-transport-attempt-complete.json').exists())
    def test_foreign_journal_and_existing_record_stop_before_any_publication(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);plan=root/'publication-plan.json';plan.write_text('{}\n')
            driver.pub.atomic(root/driver.pub.JOURNAL,{'plan_sha256':'0'*64},exclusive=True)
            with patch.object(driver.pub,'publish') as publish:
                with self.assertRaises(driver.PublicationError):driver.run(plan)
                publish.assert_not_called()
            (root/driver.pub.JOURNAL).unlink()
            (root/'http-transport-attempt.json').write_text('{}')
            with patch.object(driver.pub,'publish') as publish:
                with self.assertRaises(driver.PublicationError):driver.run(plan)
                publish.assert_not_called()
if __name__=='__main__':unittest.main()
