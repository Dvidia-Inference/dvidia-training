"""Offline receipt embedding, entry-point isolation and retained evidence."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from dvidia_training.transition_cli import write_report_page


class TransitionReportTests(unittest.TestCase):
    def test_page_embeds_exact_receipt_without_executing_label_markup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = {'kind': 'dvidia.actuator-transition-pilot.v1', 'status': 'completed',
                        'limitations': ['</script><script>window.INJECTED=true</script>'],
                        'metrics': {'test': {}}}
            raw = json.dumps(document, indent=2) + '\n'
            (root/'report.json').write_text(raw, encoding='utf8')
            write_report_page(root)
            html = (root/'report.html').read_text(encoding='utf8')
            embedded = re.search(r'<script id="receipt" type="application/json">(.*?)</script>', html, re.S).group(1)
            self.assertEqual(json.loads(embedded), raw)
            self.assertNotIn('</script><script>window.INJECTED', html)
            self.assertIn('connect-src \'none\'', html)
            self.assertNotIn('src="https:', html)
            self.assertIn('—', html)
            self.assertIn('·', html)
            self.assertNotIn('â€”', html)
            with self.assertRaises(FileExistsError):
                write_report_page(root)

    def test_wrong_kind_and_nonfinite_receipts_are_not_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for raw in ('{}', '{"kind":"dvidia.actuator-transition-pilot.v1","x":NaN}'):
                (root/'report.json').write_text(raw, encoding='utf8')
                with self.assertRaises(ValueError):
                    write_report_page(root)
                self.assertFalse((root/'report.html').exists())

    def test_help_does_not_import_training_or_native_runtime(self):
        code = '''import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'mujoco'):
        raise RuntimeError('Runtime imported during help')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from dvidia_training.transition_cli import main
main(['--help'])
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('corpus', result.stdout)


if __name__ == '__main__':
    unittest.main()
