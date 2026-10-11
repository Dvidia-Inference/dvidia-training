"""Verify grouping contracts independently of browser rendering."""
from pathlib import Path
import shutil
import subprocess
import unittest


class ReviewCoreTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser helper contracts')
    def test_grouping_and_sample_boundary_invariants(self):
        subprocess.run(['node', str(Path(__file__).with_name('observation_review_core.cjs'))], check=True,
                       capture_output=True, text=True, timeout=10)
