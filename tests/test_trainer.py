"""Tests for cloning and applying the ACE-Step compatibility patch."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mohim.trainer import apply_acestep_patch


class TrainerTests(unittest.TestCase):
    """Validate patch application without invoking Git."""

    def test_missing_patch_raises(self):
        """A missing compatibility patch must fail clearly."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                apply_acestep_patch(directory, Path(directory) / "missing.patch")

    @patch("mohim.trainer._run")
    @patch("mohim.trainer.subprocess.run")
    def test_patch_is_applied_after_clean_check(self, run_mock, command_mock):
        """A clean unapplied patch is checked and then applied."""
        run_mock.return_value = subprocess.CompletedProcess([], 1)
        with tempfile.TemporaryDirectory() as directory:
            patch_file = Path(directory) / "dual-stream.patch"
            patch_file.write_text("patch", encoding="utf-8")

            apply_acestep_patch(directory, patch_file)

        self.assertEqual(command_mock.call_count, 2)
        self.assertIn("--check", command_mock.call_args_list[0].args[0])
        self.assertNotIn("--check", command_mock.call_args_list[1].args[0])

    @patch("mohim.trainer._run")
    @patch("mohim.trainer.subprocess.run")
    def test_already_applied_patch_is_skipped(self, run_mock, command_mock):
        """An already applied patch must remain idempotent."""
        run_mock.return_value = subprocess.CompletedProcess([], 0)
        with tempfile.TemporaryDirectory() as directory:
            patch_file = Path(directory) / "dual-stream.patch"
            patch_file.write_text("patch", encoding="utf-8")

            apply_acestep_patch(directory, patch_file)

        command_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
