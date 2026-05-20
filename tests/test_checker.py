# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import tempfile
import unittest
from unittest.mock import patch

from hta.utils.checker import is_valid_directory, OperationOutcome


class TestChecker(unittest.TestCase):
    def test_operation_outcome_dataclass(self) -> None:
        o = OperationOutcome(success=True, reason="ok")
        self.assertTrue(o.success)
        self.assertEqual(o.reason, "ok")

    def test_valid_readable_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = is_valid_directory(tmp)
            self.assertTrue(result.success)
            self.assertEqual(result.reason, "")

    def test_valid_writable_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = is_valid_directory(tmp, must_be_writable=True)
            self.assertTrue(result.success)

    def test_writable_required_but_not_writable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Pretend the directory isn't writable
            with patch(
                "hta.utils.checker.os.access",
                side_effect=lambda p, m: m != os.W_OK,
            ):
                result = is_valid_directory(tmp, must_be_writable=True)
                self.assertFalse(result.success)
                self.assertIn("not writable", result.reason)

    def test_empty_path(self) -> None:
        result = is_valid_directory("")
        self.assertFalse(result.success)
        self.assertIn("non-empty string", result.reason)

    def test_path_does_not_exist(self) -> None:
        result = is_valid_directory("/no/such/path/here")
        self.assertFalse(result.success)
        self.assertIn("does not exist", result.reason)

    def test_path_is_not_dir(self) -> None:
        with tempfile.NamedTemporaryFile() as f:
            result = is_valid_directory(f.name)
            self.assertFalse(result.success)
            self.assertIn("not a directory", result.reason)

    def test_unreadable_dir_falls_through(self) -> None:
        # Path exists, is a dir, but not readable -> hits the final else
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "hta.utils.checker.os.access",
                side_effect=lambda p, m: m != os.R_OK,
            ):
                result = is_valid_directory(tmp)
                self.assertFalse(result.success)
                self.assertIn("is not a valid path", result.reason)
