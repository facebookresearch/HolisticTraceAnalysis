# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest
from collections import Counter, defaultdict
from unittest.mock import MagicMock, patch

from hta.configs.default_values import ValueType
from hta.utils.validate_trace import (
    _check_args,
    _get_argument_value_types,
    get_argument_spec,
    get_expected_arguments,
    validate_trace_format,
)


_MODULE = "hta.utils.validate_trace"


class TestValidateTrace(unittest.TestCase):
    def test_get_argument_spec_levels(self) -> None:
        for level in ("minimal", "standard", "complete"):
            spec = get_argument_spec(level)
            self.assertIsInstance(spec, list)
            self.assertGreater(len(spec), 0)

    def test_get_argument_spec_invalid_level_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_argument_spec("not-a-level")

    def test_get_expected_arguments_returns_dataframe(self) -> None:
        spec = get_argument_spec("minimal")
        df = get_expected_arguments(spec)
        self.assertIn("arg_keys", df.columns)
        self.assertIn("arg_value_types", df.columns)
        self.assertIn("arg_default_values", df.columns)
        self.assertIn("trace_df_column_name", df.columns)
        self.assertEqual(len(df), len(spec))

    def test_get_argument_value_types_round_trip(self) -> None:
        spec = get_argument_spec("minimal")
        df = get_expected_arguments(spec)
        m = _get_argument_value_types(df)
        self.assertIsInstance(m, dict)
        # Each value is a (ValueType, default) tuple
        for k, v in m.items():
            self.assertIsInstance(k, str)
            self.assertEqual(len(v), 2)

    def test_check_args_records_skipped_keys(self) -> None:
        skipped: Counter[str] = Counter()
        violations: defaultdict[str, str] = defaultdict(str)
        arg_type_map = {"existing": (ValueType.Int, 0)}
        _check_args(
            {"unknown": 5, "existing": 1},
            arg_type_map,
            skipped,
            violations,
        )
        self.assertEqual(skipped["unknown"], 1)
        self.assertEqual(len(violations), 0)

    def test_check_args_records_type_violation(self) -> None:
        skipped: Counter[str] = Counter()
        violations: defaultdict[str, str] = defaultdict(str)
        arg_type_map = {"k": (ValueType.Int, 0)}
        # Pass a string when an int is expected
        _check_args({"k": "not_an_int"}, arg_type_map, skipped, violations)
        self.assertIn("k", violations)
        self.assertIn("expect_type='Int'", violations["k"])

    def test_check_args_int_to_float_is_compatible(self) -> None:
        skipped: Counter[str] = Counter()
        violations: defaultdict[str, str] = defaultdict(str)
        arg_type_map = {"k": (ValueType.Float, 1.5)}
        _check_args({"k": 42}, arg_type_map, skipped, violations)
        self.assertEqual(len(violations), 0)

    def test_check_args_object_type_skips_check(self) -> None:
        skipped: Counter[str] = Counter()
        violations: defaultdict[str, str] = defaultdict(str)
        arg_type_map = {"k": (ValueType.Object, None)}
        _check_args({"k": "anything"}, arg_type_map, skipped, violations)
        self.assertEqual(len(violations), 0)

    @patch(f"{_MODULE}.read_trace")
    def test_validate_trace_format_read_failure(self, mock_read: MagicMock) -> None:
        mock_read.side_effect = OSError("disk error")
        ok, errors = validate_trace_format("/some/file.json")
        self.assertFalse(ok)
        self.assertIn("trace_read_error", errors)

    @patch(f"{_MODULE}.read_trace")
    def test_validate_trace_format_missing_traceevents(
        self, mock_read: MagicMock
    ) -> None:
        mock_read.return_value = {"other_key": []}
        ok, errors = validate_trace_format("/some/file.json")
        self.assertFalse(ok)
        self.assertIn("trace_data_error", errors)

    @patch(f"{_MODULE}.read_trace")
    def test_validate_trace_format_args_not_dict(self, mock_read: MagicMock) -> None:
        mock_read.return_value = {"traceEvents": [{"args": ["unexpected", "list"]}]}
        ok, errors = validate_trace_format("/some/file.json")
        self.assertFalse(ok)
        self.assertIn("args_data_type", errors)

    @patch(f"{_MODULE}.read_trace")
    def test_validate_trace_format_skipped_arguments_not_ignored(
        self, mock_read: MagicMock
    ) -> None:
        mock_read.return_value = {"traceEvents": [{"args": {"unknown_key_xyz": 1}}]}
        ok, errors = validate_trace_format(
            "/some/file.json", level="minimal", ignore_missing_arguments=False
        )
        self.assertFalse(ok)
        self.assertIn("skipped_arguments", errors)

    @patch(f"{_MODULE}.read_trace")
    def test_validate_trace_format_ok_when_no_args(self, mock_read: MagicMock) -> None:
        mock_read.return_value = {"traceEvents": [{"name": "op"}]}
        ok, errors = validate_trace_format("/some/file.json")
        self.assertTrue(ok)
        self.assertEqual(errors, {})

    @patch(f"{_MODULE}.read_trace")
    def test_validate_trace_format_type_violation_recorded(
        self, mock_read: MagicMock
    ) -> None:
        # cuda::stream is Int (in minimal level); pass a string value to
        # drive the type_violations branch (line 171).
        mock_read.return_value = {"traceEvents": [{"args": {"stream": "not_an_int"}}]}
        ok, errors = validate_trace_format("/some/file.json", level="minimal")
        self.assertFalse(ok)
        self.assertIn("type_violations", errors)

    def test_main_block_runs_via_module_invocation(self) -> None:
        # Cover the __main__ argparse block by invoking the module as a script.
        # The trace file path doesn't exist; read_trace fails and validate
        # returns False, but the argparse + print code runs first.
        import runpy
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = [
            "validate_trace.py",
            "/no/such/trace/file.json.gz",
            "--level",
            "minimal",
        ]
        sys.stdout = StringIO()
        try:
            runpy.run_module("hta.utils.validate_trace", run_name="__main__")
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
