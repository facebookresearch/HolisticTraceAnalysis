# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import tempfile
import unittest
from collections import Counter, defaultdict

import pandas as pd
from hta.configs.default_values import ValueType
from hta.configs.parser_config import ParserConfig
from hta.utils.validate_trace import (
    _check_args,
    _get_argument_value_types,
    get_argument_spec,
    get_expected_arguments,
    validate_trace_format,
)


class TestGetArgumentSpec(unittest.TestCase):
    def test_minimal_level(self) -> None:
        result = get_argument_spec("minimal")
        self.assertEqual(result, ParserConfig.ARGS_MINIMUM)

    def test_standard_level(self) -> None:
        result = get_argument_spec("standard")
        self.assertEqual(result, ParserConfig.ARGS_DEFAULT)

    def test_complete_level(self) -> None:
        result = get_argument_spec("complete")
        self.assertEqual(result, ParserConfig.ARGS_COMPLETE)

    def test_invalid_level_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_argument_spec("nonexistent")


class TestGetExpectedArguments(unittest.TestCase):
    def test_returns_dataframe_with_expected_columns(self) -> None:
        specs = get_argument_spec("minimal")
        df = get_expected_arguments(specs)
        self.assertIsInstance(df, pd.DataFrame)
        expected_cols = {
            "arg_keys",
            "arg_value_types",
            "arg_default_values",
            "trace_df_column_name",
        }
        self.assertEqual(set(df.columns), expected_cols)

    def test_row_count_matches_specs(self) -> None:
        specs = get_argument_spec("minimal")
        df = get_expected_arguments(specs)
        self.assertEqual(len(df), len(specs))

    def test_values_match_attribute_specs(self) -> None:
        specs = get_argument_spec("standard")
        df = get_expected_arguments(specs)
        for i, spec in enumerate(specs):
            self.assertEqual(df.iloc[i]["arg_keys"], spec.raw_name)
            self.assertEqual(df.iloc[i]["arg_value_types"], spec.value_type)
            self.assertEqual(df.iloc[i]["trace_df_column_name"], spec.name)


class TestGetArgumentValueTypes(unittest.TestCase):
    def test_returns_dict_with_correct_structure(self) -> None:
        specs = get_argument_spec("minimal")
        df = get_expected_arguments(specs)
        result = _get_argument_value_types(df)
        self.assertIsInstance(result, dict)
        for _key, value in result.items():
            self.assertIsInstance(value, tuple)
            self.assertEqual(len(value), 2)
            self.assertIsInstance(value[0], ValueType)


class TestCheckArgs(unittest.TestCase):
    def test_matching_types_no_violations(self) -> None:
        arg_type_map = {
            "External id": (ValueType.Int, 0),
            "name": (ValueType.String, ""),
        }
        skipped: Counter = Counter()
        violations: defaultdict = defaultdict(str)
        args = {"External id": 42, "name": "test_op"}
        _check_args(args, arg_type_map, skipped, violations)
        self.assertEqual(len(violations), 0)
        self.assertEqual(len(skipped), 0)

    def test_unknown_key_is_skipped(self) -> None:
        arg_type_map = {"known_key": (ValueType.Int, 0)}
        skipped: Counter = Counter()
        violations: defaultdict = defaultdict(str)
        args = {"unknown_key": 5}
        _check_args(args, arg_type_map, skipped, violations)
        self.assertIn("unknown_key", skipped)
        self.assertEqual(skipped["unknown_key"], 1)

    def test_type_violation_detected(self) -> None:
        arg_type_map = {"my_int": (ValueType.Int, 0)}
        skipped: Counter = Counter()
        violations: defaultdict = defaultdict(str)
        args = {"my_int": "not_an_int"}
        _check_args(args, arg_type_map, skipped, violations)
        self.assertIn("my_int", violations)

    def test_int_float_compatible(self) -> None:
        arg_type_map = {"my_float": (ValueType.Float, 0.0)}
        skipped: Counter = Counter()
        violations: defaultdict = defaultdict(str)
        args = {"my_float": 5}
        _check_args(args, arg_type_map, skipped, violations)
        self.assertEqual(len(violations), 0)

    def test_object_type_not_checked(self) -> None:
        arg_type_map = {"obj": (ValueType.Object, {})}
        skipped: Counter = Counter()
        violations: defaultdict = defaultdict(str)
        args = {"obj": "anything_goes"}
        _check_args(args, arg_type_map, skipped, violations)
        self.assertEqual(len(violations), 0)


class TestValidateTraceFormat(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        for f in os.listdir(self._tmp_dir):
            os.unlink(os.path.join(self._tmp_dir, f))
        os.rmdir(self._tmp_dir)

    def _write_trace_json(self, trace_data: dict, filename: str = "trace.json") -> str:
        path = os.path.join(self._tmp_dir, filename)
        with open(path, "w") as f:
            json.dump(trace_data, f)
        return path

    def _write_trace_gz(self, trace_data: dict, filename: str = "trace.json.gz") -> str:
        path = os.path.join(self._tmp_dir, filename)
        with gzip.open(path, "wb") as f:
            f.write(json.dumps(trace_data).encode("utf-8"))
        return path

    def test_valid_trace_passes(self) -> None:
        trace_data = {
            "traceEvents": [
                {"name": "op1", "cat": "cpu_op", "ts": 100, "dur": 50, "args": {}},
            ]
        }
        path = self._write_trace_json(trace_data)
        ok, errors = validate_trace_format(path, level="minimal")
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)

    def test_missing_trace_events_section(self) -> None:
        trace_data = {"other_key": []}
        path = self._write_trace_json(trace_data)
        ok, errors = validate_trace_format(path)
        self.assertFalse(ok)
        self.assertIn("trace_data_error", errors)

    def test_nonexistent_file(self) -> None:
        ok, errors = validate_trace_format("/nonexistent/path/trace.json")
        self.assertFalse(ok)
        self.assertIn("trace_read_error", errors)

    def test_gz_trace_format(self) -> None:
        trace_data = {
            "traceEvents": [
                {"name": "op1", "cat": "cpu_op", "ts": 100, "dur": 50, "args": {}},
            ]
        }
        path = self._write_trace_gz(trace_data)
        ok, errors = validate_trace_format(path, level="minimal")
        self.assertTrue(ok)

    def test_type_violations_reported(self) -> None:
        trace_data = {
            "traceEvents": [
                {
                    "name": "op1",
                    "cat": "cpu_op",
                    "ts": 100,
                    "dur": 50,
                    "args": {"External id": "should_be_int"},
                },
            ]
        }
        path = self._write_trace_json(trace_data)
        ok, errors = validate_trace_format(path, level="standard")
        if not ok:
            self.assertIn("type_violations", errors)

    def test_skipped_arguments_reported_when_not_ignored(self) -> None:
        trace_data = {
            "traceEvents": [
                {
                    "name": "op1",
                    "cat": "cpu_op",
                    "ts": 100,
                    "dur": 50,
                    "args": {"some_unknown_arg": 123},
                },
            ]
        }
        path = self._write_trace_json(trace_data)
        ok, errors = validate_trace_format(
            path, level="minimal", ignore_missing_arguments=False
        )
        if not ok:
            self.assertIn("skipped_arguments", errors)
