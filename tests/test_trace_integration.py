# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import unittest
from typing import cast

from hta.common.trace import parse_trace_file, Trace
from hta.utils.test_utils import get_test_data_dir


class TestParseTraceFileIntegration(unittest.TestCase):
    """Integration tests using real trace data files."""

    def test_parse_real_cpu_trace(self) -> None:
        path = os.path.join(
            get_test_data_dir(),
            "cpu_only",
            "rank-34.Jul_15_10_52_41.1074.pt.trace.json.gz",
        )
        meta, df, sym = parse_trace_file(path)
        # Real trace has metadata + non-empty df
        self.assertIsInstance(meta, dict)
        self.assertGreater(len(df), 0)
        # Symbol table populated
        self.assertGreater(len(sym.get_sym_table()), 0)
        # Standard columns added by parser
        self.assertIn("end", df.columns)
        self.assertIn("index_correlation", df.columns)


class TestTraceLoadAndQuery(unittest.TestCase):
    """End-to-end Trace loading and query tests."""

    trace: Trace

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        path = os.path.join(
            get_test_data_dir(),
            "cpu_only",
            "rank-34.Jul_15_10_52_41.1074.pt.trace.json.gz",
        )
        cls.trace = Trace(trace_files={34: path}, trace_dir="")
        cls.trace.parse_traces(use_multiprocessing=False)

    def test_get_ranks(self) -> None:
        self.assertEqual(self.trace.get_ranks(), [34])

    def test_get_trace(self) -> None:
        df = self.trace.get_trace(34)
        self.assertGreater(len(df), 0)

    def test_get_all_traces(self) -> None:
        all_traces = self.trace.get_all_traces()
        self.assertIn(34, all_traces)

    def test_get_trace_invalid_rank_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.trace.get_trace(999)

    def test_get_trace_duration(self) -> None:
        # Returns int >= 0
        d = self.trace.get_trace_duration()
        self.assertGreaterEqual(d, 0)

    def test_get_iterations_returns_list(self) -> None:
        iters = self.trace.get_iterations()
        self.assertIsInstance(iters, list)

    def test_get_raw_trace_for_one_rank(self) -> None:
        raw = self.trace.get_raw_trace_for_one_rank(34)
        self.assertIsInstance(raw, dict)
        self.assertIn("traceEvents", raw)

    def test_get_raw_trace_invalid_rank_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.trace.get_raw_trace_for_one_rank(999)

    def test_decode_symbol_ids(self) -> None:
        self.trace.decode_symbol_ids(use_shorten_name=True)
        df = self.trace.get_trace(34)
        # After decode, s_name and s_cat should exist
        self.assertIn("s_name", df.columns)


class TestTraceLoadTracesEntryPoint(unittest.TestCase):
    """Test the public load_traces() entry point."""

    def test_load_traces(self) -> None:
        path = os.path.join(
            get_test_data_dir(),
            "cpu_only",
            "rank-34.Jul_15_10_52_41.1074.pt.trace.json.gz",
        )
        trace = Trace(trace_files={0: path}, trace_dir="")
        trace.load_traces(use_multiprocessing=False)
        self.assertTrue(trace.is_parsed)
        # Re-loading is a no-op
        trace.load_traces(use_multiprocessing=False)
        self.assertTrue(trace.is_parsed)


class TestTraceCtorEmpty(unittest.TestCase):
    """Constructor edge cases."""

    def test_invalid_trace_files_type_logged(self) -> None:
        # trace_files is neither list nor dict — logged and returns
        # Intentionally pass a wrong type to test runtime validation
        Trace(trace_files=cast(dict, 42), trace_dir="")

    def test_validation_failure_raises(self) -> None:
        # Non-existent file -> validation fails
        with self.assertRaisesRegex(ValueError, "validation failed"):
            Trace(trace_files={0: "/no/such/file.json"}, trace_dir="")
