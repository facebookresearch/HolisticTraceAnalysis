# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
from hta.common.trace import (
    add_fwd_bwd_links,
    add_iteration,
    get_cpu_gpu_correlation,
    parse_trace_file,
    Trace,
    trace_event_timestamp_to_unixtime_ns,
    transform_correlation_to_index,
)
from hta.common.trace_symbol_table import TraceSymbolTable


class TestTraceEventTimestampToUnixtime(unittest.TestCase):
    def test_normal_conversion(self) -> None:
        meta = {"baseTimeNanoseconds": 1_000_000_000}
        result = trace_event_timestamp_to_unixtime_ns(5.0, meta)
        # 5us = 5000ns; plus base
        self.assertEqual(result, 1_000_005_000)

    def test_missing_base_time_raises(self) -> None:
        with self.assertRaisesRegex(KeyError, "baseTimeNanoseconds"):
            trace_event_timestamp_to_unixtime_ns(0.0, {})

    def test_zero_event_timestamp(self) -> None:
        meta = {"baseTimeNanoseconds": 100}
        self.assertEqual(trace_event_timestamp_to_unixtime_ns(0.0, meta), 100)


class TestTransformCorrelationToIndex(unittest.TestCase):
    def test_no_correlation_column(self) -> None:
        df = pd.DataFrame({"x": [1]})
        symbol_table = TraceSymbolTable()
        result = transform_correlation_to_index(df, symbol_table)
        # Returns df unchanged
        self.assertNotIn("index_correlation", result.columns)

    def test_links_cpu_gpu_pairs(self) -> None:
        # Build a symbol table with the right kinds
        symbol_table = TraceSymbolTable()
        symbol_table.add_symbols(
            ["Kernel", "cuda_runtime", "kernel_a", "cuLaunchKernel"]
        )
        # Map: Kernel=0, cuda_runtime=1, kernel_a=2, cuLaunchKernel=3
        df = pd.DataFrame(
            {
                "index": [675, 677],
                "stream": [7, -1],
                "cat": [
                    symbol_table.sym_index["Kernel"],
                    symbol_table.sym_index["cuda_runtime"],
                ],
                "name": [
                    symbol_table.sym_index["kernel_a"],
                    symbol_table.sym_index["cuLaunchKernel"],
                ],
                "correlation": [278204204, 278204204],
            },
            index=pd.Index([675, 677]),
        )
        result = transform_correlation_to_index(df, symbol_table)
        self.assertIn("index_correlation", result.columns)
        # CPU row links to GPU row and vice versa
        self.assertEqual(result.loc[677, "index_correlation"], 675)
        self.assertEqual(result.loc[675, "index_correlation"], 677)


class TestGetCpuGpuCorrelation(unittest.TestCase):
    def test_extracts_kernel_correlations(self) -> None:
        df = pd.DataFrame(
            {
                "index": [10, 20, 30],
                "stream": [1, 1, -1],
                "index_correlation": [50, 60, -1],
            },
            index=pd.Index([10, 20, 30]),
        )
        result = get_cpu_gpu_correlation(df)
        # Two GPU rows produce two correlation entries
        self.assertEqual(len(result), 2)
        self.assertIn("gpu_index", result.columns)
        self.assertIn("cpu_index", result.columns)


class TestAddIteration(unittest.TestCase):
    def test_assigns_iteration_to_cpu_events(self) -> None:
        symbol_table = TraceSymbolTable()
        symbol_table.add_symbols(["ProfilerStep#5", "cpu_op"])
        ps_id = symbol_table.sym_index["ProfilerStep#5"]
        cpu_id = symbol_table.sym_index["cpu_op"]
        df = pd.DataFrame(
            {
                "ts": [100, 150, 300],
                "dur": [200, 10, 5],
                "stream": [-1, -1, -1],
                "cat": [cpu_id, cpu_id, cpu_id],
                "name": [ps_id, cpu_id, cpu_id],
                "iteration": [-1, -1, -1],
                "index_correlation": [-1, -1, -1],
            }
        )
        result = add_iteration(df, symbol_table)
        # First row at ts=150 falls inside ProfilerStep#5 (ts=100, dur=200) -> iter=5
        self.assertEqual(int(df.loc[1, "iteration"]), 5)
        # Third row at ts=300 outside ProfilerStep -> -1
        self.assertEqual(int(df.loc[2, "iteration"]), -1)
        # Returned profiler_steps DataFrame
        self.assertEqual(len(result), 1)


class TestAddFwdBwdLinks(unittest.TestCase):
    def test_no_fwdbwd_events(self) -> None:
        df = pd.DataFrame({"cat": ["cpu_op", "kernel"]})
        # Should return early without modifying
        add_fwd_bwd_links(df)
        self.assertNotIn("fwdbwd_index", df.columns)

    def test_links_fwd_bwd_pairs(self) -> None:
        # cpu_op events at ts 0/tid 1/pid 10 and ts 100/tid 1/pid 10
        # fwdbwd 's' (start) at ts 0 -> matches first cpu_op
        # fwdbwd 'f' bp 'e' (end) at ts 100 -> matches second cpu_op
        df = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "ts": [0, 0, 100, 100],
                "tid": [1, 1, 1, 1],
                "pid": [10, 10, 10, 10],
                "cat": ["cpu_op", "fwdbwd", "cpu_op", "fwdbwd"],
                "ph": ["X", "s", "X", "f"],
                "bp": ["", "", "", "e"],
                "id": [-1, 100, -1, 100],
            },
            index=pd.Index([0, 1, 2, 3]),
        )
        add_fwd_bwd_links(df)
        self.assertIn("fwdbwd_index", df.columns)
        self.assertIn("fwdbwd", df.columns)
        # When merge succeeds, the "key" column is dropped
        self.assertNotIn("key", df.columns)

    def test_empty_fwdbwd_merge_returns_early(self) -> None:
        # fwdbwd events present but no matching cpu_op (keys differ) -> early return
        df = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "ts": [0, 5, 100, 105],
                "tid": [1, 1, 1, 1],
                "pid": [10, 10, 10, 10],
                "cat": ["cpu_op", "fwdbwd", "cpu_op", "fwdbwd"],
                "ph": ["X", "s", "X", "f"],
                "bp": ["", "", "", "e"],
                "id": [-1, 100, -1, 100],
            },
            index=pd.Index([0, 1, 2, 3]),
        )
        add_fwd_bwd_links(df)
        # Columns added but key remains because we returned before the drop
        self.assertIn("fwdbwd_index", df.columns)


class TestParseTraceFile(unittest.TestCase):
    def test_invalid_extension_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\.gz' or 'json'"):
            parse_trace_file("/some/file.txt")


class TestTraceFlowEvent(unittest.TestCase):
    def test_start_event(self) -> None:
        ev = Trace.flow_event(
            id=1, pid=10, tid=20, ts=100, is_start=True, name="link", cat="fwdbwd"
        )
        self.assertEqual(ev["id"], 1)
        self.assertEqual(ev["pid"], 10)
        self.assertEqual(ev["tid"], 20)
        self.assertEqual(ev["ts"], 100)
        self.assertEqual(ev["name"], "link")
        self.assertEqual(ev["cat"], "fwdbwd")
        self.assertNotIn("bp", ev)
        self.assertNotIn("args", ev)

    def test_end_event(self) -> None:
        ev = Trace.flow_event(
            id=1, pid=10, tid=20, ts=100, is_start=False, name="link", cat="fwdbwd"
        )
        # End events get bp="e"
        self.assertEqual(ev["bp"], "e")

    def test_with_args(self) -> None:
        ev = Trace.flow_event(
            id=1,
            pid=10,
            tid=20,
            ts=100,
            is_start=True,
            name="link",
            cat="fwdbwd",
            args={"key": "value"},
        )
        self.assertEqual(ev["args"], {"key": "value"})


class TestTraceConvertTimeSeries(unittest.TestCase):
    def test_missing_required_columns_returns_empty(self) -> None:
        # Build a minimal Trace-like instance using __new__ to avoid heavy __init__
        t = Trace.__new__(Trace)
        t.min_ts = 0
        df = pd.DataFrame({"a": [1]})
        result = t.convert_time_series_to_events(df, "ctr", "missing_col")
        self.assertEqual(result, [])

    def test_converts_with_required_columns(self) -> None:
        t = Trace.__new__(Trace)
        t.min_ts = 0
        df = pd.DataFrame(
            {
                "pid": [1, 2],
                "ts": [10, 20],
                "value": [100, 200],
            }
        )
        result = t.convert_time_series_to_events(df, "my_counter", "value")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["pid"], 1)
        self.assertEqual(result[0]["args"], {"my_counter": 100})

    def test_converts_with_optional_name_id(self) -> None:
        t = Trace.__new__(Trace)
        t.min_ts = 5
        df = pd.DataFrame(
            {
                "pid": [1],
                "ts": [10],
                "value": [100],
                "name": ["custom_name"],
                "id": [42],
            }
        )
        result = t.convert_time_series_to_events(df, "ctr", "value")
        self.assertEqual(result[0]["name"], "custom_name")
        self.assertEqual(result[0]["id"], 42)
        # ts gets min_ts added back
        self.assertEqual(result[0]["ts"], 15)


def _make_simple_trace_with_data(rank: int = 0) -> Trace:
    """Build a Trace instance bypassing __init__ with a small in-memory dataset."""
    t = Trace.__new__(Trace)
    t.trace_files = {rank: "/fake/path"}
    t.trace_path = "/fake"
    t.traces = {rank: pd.DataFrame({"ts": [0, 100, 200], "iteration": [-1, 0, 1]})}
    t.symbol_table = TraceSymbolTable()
    t.meta_data = {rank: {"device_type": "GPU"}}
    t.min_ts = 0
    t.is_parsed = True
    return t


class TestTraceGetters(unittest.TestCase):
    def test_get_ranks(self) -> None:
        t = _make_simple_trace_with_data(rank=3)
        self.assertEqual(t.get_ranks(), [3])

    def test_get_first_rank_with_arg(self) -> None:
        t = _make_simple_trace_with_data(rank=3)
        self.assertEqual(t._get_first_rank(7), 7)

    def test_get_first_rank_default(self) -> None:
        t = _make_simple_trace_with_data(rank=3)
        self.assertEqual(t._get_first_rank(), 3)

    def test_get_first_rank_no_ranks(self) -> None:
        t = Trace.__new__(Trace)
        t.traces = {}
        self.assertEqual(t._get_first_rank(), -1)

    def test_get_iterations_returns_sorted(self) -> None:
        t = _make_simple_trace_with_data()
        # iterations [-1, 0, 1] -> sorted >=0 = [0, 1]
        self.assertEqual(t.get_iterations(), [0, 1])

    def test_get_iterations_no_column(self) -> None:
        t = Trace.__new__(Trace)
        t.traces = {0: pd.DataFrame({"ts": [0]})}
        self.assertEqual(t.get_iterations(0), [])

    def test_get_iterations_invalid_rank(self) -> None:
        t = _make_simple_trace_with_data()
        self.assertEqual(t.get_iterations(99), [])

    def test_get_trace_duration(self) -> None:
        t = _make_simple_trace_with_data()
        self.assertEqual(t.get_trace_duration(), 200)

    def test_get_trace(self) -> None:
        t = _make_simple_trace_with_data()
        df = t.get_trace(0)
        self.assertEqual(len(df), 3)

    def test_get_trace_invalid_rank_raises(self) -> None:
        t = _make_simple_trace_with_data()
        with self.assertRaises(ValueError):
            t.get_trace(99)

    def test_get_all_traces(self) -> None:
        t = _make_simple_trace_with_data()
        all_traces = t.get_all_traces()
        self.assertEqual(set(all_traces.keys()), {0})

    def test_get_device_type(self) -> None:
        t = _make_simple_trace_with_data()
        self.assertEqual(t.get_device_type(), "GPU")


class TestTraceFilenameValidation(unittest.TestCase):
    def test_normalize_trace_filenames_invalid_type_raises(self) -> None:
        t = Trace.__new__(Trace)
        # Intentionally set a wrong type to test runtime validation
        t.trace_files = cast(dict, "not_a_dict")
        t.trace_path = "/some/path"
        with self.assertRaisesRegex(ValueError, "Expected trace_files"):
            t._normalize_trace_filenames()

    def test_normalize_trace_filenames_relative_to_absolute(self) -> None:
        t = Trace.__new__(Trace)
        t.trace_files = {0: "rank0.json"}
        t.trace_path = "/data/traces"
        t._normalize_trace_filenames()
        self.assertEqual(t.trace_files[0], "/data/traces/rank0.json")

    def test_normalize_trace_filenames_already_absolute(self) -> None:
        t = Trace.__new__(Trace)
        t.trace_files = {0: "/abs/path.json"}
        t.trace_path = "/some/other"
        t._normalize_trace_filenames()
        self.assertEqual(t.trace_files[0], "/abs/path.json")

    def test_validate_trace_files_missing(self) -> None:
        t = Trace.__new__(Trace)
        t.trace_files = {0: "/no/such/file.json"}
        self.assertFalse(t._validate_trace_files())

    def test_validate_trace_files_invalid_extension(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            t = Trace.__new__(Trace)
            t.trace_files = {0: f.name}
            self.assertFalse(t._validate_trace_files())

    def test_validate_trace_files_valid(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            path = f.name
        t = Trace.__new__(Trace)
        t.trace_files = {0: path}
        self.assertTrue(t._validate_trace_files())


class TestTraceAlignAllRanks(unittest.TestCase):
    def test_align_all_ranks_empty(self) -> None:
        t = Trace.__new__(Trace)
        t.traces = {}
        t.min_ts = 0
        # Should return early without raising
        t._align_all_ranks()

    def test_align_all_ranks_subtracts_min(self) -> None:
        t = Trace.__new__(Trace)
        t.traces = {
            0: pd.DataFrame({"ts": [10, 20]}),
            1: pd.DataFrame({"ts": [5, 15]}),
        }
        t.min_ts = 0
        t._align_all_ranks()
        self.assertEqual(t.min_ts, 5)
        # rank 0: 10-5=5, 20-5=15
        self.assertEqual(t.traces[0]["ts"].tolist(), [5, 15])
        # rank 1: 5-5=0, 15-5=10
        self.assertEqual(t.traces[1]["ts"].tolist(), [0, 10])


class TestTraceLoadTracesAlreadyParsed(unittest.TestCase):
    def test_load_traces_warns_when_already_parsed(self) -> None:
        t = Trace.__new__(Trace)
        t.is_parsed = True
        # Should return early without raising
        t.load_traces()


class TestGetRawTraceForOneRank(unittest.TestCase):
    def test_invalid_rank_raises(self) -> None:
        t = Trace.__new__(Trace)
        t.trace_files = {0: "/fake.json"}
        with self.assertRaises(ValueError):
            t.get_raw_trace_for_one_rank(99)


class TestAlignAndFilterTrace(unittest.TestCase):
    def test_no_traces(self) -> None:
        t = Trace.__new__(Trace)
        t.traces = {}
        # Should return without error
        t.align_and_filter_trace()


class TestParseTracesEmpty(unittest.TestCase):
    def test_no_ranks_logs_error(self) -> None:
        t = Trace.__new__(Trace)
        t.trace_files = {}
        # No-op other than logging an error
        t.parse_traces()
        self.assertFalse(t.is_parsed)


class TestGetTraceStartUnixtimeNs(unittest.TestCase):
    def test_invalid_rank_raises(self) -> None:
        t = Trace.__new__(Trace)
        t.traces = {}
        t.meta_data = {}
        with self.assertRaisesRegex(ValueError, "No trace found"):
            t.get_trace_start_unixtime_ns(99)

    def test_valid_rank_returns_unixtime(self) -> None:
        t = Trace.__new__(Trace)
        t.traces = {0: pd.DataFrame()}
        t.meta_data = {0: {"baseTimeNanoseconds": 1_000_000_000}}
        t.min_ts = 5  # 5us
        # 5us = 5000ns + base = 1_000_005_000
        self.assertEqual(t.get_trace_start_unixtime_ns(0), 1_000_005_000)


class TestDecodeSymbolIds(unittest.TestCase):
    def test_decodes_for_each_rank(self) -> None:
        t = Trace.__new__(Trace)
        sym = TraceSymbolTable()
        sym.add_symbols(["op_a", "cpu_op"])
        t.symbol_table = sym
        df = pd.DataFrame({"name": [0, 1], "cat": [1, 0]})
        t.traces = {0: df}
        t.decode_symbol_ids(use_shorten_name=False)
        # Should add s_name and s_cat columns
        self.assertIn("s_name", df.columns)
        self.assertIn("s_cat", df.columns)


class TestParseTraceFileSuccess(unittest.TestCase):
    @patch("hta.common.trace.parse_trace_dataframe")
    def test_empty_df_returns_early(self, mock_parse: MagicMock) -> None:
        mock_parse.return_value = (
            {"meta": "data"},
            pd.DataFrame(),
            TraceSymbolTable(),
        )
        meta, df, st = parse_trace_file("/some/file.json")
        self.assertTrue(df.empty)
        self.assertEqual(meta, {"meta": "data"})


class TestWriteRawTrace(unittest.TestCase):
    def test_writes_gzipped_json(self) -> None:
        import gzip
        import json
        import os
        import tempfile

        t = Trace.__new__(Trace)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gz")
            t.write_raw_trace(path, {"a": 1})
            with gzip.open(path, "rt") as f:
                self.assertEqual(json.load(f), {"a": 1})


class TestFilterIrrelevantGpuKernels(unittest.TestCase):
    def test_no_profiler_steps_skips(self) -> None:
        t = Trace.__new__(Trace)
        sym = TraceSymbolTable()
        sym.add_symbols(["cpu_op", "kernel"])
        t.symbol_table = sym
        t.traces = {0: pd.DataFrame()}
        t.meta_data = {0: {"device_type": "GPU"}}
        # No ProfilerStep symbols -> early return
        t._filter_irrelevant_gpu_kernels()

    def test_one_profiler_step_skips(self) -> None:
        t = Trace.__new__(Trace)
        sym = TraceSymbolTable()
        sym.add_symbols(["cpu_op", "kernel", "ProfilerStep#1"])
        t.symbol_table = sym
        t.traces = {0: pd.DataFrame()}
        t.meta_data = {0: {"device_type": "GPU"}}
        # Only one ProfilerStep -> skip filter
        t._filter_irrelevant_gpu_kernels()
