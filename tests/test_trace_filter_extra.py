# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest
from typing import cast

import pandas as pd
from hta.common.trace_filter import (
    CompositeFilter,
    CPUOperatorFilter,
    Filter,
    FirstIterationFilter,
    GPUKernelFilter,
    IterationFilter,
    IterationIndexFilter,
    MemCopyEventFilter,
    NameFilter,
    NameIdColumnFilter,
    NameStringColumnFilter,
    QueryFilter,
    RankFilter,
    TimeRangeFilter,
    ZeroDurationFilter,
)
from hta.common.trace_symbol_table import TraceSymbolTable


def _make_symbol_table(symbols: list[str]) -> TraceSymbolTable:
    t = TraceSymbolTable()
    t.add_symbols(symbols)
    return t


class TestIterationFilter(unittest.TestCase):
    def test_int_arg(self) -> None:
        f = IterationFilter(1)
        self.assertEqual(f.iterations, [1])

    def test_list_arg(self) -> None:
        f = IterationFilter([1, 2])
        self.assertEqual(f.iterations, [1, 2])

    def test_invalid_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "Iterations must"):
            # Intentionally pass a wrong type to test runtime validation
            IterationFilter(cast(int, "bad"))

    def test_call_filters(self) -> None:
        df = pd.DataFrame({"iteration": [1, 2, 3, 1], "x": [10, 20, 30, 40]})
        result = IterationFilter([1])(df)
        self.assertEqual(result["x"].tolist(), [10, 40])

    def test_call_warns_when_no_iteration_column(self) -> None:
        df = pd.DataFrame({"x": [1]})
        result = IterationFilter(1)(df)
        # Returns df unchanged
        self.assertEqual(len(result), 1)


class TestIterationIndexFilter(unittest.TestCase):
    def test_invalid_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "iteration_index"):
            IterationIndexFilter("bad")  # pyre-ignore[6]

    def test_int_arg(self) -> None:
        f = IterationIndexFilter(0)
        self.assertEqual(f.iteration_index, [0])

    def test_call_no_iteration_column(self) -> None:
        df = pd.DataFrame({"x": [1]})
        result = IterationIndexFilter(0)(df)
        self.assertEqual(len(result), 1)

    def test_call_only_neg1(self) -> None:
        df = pd.DataFrame({"iteration": [-1, -1]})
        # Returns df unchanged
        result = IterationIndexFilter(0)(df)
        self.assertEqual(len(result), 2)

    def test_call_filters_by_index(self) -> None:
        df = pd.DataFrame({"iteration": [-1, 100, 200, 300], "x": [0, 1, 2, 3]})
        # index 0 -> iteration 100, index 1 -> iteration 200
        result = IterationIndexFilter([0, 1])(df)
        self.assertEqual(set(result["x"].tolist()), {1, 2})

    def test_call_no_match(self) -> None:
        df = pd.DataFrame({"iteration": [100, 200]})
        result = IterationIndexFilter([99])(df)
        self.assertTrue(result.empty)


class TestFirstIterationFilter(unittest.TestCase):
    def test_picks_first(self) -> None:
        df = pd.DataFrame({"iteration": [100, 200, 100], "x": [1, 2, 3]})
        f = FirstIterationFilter()
        result = f(df)
        self.assertEqual(set(result["x"].tolist()), {1, 3})


class TestRankFilter(unittest.TestCase):
    def test_invalid_type(self) -> None:
        with self.assertRaisesRegex(TypeError, "ranks"):
            RankFilter("bad")  # pyre-ignore[6]

    def test_int_arg(self) -> None:
        self.assertEqual(RankFilter(0).ranks, [0])

    def test_no_rank_column(self) -> None:
        df = pd.DataFrame({"x": [1]})
        # Returns df unchanged
        self.assertEqual(len(RankFilter(0)(df)), 1)

    def test_filters(self) -> None:
        df = pd.DataFrame({"rank": [0, 1, 2], "x": [10, 20, 30]})
        result = RankFilter([0, 2])(df)
        self.assertEqual(set(result["x"].tolist()), {10, 30})


class TestTimeRangeFilter(unittest.TestCase):
    def test_invalid_tuple_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "tuple of two"):
            TimeRangeFilter([1, 2])  # pyre-ignore[6]

    def test_invalid_order_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "less than or equal"):
            TimeRangeFilter((10, 5))

    def test_no_ts_column(self) -> None:
        df = pd.DataFrame({"x": [1]})
        # Returns df unchanged
        self.assertEqual(len(TimeRangeFilter((0, 100))(df)), 1)

    def test_filters_by_time(self) -> None:
        df = pd.DataFrame({"ts": [0, 50, 90], "dur": [10, 10, 20]})
        # End times: 10, 60, 110. Range (0, 100) selects ts>=0 & end<=100
        result = TimeRangeFilter((0, 100))(df)
        self.assertEqual(len(result), 2)


class TestNameStringColumnFilter(unittest.TestCase):
    def test_no_name_column(self) -> None:
        df = pd.DataFrame({"x": [1]})
        # Returns df unchanged when no name col
        result = NameStringColumnFilter("nccl")(df)
        self.assertEqual(len(result), 1)

    def test_filters_by_pattern(self) -> None:
        df = pd.DataFrame({"name": ["ncclAll", "memcpy", "compute"]})
        result = NameStringColumnFilter("nccl")(df)
        self.assertEqual(result["name"].tolist(), ["ncclAll"])


class TestNameIdColumnFilter(unittest.TestCase):
    def test_returns_unchanged_without_symbol_table(self) -> None:
        df = pd.DataFrame({"name": [1, 2]})
        result = NameIdColumnFilter("nccl")(df)
        self.assertEqual(len(result), 2)

    def test_filters_with_symbol_table(self) -> None:
        t = _make_symbol_table(["ncclAll", "memcpy"])
        df = pd.DataFrame({"name": [0, 1]})
        result = NameIdColumnFilter("nccl")(df, t)
        self.assertEqual(result["name"].tolist(), [0])


class TestNameFilter(unittest.TestCase):
    def test_empty_df(self) -> None:
        result = NameFilter("nccl")(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_filters_with_string_column(self) -> None:
        df = pd.DataFrame({"name": ["ncclAll", "memcpy"]})
        result = NameFilter("nccl")(df)
        self.assertEqual(result["name"].tolist(), ["ncclAll"])

    def test_filters_with_provided_symbol_table_ctor(self) -> None:
        t = _make_symbol_table(["ncclAll", "memcpy"])
        df = pd.DataFrame({"name": [0, 1]})
        result = NameFilter("nccl", symbol_table=t)(df)
        self.assertEqual(result["name"].tolist(), [0])

    def test_filters_with_call_symbol_table(self) -> None:
        t = _make_symbol_table(["ncclAll", "memcpy"])
        df = pd.DataFrame({"name": [0, 1]})
        result = NameFilter("nccl")(df, symbol_table=t)
        self.assertEqual(result["name"].tolist(), [0])


class TestQueryFilter(unittest.TestCase):
    def test_filter_by_query(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = QueryFilter("x > 1")(df)
        self.assertEqual(result["x"].tolist(), [2, 3])

    def test_zero_duration_filter(self) -> None:
        df = pd.DataFrame({"dur": [0, 5, 10]})
        result = ZeroDurationFilter(df)
        self.assertEqual(set(result["dur"].tolist()), {5, 10})


class TestGPUKernelFilter(unittest.TestCase):
    def test_no_stream_column(self) -> None:
        df = pd.DataFrame({"x": [1]})
        result = GPUKernelFilter()(df)
        self.assertEqual(len(result), 1)

    def test_no_symbol_table(self) -> None:
        df = pd.DataFrame({"stream": [0, -1, 1], "correlation": [0, -1, 5]})
        result = GPUKernelFilter()(df)
        # stream >= 0 AND correlation >= 0
        self.assertEqual(len(result), 2)

    def test_with_symbol_table(self) -> None:
        t = _make_symbol_table(["Event Sync", "compute"])
        df = pd.DataFrame(
            {
                "stream": [0, -1, -1],
                "correlation": [5, -1, -1],
                "name": [1, 0, 1],
            }
        )
        result = GPUKernelFilter()(df, t)
        self.assertGreaterEqual(len(result), 1)


class TestCPUOperatorFilter(unittest.TestCase):
    def test_no_stream_column(self) -> None:
        df = pd.DataFrame({"x": [1]})
        result = CPUOperatorFilter()(df)
        self.assertEqual(len(result), 1)

    def test_no_symbol_table(self) -> None:
        df = pd.DataFrame({"stream": [-1, 0, -1], "correlation": [0, 5, 0]})
        result = CPUOperatorFilter()(df)
        # stream == -1 only
        self.assertEqual(len(result), 2)

    def test_with_symbol_table(self) -> None:
        t = _make_symbol_table(["Event Sync", "compute"])
        df = pd.DataFrame({"stream": [-1, 0], "correlation": [-1, 5], "name": [1, 1]})
        result = CPUOperatorFilter()(df, t)
        self.assertGreaterEqual(len(result), 1)


class TestCompositeFilter(unittest.TestCase):
    def test_invalid_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "instances of Filter"):
            CompositeFilter([object()])  # pyre-ignore[6]

    def test_applies_filters_in_order(self) -> None:
        df = pd.DataFrame(
            {"rank": [0, 1, 2], "iteration": [10, 20, 30], "x": [1, 2, 3]}
        )
        cf = CompositeFilter([RankFilter([0, 1]), IterationFilter([10])])
        result = cf(df)
        self.assertEqual(result["x"].tolist(), [1])


class TestMemCopyEventFilter(unittest.TestCase):
    def test_empty_df(self) -> None:
        result = MemCopyEventFilter("Memcpy DtoH")(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_no_symbol_match_returns_empty(self) -> None:
        t = _make_symbol_table(["other"])
        df = pd.DataFrame({"name": [0], "cat": [0]})
        result = MemCopyEventFilter("Memcpy DtoH", symbol_table=t)(df)
        self.assertTrue(result.empty)

    def test_filters_matching(self) -> None:
        t = _make_symbol_table(["Memcpy DtoH", "gpu_memcpy", "other"])
        df = pd.DataFrame({"name": [0, 0, 2], "cat": [1, 1, 0], "x": [10, 20, 30]})
        result = MemCopyEventFilter("Memcpy DtoH")(df, symbol_table=t)
        self.assertEqual(result["x"].tolist(), [10, 20])


class TestFilterAbstract(unittest.TestCase):
    def test_filter_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Filter()  # pyre-ignore[45]
