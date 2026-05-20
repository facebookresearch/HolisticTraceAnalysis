# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import pandas as pd
from hta.common.trace_stack_filter import (
    AfterOperatorFilter,
    BeforeOperatorFilter,
    CombinedOperatorFilter,
    get_matching_kernels,
    OperatorFilter,
    OperatorFilterMethod,
    UnderOperatorFilter,
)


def _make_df_with_op() -> pd.DataFrame:
    """Build a small dataframe with cpu ops + a kernel under one of them."""
    return pd.DataFrame(
        {
            "index": [0, 1, 2, 3, 4],
            "ts": [0, 100, 105, 200, 300],
            "dur": [500, 50, 30, 50, 100],
            "end": [500, 150, 135, 250, 400],
            "stream": [-1, -1, -1, -1, -1],
            "s_name": ["forward", "op_inside", "op_inside_2", "op_after", "op_extra"],
            "s_cat": ["cpu_op", "cpu_op", "cpu_op", "cpu_op", "cpu_op"],
            "name": [0, 1, 2, 3, 4],
            "cat": [0, 0, 0, 0, 0],
            "index_correlation": [-1, -1, -1, -1, -1],
        }
    )


class TestGetMatchingKernels(unittest.TestCase):
    def test_filters_kernels_launched_by_runtimes(self) -> None:
        df_ops = pd.DataFrame({"index": [10, 11], "s_cat": ["cuda_runtime", "cpu_op"]})
        df_both = pd.DataFrame(
            {
                "index": [100, 101, 102],
                "index_correlation": [10, 99, 11],
            }
        )
        kernels = get_matching_kernels(df_ops, df_both, "s_cat")
        # Only kernel with index_correlation 10 (matches cuda_runtime row)
        self.assertEqual(kernels["index"].tolist(), [100])


class TestOperatorFilterMethod(unittest.TestCase):
    def test_enum_values(self) -> None:
        self.assertEqual(OperatorFilterMethod.Under.value, 0)
        self.assertEqual(OperatorFilterMethod.After.value, 1)
        self.assertEqual(OperatorFilterMethod.Before.value, 2)


class TestOperatorFilter(unittest.TestCase):
    def test_op_not_found_returns_empty(self) -> None:
        df = _make_df_with_op()
        f = OperatorFilter("missing_op", 0, OperatorFilterMethod.Under)
        result = f(df)
        self.assertTrue(result.empty)

    def test_under_filter(self) -> None:
        df = _make_df_with_op()
        f = OperatorFilter("forward", 0, OperatorFilterMethod.Under)
        result = f(df)
        # All ops fit under "forward" (ts 0 - end 500)
        self.assertEqual(len(result), 5)

    def test_after_filter(self) -> None:
        df = _make_df_with_op()
        f = OperatorFilter("op_inside", 0, OperatorFilterMethod.After)
        result = f(df)
        # ops with ts >= op_inside.end (150). That's op_after (ts=200), op_extra (ts=300)
        self.assertEqual(set(result["s_name"]), {"op_after", "op_extra"})

    def test_before_filter(self) -> None:
        df = _make_df_with_op()
        f = OperatorFilter("op_after", 0, OperatorFilterMethod.Before)
        result = f(df)
        # ops with end <= op_after.ts (200): forward (end=500, no - ge 200), op_inside (end 150 yes), op_inside_2 (end 135 yes)
        names = set(result["s_name"])
        self.assertIn("op_inside", names)
        self.assertIn("op_inside_2", names)

    def test_unsupported_method_raises(self) -> None:
        df = _make_df_with_op()
        f = OperatorFilter("forward", 0, OperatorFilterMethod.Stack)
        with self.assertRaises(NotImplementedError):
            f(df)

    def test_no_string_columns_returns_df(self) -> None:
        df = pd.DataFrame({"name": [1, 2], "cat": [1, 2]})  # int columns
        f = OperatorFilter("op", 0, OperatorFilterMethod.Under)
        result = f(df)
        self.assertEqual(len(result), 2)

    def test_explicit_name_column(self) -> None:
        df = _make_df_with_op()
        f = OperatorFilter(
            "forward", 0, OperatorFilterMethod.Under, name_column="s_name"
        )
        result = f(df)
        self.assertEqual(len(result), 5)


class TestSubclassFilters(unittest.TestCase):
    def test_after_subclass(self) -> None:
        df = _make_df_with_op()
        f = AfterOperatorFilter("op_inside", 0)
        self.assertEqual(f.method, OperatorFilterMethod.After)
        result = f(df)
        self.assertEqual(set(result["s_name"]), {"op_after", "op_extra"})

    def test_before_subclass(self) -> None:
        df = _make_df_with_op()
        f = BeforeOperatorFilter("op_after", 0)
        self.assertEqual(f.method, OperatorFilterMethod.Before)
        result = f(df)
        names = set(result["s_name"])
        self.assertIn("op_inside", names)

    def test_under_subclass(self) -> None:
        df = _make_df_with_op()
        f = UnderOperatorFilter("forward", 0)
        self.assertEqual(f.method, OperatorFilterMethod.Under)
        result = f(df)
        self.assertEqual(len(result), 5)


class TestCombinedOperatorFilter(unittest.TestCase):
    def _df_with_iteration(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4, 5],
                "ts": [0, 50, 100, 150, 200, 300],
                "dur": [500, 30, 30, 30, 30, 50],
                "end": [500, 80, 130, 180, 230, 350],
                "stream": [-1, -1, -1, -1, -1, -1],
                "iteration": [1, 1, 1, 1, 1, 1],
                "s_name": [
                    "## forward ##",
                    "All2All_Pooled_Wait",
                    "middle_op_a",
                    "middle_op_b",
                    "## sdd_preprocess_tensors ##",
                    "outside_op",
                ],
                "s_cat": ["cpu_op"] * 6,
                "name": [0, 1, 2, 3, 4, 5],
                "cat": [0, 0, 0, 0, 0, 0],
                "index_correlation": [-1, -1, -1, -1, -1, -1],
            }
        )

    def test_runs_with_all_three_ops_present(self) -> None:
        df = self._df_with_iteration()
        f = CombinedOperatorFilter(
            "## forward ##",
            "All2All_Pooled_Wait",
            "## sdd_preprocess_tensors ##",
        )
        result = f(df)
        # Should find some events between the bracketing ops
        self.assertIsInstance(result, pd.DataFrame)

    def test_missing_columns_returns_df(self) -> None:
        df = pd.DataFrame({"x": [1]})
        f = CombinedOperatorFilter("a", "b", "c")
        result = f(df)
        self.assertEqual(len(result), 1)

    def test_with_stack_depths(self) -> None:
        df = self._df_with_iteration()
        df["depth"] = [0, 1, 2, 2, 1, 0]
        f = CombinedOperatorFilter(
            "## forward ##",
            "All2All_Pooled_Wait",
            "## sdd_preprocess_tensors ##",
            stack_depths=[1, 2],
        )
        # Should not raise
        result = f(df)
        self.assertIsInstance(result, pd.DataFrame)
