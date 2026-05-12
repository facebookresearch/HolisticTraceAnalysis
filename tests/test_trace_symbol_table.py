# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import tempfile
import unittest

import pandas as pd
from hta.common.trace_symbol_table import (
    decode_symbol_id_to_symbol_name,
    TraceSymbolTable,
)


class TestTraceSymbolTableBasics(unittest.TestCase):
    def test_is_empty_and_add_symbols(self) -> None:
        t = TraceSymbolTable()
        self.assertTrue(t.is_empty())
        t.add_symbols(["a", "b", "a", "c"])
        self.assertFalse(t.is_empty())
        self.assertEqual(t.get_sym_table(), ["a", "b", "c"])
        self.assertEqual(t.get_sym_id_map(), {"a": 0, "b": 1, "c": 2})

    def test_get_sym_index_alias(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["x"])
        self.assertEqual(t.get_sym_index(), {"x": 0})

    def test_find_matches_and_matched_symbols(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["foo_bar", "foo_baz", "qux"])
        self.assertEqual(set(t.find_matches(["foo"])), {0, 1})
        self.assertEqual(set(t.find_matched_symbols(["foo"])), {"foo_bar", "foo_baz"})

    def test_get_sym_index_series_and_table_series(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["a", "b"])
        self.assertEqual(len(t.get_sym_index_series()), 2)
        self.assertEqual(len(t.get_sym_table_series()), 2)


class TestTraceSymbolTableDecode(unittest.TestCase):
    def test_decode_df_creates_new_columns(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["op_a", "cpu_op"])
        df = pd.DataFrame({"name": [0, 1], "cat": [1, 0]})
        t.decode_df(df, create_new_columns=True)
        self.assertIn("s_name", df.columns)
        self.assertIn("s_cat", df.columns)
        self.assertEqual(df["s_name"].tolist(), ["op_a", "cpu_op"])

    def test_decode_df_overwrites_columns(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["op_a", "cpu_op"])
        df = pd.DataFrame({"name": [0, 1], "cat": [1, 0]})
        t.decode_df(df, create_new_columns=False)
        self.assertEqual(df["name"].tolist(), ["op_a", "cpu_op"])

    def test_encode_df(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["op_a", "cpu_op"])
        df = pd.DataFrame({"name": ["op_a"], "cat": ["cpu_op"]})
        t.encode_df(df)
        self.assertEqual(df["name"].tolist(), [0])
        self.assertEqual(df["cat"].tolist(), [1])


class TestTraceSymbolTableUpdateAndAdd(unittest.TestCase):
    def test_update_encoded_df(self) -> None:
        old = TraceSymbolTable()
        old.add_symbols(["op_a", "op_b"])
        new = TraceSymbolTable()
        new.add_symbols(["op_b", "op_a"])  # different order

        df = pd.DataFrame({"name": [0, 1], "cat": [1, 0]})  # encoded with old
        new.update_encoded_df(df, old)
        # After update: 0 ("op_a" in old) -> new index of "op_a" = 1
        self.assertEqual(df["name"].tolist(), [1, 0])

    def test_add_symbols_to_trace_df(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["op_a", "op_b"])
        df = pd.DataFrame({"name": [0, 1, 99]})
        t.add_symbols_to_trace_df(df, "name")
        # Out-of-range gets empty string
        self.assertEqual(df["name"].tolist(), ["op_a", "op_b", ""])


class TestTraceSymbolTableMasks(unittest.TestCase):
    def test_get_operator_or_cuda_runtime_mask(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["cpu_op", "cuda_runtime", "cuda_driver", "kernel"])
        df = pd.DataFrame(
            {
                "cat": [
                    t.get_sym_id_map()["cpu_op"],
                    t.get_sym_id_map()["kernel"],
                    t.get_sym_id_map()["cuda_driver"],
                ]
            }
        )
        mask = t.get_operator_or_cuda_runtime_mask(df)
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_get_runtime_launch_events_mask(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["cudaLaunchKernel", "noise"])
        df = pd.DataFrame(
            {
                "name": [
                    t.get_sym_id_map()["cudaLaunchKernel"],
                    t.get_sym_id_map()["noise"],
                ],
                "index_correlation": [5, 5],
            }
        )
        mask = t.get_runtime_launch_events_mask(df)
        self.assertEqual(mask.tolist(), [True, False])

    def test_get_events_mask_none(self) -> None:
        t = TraceSymbolTable()
        df = pd.DataFrame({"name": [0, 1]})
        mask = t.get_events_mask(df, None)
        self.assertFalse(mask.any())

    def test_get_events_mask_with_pattern(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["ncclAllReduce", "memcpy"])
        df = pd.DataFrame({"name": [0, 1]})
        mask = t.get_events_mask(df, ["nccl"])
        self.assertEqual(mask.tolist(), [True, False])


class TestTraceSymbolTableGetCategoryHelpers(unittest.TestCase):
    def test_helpers_return_lists(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(
            ["cpu_op", "kernel", "cudaLaunchKernel", "Memcpy DtoH", "ProfilerStep#1"]
        )
        self.assertIsInstance(t.get_cpu_event_cat_ids(), list)
        self.assertIsInstance(t.get_gpu_kernel_cat_ids(), list)
        self.assertIsInstance(t.get_kernel_launch_ids(), list)
        self.assertIsInstance(t.get_memory_name_ids(), list)
        self.assertIsInstance(t.get_profiler_step_ids(), list)


class TestTraceSymbolTableFromCsv(unittest.TestCase):
    def test_round_trip_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = TraceSymbolTable()
            t.add_symbols(["a", "b", "c"])
            path = os.path.join(tmp, "out.csv")
            t._to_csv_file(path)
            loaded = TraceSymbolTable.from_csv_file(path)
            self.assertEqual(loaded.get_sym_table(), ["a", "b", "c"])

    def test_from_csv_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            TraceSymbolTable.from_csv_file("/no/such/file.csv")

    def test_from_csv_missing_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            pd.DataFrame({"other": [1, 2]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "expect a column"):
                TraceSymbolTable.from_csv_file(path)


class TestCombineAndCloneAndCreateFromDf(unittest.TestCase):
    def test_combine_symbol_tables(self) -> None:
        t1 = TraceSymbolTable()
        t1.add_symbols(["a", "b"])
        t2 = TraceSymbolTable()
        t2.add_symbols(["b", "c"])
        combined = TraceSymbolTable.combine_symbol_tables([t1, t2])
        self.assertEqual(combined.get_sym_table(), ["a", "b", "c"])

    def test_clone(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["a", "b"])
        c = TraceSymbolTable.clone(t)
        self.assertEqual(c.get_sym_table(), t.get_sym_table())
        # Mutating clone doesn't affect original
        c.add_symbols(["new"])
        self.assertNotIn("new", t.get_sym_index())

    def test_create_from_df(self) -> None:
        df = pd.DataFrame({"name": ["a", "b"], "cat": ["c", "d"]})
        t = TraceSymbolTable.create_from_df(df)
        self.assertEqual(set(t.get_sym_table()), {"a", "b", "c", "d"})

    def test_create_from_df_invalid_raises(self) -> None:
        df = pd.DataFrame({"x": [1]})
        with self.assertRaisesRegex(ValueError, "name and cat columns"):
            TraceSymbolTable.create_from_df(df)

    def test_create_from_symbol_id_map(self) -> None:
        t = TraceSymbolTable.create_from_symbol_id_map({"a": 0, "b": 2})
        self.assertEqual(t.get_sym_table()[0], "a")
        self.assertEqual(t.get_sym_table()[2], "b")
        # Index 1 not present in map -> Undefined-1
        self.assertEqual(t.get_sym_table()[1], "Undefined-1")


class TestDecodeSymbolIdToSymbolName(unittest.TestCase):
    def test_decode_with_short_names(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["foo<int>(bar)", "noop"])
        df = pd.DataFrame({"name": [0, 1], "cat": [1, 0]})
        decode_symbol_id_to_symbol_name(df, t, use_shorten_name=True)
        self.assertIn("s_name", df.columns)
        self.assertIn("s_cat", df.columns)

    def test_decode_with_user_annotation(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["op_a", "op_b"])
        df = pd.DataFrame({"name": [0], "cat": [1], "user_annotation": [0]})
        decode_symbol_id_to_symbol_name(df, t, use_shorten_name=False)
        self.assertIn("s_user_annotation", df.columns)

    def test_decode_skips_non_int_columns(self) -> None:
        t = TraceSymbolTable()
        t.add_symbols(["a"])
        df = pd.DataFrame({"name": ["already_str"]})
        decode_symbol_id_to_symbol_name(df, t, use_shorten_name=False)
        # Should NOT add s_name since name is not int dtype
        self.assertNotIn("s_name", df.columns)
