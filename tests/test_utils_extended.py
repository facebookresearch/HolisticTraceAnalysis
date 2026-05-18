# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd
from hta.utils.utils import (
    flatten_column_names,
    get_kernel_type,
    get_memory_kernel_type,
    get_mp_pool_size,
    get_value_from_dict,
    is_comm_kernel,
    is_compute_kernel,
    is_computer_kernel,
    is_memory_kernel,
    KernelType,
    merge_kernel_intervals,
    normalize_gpu_stream_numbers,
    normalize_path,
)


class TestNormalizePath(unittest.TestCase):
    def test_absolute_path_unchanged(self) -> None:
        self.assertEqual(normalize_path("/tmp/trace"), "/tmp/trace")

    def test_relative_dot_slash_path(self) -> None:
        result = normalize_path("./subdir")
        self.assertTrue(result.endswith("subdir"))
        self.assertFalse(result.startswith("./"))

    def test_dot_slash_only(self) -> None:
        result = normalize_path("./")
        self.assertFalse(result.endswith("/"))
        self.assertTrue(len(result) > 0)

    def test_tilde_path(self) -> None:
        result = normalize_path("~/mydir")
        self.assertTrue(result.endswith("mydir"))
        self.assertFalse(result.startswith("~"))

    def test_tilde_only(self) -> None:
        result = normalize_path("~/")
        self.assertFalse(result.endswith("/"))
        self.assertTrue(len(result) > 0)

    def test_plain_path_unchanged(self) -> None:
        self.assertEqual(normalize_path("some/path"), "some/path")


class TestKernelClassification(unittest.TestCase):
    def test_nccl_is_comm_kernel(self) -> None:
        self.assertTrue(is_comm_kernel("ncclAllReduce"))
        self.assertTrue(is_comm_kernel("ncclKernel_SendRecv"))

    def test_compute_kernel_not_comm(self) -> None:
        self.assertFalse(is_comm_kernel("volta_sgemm"))

    def test_memcpy_is_memory_kernel(self) -> None:
        self.assertTrue(is_memory_kernel("Memcpy DtoH"))
        self.assertTrue(is_memory_kernel("Memcpy HtoD"))
        self.assertTrue(is_memory_kernel("Memset"))

    def test_compute_kernel_not_memory(self) -> None:
        self.assertFalse(is_memory_kernel("volta_sgemm"))

    def test_sgemm_is_compute_kernel(self) -> None:
        self.assertTrue(is_compute_kernel("volta_sgemm"))

    def test_nccl_not_compute_kernel(self) -> None:
        self.assertFalse(is_compute_kernel("ncclAllReduce"))

    def test_memcpy_not_compute_kernel(self) -> None:
        self.assertFalse(is_compute_kernel("Memcpy DtoH"))

    def test_is_computer_kernel_deprecated_alias(self) -> None:
        self.assertEqual(
            is_computer_kernel("volta_sgemm"), is_compute_kernel("volta_sgemm")
        )
        self.assertEqual(
            is_computer_kernel("ncclAllReduce"), is_compute_kernel("ncclAllReduce")
        )


class TestGetKernelType(unittest.TestCase):
    def test_communication_kernel(self) -> None:
        self.assertEqual(
            get_kernel_type("ncclAllReduce"), KernelType.COMMUNICATION.name
        )

    def test_memory_kernel(self) -> None:
        self.assertEqual(get_kernel_type("Memcpy DtoH"), KernelType.MEMORY.name)

    def test_computation_kernel(self) -> None:
        self.assertEqual(get_kernel_type("volta_sgemm"), KernelType.COMPUTATION.name)

    def test_other_kernel(self) -> None:
        self.assertEqual(
            get_kernel_type("cudaStreamSynchronize"), KernelType.OTHER.name
        )


class TestGetMemoryKernelType(unittest.TestCase):
    def test_memset(self) -> None:
        self.assertEqual(get_memory_kernel_type("Memset (Device)"), "Memset")

    def test_memcpy_dtoh(self) -> None:
        self.assertEqual(get_memory_kernel_type("Memcpy DtoH"), "Memcpy DtoH")

    def test_memcpy_htod(self) -> None:
        self.assertEqual(get_memory_kernel_type("Memcpy HtoD"), "Memcpy HtoD")

    def test_memcpy_dtod(self) -> None:
        self.assertEqual(get_memory_kernel_type("Memcpy DtoD"), "Memcpy DtoD")

    def test_non_memcpy_prefix(self) -> None:
        self.assertEqual(get_memory_kernel_type("SomeOther"), "Memcpy Unknown")


class TestMergeKernelIntervals(unittest.TestCase):
    def test_non_overlapping_intervals(self) -> None:
        df = pd.DataFrame({"ts": [0, 100, 200], "dur": [50, 50, 50]})
        result = merge_kernel_intervals(df)
        self.assertEqual(len(result), 3)

    def test_overlapping_intervals_merged(self) -> None:
        df = pd.DataFrame({"ts": [0, 30, 200], "dur": [50, 50, 50]})
        result = merge_kernel_intervals(df)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["ts"], 0)
        self.assertEqual(result.iloc[0]["end"], 80)
        self.assertEqual(result.iloc[1]["ts"], 200)
        self.assertEqual(result.iloc[1]["end"], 250)

    def test_fully_contained_interval(self) -> None:
        df = pd.DataFrame({"ts": [0, 10], "dur": [100, 20]})
        result = merge_kernel_intervals(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ts"], 0)
        self.assertEqual(result.iloc[0]["end"], 100)

    def test_single_interval(self) -> None:
        df = pd.DataFrame({"ts": [10], "dur": [50]})
        result = merge_kernel_intervals(df)
        self.assertEqual(len(result), 1)


class TestFlattenColumnNames(unittest.TestCase):
    def test_multi_index_columns_flattened(self) -> None:
        arrays = [["A", "A", "B"], ["one", "two", "one"]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[1, 2, 3]], columns=index)
        flatten_column_names(df)
        self.assertEqual(list(df.columns), ["A_one", "A_two", "B_one"])

    def test_single_index_columns_unchanged(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        flatten_column_names(df)
        self.assertEqual(list(df.columns), ["a", "b"])


class TestGetMpPoolSize(unittest.TestCase):
    def test_returns_positive_integer(self) -> None:
        result = get_mp_pool_size(1024, 10)
        self.assertGreater(result, 0)

    def test_limited_by_num_objs(self) -> None:
        result = get_mp_pool_size(1, 2)
        self.assertLessEqual(result, 2)

    def test_large_obj_size_limits_pool(self) -> None:
        result = get_mp_pool_size(10**15, 100)
        self.assertLessEqual(result, 100)


class TestNormalizeGpuStreamNumbers(unittest.TestCase):
    def test_integer_streams_unchanged(self) -> None:
        df = pd.DataFrame({"stream": [1, 2, 3]})
        normalize_gpu_stream_numbers(df)
        self.assertEqual(df["stream"].tolist(), [1, 2, 3])

    def test_non_numeric_stream_becomes_minus_one(self) -> None:
        df = pd.DataFrame({"stream": ["abc", "2", "xyz"]})
        normalize_gpu_stream_numbers(df)
        self.assertEqual(df["stream"].tolist(), [-1, 2, -1])

    def test_no_stream_column(self) -> None:
        df = pd.DataFrame({"other": [1, 2]})
        normalize_gpu_stream_numbers(df)
        self.assertNotIn("stream", df.columns)


class TestGetValueFromDict(unittest.TestCase):
    def test_simple_key(self) -> None:
        d = {"a": 1, "b": 2}
        self.assertEqual(get_value_from_dict(d, "a"), 1)

    def test_nested_key(self) -> None:
        d = {"a": {"b": {"c": 42}}}
        self.assertEqual(get_value_from_dict(d, "a.b.c"), 42)

    def test_missing_key_returns_default(self) -> None:
        d = {"a": 1}
        self.assertEqual(get_value_from_dict(d, "b", "default"), "default")

    def test_missing_nested_key_returns_default(self) -> None:
        d = {"a": {"b": 1}}
        self.assertIsNone(get_value_from_dict(d, "a.c"))

    def test_default_none(self) -> None:
        d = {"a": 1}
        self.assertIsNone(get_value_from_dict(d, "z"))

    def test_non_dict_intermediate(self) -> None:
        d = {"a": 5}
        self.assertEqual(get_value_from_dict(d, "a.b", "fallback"), "fallback")
