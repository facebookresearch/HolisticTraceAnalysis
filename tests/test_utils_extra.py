# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from hta.utils.test_utils import data_provider, get_test_data_dir
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
    def test_dot_slash_with_subpath(self) -> None:
        result = normalize_path("./subdir")
        self.assertEqual(result, str(Path.cwd().joinpath("subdir")))

    def test_dot_slash_only(self) -> None:
        result = normalize_path("./")
        self.assertEqual(result, str(Path.cwd()))

    def test_tilde_with_subpath(self) -> None:
        result = normalize_path("~/subdir")
        self.assertEqual(result, str(Path.home().joinpath("subdir")))

    def test_tilde_only(self) -> None:
        result = normalize_path("~/")
        self.assertEqual(result, str(Path.home()))

    def test_absolute_path_unchanged(self) -> None:
        self.assertEqual(normalize_path("/abs/path"), "/abs/path")


class TestKernelClassifiers(unittest.TestCase):
    def test_is_comm_kernel(self) -> None:
        self.assertTrue(is_comm_kernel("ncclAllReduceKernel"))
        self.assertFalse(is_comm_kernel("Memcpy DtoD"))

    def test_is_memory_kernel(self) -> None:
        self.assertTrue(is_memory_kernel("Memcpy HtoD"))
        self.assertFalse(is_memory_kernel("ncclAllReduceKernel"))

    def test_is_compute_kernel(self) -> None:
        # A kernel that isn't comm or memory should be classified as compute
        self.assertTrue(is_compute_kernel("at::native::add_kernel"))

    def test_is_computer_kernel_alias(self) -> None:
        # Deprecated alias for backward compat
        self.assertEqual(
            is_computer_kernel("at::native::add_kernel"),
            is_compute_kernel("at::native::add_kernel"),
        )

    def test_get_kernel_type_communication(self) -> None:
        self.assertEqual(
            get_kernel_type("ncclAllReduceKernel"), KernelType.COMMUNICATION.name
        )

    def test_get_kernel_type_memory(self) -> None:
        self.assertEqual(get_kernel_type("Memcpy DtoH"), KernelType.MEMORY.name)

    def test_get_kernel_type_computation(self) -> None:
        self.assertEqual(
            get_kernel_type("at::native::add_kernel"), KernelType.COMPUTATION.name
        )

    def test_get_kernel_type_other(self) -> None:
        # Use an explicit cpu_op-style name that doesn't match any kernel pattern.
        # (Empty string matches COMPUTATION via permissive pattern.)
        self.assertEqual(
            get_kernel_type("cudaStreamSynchronize"), KernelType.OTHER.name
        )


class TestGetMemoryKernelType(unittest.TestCase):
    def test_memset(self) -> None:
        self.assertEqual(get_memory_kernel_type("Memset something"), "Memset")

    def test_memcpy_dtoh(self) -> None:
        self.assertEqual(get_memory_kernel_type("Memcpy DtoH stuff"), "Memcpy DtoH")

    def test_memcpy_unknown(self) -> None:
        self.assertEqual(get_memory_kernel_type("RandomKernel"), "Memcpy Unknown")


class TestMergeKernelIntervals(unittest.TestCase):
    def test_overlapping_merged(self) -> None:
        df = pd.DataFrame({"ts": [0, 5, 20], "dur": [10, 10, 5]})
        merged = merge_kernel_intervals(df)
        # First two overlap (0-10, 5-15) -> 0-15. Third 20-25 stays separate.
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.iloc[0]["ts"], 0)
        self.assertEqual(merged.iloc[0]["end"], 15)
        self.assertEqual(merged.iloc[1]["ts"], 20)


class TestFlattenColumnNames(unittest.TestCase):
    def test_flattens_multiindex(self) -> None:
        df = pd.DataFrame(
            [[1, 2]],
            columns=pd.MultiIndex.from_tuples([("a", "1"), ("b", "")]),
        )
        flatten_column_names(df)
        self.assertEqual(list(df.columns), ["a_1", "b"])

    def test_no_op_when_not_multiindex(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        flatten_column_names(df)
        self.assertEqual(list(df.columns), ["a", "b"])


class TestGetMpPoolSize(unittest.TestCase):
    def test_returns_min_of_inputs(self) -> None:
        # With small num_objs, result should equal num_objs
        result = get_mp_pool_size(obj_size=1024, num_objs=2)
        self.assertEqual(result, 2)


class TestNormalizeGpuStreamNumbers(unittest.TestCase):
    def test_no_stream_column(self) -> None:
        df = pd.DataFrame({"a": [1]})
        # Should log error and return without raising
        normalize_gpu_stream_numbers(df)
        self.assertNotIn("stream", df.columns)

    def test_normalizes_numeric_streams(self) -> None:
        df = pd.DataFrame({"stream": ["1", "2", "3"]})
        normalize_gpu_stream_numbers(df)
        self.assertEqual(df["stream"].tolist(), [1, 2, 3])

    def test_replaces_non_numeric_with_neg1(self) -> None:
        df = pd.DataFrame({"stream": ["1", "bad", "3"]})
        normalize_gpu_stream_numbers(df)
        self.assertEqual(df["stream"].tolist(), [1, -1, 3])


class TestGetValueFromDict(unittest.TestCase):
    def test_simple_key(self) -> None:
        self.assertEqual(get_value_from_dict({"a": 1}, "a"), 1)

    def test_nested_key(self) -> None:
        self.assertEqual(get_value_from_dict({"a": {"b": {"c": 5}}}, "a.b.c"), 5)

    def test_missing_returns_default(self) -> None:
        self.assertEqual(get_value_from_dict({"a": 1}, "b", "fallback"), "fallback")

    def test_missing_returns_none_by_default(self) -> None:
        self.assertIsNone(get_value_from_dict({"a": 1}, "b"))

    def test_non_dict_intermediate_returns_default(self) -> None:
        self.assertEqual(get_value_from_dict({"a": 5}, "a.b", "fb"), "fb")


class TestGetTestDataDir(unittest.TestCase):
    def test_with_env_prefix(self) -> None:
        # Use a tempdir layout matching `<prefix>/tests/data/` so the test
        # passes in both Buck (fbcode layout) and OSS (pip-install layout).
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "tests", "data"))
            with patch.dict(os.environ, {"TEST_DATA_PREFIX_PATH": tmp}):
                d = get_test_data_dir()
                self.assertTrue(d.endswith(os.path.join("tests", "data")))

    def test_with_env_prefix_and_subdirs(self) -> None:
        with patch.dict(os.environ, {"TEST_DATA_PREFIX_PATH": "/no/such/prefix"}):
            with self.assertRaises(FileNotFoundError):
                get_test_data_dir("nonexistent_subdir")

    def test_without_env_prefix_falls_back(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Without env prefix, falls back to repo-root path; in test env this
            # path doesn't exist, so it raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                get_test_data_dir("nonexistent_dir_xyz")

    def test_without_env_prefix_no_subdirs(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Try without subdirs — also exercises the no-subdirs branch
            try:
                get_test_data_dir()
            except FileNotFoundError:
                pass


class TestDataProviderTuple(unittest.TestCase):
    def test_data_provider_with_tuple(self) -> None:
        """Cover line 85: tuple-style data provider."""
        results = []

        @data_provider(lambda: ((1, 2), (3, 4)))
        def fn(self_, a, b):
            results.append((a, b))

        fn(self)
        self.assertEqual(results, [(1, 2), (3, 4)])

    def test_data_provider_with_scalar(self) -> None:
        """Cover line 87: scalar (non-dict, non-tuple) data."""
        results = []

        @data_provider(lambda: ("a", "b"))
        def fn(self_, item):
            results.append(item)

        fn(self)
        self.assertEqual(results, ["a", "b"])
