# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import re
import unittest
from typing import cast

import pandas as pd
from hta.common.types import (
    DeviceType,
    GroupingPattern,
    infer_device_type,
    MEMCPY_TYPE_TO_STR,
    MemcpyType,
    to_grouping_pattern,
)


class TestDeviceType(unittest.TestCase):
    def test_unknown_when_no_relevant_columns(self) -> None:
        df = pd.DataFrame({"x": [1]})
        self.assertEqual(infer_device_type(df), DeviceType.UNKNOWN)

    def test_gpu_when_only_stream_positive(self) -> None:
        df = pd.DataFrame({"stream": [1, 2, 3]})
        self.assertEqual(infer_device_type(df), DeviceType.GPU)

    def test_cpu_when_only_stream_neg1(self) -> None:
        df = pd.DataFrame({"stream": [-1, -1]})
        self.assertEqual(infer_device_type(df), DeviceType.CPU)

    def test_gpu_when_full_columns_stream_positive(self) -> None:
        df = pd.DataFrame({"stream": [1, 2], "pid": [10, 10], "tid": [20, 20]})
        self.assertEqual(infer_device_type(df), DeviceType.GPU)

    def test_gpu_when_pid_zero(self) -> None:
        df = pd.DataFrame({"stream": [-1, -1], "pid": [0, 0], "tid": [20, 20]})
        self.assertEqual(infer_device_type(df), DeviceType.GPU)

    def test_cpu_with_full_columns_stream_neg1_pid_nonzero(self) -> None:
        df = pd.DataFrame({"stream": [-1, -1], "pid": [10, 10], "tid": [20, 20]})
        self.assertEqual(infer_device_type(df), DeviceType.CPU)


class TestMemcpyType(unittest.TestCase):
    def test_memcpy_type_to_str_complete(self) -> None:
        for mt in MemcpyType:
            self.assertIn(mt, MEMCPY_TYPE_TO_STR)
            self.assertIsInstance(MEMCPY_TYPE_TO_STR[mt], str)


class TestGroupingPattern(unittest.TestCase):
    def test_match_normal(self) -> None:
        gp = GroupingPattern(re.compile(r"^foo"))
        self.assertTrue(gp.match("foobar"))
        self.assertFalse(gp.match("baz"))

    def test_match_inverse(self) -> None:
        gp = GroupingPattern(re.compile(r"^foo"), inverse_match=True)
        self.assertFalse(gp.match("foobar"))
        self.assertTrue(gp.match("baz"))

    def test_hashable(self) -> None:
        gp1 = GroupingPattern(re.compile(r"^x"), inverse_match=False)
        gp2 = GroupingPattern(re.compile(r"^x"), inverse_match=False)
        self.assertEqual(hash(gp1), hash(gp2))
        # Can be used as a dict key
        d = {gp1: "v"}
        self.assertEqual(d[gp2], "v")


class TestToGroupingPattern(unittest.TestCase):
    def test_passthrough_grouping_pattern(self) -> None:
        gp = GroupingPattern(re.compile(r"^x"))
        self.assertIs(to_grouping_pattern(gp), gp)

    def test_from_string(self) -> None:
        gp = to_grouping_pattern("^foo", group_name="g", inverse_match=True)
        self.assertTrue(isinstance(gp, GroupingPattern))
        self.assertEqual(gp.group_name, "g")
        self.assertTrue(gp.inverse_match)

    def test_from_compiled_pattern(self) -> None:
        gp = to_grouping_pattern(re.compile(r"^bar"))
        self.assertTrue(isinstance(gp, GroupingPattern))
        self.assertFalse(gp.inverse_match)
        self.assertEqual(gp.group_name, "")

    def test_from_list(self) -> None:
        gp = to_grouping_pattern(["alpha", "beta"], group_name="lst")
        self.assertTrue(isinstance(gp, GroupingPattern))
        self.assertEqual(gp.group_name, "lst")

    def test_unsupported_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported pattern type"):
            # Intentionally pass a wrong type to test runtime validation
            to_grouping_pattern(cast(str, 123))
