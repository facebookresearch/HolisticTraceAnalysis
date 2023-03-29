# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

from hta.common.trace_file import (
    create_rank_to_trace_dict,
    read_trace,
    update_trace_rank,
    write_trace,
)


class TestTraceFile(unittest.TestCase):
    test_trace_data = {
        "distributedInfo": {"rank": 1},
        "traceEvents": [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::zeros",
                "pid": 2568503,
                "tid": 2568503,
                "ts": 1661938466265032,
                "dur": 25,
                "args": {
                    "Trace name": "PyTorch Profiler",
                    "Trace iteration": 0,
                    "External id": 1,
                    "Profiler Event Index": 0,
                },
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "ProfilerStep#1009",
                "pid": 2568503,
                "tid": 2568503,
                "ts": 1661938466265087,
                "dur": 100298,
                "args": {
                    "Trace name": "PyTorch Profiler",
                    "Trace iteration": 0,
                    "External id": 5,
                    "Profiler Event Index": 4,
                },
            },
        ],
    }

    def setUp(self) -> None:
        self.trace_without_distributed_info = "tests/data/distributed_info_unavailable"
        self.trace_without_rank = "tests/data/rank_unavailable"

    def test_create_rank_to_trace_dict_without_distributed_info(self):
        self.assertRaises(
            SystemExit, create_rank_to_trace_dict, self.trace_without_distributed_info
        )

    def test_create_rank_to_trace_dict_without_rank(self) -> None:
        self.assertRaises(
            SystemExit, create_rank_to_trace_dict, self.trace_without_rank
        )

    def test_read_write_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_trace_file = os.path.join(tmpdirname, "test.json.gz")
            write_trace(TestTraceFile.test_trace_data, test_trace_file)
            read_trace_data = read_trace(test_trace_file)
            self.assertDictEqual(read_trace_data, TestTraceFile.test_trace_data)

    def test_update_trace_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_trace_file = os.path.join(tmpdirname, "test.json.gz")
            test_rank = 99
            write_trace(TestTraceFile.test_trace_data, test_trace_file)
            update_trace_rank(test_trace_file, test_rank)
            read_trace_data = read_trace(test_trace_file)
            self.assertTrue("distributedInfo" in read_trace_data)
            self.assertTrue("rank" in read_trace_data["distributedInfo"])
            self.assertEqual(read_trace_data["distributedInfo"]["rank"], test_rank)
