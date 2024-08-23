# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

from hta.common.trace_file import (
    create_rank_to_trace_dict,
    create_rank_to_trace_dict_from_dir,
    read_trace,
    update_trace_rank,
    write_trace,
)
from hta.configs.config import logger


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
        self.trace_mixed_files = "tests/data/mixed_files"
        self.trace_file_list = ["tests/data/trace_file_list/inference_rank_1.json.gz"]
        self.logger = logger

    def test_create_rank_to_trace_dict_without_distributed_info(self):
        with self.assertLogs(logger, level="WARNING") as cm:
            self.assertEqual(
                create_rank_to_trace_dict_from_dir(self.trace_without_distributed_info),
                (
                    True,
                    {
                        0: "tests/data/distributed_info_unavailable/distributed_info_not_found.json.gz"
                    },
                ),
            )
            self.assertIn("trace file does not have the rank", cm.output[0])

    def test_create_rank_to_trace_dict_without_rank(self) -> None:
        with self.assertLogs(logger, level="WARNING") as cm:
            self.assertEqual(
                create_rank_to_trace_dict_from_dir(self.trace_without_rank),
                (True, {0: "tests/data/rank_unavailable/rank_not_found.json.gz"}),
            )
            self.assertIn("trace file does not have the rank", cm.output[0])

    def test_create_rank_to_trace_dict_with_mixed_dir(self) -> None:
        with self.assertLogs(logger, level="WARNING") as cm:
            ok, res_dict = create_rank_to_trace_dict_from_dir(self.trace_mixed_files)
            self.assertTrue(ok)
            self.assertEqual(list(res_dict.keys()), [0])
            self.assertIn("has the same rank", cm.output[0])

    def test_create_rank_to_trace_dict_with_file_list(self) -> None:
        self.assertEqual(
            create_rank_to_trace_dict(self.trace_file_list),
            (True, {1: "tests/data/trace_file_list/inference_rank_1.json.gz"}),
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
