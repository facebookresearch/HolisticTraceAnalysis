# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict

import pandas as pd
from hta.trace_diff import DeviceType, LabeledTrace, TraceDiff


class TraceCompareTestCase(unittest.TestCase):
    def setUp(self):
        self.base_trace: Dict[str, Any] = {
            "label": "Base Trace",
            "trace_dir": "tests/data/trace_compare/base",
            "ranks": [1],
            "iterations": [1009, 1010],
            "test_rank": 1,
            "test_iteration": 1010,
            "test_num_cpu_events": 6,
            "test_num_gpu_events": 1,
            "test_user_annotation_summary": {
                "counts": {
                    "ProfilerStep#1010": 1,
                    "aten::as_strided": 1,
                    "aten::empty": 1,
                    "cudaEventQuery": 1,
                    "cudaLaunchKernel": 1,
                    "nccl:all_reduce": 1,
                },
                "total_duration": {
                    "ProfilerStep#1010": 122149,
                    "aten::as_strided": 0,
                    "aten::empty": 0,
                    "cudaEventQuery": 4,
                    "cudaLaunchKernel": 4,
                    "nccl:all_reduce": 67,
                },
            },
        }
        self.test_trace: Dict[str, Any] = {
            "label": "Test Trace",
            "trace_dir": "tests/data/trace_compare/test",
            "ranks": [1],
            "iterations": [1009, 1010],
            "test_rank": 1,
            "test_iteration": 1010,
            "counts_change_categories": {
                "cpu": {"=": 5, "+": 3, "-": 1},
                "gpu": {"-": 1},
            },
            "ops_diff": {
                "added": ["AddmmBackward0", "All2All_Pooled_Req", "CatBackward0"],
                "deleted": ["nccl:all_reduce"],
                "increased": [],
                "decreased": [],
                "unchanged": [
                    "ProfilerStep#1010",
                    "aten::as_strided",
                    "aten::empty",
                    "cudaEventQuery",
                    "cudaLaunchKernel",
                ],
            },
        }

    def test_labeled_trace(self) -> None:
        base_t = LabeledTrace(
            label=self.base_trace["label"], trace_dir=self.base_trace["trace_dir"]
        )
        self.assertListEqual(base_t.ranks(), self.base_trace["ranks"])
        self.assertListEqual(base_t.iterations(), self.base_trace["iterations"])

    def test_labeled_trace_random_label(self) -> None:
        base_t = LabeledTrace(trace_dir=self.base_trace["trace_dir"])
        self.assertTrue(base_t.label.startswith("t"))

    def test_extract_ops(self) -> None:
        base_t = LabeledTrace(
            label=self.base_trace["label"], trace_dir=self.base_trace["trace_dir"]
        )
        cpu_events: pd.DataFrame = base_t.extract_ops(
            self.base_trace["test_rank"],
            self.base_trace["test_iteration"],
            DeviceType.CPU,
        )
        self.assertEqual(cpu_events.shape[0], self.base_trace["test_num_cpu_events"])
        gpu_events: pd.DataFrame = base_t.extract_ops(
            self.base_trace["test_rank"],
            self.base_trace["test_iteration"],
            DeviceType.GPU,
        )
        self.assertEqual(gpu_events.shape[0], self.base_trace["test_num_gpu_events"])

    def test_get_ops_summary(self) -> None:
        base_t = LabeledTrace(
            label=self.base_trace["label"], trace_dir=self.base_trace["trace_dir"]
        )
        df: pd.DataFrame = base_t.get_ops_summary(
            base_t.extract_ops(
                self.base_trace["test_rank"],
                self.base_trace["test_iteration"],
                DeviceType.CPU,
            )
        )
        self.assertDictEqual(
            df.sort_values(by="name")[["name", "counts", "total_duration"]]
            .set_index("name")
            .to_dict(),
            self.base_trace["test_user_annotation_summary"],
        )

    def test_compare_trace(self) -> None:
        base_t = LabeledTrace(
            label=self.base_trace["label"], trace_dir=self.base_trace["trace_dir"]
        )
        base_rank = self.base_trace["test_rank"]
        base_iteration = self.base_trace["test_iteration"]

        test_t = LabeledTrace(
            label=self.test_trace["label"], trace_dir=self.test_trace["trace_dir"]
        )
        test_rank = self.test_trace["test_rank"]
        test_iteration = self.test_trace["test_iteration"]

        cpu_df = TraceDiff.compare_traces(
            base_t,
            test_t,
            base_rank,
            test_rank,
            base_iteration,
            test_iteration,
            device_type=DeviceType.CPU,
            use_short_name=True,
        )
        self.assertDictEqual(
            cpu_df["counts_change_categories"].value_counts().to_dict(),
            self.test_trace["counts_change_categories"]["cpu"],
        )

        gpu_df = TraceDiff.compare_traces(
            base_t,
            test_t,
            base_rank,
            test_rank,
            base_iteration,
            test_iteration,
            device_type=DeviceType.GPU,
            use_short_name=True,
        )
        self.assertDictEqual(
            gpu_df["counts_change_categories"].value_counts().to_dict(),
            self.test_trace["counts_change_categories"]["gpu"],
        )

    def test_ops_diff(self) -> None:
        base_t = LabeledTrace(
            label=self.base_trace["label"], trace_dir=self.base_trace["trace_dir"]
        )
        base_rank = self.base_trace["test_rank"]
        base_iteration = self.base_trace["test_iteration"]

        test_t = LabeledTrace(
            label=self.test_trace["label"], trace_dir=self.test_trace["trace_dir"]
        )
        test_rank = self.test_trace["test_rank"]
        test_iteration = self.test_trace["test_iteration"]

        diff = TraceDiff.ops_diff(
            base_t,
            test_t,
            base_rank,
            test_rank,
            base_iteration,
            test_iteration,
            device_type=DeviceType.CPU,
        )
        self.assertDictEqual(diff, self.test_trace["ops_diff"])

    def test_visualize(self) -> None:
        base_t = LabeledTrace(
            label=self.base_trace["label"], trace_dir=self.base_trace["trace_dir"]
        )
        base_rank = self.base_trace["test_rank"]
        base_iteration = self.base_trace["test_iteration"]

        test_t = LabeledTrace(
            label=self.test_trace["label"], trace_dir=self.test_trace["trace_dir"]
        )
        test_rank = self.test_trace["test_rank"]
        test_iteration = self.test_trace["test_iteration"]

        cpu_df = TraceDiff.compare_traces(
            base_t,
            test_t,
            base_rank,
            test_rank,
            base_iteration,
            test_iteration,
            device_type=DeviceType.CPU,
            use_short_name=True,
        )
        self.assertIsNone(TraceDiff.visualize_counts_diff(cpu_df, show_image=False))
        self.assertIsNone(TraceDiff.visualize_duration_diff(cpu_df, show_image=False))
