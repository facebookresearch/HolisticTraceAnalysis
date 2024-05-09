import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Union

import hta

import numpy as np

import pandas as pd
from hta.common.trace import Trace
from hta.common.trace_filter import (
    CompositeFilter,
    CPUOperatorFilter,
    FirstIterationFilter,
    GPUKernelFilter,
    IterationFilter,
    IterationIndexFilter,
    NameFilter,
    RankFilter,
    TimeRangeFilter,
)
from hta.common.trace_stack_filter import CombinedOperatorFilter, UnderOperatorFilter


class TestTraceFilters(unittest.TestCase):
    base_data_dir = str(Path(hta.__file__).parent.parent.joinpath("tests/data"))
    trace_dir: str = os.path.join(base_data_dir, "trace_filter")
    htaTrace: Trace = Trace(trace_dir=trace_dir)
    htaTrace.parse_traces()

    def setUp(self):
        self.htaTrace = TestTraceFilters.htaTrace
        self.df = self.htaTrace.get_trace(0)

    def testIterationFilter(self) -> None:
        f = IterationFilter([551])
        filtered_df = f(self.df)

        # 551 is present
        self.assertTrue(filtered_df[(filtered_df["iteration"] == 551)].size > 0)
        # others are not present
        self.assertEqual(filtered_df[(filtered_df["iteration"] != 551)].size, 0)

    def testRankFilter(self) -> None:
        f = RankFilter([0])

        ranks_df = []
        for rank, df in self.htaTrace.traces.items():
            # add rank column
            df["rank"] = rank
            ranks_df.append(df)

        # combine both ranks
        combined_df = pd.concat(ranks_df)
        filtered_df = f(combined_df)

        # only rank 0 is present
        self.assertEqual(filtered_df["rank"].unique().tolist(), [0])

    def testTimeRangeFilter(self) -> None:
        start_time = 1682725898237042
        end_time = 1682725898240570
        f = TimeRangeFilter((start_time, end_time))
        filtered_df = f(self.htaTrace.traces[0])

        # rows are present in time range
        self.assertEqual(filtered_df.shape[0], 93)

    def testNameFilter(self) -> None:
        name_filter = NameFilter("^nccl", symbol_table=self.htaTrace.symbol_table)
        original_df = self.df
        filtered_df = name_filter(original_df)
        filtered_df_names = filtered_df["name"].apply(
            lambda idx: self.htaTrace.symbol_table.sym_table[idx]
        )

        # should match "^nccl"
        self.assertTrue(
            filtered_df_names[filtered_df_names.str.match("^nccl")].size > 0
        )

        # others should not match "^nccl"
        self.assertTrue(
            filtered_df_names[~filtered_df_names.str.match("^nccl")].size == 0
        )

    def testNameFilterWithoutSymbolTable(self) -> None:
        name_filter = NameFilter("^nccl", name_column="s_name")
        original_df = self.df.copy()
        original_df["s_name"] = original_df["name"].apply(
            lambda idx: self.htaTrace.symbol_table.sym_table[idx]
        )
        filtered_df = name_filter(original_df)
        filtered_df_names = filtered_df["name"].apply(
            lambda idx: self.htaTrace.symbol_table.sym_table[idx]
        )

        # should match "^nccl"
        self.assertTrue(
            filtered_df_names[filtered_df_names.str.match("^nccl")].size > 0
        )

        # others should not match "^nccl"
        self.assertTrue(
            filtered_df_names[~filtered_df_names.str.match("^nccl")].size == 0
        )

    def testGPUKernelFilter(self) -> None:
        f = GPUKernelFilter()
        filtered_df = f(self.htaTrace.traces[0])

        # GPU kernel is present, note we are not reading CUDA sync events.
        self.assertTrue(filtered_df[(filtered_df["stream"] > 0)].size > 0)

        # others are not present
        self.assertTrue(filtered_df[(filtered_df["stream"] < 0)].size == 0)

    def testCPUOperatorFilter(self) -> None:
        f = CPUOperatorFilter()
        filtered_df = f(self.htaTrace.traces[0])

        # CPU operator is present
        self.assertTrue(filtered_df[(filtered_df["stream"] < 0)].size > 0)

        # others are not present
        self.assertTrue(filtered_df[(filtered_df["stream"] > 0)].size == 0)

    def testCompositeFilter(self) -> None:
        f1 = IterationFilter([551])

        start_time = 1682725898237042
        end_time = 1682725898240570
        f2 = TimeRangeFilter((start_time, end_time))

        cf = CompositeFilter([f1, f2])
        filtered_df = cf(self.htaTrace.traces[0])
        self.assertTrue(filtered_df[(filtered_df["iteration"] == 551)].size > 0)
        self.assertTrue(
            filtered_df[
                (filtered_df["ts"].ge(start_time)) & (filtered_df["ts"].le(end_time))
            ].size
            > 0
        )

    def testIterationIndexFilter(self) -> None:
        data = np.array(
            [
                [-1, 80210, 307, 12570],
                [550, 0, 169, 295776],
                [551, 7446, 230, 278517],
                [552, 14892, 312, 281212],
                [553, 22338, 155, 275735],
                [554, 29784, 49, 281059],
            ]
        )

        class TestCase(NamedTuple):
            input_df: pd.DataFrame
            input_interation_index: Union[int, List[int]]
            output_df: pd.DataFrame

        test_cases = [
            TestCase(
                pd.DataFrame(data, columns=["iteration", "index", "name", "duration"]),
                1,
                pd.DataFrame(
                    data[2:3],
                    columns=["iteration", "index", "name", "duration"],
                    index=[2],
                ),
            ),
            TestCase(
                pd.DataFrame(data, columns=["iteration", "index", "name", "duration"]),
                -1,
                pd.DataFrame(),
            ),
            TestCase(
                pd.DataFrame(data, columns=["iteration", "index", "name", "duration"]),
                5,
                pd.DataFrame(),
            ),
            TestCase(
                pd.DataFrame(
                    data[:1], columns=["iteration", "index", "name", "duration"]
                ),
                0,
                pd.DataFrame(
                    data[:1], columns=["iteration", "index", "name", "duration"]
                ),
            ),
            TestCase(
                pd.DataFrame(data[:, 1:], columns=["index", "name", "duration"]),
                0,
                pd.DataFrame(data[:, 1:], columns=["index", "name", "duration"]),
            ),
            TestCase(
                pd.DataFrame(data, columns=["iteration", "index", "name", "duration"]),
                [1, 2, 3],
                pd.DataFrame(
                    data, columns=["iteration", "index", "name", "duration"]
                ).iloc[2:5],
            ),
        ]

        for i, tc in enumerate(test_cases):
            got_df = IterationIndexFilter(tc.input_interation_index)(tc.input_df)
            self.assertTrue(
                got_df.equals(tc.output_df),
                f"Test case #{i} failed: expect=\n{tc.output_df}\n\ngot=\n{got_df}",
            )

    def testFirstIterationFilter(self) -> None:
        data = np.array(
            [
                [-1, 80210, 307, 12570],
                [550, 0, 169, 295776],
                [551, 7446, 230, 278517],
                [552, 14892, 312, 281212],
                [553, 22338, 155, 275735],
                [554, 29784, 49, 281059],
            ]
        )
        df = pd.DataFrame(data, columns=["iteration", "index", "name", "duration"])
        expected_df = pd.DataFrame(
            data[1:2], columns=["iteration", "index", "name", "duration"], index=[1]
        )
        got_df = FirstIterationFilter()(df)
        self.assertTrue(
            got_df.equals(expected_df),
            f"testIterationIndexFilter failed: expect=\n{expected_df}\n\ngot=\n{got_df}",
        )

    def testCombinedOperatorFilter(self) -> None:
        @dataclass
        class TC:
            root_op_name: str
            after_op_name: str
            before_op_name: str
            include_gpu_kernels: bool
            expected_num_events: int
            expected_num_kernels: int

        test_cases = [
            TC(
                r"## forward ##",
                r"All2All_Pooled_Wait",
                r"forward",
                False,
                2,
                0,
            ),
            TC(
                r"## forward ##",
                r"All2All_Pooled_Wait",
                r"forward",
                True,
                3,
                1,
            ),
            TC(
                r"## forward ##",
                r"All2All_Pooled_Wait",
                r"## sdd_preprocess_tensors ##",
                True,
                411,
                104,
            ),
        ]

        self.htaTrace.decode_symbol_ids(use_shorten_name=False)
        df = self.htaTrace.traces[0]
        for i, tc in enumerate(test_cases):
            f = CombinedOperatorFilter(
                tc.root_op_name,
                tc.after_op_name,
                tc.before_op_name,
                tc.include_gpu_kernels,
            )
            got_df = f(df)
            self.assertEqual(
                got_df.shape[0],
                tc.expected_num_events,
                f"test case #{i}: expect {tc.expected_num_events} events; got {got_df.shape[0]}",
            )
            cuda_kernels = got_df.loc[got_df["stream"].gt(0)]
            self.assertEqual(
                tc.expected_num_kernels,
                cuda_kernels.shape[0],
                f"test case #{i}: expect {tc.expected_num_kernels} cuda kernels, got {cuda_kernels.shape[0]}",
            )

    def testUnderOperatorFilter(self) -> None:
        op_name = "forward"
        self.htaTrace.decode_symbol_ids(use_shorten_name=False)
        df = FirstIterationFilter()(self.htaTrace.traces[0])
        f = UnderOperatorFilter(op_name=op_name, position=0, include_gpu_kernels=True)
        self.assertEqual(f(df).shape[0], 146)


class TestTraceFiltersSyncEvents(unittest.TestCase):
    """This test checks for corner cases where cuda_sync events are present.
    The "Context Sync" and "Event Sync" events are on stream = -1 but are
    also on the GPU and running device events.
    """

    base_data_dir = str(Path(hta.__file__).parent.parent.joinpath("tests/data"))
    trace_dir: str = os.path.join(base_data_dir, "critical_path/cuda_event_sync")
    htaTrace: Trace = Trace(trace_dir=trace_dir)
    htaTrace.parse_traces()

    def setUp(self):
        self.htaTrace = TestTraceFiltersSyncEvents.htaTrace
        self.df = self.htaTrace.get_trace(0)

    def testGPUKernelFilter(self) -> None:
        f = GPUKernelFilter()
        filtered_df = f(
            self.htaTrace.traces[0], symbol_table=self.htaTrace.symbol_table
        )

        # GPU kernel is present
        self.assertGreater(filtered_df[(filtered_df["stream"] > 0)].size, 0)

        # We will also see 3 sync events that are with stream == -1
        self.assertEqual(len(filtered_df[(filtered_df["stream"] < 0)]), 3)
        # print(filtered_df[filtered_df.stream == -1])

    def testCPUOperatorFilter(self) -> None:
        f = CPUOperatorFilter()
        filtered_df = f(
            self.htaTrace.traces[0], symbol_table=self.htaTrace.symbol_table
        )

        # CPU operator is present
        self.assertTrue(filtered_df[(filtered_df["stream"] < 0)].size > 0)

        # GPU kernels are not present
        self.assertTrue(filtered_df[(filtered_df["stream"] > 0)].size == 0)

        # CUDA sync events (with stream == -1) are also not present
        sym_id_map = self.htaTrace.symbol_table.get_sym_id_map()
        event_sync_id = sym_id_map.get("Event Sync", -1)
        context_sync_id = sym_id_map.get("Context Sync", -1)
        self.assertEqual(
            filtered_df.query(
                f"stream == -1 and (name == {event_sync_id} or name == {context_sync_id})"
            ).size,
            0,
        )
