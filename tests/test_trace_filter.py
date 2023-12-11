import os
import unittest
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


class TestTraceFilters(unittest.TestCase):
    def setUp(self):
        base_data_dir = str(Path(hta.__file__).parent.parent.joinpath("tests/data"))
        trace_dir: str = os.path.join(base_data_dir, "trace_filter")
        self.htaTrace = Trace(trace_dir=trace_dir)
        self.htaTrace.parse_traces()

    def testIterationFilter(self) -> None:
        f = IterationFilter([551])
        filtered_df = f(self.htaTrace.traces[0])

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
        start_time = 1682725898079588
        end_time = 1682725898081175
        f = TimeRangeFilter((start_time, end_time))
        filtered_df = f(self.htaTrace.traces[0])

        # rows are present in time range
        self.assertEqual(filtered_df.shape[0], 45)

    def testNameFilter(self) -> None:
        f = NameFilter("^nccl", symbol_table=self.htaTrace.symbol_table)
        rank0 = f(self.htaTrace.traces[0])
        symbol_named_rank0 = rank0["name"].apply(
            lambda idx: self.htaTrace.symbol_table.sym_table[idx]
        )

        # should match "^nccl"
        self.assertTrue(
            symbol_named_rank0[symbol_named_rank0.str.match("^nccl")].size > 0
        )

        # others should not match "^nccl"
        self.assertTrue(
            symbol_named_rank0[~symbol_named_rank0.str.match("^nccl")].size == 0
        )

    def testGPUKernelFilter(self) -> None:
        f = GPUKernelFilter()
        filtered_df = f(self.htaTrace.traces[0])

        # GPU kernel is present
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

        start_time = 1682725898079588
        end_time = 1682725898081175
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
