# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest
from collections import namedtuple
from pathlib import Path
from typing import List
from unittest.mock import patch

import hta
from hta.common.trace import PHASE_COUNTER
from hta.trace_analysis import TimeSeriesTypes, TraceAnalysis


class TraceAnalysisTestCase(unittest.TestCase):
    vision_transformer_t: TraceAnalysis
    inference_t: TraceAnalysis
    df_index_resolver_t: TraceAnalysis
    rank_non_gpu_t: TraceAnalysis

    @classmethod
    def setUpClass(cls):
        super(TraceAnalysisTestCase, cls).setUpClass()
        cls.base_data_dir = str(Path(__file__).parent.parent.joinpath("tests/data"))
        cls.vision_transformer_trace_dir: str = os.path.join(
            cls.base_data_dir, "vision_transformer"
        )
        cls.inference_trace_dir: str = os.path.join(
            cls.base_data_dir, "inference_single_rank"
        )
        cls.df_index_resolver_trace_dir: str = os.path.join(
            cls.base_data_dir, "df_index_resolver"
        )
        cls.rank_non_gpu_trace_dir: str = os.path.join(
            cls.base_data_dir, "rank_non_gpu/"
        )
        cls.h100_trace_dir: str = os.path.join(cls.base_data_dir, "h100")
        cls.vision_transformer_t = TraceAnalysis(
            trace_dir=cls.vision_transformer_trace_dir
        )
        cls.inference_t = TraceAnalysis(trace_dir=cls.inference_trace_dir)
        cls.df_index_resolver_t = TraceAnalysis(
            trace_dir=cls.df_index_resolver_trace_dir
        )
        cls.rank_non_gpu_t = TraceAnalysis(trace_dir=cls.rank_non_gpu_trace_dir)
        cls.h100_trace_t = TraceAnalysis(trace_dir=cls.h100_trace_dir)
        cls.mtia_single_rank_dir: str = os.path.join(
            cls.base_data_dir, "mtia_trace_single_rank/"
        )
        cls.mtia_single_rank_trace_t = TraceAnalysis(trace_dir=cls.mtia_single_rank_dir)

    def setUp(self):
        self.overlaid_trace_dir = self.base_data_dir
        self.overlaid_trace_file = os.path.join(
            str(Path(self.overlaid_trace_dir)), "overlaid_rank-0.json.gz"
        )

    @patch.object(hta.common.trace.Trace, "write_raw_trace")
    def test_frequent_cuda_kernel_sequences(self, mock_write_trace):
        frequent_patterns_dfs = (
            self.vision_transformer_t.get_frequent_cuda_kernel_sequences(
                operator_name="aten::linear",
                output_dir=self.overlaid_trace_dir,
                visualize=False,
                compress_other_kernels=True,
            )
        )
        self.assertIn(
            "|".join(
                [
                    "aten::linear",
                    "Memset (Device)",
                    "volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn",
                    "void at::native::elementwise_kernel",
                ]
            ),
            frequent_patterns_dfs.iloc[2]["pattern"],
        )
        self.assertEqual(frequent_patterns_dfs.iloc[2]["count"], 48)
        self.assertEqual(
            frequent_patterns_dfs.iloc[2]["GPU kernel duration (us)"], 11300
        )
        self.assertEqual(frequent_patterns_dfs.iloc[2]["CPU op duration (us)"], 9652)
        mock_write_trace.assert_called_once()
        trace_output_filename, _ = mock_write_trace.call_args.args
        self.assertEqual(trace_output_filename, self.overlaid_trace_file)

    def test_no_frequent_cuda_kernel_sequences_found(self):
        frequent_patterns_dfs = (
            self.df_index_resolver_t.get_frequent_cuda_kernel_sequences(
                operator_name="aten::clone",
                output_dir=self.overlaid_trace_dir,
                rank=1,
                visualize=False,
                compress_other_kernels=True,
            )
        )
        self.assertTrue(frequent_patterns_dfs.empty)

    def test_get_cuda_kernel_launch_stats_training_multiple_ranks(self):
        dataframe_dict = self.vision_transformer_t.get_cuda_kernel_launch_stats(
            ranks=[1, 7], visualize=False
        )
        rank_1_df, rank_7_df = dataframe_dict[1], dataframe_dict[7]
        row1 = rank_1_df[rank_1_df["correlation"] == 373234]
        row2 = rank_7_df[rank_7_df["correlation"] == 327327]

        self.assertEqual(row1["cpu_duration"].item(), 16)
        self.assertEqual(row1["gpu_duration"].item(), 2394)
        self.assertEqual(row1["launch_delay"].item(), 16491)
        self.assertEqual(row2["cpu_duration"].item(), 21)
        self.assertEqual(row2["gpu_duration"].item(), 94)
        self.assertEqual(row2["launch_delay"].item(), 41)

    def test_get_cuda_kernel_launch_stats_inference_single_rank(self):
        dataframe_list = self.inference_t.get_cuda_kernel_launch_stats(visualize=False)
        rank_0_df = dataframe_list[0]
        row = rank_0_df[rank_0_df["correlation"] == 684573]

        self.assertEqual(row["cpu_duration"].item(), 9)
        self.assertEqual(row["gpu_duration"].item(), 3)
        self.assertEqual(row["launch_delay"].item(), 20)

    def test_get_mtia_kernel_launch_stats_inference_single_rank(self):
        dataframe_list = self.mtia_single_rank_trace_t.get_cuda_kernel_launch_stats(
            visualize=False
        )
        rank_0_df = dataframe_list[0]
        row = rank_0_df[rank_0_df["correlation"] == 423]

        self.assertEqual(row["cpu_duration"].item(), 435.200)
        self.assertEqual(row["gpu_duration"].item(), 124.768)
        self.assertEqual(row["launch_delay"].item(), 340.291)

    def test_get_cuda_kernel_launch_stats_for_h100(self):
        dataframe_dict = self.h100_trace_t.get_cuda_kernel_launch_stats(
            ranks=[1], visualize=False
        )
        rank_1_df = dataframe_dict[1]
        row = rank_1_df[rank_1_df["correlation"] == 1281474]

        self.assertEqual(rank_1_df.shape[0], 32835)
        self.assertEqual(row["cpu_duration"].item(), 20)
        self.assertEqual(row["gpu_duration"].item(), 31)
        self.assertEqual(row["launch_delay"].item(), 41)

    def test_get_profiler_steps(self):
        results = self.vision_transformer_t.get_profiler_steps()
        expected = [15, 16, 17, 18]
        self.assertListEqual(results, expected)

    def test_include_last_profiler_step(self):
        t = TraceAnalysis(
            trace_files={
                0: os.path.join(
                    self.base_data_dir, "vision_transformer", "rank-0.json.gz"
                )
            },
            trace_dir="",
            include_last_profiler_step=True,
        )
        expected_results = [15, 16, 17, 18, 19]
        self.assertListEqual(t.get_profiler_steps(), expected_results)

    def test_get_potential_stragglers(self):
        TCase = namedtuple(
            "TCase", ["profiler_steps", "num_candidates", "expected_results"]
        )
        p_steps = self.vision_transformer_t.get_profiler_steps()
        test_cases: List[TCase] = [
            TCase(p_steps[:1], -1, [7]),
            TCase(p_steps[:1], 2, [6, 7]),
            TCase(p_steps, 2, [0, 1]),
        ]

        for tc in test_cases:
            got_stragglers = sorted(
                self.vision_transformer_t.get_potential_stragglers(
                    profiler_steps=tc.profiler_steps, num_candidates=tc.num_candidates
                )
            )
            self.assertListEqual(got_stragglers, tc.expected_results)

    def test_comm_comp_overlap(self):
        comm_comp_overlap = self.vision_transformer_t.get_comm_comp_overlap(
            visualize=False
        )
        self.assertAlmostEqual(
            comm_comp_overlap.iloc[0]["comp_comm_overlap_pctg"],
            round((240322 * 100) / 1091957, 3),
            delta=0.01,
        )
        self.assertAlmostEqual(
            comm_comp_overlap.iloc[1]["comp_comm_overlap_pctg"],
            round((262584 * 100) / 1233755, 3),
            delta=0.01,
        )

    def test_temporal_breakdown(self):
        idle_time = self.vision_transformer_t.get_temporal_breakdown(visualize=False)
        self.assertAlmostEqual(
            idle_time.iloc[0]["idle_time_pctg"],
            round((552069 * 100) / 2033570, 3),
            delta=0.01,
        )
        self.assertAlmostEqual(
            idle_time.iloc[0]["compute_time_pctg"],
            round((596651 * 100 / 2033570), 3),
            delta=0.01,
        )
        self.assertAlmostEqual(
            idle_time.iloc[0]["non_compute_time_pctg"],
            round(884850 * 100 / 2033570, 3),
            delta=0.01,
        )
        self.assertAlmostEqual(
            idle_time.iloc[1]["idle_time_pctg"],
            round(431771 * 100 / 2032757, 3),
            delta=0.01,
        )
        self.assertAlmostEqual(
            idle_time.iloc[1]["compute_time_pctg"],
            round(596759 * 100 / 2032757, 3),
            delta=0.01,
        )
        self.assertAlmostEqual(
            idle_time.iloc[1]["non_compute_time_pctg"],
            round(1004227 * 100 / 2032757, 3),
            delta=0.01,
        )

    def test_mtia_temporal_breakdown(self):
        idle_time = self.mtia_single_rank_trace_t.get_temporal_breakdown(
            visualize=False
        )
        self.assertAlmostEqual(
            idle_time.iloc[0]["idle_time_pctg"],
            round((5649476.0 * 100) / 13328071, 3),
            delta=0.01,
        )
        self.assertAlmostEqual(
            idle_time.iloc[0]["compute_time_pctg"],
            round((7305597.0 * 100 / 13328071), 3),
            delta=0.01,
        )
        self.assertAlmostEqual(
            idle_time.iloc[0]["non_compute_time_pctg"],
            round(372998.0 * 100 / 13328071, 3),
            delta=0.01,
        )

    def test_get_gpu_kernel_breakdown(self):
        (
            kernel_type_breakdown,
            kernel_breakdown,
        ) = self.vision_transformer_t.get_gpu_kernel_breakdown(
            visualize=False, include_memory_kernels=True
        )

        self.assertEqual(kernel_type_breakdown.iloc[0]["kernel_type"], "COMMUNICATION")
        self.assertEqual(kernel_type_breakdown.iloc[0]["sum"], 8040285)
        self.assertEqual(kernel_breakdown.iloc[0]["kernel_type"], "COMMUNICATION")
        self.assertEqual(kernel_breakdown.iloc[0]["sum (us)"], 627683)
        self.assertEqual(kernel_breakdown.iloc[151]["kernel_type"], "MEMORY")
        self.assertEqual(kernel_breakdown.iloc[151]["sum (us)"], 1064)

    def test_get_mtia_kernel_breakdown(self):
        (
            kernel_type_breakdown,
            kernel_breakdown,
        ) = self.mtia_single_rank_trace_t.get_gpu_kernel_breakdown(
            visualize=False, include_memory_kernels=True
        )

        self.assertEqual(kernel_type_breakdown.iloc[0]["kernel_type"], "COMPUTATION")
        self.assertEqual(kernel_type_breakdown.iloc[0]["sum"], 7305597)
        self.assertEqual(kernel_breakdown.iloc[0]["kernel_type"], "COMPUTATION")
        self.assertEqual(kernel_breakdown.iloc[0]["sum (us)"], 77283.0)
        self.assertEqual(kernel_breakdown.iloc[11]["kernel_type"], "MEMORY")
        self.assertEqual(kernel_breakdown.iloc[11]["sum (us)"], 400892.0)

    def test_get_queue_length_stats(self):
        qd_summary = self.vision_transformer_t.get_queue_length_summary(ranks=[0])
        streams = qd_summary.index.to_list()
        self.assertEqual(streams, list(zip([0] * 6, [7, 20, 24, 26, 28, 30])))

        stream7_stats = qd_summary.loc[0, 7]["queue_length"].to_dict()
        expected_stats = {
            "count": 17160.0,
            "mean": 61.043473193,
            "std": 92.33450047722,
            "min": 0.0,
            "25%": 1.0,
            "50%": 9.0,
            "75%": 87.0,
            "max": 403.0,
        }
        for key, expval in expected_stats.items():
            self.assertAlmostEqual(
                stream7_stats[key],
                expval,
                places=2,
                msg=f"Stream 7 stats mismatch key={key}",
            )

        queue_len_ts_dict = self.vision_transformer_t.get_queue_length_time_series()
        queue_full_df = self.vision_transformer_t.get_time_spent_blocked_on_full_queue(
            queue_len_ts_dict, max_queue_length=400  # Just a hack for testing
        )
        self.assertEqual(len(queue_full_df), 1)
        self.assertAlmostEqual(
            queue_full_df.loc[0]["duration_at_max_queue_length"],
            2300.0,
            msg=f"queue_full_df = {queue_full_df}",
        )
        self.assertAlmostEqual(
            queue_full_df.loc[0]["relative_duration_at_max_queue_length"],
            0.001129,
            places=5,
            msg=f"queue_full_df = {queue_full_df}",
        )

    def test_get_mtia_queue_length_stats(self):
        qd_summary = self.mtia_single_rank_trace_t.get_queue_length_summary(ranks=[0])
        streams = qd_summary.index.to_list()
        self.assertEqual(streams, list(zip([0] * 2, [1, 102])))

        stream102_stats = qd_summary.loc[0, 102]["queue_length"].to_dict()
        expected_stats = {
            "count": 6.0,
            "mean": 0.5,
            "std": 0.547723,
            "min": 0.0,
            "25%": 0.0,
            "50%": 0.5,
            "75%": 1.0,
            "max": 1.0,
        }
        for key, expval in expected_stats.items():
            self.assertAlmostEqual(
                stream102_stats[key],
                expval,
                places=2,
                msg=f"Stream 102 stats mismatch key={key}",
            )

        queue_len_ts_dict = self.mtia_single_rank_trace_t.get_queue_length_time_series()
        queue_full_df = (
            self.mtia_single_rank_trace_t.get_time_spent_blocked_on_full_queue(
                queue_len_ts_dict, max_queue_length=1  # Just a hack for testing
            )
        )
        self.assertEqual(len(queue_full_df), 1)
        self.assertAlmostEqual(
            queue_full_df.loc[0]["duration_at_max_queue_length"],
            1060.0,
            msg=f"queue_full_df = {queue_full_df}",
        )
        self.assertAlmostEqual(
            queue_full_df.loc[0]["relative_duration_at_max_queue_length"],
            0.000079,
            places=5,
            msg=f"queue_full_df = {queue_full_df}",
        )

    @patch.object(hta.common.trace.Trace, "write_raw_trace")
    def test_generate_trace_with_counters(self, mock_write_trace):
        # Use a trace with some kernels missing attribution to operators
        # to check if our logic is robust and does not lead to negative values.
        queue_length_trace: str = os.path.join(
            self.base_data_dir, "negative_queue_length_values_check"
        )
        analyzer_t = TraceAnalysis(trace_dir=queue_length_trace)
        analyzer_t.generate_trace_with_counters(
            time_series=TimeSeriesTypes.QUEUE_LENGTH | TimeSeriesTypes.MEMCPY_BANDWIDTH
        )
        mock_write_trace.assert_called_once()
        # change to kwargs if you use kwargs while calling write_raw_trace
        trace_filename, trace_json = mock_write_trace.call_args.args
        self.assertTrue("with_counters" in trace_filename)

        counter_events = [
            ev for ev in trace_json["traceEvents"] if ev["ph"] == PHASE_COUNTER
        ]
        print(f"Trace has {len(counter_events)} counter events")
        self.assertGreaterEqual(len(counter_events), 12000)

        counter_names = {ev["name"] for ev in counter_events}
        self.assertEqual(
            counter_names,
            {"Queue Length", "Memcpy DtoH", "Memcpy HtoD", "Memcpy DtoD", "Memset"},
        )

        membw_ts = analyzer_t.get_memory_bw_time_series()[0]
        self.assertEqual(len(membw_ts[membw_ts.memory_bw_gbps < 0]), 0)

        queue_len_ts = analyzer_t.get_queue_length_time_series()[0]
        self.assertEqual(len(queue_len_ts[queue_len_ts.queue_length < 0]), 0)

        mem_bw_summary_df = analyzer_t.get_memory_bw_summary(ranks=[0])
        # 1 ranks x 4 types of memcpy/memset
        self.assertEqual(len(mem_bw_summary_df), 4)

        queue_len_summary_df = analyzer_t.get_queue_length_summary(ranks=[0])
        # 1 ranks x 6 streams
        self.assertEqual(len(queue_len_summary_df), 6)

        # Test traces without GPU kernels, these should return empty dicts or dataframes
        queue_len_ts_dict = self.rank_non_gpu_t.get_queue_length_time_series()
        self.assertEqual(len(queue_len_ts_dict), 0)

        queue_len_summary_df = self.rank_non_gpu_t.get_queue_length_summary(ranks=[0])
        self.assertIsNone(queue_len_summary_df)

        mem_bw_summary_df = self.rank_non_gpu_t.get_memory_bw_summary(ranks=[0])
        self.assertIsNone(mem_bw_summary_df)

    def test_get_idle_time_breakdown(self):
        (
            idle_time_df,
            idle_interval_df,
        ) = self.vision_transformer_t.get_idle_time_breakdown(
            ranks=[0, 1], visualize=False, show_idle_interval_stats=True
        )
        ranks = idle_time_df[
            "rank"
        ].unique()  # cannot use dataframe.rank as rank() is a utility too
        streams = idle_time_df.stream.unique()
        idle_categories = idle_time_df.idle_category.unique()

        self.assertEqual(set(ranks), {0, 1})
        self.assertEqual(set(streams), {7, 20, 24, 26, 28, 30})
        self.assertEqual(set(idle_categories), {"host_wait", "kernel_wait", "other"})

        # Ratios sum up to 1.0, 6 streams x 2 ranks = 12.0
        self.assertAlmostEqual(idle_time_df.idle_time_ratio.sum(), 12.0)

        stream7_stats = idle_time_df[idle_time_df.stream == 7].iloc[0].to_dict()
        expected_stats = {
            "idle_category": "host_wait",
            "idle_time": 1000581.0,
            "stream": 7,
            "idle_time_ratio": 0.71,
            "rank": 0,
        }
        for key, expval in expected_stats.items():
            self.assertAlmostEqual(
                stream7_stats[key],
                expval,
                msg=f"Stream 7 idle stats mismatch key={key}",
            )

    def test_get_mtia_idle_time_breakdown(self):
        (
            idle_time_df,
            idle_interval_df,
        ) = self.mtia_single_rank_trace_t.get_idle_time_breakdown(
            ranks=[0], visualize=False, show_idle_interval_stats=True
        )
        ranks = idle_time_df["rank"].unique()
        streams = idle_time_df.stream.unique()
        idle_categories = idle_time_df.idle_category.unique()

        self.assertEqual(set(ranks), {0})
        self.assertEqual(set(streams), {1, 102})
        self.assertEqual(set(idle_categories), {"host_wait", "kernel_wait", "other"})

        # Ratios sum up to 1.0, 2 streams x 1 ranks = 2.0
        self.assertAlmostEqual(idle_time_df.idle_time_ratio.sum(), 2.0)

        stream1_stats = idle_time_df[idle_time_df.stream == 1].iloc[0].to_dict()
        expected_stats = {
            "idle_category": "host_wait",
            "idle_time": 417937.0,
            "stream": 1,
            "idle_time_ratio": 0.07,
            "rank": 0,
        }
        for key, expval in expected_stats.items():
            self.assertAlmostEqual(
                stream1_stats[key],
                expval,
                msg=f"Stream 1 idle stats mismatch key={key}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
