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
from hta.analyzers.counters_analysis import CUDA_SASS_INSTRUCTION_COUNTER_FLOPS
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
        vision_transformer_trace_dir: str = "tests/data/vision_transformer"
        inference_trace_dir: str = "tests/data/inference_single_rank"
        df_index_resolver_trace_dir: str = "tests/data/df_index_resolver"
        rank_non_gpu_trace_dir: str = "tests/data/rank_non_gpu/"
        cls.vision_transformer_t = TraceAnalysis(trace_dir=vision_transformer_trace_dir)
        cls.inference_t = TraceAnalysis(trace_dir=inference_trace_dir)
        cls.df_index_resolver_t = TraceAnalysis(trace_dir=df_index_resolver_trace_dir)
        cls.rank_non_gpu_t = TraceAnalysis(trace_dir=rank_non_gpu_trace_dir)

    def setUp(self):
        self.overlaid_trace_dir = "tests/data"
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
            frequent_patterns_dfs.iloc[2]["GPU kernel duration (ns)"], 11300
        )
        self.assertEqual(frequent_patterns_dfs.iloc[2]["CPU op duration (ns)"], 9652)
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

    def test_get_profiler_steps(self):
        results = self.vision_transformer_t.get_profiler_steps()
        expected = [15, 16, 17, 18]
        self.assertListEqual(results, expected)

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
        self.assertEqual(kernel_breakdown.iloc[0]["sum (ns)"], 627683)
        self.assertEqual(kernel_breakdown.iloc[151]["kernel_type"], "MEMORY")
        self.assertEqual(kernel_breakdown.iloc[151]["sum (ns)"], 1064)

    def test_get_queue_length_summary(self):
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
                stream7_stats[key], expval, msg=f"Stream 7 stats mismatch key={key}"
            )

    @patch.object(hta.common.trace.Trace, "write_raw_trace")
    def test_generate_trace_with_counters(self, mock_write_trace):
        self.vision_transformer_t.generate_trace_with_counters(
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
        self.assertGreaterEqual(len(counter_events), 21000)

        counter_names = {ev["name"] for ev in counter_events}
        self.assertEqual(
            counter_names,
            {"Queue Length", "Memcpy DtoH", "Memcpy HtoD", "Memcpy DtoD", "Memset"},
        )

        membw_ts = self.vision_transformer_t.get_memory_bw_time_series()[0]
        self.assertEqual(len(membw_ts[membw_ts.memory_bw_gbps < 0]), 0)

        queue_len_ts = self.vision_transformer_t.get_queue_length_time_series()[0]
        self.assertEqual(len(queue_len_ts[queue_len_ts.queue_length < 0]), 0)

        mem_bw_summary_df = self.vision_transformer_t.get_memory_bw_summary(
            ranks=[0, 2]
        )
        # 2 ranks x 4 types of memcpy/memset
        self.assertEqual(len(mem_bw_summary_df), 8)

        queue_len_summary_df = self.vision_transformer_t.get_queue_length_summary(
            ranks=[0, 2]
        )
        # 2 ranks x 6 streams
        self.assertEqual(len(queue_len_summary_df), 12)

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

    def test_get_counter_data_with_operators(self):
        # regular trace should return empty list since it will not have cuda_profiler_range events
        self.assertEqual(self.inference_t.get_counter_data_with_operators(), [])

        cupti_profiler_trace_dir: str = "tests/data/cupti_profiler/"
        cupti_profiler_t = TraceAnalysis(trace_dir=cupti_profiler_trace_dir)

        counters_df = cupti_profiler_t.get_counter_data_with_operators(stringify=True)[
            0
        ]

        test_row = counters_df.iloc[5].to_dict()
        self.assertEqual(test_row["cat"], "cuda_profiler_range")
        self.assertTrue("fft2d_r2c_32x32" in test_row["name"])

        self.assertEqual(
            test_row["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"], 86114304
        )
        self.assertEqual(test_row["top_level_op"], "aten::conv2d")
        self.assertEqual(test_row["bottom_level_op"], "aten::_convolution")

        # Example trace has CUPTI SASS FLOPS instruction counters
        counter_names = set(CUDA_SASS_INSTRUCTION_COUNTER_FLOPS.keys())
        self.assertEqual(
            set(counters_df.columns.unique()) & counter_names, counter_names
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
