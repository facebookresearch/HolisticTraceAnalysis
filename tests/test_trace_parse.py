# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import pandas as pd

from hta.common.trace import Trace, parse_trace_dict


class TraceParseTestCase(unittest.TestCase):
    vision_transformer_t: Trace
    vision_transformer_raw_df: pd.DataFrame
    inference_t: Trace
    inference_raw_df: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        super(TraceParseTestCase, cls).setUpClass()
        vision_transformer_trace_dir: str = "tests/data/vision_transformer"
        inference_trace_dir: str = "tests/data/inference_single_rank"
        vision_transformer_rank_0_file: str = "rank-0.json.gz"
        inference_rank_0_file: str = "inference_rank_0.json.gz"
        max_ranks = 8

        # Trace parser for vision transformer
        cls.vision_transformer_t: Trace = Trace(trace_dir=vision_transformer_trace_dir)
        cls.vision_transformer_t.parse_traces(max_ranks=max_ranks, use_multiprocessing=True)
        cls.vision_transformer_raw_df = cls.prepare_ground_truth_df(
            vision_transformer_trace_dir, vision_transformer_rank_0_file
        )
        # Trace parser for inference
        cls.inference_t: Trace = Trace(trace_dir=inference_trace_dir)
        cls.inference_t.parse_traces(max_ranks=max_ranks, use_multiprocessing=True)
        cls.inference_raw_df = cls.prepare_ground_truth_df(inference_trace_dir, inference_rank_0_file)

    @classmethod
    def prepare_ground_truth_df(cls, trace_dir, rank_0_file) -> pd.DataFrame:
        df = pd.DataFrame(parse_trace_dict(os.path.join(trace_dir, rank_0_file))["traceEvents"])
        df.dropna(axis=0, subset=["dur", "cat"], inplace=True)
        df.drop(df[df["cat"] == "Trace"].index, inplace=True)
        return df

    def setUp(self) -> None:
        self.traces = [self.vision_transformer_t, self.inference_t]
        self.raw_dfs = [self.vision_transformer_raw_df, self.inference_raw_df]
        self.total_ranks = [8, 1]

    def test_trace_load(self) -> None:
        # run tests for each collection of traces
        for t, raw_df, total_ranks in zip(self.traces, self.raw_dfs, self.total_ranks):
            # test raw trace after parsing
            self.assertEqual(len(t.traces), total_ranks)

            sym_id_map = t.symbol_table.get_sym_id_map()
            sym_table = t.symbol_table.get_sym_table()
            rank_0_df_name_id = t.traces[0]["name"]
            rank_0_df_name = t.traces[0]["name"].apply(lambda x: sym_table[x])

            ground_truth_name = raw_df["name"]
            ground_truth_name_id = raw_df["name"].apply(lambda x: sym_id_map[x])

            self.assertListEqual(rank_0_df_name_id.to_list(), ground_truth_name_id.to_list())
            self.assertListEqual(rank_0_df_name.to_list(), ground_truth_name.to_list())

            # test aligned and filtered trace
            t.align_and_filter_trace()

            raw_profiler_steps = raw_df["name"].str.contains("ProfilerStep").sum()
            sym_id_map = t.symbol_table.get_sym_id_map()
            profiler_steps = [v for k, v in sym_id_map.items() if "ProfilerStep" in k]
            filtered_profiler_steps = t.traces[0]["name"].isin(profiler_steps).sum()

            self.assertEqual(
                filtered_profiler_steps + int(raw_profiler_steps > 1),
                raw_profiler_steps,
            )
            self.assertLessEqual(len(t.traces[0]), len(raw_df))
            self.assertGreaterEqual(t.traces[0]["ts"].min(), 0)

    def test_trace_iteration(self) -> None:
        # run tests for each collection of traces
        for t in self.traces:
            df = t.traces[0]
            sym_id_map = t.symbol_table.get_sym_id_map()
            iterations = {f"ProfilerStep#{i}" for i in set(df["iteration"].unique()) if i != -1}

            valid_gpu_kernels = df.loc[df["stream"].gt(0) & df["index_correlation"].gt(0)]
            correlated_cpu_ops = df.loc[df.loc[valid_gpu_kernels.index, "index_correlation"]]
            gpu_kernels_per_iteration = valid_gpu_kernels.groupby("iteration")["index"].agg("count").to_dict()
            correlated_cpu_ops_per_iteration = correlated_cpu_ops.groupby("iteration")["index"].agg("count").to_dict()

            self.assertTrue("iteration" in df.columns)
            self.assertTrue(all(i in sym_id_map for i in iterations))
            self.assertDictEqual(gpu_kernels_per_iteration, correlated_cpu_ops_per_iteration)


if __name__ == "__main__":
    unittest.main()
