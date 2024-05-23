# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from typing import Any, Dict

# import unittest.mock as mock

import pandas as pd
from hta.common.trace import parse_trace_dict, Trace
from hta.common.trace_parser import (
    _auto_detect_parser_backend,
    _open_trace_file,
    get_default_trace_parsing_backend,
    parse_metadata_ijson,
    ParserBackend,
    set_default_trace_parsing_backend,
)
from hta.configs.parser_config import AVAILABLE_ARGS, ParserConfig

EXPECTED_META_VISION_TRANFORMER: Dict[str, Any] = {
    "schemaVersion": 1,
    "distributedInfo": {"backend": "nccl", "rank": 0, "world_size": 64},
    "deviceProperties": [
        {
            "id": 0,
            "name": "Tesla V100-SXM2-32GB",
            "totalGlobalMem": 34089730048,
            "computeMajor": 7,
            "computeMinor": 0,
            "maxThreadsPerBlock": 1024,
            "maxThreadsPerMultiprocessor": 2048,
            "regsPerBlock": 65536,
            "regsPerMultiprocessor": 65536,
            "warpSize": 32,
            "sharedMemPerBlock": 49152,
            "sharedMemPerMultiprocessor": 98304,
            "numSms": 80,
            "sharedMemPerBlockOptin": 98304,
        },
    ],
}

GROUND_TRUTH_CACHE: Dict[str, pd.DataFrame] = {}


def prepare_ground_truth_df(trace_dir, rank_0_file) -> pd.DataFrame:
    global GROUND_TRUTH_CACHE
    filep = os.path.join(trace_dir, rank_0_file)
    if str(filep) in GROUND_TRUTH_CACHE:
        df = GROUND_TRUTH_CACHE[str(filep)].copy()
    else:
        df = pd.DataFrame(parse_trace_dict(filep)["traceEvents"])
        GROUND_TRUTH_CACHE[str(filep)] = df.copy()

    # perform some manipulations on raw df
    df.dropna(axis=0, subset=["dur", "cat"], inplace=True)
    to_drop_cats = ["Trace"]
    if get_default_trace_parsing_backend() != ParserBackend.JSON:
        to_drop_cats.append("python_function")
    df.drop(df[df["cat"].isin(to_drop_cats)].index, inplace=True)
    return df


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
        cls.vision_transformer_t.parse_traces(
            max_ranks=max_ranks, use_multiprocessing=True
        )
        cls.vision_transformer_raw_df = prepare_ground_truth_df(
            vision_transformer_trace_dir, vision_transformer_rank_0_file
        )
        # Trace parser for inference
        cls.inference_t: Trace = Trace(trace_dir=inference_trace_dir)
        cls.inference_t.parse_traces(max_ranks=max_ranks, use_multiprocessing=True)
        cls.inference_raw_df = prepare_ground_truth_df(
            inference_trace_dir, inference_rank_0_file
        )

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

            self.assertListEqual(
                rank_0_df_name_id.to_list(), ground_truth_name_id.to_list()
            )
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
            iterations = {
                f"ProfilerStep#{i}" for i in set(df["iteration"].unique()) if i != -1
            }

            valid_gpu_kernels = df.loc[
                df["stream"].gt(0) & df["index_correlation"].gt(0)
            ]
            correlated_cpu_ops = df.loc[
                df.loc[valid_gpu_kernels.index, "index_correlation"]
            ]
            gpu_kernels_per_iteration = (
                valid_gpu_kernels.groupby("iteration")["index"].agg("count").to_dict()
            )
            correlated_cpu_ops_per_iteration = (
                correlated_cpu_ops.groupby("iteration")["index"].agg("count").to_dict()
            )

            self.assertTrue("iteration" in df.columns)
            self.assertTrue(all(i in sym_id_map for i in iterations))
            self.assertDictEqual(
                gpu_kernels_per_iteration, correlated_cpu_ops_per_iteration
            )

    def test_trace_metadata(self) -> None:
        trace_meta = self.vision_transformer_t.meta_data[0]
        exp_meta = EXPECTED_META_VISION_TRANFORMER
        self.assertEqual(trace_meta["schemaVersion"], exp_meta["schemaVersion"])
        self.assertEqual(trace_meta["distributedInfo"], exp_meta["distributedInfo"])
        self.assertEqual(
            trace_meta["deviceProperties"][0], exp_meta["deviceProperties"][0]
        )
        # print(trace_meta)


@unittest.skipIf(
    # _auto_detect_parser_backend() == ParserBackend.JSON,
    # Tests are timing out the CI so have to disable this
    1,
    "Skipping ijson based trace load tests",
)
class TraceParseIjsonBatchCompressTestCase(TraceParseTestCase):
    @classmethod
    def setUpClass(cls):
        set_default_trace_parsing_backend(ParserBackend.IJSON_BATCH_AND_COMPRESS)
        super(TraceParseIjsonBatchCompressTestCase, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        set_default_trace_parsing_backend(ParserBackend.JSON)


@unittest.skipIf(
    _auto_detect_parser_backend() == ParserBackend.JSON,
    "Skipping ijson based trace load tests",
)
class TraceParseIjsonOthersTestCase(unittest.TestCase):
    """Additional test for coverage of 2 other backends"""

    inference_trace_dir: str
    vision_transformer_trace_dir: str

    @classmethod
    def setUpClass(cls):
        cls.inference_trace_dir: str = "tests/data/critical_path/alexnet"
        cls.vision_transformer_trace_dir: str = "tests/data/vision_transformer"

    def test_ijson_parser(self):
        set_default_trace_parsing_backend(ParserBackend.IJSON)

        inference_t: Trace = Trace(trace_dir=self.inference_trace_dir)
        inference_t.parse_traces(max_ranks=1)

        self.assertEqual(len(inference_t.traces), 1)
        set_default_trace_parsing_backend(ParserBackend.JSON)

    def test_ijson_batched_parser(self):
        set_default_trace_parsing_backend(ParserBackend.IJSON_BATCHED)

        inference_t: Trace = Trace(trace_dir=self.inference_trace_dir)
        inference_t.parse_traces(max_ranks=1)

        self.assertEqual(len(inference_t.traces), 1)
        set_default_trace_parsing_backend(ParserBackend.JSON)

    def test_ijson_batch_and_compress_parser(self):
        set_default_trace_parsing_backend(ParserBackend.IJSON_BATCH_AND_COMPRESS)

        inference_t: Trace = Trace(trace_dir=self.inference_trace_dir)
        inference_t.parse_traces(max_ranks=1)

        self.assertEqual(len(inference_t.traces), 1)
        set_default_trace_parsing_backend(ParserBackend.JSON)

    def test_ijson_metadata_reader(self):
        trace_file_path = self.vision_transformer_trace_dir + "/rank-0.json.gz"
        trace_meta = {}
        with _open_trace_file(trace_file_path) as fh:
            trace_meta = parse_metadata_ijson(fh)
        # print(trace_meta)

        exp_meta = EXPECTED_META_VISION_TRANFORMER
        self.assertEqual(trace_meta["schemaVersion"], exp_meta["schemaVersion"])
        self.assertEqual(trace_meta["distributedInfo"], exp_meta["distributedInfo"])
        self.assertEqual(
            trace_meta["deviceProperties"][0], exp_meta["deviceProperties"][0]
        )

    # @mock.patch('ijson.backend')
    # def test_optimal_backend_detection(self, mock_backend) -> None:
    #     mock_backend = "xxx"
    #     self.assertEqual(_auto_detect_parser_backend(), "json")
    #     mock_backend = "yajl_2c"
    #     self.assertEqual(_auto_detect_parser_backend(), "ijson_batch_and_compress")


class TraceParseConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        resnet_nccl_trace: str = "tests/data/nccl_parser_config"
        # Trace parser for nccl fields
        self.resnet_nccl_t: Trace = Trace(trace_dir=resnet_nccl_trace)

        # Parse all nccl fields in the test
        custom_cfg = ParserConfig(ParserConfig.get_minimum_args())
        custom_cfg.add_args(
            [spec for (arg, spec) in AVAILABLE_ARGS.items() if arg.startswith("nccl")]
        )
        ParserConfig.set_default_cfg(custom_cfg)

    def tearDown(self) -> None:
        ParserConfig.set_default_cfg(ParserConfig(ParserConfig.get_minimum_args()))

    def test_nccl_parser_config(self) -> None:
        "Tests if nccl metadata is parsed correctly"
        self.resnet_nccl_t.parse_traces(max_ranks=1, use_multiprocessing=False)
        self.resnet_nccl_t.decode_symbol_ids(use_shorten_name=False)

        trace_df = self.resnet_nccl_t.get_trace(0)
        self.assertGreater(len(trace_df), 0)

        nccl_kernels = trace_df.query(
            "s_cat == 'kernel' and s_name.str.contains('ncclKernel')"
        ).sort_values("dur", ascending=False)

        self.assertEqual(len(nccl_kernels), 21)

        # check first allreaduce kernel
        nccl_data = nccl_kernels.iloc[0].to_dict()
        print(nccl_data)
        self.assertEqual(nccl_data["collective_name"], "allreduce")
        self.assertEqual(nccl_data["in_msg_nelems"], 2049000)
        self.assertEqual(nccl_data["out_msg_nelems"], 2049000)
        self.assertEqual(nccl_data["in_split_size"], "[]")
        self.assertEqual(nccl_data["out_split_size"], "[]")
        self.assertEqual(nccl_data["process_group_name"], "0")
        self.assertEqual(nccl_data["process_group_desc"], "default_pg")
        self.assertEqual(nccl_data["process_group_ranks"], "[0, 1]")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
