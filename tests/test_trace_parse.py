# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import unittest

from typing import Any, Dict, Optional, Set

# import unittest.mock as mock

import pandas as pd
from hta.common import singletrace
from hta.common.singletrace import Trace
from hta.common.trace_collection import parse_trace_dict, TraceCollection
from hta.common.trace_parser import (
    _auto_detect_parser_backend,
    _open_trace_file,
    get_default_trace_parsing_backend,
    infer_gpu_type,
    parse_metadata_ijson,
    parse_trace_dataframe,
    ParserBackend,
    set_default_trace_parsing_backend,
)
from hta.common.trace_symbol_table import TraceSymbolTable
from hta.configs.parser_config import AVAILABLE_ARGS, ParserConfig
from hta.utils.test_utils import data_provider

JSON = Dict[str, Any]
EXPECTED_META_VISION_TRANFORMER: JSON = {
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
        {},  # omitting the actual device property for brevity.
        {},
        {},
        {},
        {},
        {},
        {},
    ],
}

EXPECTED_META_CPU_ONLY_TRACE: JSON = {
    "deviceProperties": [],
    "distributedInfo": {
        "backend": "gloo",
        "pg_config": None,
        "pg_count": 10,
        "rank": 34,
        "world_size": 300,
    },
    "schemaVersion": 1,
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
    vision_transformer_t: TraceCollection
    vision_transformer_raw_df: pd.DataFrame
    inference_t: TraceCollection
    inference_raw_df: pd.DataFrame
    triton_t: TraceCollection
    triton_raw_df: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        super(TraceParseTestCase, cls).setUpClass()
        vision_transformer_trace_dir: str = add_test_data_path_prefix_if_exists(
            "tests/data/vision_transformer"
        )
        inference_trace_dir: str = add_test_data_path_prefix_if_exists(
            "tests/data/inference_single_rank"
        )
        triton_trace_dir: str = add_test_data_path_prefix_if_exists(
            "tests/data/triton_example"
        )

        vision_transformer_rank_0_file: str = "rank-0.json.gz"
        inference_rank_0_file: str = "inference_rank_0.json.gz"
        triton_example_file: str = "triton_example.json.gz"
        inference_trace_files = [
            os.path.join(inference_trace_dir, inference_rank_0_file)
        ]
        max_ranks = 8

        cls.vision_transformer_t: TraceCollection = TraceCollection(
            trace_dir=vision_transformer_trace_dir
        )
        cls.vision_transformer_t.parse_traces(
            max_ranks=max_ranks, use_multiprocessing=True
        )
        cls.vision_transformer_raw_df = prepare_ground_truth_df(
            vision_transformer_trace_dir, vision_transformer_rank_0_file
        )
        cls.inference_t: TraceCollection = TraceCollection(
            trace_files=inference_trace_files, trace_dir=os.getcwd()
        )
        cls.inference_t.parse_traces(max_ranks=max_ranks, use_multiprocessing=True)
        cls.inference_raw_df = prepare_ground_truth_df(
            inference_trace_dir, inference_rank_0_file
        )
        cls.triton_t: TraceCollection = TraceCollection(trace_dir=triton_trace_dir)
        cls.triton_t.parse_traces(max_ranks=max_ranks, use_multiprocessing=True)
        cls.triton_t.align_and_filter_trace(include_last_profiler_step=True)
        cls.triton_raw_df = prepare_ground_truth_df(
            triton_trace_dir, triton_example_file
        )

    def setUp(self) -> None:
        self.traces = [
            self.vision_transformer_t,
            self.inference_t,
            self.triton_t,
        ]
        self.raw_dfs = [
            self.vision_transformer_raw_df,
            self.inference_raw_df,
            self.triton_raw_df,
        ]
        self.total_ranks = [
            8,
            1,
            1,
        ]

    def test_trace_load(self) -> None:
        # run tests for each collection of traces
        for t, raw_df, total_ranks in zip(self.traces, self.raw_dfs, self.total_ranks):
            # test raw trace after parsing
            self.assertEqual(len(t.traces), total_ranks)

            sym_id_map = t.symbol_table.get_sym_id_map()
            sym_table = t.symbol_table.get_sym_table()
            rank_0_df_name_id = t.get_trace_df(0)["name"]
            rank_0_df_name = t.get_trace_df(0)["name"].apply(lambda x: sym_table[x])

            ground_truth_name = raw_df["name"]
            ground_truth_name_id = raw_df["name"].apply(lambda x: sym_id_map[x])

            self.assertSetEqual(
                set(rank_0_df_name_id.to_list()), set(ground_truth_name_id.to_list())
            )
            self.assertSetEqual(
                set(rank_0_df_name.to_list()), set(ground_truth_name.to_list())
            )

            raw_profiler_steps = raw_df["name"].str.contains("ProfilerStep").sum()
            # test aligned and filtered trace
            t.align_and_filter_trace(
                include_last_profiler_step=True if raw_profiler_steps == 1 else False
            )

            sym_id_map = t.symbol_table.get_sym_id_map()
            profiler_steps = [v for k, v in sym_id_map.items() if "ProfilerStep" in k]
            filtered_profiler_steps = (
                t.get_trace_df(0)["name"].isin(profiler_steps).sum()
            )

            self.assertEqual(
                filtered_profiler_steps + int(raw_profiler_steps > 1),
                raw_profiler_steps,
            )
            self.assertLessEqual(len(t.get_trace_df(0)), len(raw_df))
            self.assertGreaterEqual(t.get_trace_df(0)["ts"].min(), 0)

    def test_trace_iteration(self) -> None:
        # run tests for each collection of traces
        for t in self.traces:
            df = t.get_trace_df(0)
            sym_id_map = t.symbol_table.get_sym_id_map()
            iterations = {
                f"ProfilerStep#{i}"
                for i in set(df["iteration"].unique())
                if i != -1 and not math.isnan(i)
            }

            valid_gpu_kernels = df.loc[
                df["stream"].gt(0) & df["index_correlation"].gt(0)
            ]
            correlated_cpu_ops = df.loc[
                df.loc[valid_gpu_kernels.index, "index_correlation"]
            ]
            gpu_kernels_per_iteration = (
                valid_gpu_kernels.groupby("iteration")["index"].count().to_dict()
            )
            correlated_cpu_ops_per_iteration = (
                correlated_cpu_ops.groupby("iteration")["index"].count().to_dict()
            )

            self.assertTrue("iteration" in df.columns)
            self.assertTrue(all(i in sym_id_map for i in iterations))
            self.assertDictEqual(
                gpu_kernels_per_iteration, correlated_cpu_ops_per_iteration
            )

    def test_trace_metadata(self) -> None:
        trace: Trace = self.vision_transformer_t.get_trace(0)
        trace_meta = trace.meta
        exp_meta = EXPECTED_META_VISION_TRANFORMER
        self.assertEqual(trace_meta["schemaVersion"], exp_meta["schemaVersion"])
        self.assertEqual(trace_meta["distributedInfo"], exp_meta["distributedInfo"])
        self.assertEqual(
            len(trace_meta["deviceProperties"]), len(exp_meta["deviceProperties"])
        )
        self.assertEqual(
            trace_meta["deviceProperties"][0], exp_meta["deviceProperties"][0]
        )
        # print(trace_meta)

    def test_get_trace_start_unixtime_ns(self) -> None:
        with self.assertRaises(KeyError):
            # This trace metadata doesn't have the "baseTimeNanoseconds" field, so we expect a KeyError
            self.vision_transformer_t.get_trace_start_unixtime_ns(0)

        triton_trace_first_event_ts_ns = 2413669096090100
        triton_trace_base_time_nanoseconds = 1727743122000000000
        expected_triton_start_unixtime_ns = (
            triton_trace_first_event_ts_ns + triton_trace_base_time_nanoseconds
        )

        actual_triton_start_unixtime_ns = self.triton_t.get_trace_start_unixtime_ns(0)

        # Rounding of ns resolution events yields an imprecise result.
        # We expect the difference to be less than 1us
        self.assertAlmostEqual(
            actual_triton_start_unixtime_ns,
            expected_triton_start_unixtime_ns,
            delta=1000,
        )


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
        cls.inference_trace_dir: str = add_test_data_path_prefix_if_exists(
            "tests/data/critical_path/alexnet"
        )
        cls.vision_transformer_trace_dir: str = add_test_data_path_prefix_if_exists(
            "tests/data/vision_transformer"
        )
        cls.cpu_only_trace_path: str = add_test_data_path_prefix_if_exists(
            "tests/data/cpu_only/rank-34.Jul_15_10_52_41.1074.pt.trace.json.gz"
        )

    def test_ijson_parser(self):
        set_default_trace_parsing_backend(ParserBackend.IJSON)

        inference_t: TraceCollection = TraceCollection(
            trace_dir=self.inference_trace_dir
        )
        inference_t.parse_traces(max_ranks=1)

        self.assertEqual(len(inference_t.traces), 1)
        set_default_trace_parsing_backend(ParserBackend.JSON)

    def test_ijson_batched_parser(self):
        set_default_trace_parsing_backend(ParserBackend.IJSON_BATCHED)

        inference_t: TraceCollection = TraceCollection(
            trace_dir=self.inference_trace_dir
        )
        inference_t.parse_traces(max_ranks=1)

        self.assertEqual(len(inference_t.traces), 1)
        set_default_trace_parsing_backend(ParserBackend.JSON)

    def test_ijson_batch_and_compress_parser(self):
        set_default_trace_parsing_backend(ParserBackend.IJSON_BATCH_AND_COMPRESS)

        inference_t: TraceCollection = TraceCollection(
            trace_dir=self.inference_trace_dir
        )
        inference_t.parse_traces(max_ranks=1)

        self.assertEqual(len(inference_t.traces), 1)
        set_default_trace_parsing_backend(ParserBackend.JSON)

    def _ijson_metadata_test_common(self, trace_file_path: str, exp_meta: JSON):
        trace_meta = {}
        with _open_trace_file(trace_file_path) as fh:
            trace_meta = parse_metadata_ijson(fh)
        # print(trace_meta)

        self.assertEqual(trace_meta["schemaVersion"], exp_meta["schemaVersion"])
        self.assertEqual(trace_meta["distributedInfo"], exp_meta["distributedInfo"])
        self.assertEqual(
            len(trace_meta["deviceProperties"]), len(exp_meta["deviceProperties"])
        )
        if len(trace_meta["deviceProperties"]) > 0:
            self.assertEqual(
                trace_meta["deviceProperties"][0], exp_meta["deviceProperties"][0]
            )

    def test_ijson_metadata_reader_basic(self):
        trace_file_path = self.vision_transformer_trace_dir + "/rank-0.json.gz"
        self._ijson_metadata_test_common(
            trace_file_path, EXPECTED_META_VISION_TRANFORMER
        )

    def test_ijson_metadata_reader_corner_cases(self):
        # This trace has an empty deviceProperties [] array as it runs on CPU.
        # It also has a large pg_config array in distributedInfo.
        trace_file_path = self.cpu_only_trace_path
        self._ijson_metadata_test_common(trace_file_path, EXPECTED_META_CPU_ONLY_TRACE)

    # @mock.patch('ijson.backend')
    # def test_optimal_backend_detection(self, mock_backend) -> None:
    #     mock_backend = "xxx"
    #     self.assertEqual(_auto_detect_parser_backend(), "json")
    #     mock_backend = "yajl_2c"
    #     self.assertEqual(_auto_detect_parser_backend(), "ijson_batch_and_compress")


def add_test_data_path_prefix_if_exists(test_path):
    """Add TEST_DATA_PREFIX_PATH to the test path if it exists"""
    needs_prefix = os.environ.get("TEST_DATA_PREFIX_PATH", "")
    if needs_prefix:
        return needs_prefix + "/" + test_path
    return test_path


class TestMtiaAlignAndFilter(unittest.TestCase):
    def test_align_and_filter_mtia(self) -> None:
        # Trace parser for MTIA
        mtia_trace_dir: str = add_test_data_path_prefix_if_exists(
            "tests/data/mtia_trace_single_rank"
        )
        t: TraceCollection = TraceCollection(trace_dir=mtia_trace_dir)
        t.parse_traces()
        t.align_and_filter_trace()
        t.decode_symbol_ids(use_shorten_name=False)

        # Ensure that the trace is MTIA trace
        self.assertEqual(t.get_device_type(), "MTIA")
        self.assertGreaterEqual(len(t.get_ranks()), 1)

        # Ensure that the trace has the correct iterations
        result_df = t.get_trace_df(t.get_ranks()[0])
        self.assertTrue(result_df["ts"].ge(0).all())
        self.assertTrue(result_df["iteration"].ge(0).all())

        # Ensure that cpu ops has the correct stream
        cpu_cat_ids = t.symbol_table.get_cpu_event_cat_ids()
        cpu_ops = result_df[result_df["cat"].isin(cpu_cat_ids)]
        self.assertTrue(len(cpu_ops) > 0)
        self.assertTrue(cpu_ops["stream"].le(0).all())

        # Ensure that cuda ops has the correct stream
        memory_kernels = result_df[
            result_df["name"].isin(t.symbol_table.get_memory_name_ids())
        ]
        self.assertTrue(len(memory_kernels) > 0)
        self.assertTrue(memory_kernels["stream"].ge(0).all())
        self.assertTrue(memory_kernels["iteration"].ge(0).all())

        mtia_kernels = result_df[
            result_df["stream"].ge(0) & result_df["correlation"].gt(0)
        ]
        self.assertTrue(len(mtia_kernels) > 0)


class TraceParseConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Parse all nccl fields in the test
        self.custom_cfg = ParserConfig(ParserConfig.get_minimum_args())
        self.custom_cfg.add_args(
            [spec for (arg, spec) in AVAILABLE_ARGS.items() if arg.startswith("nccl")]
            + ParserConfig.ARGS_TRITON_KERNELS
        )
        # ParserConfig.set_default_cfg(custom_cfg)

        # Trace parser test file for nccl fields
        self.resnet_nccl_trace: str = add_test_data_path_prefix_if_exists(
            "tests/data/nccl_parser_config"
        )
        self.resnet_nccl_t: TraceCollection = TraceCollection(
            trace_dir=self.resnet_nccl_trace, parser_config=self.custom_cfg
        )

        # Trace parser test file for Triton fields
        triton_trace: str = add_test_data_path_prefix_if_exists(
            "tests/data/triton_example"
        )
        self.triton_t: TraceCollection = TraceCollection(
            trace_dir=triton_trace, parser_config=self.custom_cfg
        )

    def tearDown(self) -> None:
        ParserConfig.set_default_cfg(ParserConfig(ParserConfig.get_minimum_args()))

    def test_nccl_parser_config(self) -> None:
        "Tests if nccl metadata is parsed correctly"
        self.resnet_nccl_t.parse_traces(max_ranks=1, use_multiprocessing=False)
        self.resnet_nccl_t.decode_symbol_ids(use_shorten_name=False)

        trace_df = self.resnet_nccl_t.get_trace_df(0)
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

    def test_triton_trace(self) -> None:
        """Tests if a file with Triton/torch.compile() is parsed correctly,
        and we can obtain special attributes from the cpu ops tha launch Triton kernels
        """
        self.triton_t.parse_traces(max_ranks=1, use_multiprocessing=False)
        self.triton_t.decode_symbol_ids(use_shorten_name=False)

        trace_df = self.triton_t.get_trace_df(0)
        self.assertGreater(len(trace_df), 0)
        self.assertTrue("kernel_backend" in trace_df.columns)
        self.assertTrue("kernel_hash" in trace_df.columns)

        triton_cpu_ops = trace_df[trace_df.kernel_backend.ne("")]
        # We have one triton cpu op
        self.assertEqual(len(triton_cpu_ops), 1)

        triton_op = triton_cpu_ops.iloc[0].to_dict()
        self.assertEqual(triton_op["s_name"], "triton_poi_fused_add_cos_sin_0")
        self.assertEqual(triton_op["s_cat"], "cpu_op")
        self.assertEqual(triton_op["kernel_backend"], "triton")
        self.assertEqual(
            triton_op["kernel_hash"],
            "cqaokwf2bph4egogzevc22vluasiyuui4i54zpemp6knbsggfbuu",
        )

    @data_provider(
        lambda: (
            {
                "parse_all_args": False,
                "expected_columns": {
                    "name",
                    "ts",
                    "index",
                    "tid",
                    "stream",
                    "cat",
                    "dur",
                    "end",
                    "pid",
                    "correlation",
                },
                "expected_missing_columns": {"block", "grid"},
            },
            {
                "parse_all_args": True,
                "expected_columns": {
                    "index",
                    "cat",
                    "name",
                    "pid",
                    "tid",
                    "ts",
                    "dur",
                    "end",
                    "ev_idx",
                    "external_id",
                    "fwd_thread_id",
                    "in_msg_nelems",
                },
                "expected_missing_columns": set(),
            },
        )
    )
    def test_parse_all_args(
        self,
        parse_all_args: bool,
        expected_columns: Set[str],
        expected_missing_columns: Set[str],
    ) -> None:
        """Tests if we can parse all args in the trace"""
        trace_file = os.path.join(self.resnet_nccl_trace, "nccl_data.json.gz")
        cfg = ParserConfig(ParserConfig.get_minimum_args())
        cfg.set_parse_all_args(parse_all_args)
        trace: Trace = parse_trace_dataframe(trace_file, cfg)
        df = trace.df
        self.assertTrue(expected_columns.issubset(set(df.columns)))
        self.assertTrue(expected_missing_columns.isdisjoint(set(df.columns)))

    # pyre-ignore[56]
    @data_provider(
        lambda: (
            {
                "metadata": {
                    "distributedInfo": {"backend": "mtia:hccl"},
                },
                "syms": {"some_event": 1},
                "expected_device_type": "MTIA",
            },
            {
                "metadata": None,
                "syms": {
                    "some_event": 1,
                    "runFunction - job_prep_and_submit_for_execution": 2,
                },
                "expected_device_type": "MTIA",
            },
            {
                "metadata": None,
                "syms": {"cudaLaunchKernel": 1, "other_event": 2},
                "expected_device_type": "NVIDIA GPU",
            },
            {
                "metadata": None,
                "syms": {"hipLaunchKernel": 1, "another_event": 2},
                "expected_device_type": "AMD GPU",
            },
            {
                "metadata": None,
                "syms": {"some_event": 1},
                "expected_device_type": "UNKNOWN GPU",
            },
        )
    )
    def test_infer_gpu_type(
        self,
        syms: Dict[str, int],
        metadata: Optional[Dict[str, object]],
        expected_device_type: str,
    ) -> None:
        self.assertEqual(
            infer_gpu_type(metadata, syms),
            expected_device_type,
        )

    def test_fix_mtia_memory_kernels(self) -> None:
        df = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4],
                "name": [1001, 2001, 2001, 2001, 2004],
                "ts": [0, 10, 20, 30, 40],
                "dur": [50, 10, 10, 10, 10],
                "iteration": [1, -1, 1, -1, 1],
                "stream": [-1, 3, -1, 4, 1],
                "tid": [1, 2, 3, 4, 5],
            }
        )
        symbol_table = TraceSymbolTable.create_from_symbol_id_map(
            {
                "ProfilerStep#1": 1001,
                "dma_request": 2001,
                "aten::add": 2004,
            }
        )
        # Create a TraceCollection object
        t = TraceCollection(trace_dir="", trace_files={})
        t.traces[0] = singletrace.create_default(df=df.copy())
        t.symbol_table = symbol_table

        # Expected result after applying fix
        expected_df = df.copy()
        expected_df.loc[[1, 3], "iteration"] = 1
        expected_df.loc[[2], "stream"] = expected_df.loc[[2], "tid"]

        t._fix_mtia_memory_kernels(t.get_trace_df(0))
        fixed_df = t.get_trace_df(0)

        # Validate results
        pd.testing.assert_frame_equal(fixed_df, expected_df)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
