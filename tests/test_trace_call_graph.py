# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest
from pathlib import Path

import pandas as pd
from hta.common.trace import Trace
from hta.common.trace_call_graph import CallGraph
from hta.common.trace_call_stack import CallStackGraph, CallStackIdentity


class TraceCallGraphTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        test_data_path = Path(__file__).parent.parent.joinpath(
            "tests/data/call_stack/backward_thread.json"
        )
        self.test_trace_backward_threads: str = str(test_data_path)
        self.t_backward_threads: Trace = Trace(
            trace_files={0: self.test_trace_backward_threads},
            trace_dir="",
        )
        self.t_backward_threads.parse_traces()
        self.t_backward_threads.decode_symbol_ids(use_shorten_name=False)

        self.cg_backward_threads: CallGraph = CallGraph(
            self.t_backward_threads, ranks=[0]
        )
        self.df_backward_threads: pd.DataFrame = (
            self.cg_backward_threads.trace_data.get_trace(0)
        )

    @staticmethod
    def _get_first_index(df: pd.DataFrame, s_name: str) -> int:
        indices = df.loc[df["s_name"].eq(s_name)].index.to_list()
        return indices[0] if len(indices) > 0 else -1

    def test_call_graph_attributes(self) -> None:
        cg: CallGraph = self.cg_backward_threads
        df: pd.DataFrame = self.df_backward_threads

        self.assertEqual(len(cg.trace_data.get_all_traces()), 1)
        self.assertEqual(df.shape[0], 7)

        self.assertListEqual(list(cg.rank_to_stacks.keys()), [0])
        self.assertListEqual(list(cg.rank_to_nodes.keys()), [0])

    def test_call_graph_using_kernel_node(self) -> None:
        cg: CallGraph = self.cg_backward_threads
        df: pd.DataFrame = self.df_backward_threads

        kernel_indices = df.loc[df["s_name"].str.contains("cutlass::Kernel")]["index"]
        self.assertTrue(kernel_indices.shape[0] >= 1)

        kernel_index = kernel_indices.tolist()[0]
        kernel_node = df.loc[kernel_index]
        self.assertEqual(kernel_node["height"], 0)
        self.assertEqual(kernel_node["depth"], 6)

        stack = cg.get_stack_of_node(kernel_index)
        self.assertListEqual(
            stack["s_name"].tolist(),
            [
                "ProfilerStep#552",
                "## backward ##",
                "autograd::engine::evaluate_function: AddmmBackward0",
                "AddmmBackward0",
                "aten::mm",
                "cudaLaunchKernel",
                "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4>",
            ],
        )
        self.assertEqual(stack["depth"].tolist(), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(stack["height"].tolist(), [6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(stack["num_kernels"].tolist(), [1] * 7)

    def test_link_main_and_bwd_stacks_has_bwd_annotation(self) -> None:
        cg: CallGraph = self.cg_backward_threads

        self.assertEqual(len(cg.call_stacks), 2)
        self.assertTupleEqual(cg.mapping.shape, (2, 7))
        self.assertListEqual(["bwd", "main"], sorted(cg.mapping["label"].to_list()))

        main_stack_root = sorted(
            cg.mapping.loc[cg.mapping["label"].eq("main"), "stack_root"].to_list()
        )
        bwd_stack_root = sorted(
            cg.mapping.loc[cg.mapping["label"].eq("bwd"), "stack_root"].to_list()
        )
        self.assertListEqual(main_stack_root, bwd_stack_root)

    def test_link_main_and_bwd_stacks_no_bwd_annotation(self) -> None:
        t: Trace = self.t_backward_threads
        # remove backward annotation
        for _, df in t.get_all_traces().items():
            df.drop(df.loc[df["s_name"].eq("## backward ##")].index, inplace=True)
        cg: CallGraph = CallGraph(t)

        autograd_index = self._get_first_index(
            t.get_trace(0), "autograd::engine::evaluate_function: AddmmBackward0"
        )
        profiler_step_index = self._get_first_index(t.get_trace(0), "ProfilerStep#552")
        self.assertTrue(autograd_index > -1)
        self.assertTrue(profiler_step_index > -1)
        csg: CallStackGraph = cg.get_csg_of_node(autograd_index, 0)
        self.assertEqual(csg.get_parent(autograd_index), profiler_step_index)
        self.assertIn(autograd_index, csg.get_children(profiler_step_index))

    def test_skip_gpu_threads(self) -> None:
        trace_file = self.test_trace_backward_threads
        t: Trace = Trace(trace_files={i: trace_file for i in range(4)})
        t.parse_traces()
        # set a new pid for the traces
        for rank, df in t.get_all_traces().items():
            df["pid"] = rank + 1
        # For the test traces, there should be exactly two stacks for each rank.
        cg: CallGraph = CallGraph(t)
        self.assertListEqual(cg.mapping.groupby("rank").size().unique().tolist(), [2])

    def test_get_call_stacks(self) -> None:
        cg: CallGraph = self.cg_backward_threads

        count: int = 0
        for csg in cg.get_call_stacks():
            count += 1
            self.assertEqual(csg.identity.rank, 0)
        self.assertEqual(count, 2)

        count = 0
        for csg in cg.get_call_stacks(rank=0, pid=3914, tid=24922):
            count += 1
            self.assertEqual(
                csg.identity, CallStackIdentity(rank=0, pid=3914, tid=24922)
            )
        self.assertEqual(count, 1)

        for csg in cg.get_call_stacks(stack_index=1):
            self.assertEqual(
                csg.identity, CallStackIdentity(rank=0, pid=3914, tid=24922)
            )

    def test_get_csg_of_node(self) -> None:
        cg: CallGraph = self.cg_backward_threads
        csg = cg.get_csg_of_node(5)
        self.assertEqual(csg.identity, CallStackIdentity(rank=0, pid=3914, tid=24922))

    def test_get_stack_of_node(self) -> None:
        cg: CallGraph = self.cg_backward_threads

        # stack of GPU kernel, skipping ancestors: index = 5
        stack = cg.get_stack_of_node(5, skip_ancestors=True)
        self.assertEqual(len(stack), 1)
        self.assertListEqual(stack["parent"].tolist(), [6])

        # stack of GPU kernel, with ancestors: index = 5
        stack = cg.get_stack_of_node(5, skip_ancestors=False)
        self.assertEqual(len(stack), 7)
        self.assertListEqual(stack["parent"].tolist(), [-3914, 0, 1, 2, 3, 4, 6])

        # stack of GPU kernel, skipping ancestors: index = 2
        stack = cg.get_stack_of_node(2, skip_ancestors=True)
        self.assertEqual(len(stack), 5)
        self.assertListEqual(stack["parent"].tolist(), [1, 2, 3, 4, 6])

        # stack of GPU kernel, with ancestors: index = 2
        stack = cg.get_stack_of_node(2, skip_ancestors=False)
        self.assertEqual(len(stack), 7)
        self.assertListEqual(stack["parent"].tolist(), [-3914, 0, 1, 2, 3, 4, 6])

    def test_call_graph_from_dataframe(self) -> None:
        df = self.df_backward_threads
        symbol_table = self.t_backward_threads.symbol_table
        cg = CallGraph.from_dataframe(df, symbol_table)
        cs = cg.get_csg_of_node(1)

        node_parent_pairs = [
            (idx, cs.nodes[idx].parent) for idx in sorted(cs.nodes.keys())
        ]
        expected_node_pairs = [
            (-3914, -1),
            (0, -3914),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 6),
            (6, 4),
        ]

        self.assertEqual(cg.mapping.shape[0], 2)
        self.assertListEqual(node_parent_pairs, expected_node_pairs)
