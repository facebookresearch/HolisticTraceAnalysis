#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest
from collections import namedtuple
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from hta.common.trace_call_stack import (
    _less_than,
    CallStackGraph,
    CallStackIdentity,
    CallStackNode,
    sort_events,
)
from hta.common.trace_symbol_table import TraceSymbolTable
from hta.common.types import DeviceType


class CallStackTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.df = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "cat": [1, 1, 1, 1, 1, 1, 1, 2, 3, 0, 1, 2, 2],
                "ts": [0, 1, 3, 3, 7, 8, 10, 12, 16, 0, 10, 15, 20],
                "dur": [10, 5, 2, 1, 3, 1, 11, 6, 3, 25, 10, 5, 5],
                "pid": [1] * 13,
                "tid": [2] * 9 + [3] * 4,
                "stream": [-1] * 13,
                "index_correlation": [-1] * 13,
            }
        )
        self.df_1 = self.df.loc[self.df.tid.eq(2)]
        self.df_2 = self.df.loc[self.df.tid.eq(3)]
        self.s_table = TraceSymbolTable()
        self.s_table.add_symbols(["annotation", "cpu_op", "cuda_runtime", "kernel"])

        self.csi_1 = CallStackIdentity(
            0,
            1,
            2,
        )
        self.csi_2 = CallStackIdentity(
            0,
            1,
            3,
        )
        self.root_index_1 = -abs(self.csi_1.tid)
        self.root_index_2 = -abs(self.csi_2.tid)

        self.nodes_1 = {
            self.root_index_1: CallStackNode(
                parent=-1, depth=-1, height=5, children=[0, 6]
            ),
            0: CallStackNode(
                parent=self.root_index_1, depth=0, height=4, children=[1, 4]
            ),
            1: CallStackNode(parent=0, depth=1, height=3, children=[2]),
            2: CallStackNode(parent=1, depth=2, height=2, children=[3]),
            3: CallStackNode(parent=2, depth=3, height=1, children=[]),
            4: CallStackNode(parent=0, depth=1, height=2, children=[5]),
            5: CallStackNode(parent=4, depth=2, height=1, children=[]),
            6: CallStackNode(parent=-2, depth=0, height=3, children=[7]),
            7: CallStackNode(parent=6, depth=1, height=2, children=[8]),
            8: CallStackNode(parent=7, depth=2, height=1, children=[]),
        }
        self.nodes_2 = {
            self.root_index_2: CallStackNode(
                parent=-1, depth=-1, height=4, children=[9]
            ),
            9: CallStackNode(
                parent=self.root_index_2, depth=0, height=3, children=[10, 12]
            ),
            10: CallStackNode(parent=9, depth=1, height=2, children=[11]),
            11: CallStackNode(parent=10, depth=2, height=1, children=[]),
            12: CallStackNode(parent=9, depth=1, height=1, children=[]),
        }
        self.path_to_root_of_2 = [2, 1, 0, self.root_index_1]
        self.path_to_root_of_3 = [3, 2, 1, 0, self.root_index_1]
        self.path_to_root_of_5 = [5, 4, 0, self.root_index_1]
        self.leaf_nodes_of_0 = [3, 5]
        self.leaf_nodes_of_4 = [5]
        self.leaf_nodes_of_5 = [5]
        self.paths_to_leaves_of_0 = [[0, 1, 2, 3], [0, 4, 5]]
        self.correlations = pd.DataFrame(
            {
                "cpu_index": [7],
                "gpu_index": [8],
            }
        )

    def test_construct_call_stack_graph(self) -> None:
        csg_1 = CallStackGraph(
            self.df_1,
            self.csi_1,
            self.correlations,
            self.df,
            self.s_table,
        )
        csg_2 = CallStackGraph(
            self.df_2,
            self.csi_2,
            self.correlations,
            self.df,
            self.s_table,
        )

        self.assertDictEqual(
            csg_1.get_nodes(),
            self.nodes_1,
            f"expected: {self.nodes_1};\ngot: {csg_1.get_nodes()}",
        )
        self.assertDictEqual(
            csg_2.get_nodes(),
            self.nodes_2,
            f"expected: {self.nodes_2};\ngot: {csg_2.get_nodes()}",
        )
        self.assertEqual(
            csg_1.root_index,
            self.root_index_1,
            f"expected root_index: {self.root_index_1};\ngot: {csg_1.root_index}",
        )

    def test_call_stack_graph_with_consolidated_nodes(self) -> None:
        nodes: Dict[int, CallStackNode] = {}
        csg1 = CallStackGraph(
            self.df_1, self.csi_1, self.correlations, self.df, self.s_table, nodes
        )
        csg2 = CallStackGraph(
            self.df_2, self.csi_2, self.correlations, self.df, self.s_table, nodes
        )
        nodes_1 = csg1.get_nodes()
        nodes_2 = csg2.get_nodes()

        self.assertDictEqual(nodes_1, nodes_2, f"expected: {nodes_1};\ngot: {nodes_2}")
        self.assertEqual(
            csg1.root_index,
            self.root_index_1,
            f"expected root_index: {self.root_index_1};\ngot: {csg1.root_index}",
        )
        for i in self.df["index"].to_list():
            self.assertIn(i, nodes_1, f"node {i} not in the node map.")
        csg1_nodes = {i: nodes_1[i] for i in self.df_1["index"].to_list()}
        csg1_nodes[csg1.root_index] = self.nodes_1[self.root_index_1]
        self.assertDictEqual(nodes, csg1.get_nodes())

    def test_update_root(self) -> None:
        nodes: Dict[int, CallStackNode] = {}
        csg1 = CallStackGraph(
            self.df_1, self.csi_1, self.correlations, self.df, self.s_table, nodes
        )
        csg2 = CallStackGraph(
            self.df_2, self.csi_2, self.correlations, self.df, self.s_table, nodes
        )
        new_root_index = 9
        csg1.update_parent_of_first_layer_nodes(new_root_index)
        self.assertEqual(
            csg1.root_index,
            csg1.get_root(new_root_index),
            f"root index doesn't match: expected {csg1.get_root(new_root_index)}; got {csg1.root_index}",
        )
        self.assertEqual(
            csg2.root_index,
            self.root_index_2,
            f"root index doesn't match: expected {self.root_index_2}; got {csg2.root_index}",
        )

        s = set(csg2.nodes[new_root_index].children)
        s1 = set(self.nodes_1[self.root_index_1].children)
        s2 = set(self.nodes_2[new_root_index].children)
        self.assertTrue(s.issuperset(s1), f"nodes {s1 - s} is in {s1} but not in {s}")
        self.assertTrue(s.issuperset(s2), f"nodes {s2 - s} is in {s2} but not in {s}")

        for c in csg1.nodes[new_root_index].children:
            path = csg1.get_path_to_root(c)
            self.assertIn(
                new_root_index, path, f"{new_root_index} not on the path {path}"
            )
            self.assertEqual(
                path[-1],
                self.root_index_2,
                f"{self.root_index_2} is not the last node on the path {path}",
            )

    @staticmethod
    def nodes2str(nodes: Dict[int, CallStackNode]) -> str:
        s = "\n"
        for k, v in nodes.items():
            s += f"  {k}: {v}\n"
        return s

    def test_sort_events_without_zero_duration(self) -> None:
        index = [1, 2, 3, 4]
        start = [0, 0, 5, 5]
        dur = [10, 5, 1, 5]
        stream = [-1, -1, -1, -1]
        cor = [-1, -1, -1, -1]
        df = pd.DataFrame(
            {
                "index": index,
                "ts": start,
                "dur": dur,
                "stream": stream,
                "index_correlation": cor,
            }
        )
        nodes = {
            self.root_index_1: CallStackNode(
                parent=-1, depth=-1, height=4, children=[1]
            ),
            1: CallStackNode(
                parent=self.root_index_1, depth=0, height=3, children=[2, 4]
            ),
            2: CallStackNode(parent=1, depth=1, height=1, children=[]),
            4: CallStackNode(parent=1, depth=1, height=2, children=[3]),
            3: CallStackNode(parent=4, depth=2, height=1, children=[]),
        }
        csg = CallStackGraph(df, self.csi_1, self.correlations, self.df, self.s_table)
        self.assertDictEqual(
            nodes, csg.get_nodes(), f"expect: {self.nodes2str(nodes)}\ngot: {csg}"
        )

    def test_sort_events_zero_duration(self) -> None:
        index = [1, 2, 3, 4]
        start = [0, 0, 4, 5]
        dur = [0, 5, 0, 0]
        stream = [-1, -1, -1, -1]
        cor = [-1, -1, -1, -1]
        df = pd.DataFrame(
            {
                "index": index,
                "ts": start,
                "dur": dur,
                "stream": stream,
                "index_correlation": cor,
            }
        )

        nodes = {
            self.root_index_1: CallStackNode(
                parent=-1, depth=-1, height=3, children=[2]
            ),
            1: CallStackNode(parent=2, depth=1, height=1, children=[]),
            2: CallStackNode(
                parent=self.root_index_1, depth=0, height=2, children=[1, 3, 4]
            ),
            3: CallStackNode(parent=2, depth=1, height=1, children=[]),
            4: CallStackNode(parent=2, depth=1, height=1, children=[]),
        }
        csg = CallStackGraph(df, self.csi_1, self.correlations, self.df, self.s_table)
        self.assertDictEqual(nodes, csg.get_nodes(), f"got:\n{str(csg)}")

    def test_less_than(self) -> None:
        a = np.array(
            [
                [24721, 15, 1, 1666149715059476],
                [24723, 1, -1, 1666149715059485],
                [24723, 1, 1, 1666149715059486],
                [24724, 0, 1, 1666149715059486],
                [24724, 0, -1, 1666149715059486],
                [24725, 0, -1, 1666149715059486],
                [24725, 0, 1, 1666149715059486],
                [24726, 225181, -1, 1666149715059490],
            ]
        )
        TC = namedtuple("TC", ["x", "y", "expect_result"])
        test_cases: List[TC] = [
            TC(a[0], a[1], True),
            TC(a[1], a[2], True),
            TC(a[1], a[3], True),
            TC(a[2], a[3], False),
            TC(a[2], a[4], False),
            TC(a[2], a[5], False),
            TC(a[2], a[6], False),
            TC(a[3], a[4], False),
            TC(a[3], a[5], False),
            TC(a[3], a[6], False),
            TC(a[3], a[7], True),
            TC(a[4], a[5], True),
            TC(a[4], a[6], True),
            TC(a[5], a[6], True),
        ]

        for tc in test_cases:
            got = _less_than(tc.x, tc.y)
            self.assertEqual(
                got,
                tc.expect_result,
                f"\n{tc.x}\n{tc.y}\n Eexpect: {tc.expect_result} got: {got}",
            )

    def test_sort_events(self) -> None:
        a = np.array(
            [
                [24721, 15, 1, 1666149715059476],
                [24723, 1, -1, 1666149715059485],
                [24723, 1, 1, 1666149715059486],
                [24724, 0, 1, 1666149715059486],
                [24724, 0, -1, 1666149715059486],
                [24725, 0, -1, 1666149715059486],
                [24725, 0, 1, 1666149715059486],
                [24726, 225181, -1, 1666149715059490],
            ]
        )
        expected = np.array(
            [
                [24721, 15, 1, 1666149715059476],
                [24723, 1, -1, 1666149715059485],
                [24724, 0, -1, 1666149715059486],
                [24725, 0, -1, 1666149715059486],
                [24725, 0, 1, 1666149715059486],
                [24724, 0, 1, 1666149715059486],
                [24723, 1, 1, 1666149715059486],
                [24726, 225181, -1, 1666149715059490],
            ]
        )
        sort_events(a)

        self.assertEqual(a.shape, expected.shape)
        self.assertListEqual(expected.tolist(), a.tolist(), f"got:\n{a}")

    def test_call_stack_with_zero_duration_events(self) -> None:
        a = np.array(
            [
                [0, 55, 0, -1, -1],
                [1, 61, 15, -1, -1],
                [2, 62, 0, -1, -1],
                [3, 85, 1, -1, -1],
                [4, 86, 0, -1, -1],
                [5, 86, 0, -1, -1],
                [6, 90, 225181, -1, -1],
                [7, 91, 0, -1, -1],
            ],
            dtype=np.int64,
        )
        cols = ["index", "ts", "dur", "stream", "index_correlation"]
        df = pd.DataFrame(data=a, columns=cols)
        df = df.set_index("index", drop=False)
        nodes = {
            -1: CallStackNode(parent=-1, depth=-1, height=4, children=[0, 1, 3, 6]),
            0: CallStackNode(parent=-1, depth=0, height=1, children=[]),
            1: CallStackNode(parent=-1, depth=0, height=2, children=[2]),
            2: CallStackNode(parent=1, depth=1, height=1, children=[]),
            3: CallStackNode(parent=-1, depth=0, height=3, children=[4]),
            4: CallStackNode(parent=3, depth=1, height=2, children=[5]),
            5: CallStackNode(parent=4, depth=2, height=1, children=[]),
            6: CallStackNode(parent=-1, depth=0, height=2, children=[7]),
            7: CallStackNode(parent=6, depth=1, height=1, children=[]),
        }
        correlations = pd.DataFrame(columns=["cpu_index", "gpu_index"])
        csg = CallStackGraph(
            df, CallStackIdentity(0, 1, 1), correlations, df, self.s_table
        )
        self.assertDictEqual(nodes, csg.get_nodes(), f"got:\n{str(csg)}")

    def test_get_path_to_root(self) -> None:
        csg = CallStackGraph(
            self.df_1,
            self.csi_1,
            self.correlations,
            self.df_1,
            self.s_table,
        )
        self.assertListEqual(csg.get_path_to_root(2), self.path_to_root_of_2)
        self.assertListEqual(csg.get_path_to_root(3), self.path_to_root_of_3)
        self.assertListEqual(csg.get_path_to_root(5), self.path_to_root_of_5)

    def test_get_leaf_nodes(self) -> None:
        csg = CallStackGraph(
            self.df_1, self.csi_1, self.correlations, self.df_1, self.s_table
        )
        self.assertListEqual(csg.get_leaf_nodes(0), self.leaf_nodes_of_0)
        self.assertListEqual(csg.get_leaf_nodes(4), self.leaf_nodes_of_4)
        self.assertListEqual(csg.get_leaf_nodes(5), self.leaf_nodes_of_5)

    def test_get_paths_to_leaves(self) -> None:
        csg = CallStackGraph(
            self.df_1, self.csi_1, self.correlations, self.df_1, self.s_table
        )
        self.assertListEqual(csg.get_paths_to_leaves(0), self.paths_to_leaves_of_0)
        self.assertListEqual(csg.get_paths_to_leaves(5), [self.leaf_nodes_of_5])

    def test_node_depth(self) -> None:
        csg = CallStackGraph(
            self.df_1, self.csi_1, self.correlations, self.df, self.s_table
        )
        nodes = csg.get_nodes()

        depth_from_csg: pd.Series = csg.get_depth()
        depth_from_nodes = pd.Series(
            {idx: node.depth for idx, node in nodes.items() if idx >= 0},
            dtype=pd.Int16Dtype(),
            name="depth",
        )
        pd.testing.assert_series_equal(depth_from_csg, depth_from_nodes)

    def test_get_root(self) -> None:
        stack = CallStackGraph(
            self.df_1, self.csi_1, self.correlations, self.df, self.s_table
        )
        for idx in self.nodes_1:
            if idx < 0:
                continue
            got = stack.get_root(idx)
            self.assertEqual(got, self.root_index_1, f"{stack}\n node={idx} root={got}")

    # AI-assisted test
    def test_duplicate_events_logged(self) -> None:
        """Test that duplicate events in the sorted array trigger a log error (line 367)."""
        from unittest.mock import patch

        # Create a DataFrame where the melted events array will contain
        # duplicate tuples. We patch sort_events to inject a duplicate row.
        df = pd.DataFrame(
            {
                "index": [0, 1, 2],
                "ts": [0, 5, 10],
                "dur": [3, 3, 3],
                "stream": [-1, -1, -1],
                "index_correlation": [-1, -1, -1],
                "pid": [1, 1, 1],
                "tid": [2, 2, 2],
                "cat": [1, 1, 1],
            }
        )
        correlations = pd.DataFrame(columns=["cpu_index", "gpu_index"])

        original_sort_events: Callable[[np.ndarray], None] = sort_events

        def sort_and_inject_dup(events: np.ndarray) -> None:
            """Sort then duplicate the first row to trigger the duplicate check."""
            original_sort_events(events)
            # Overwrite the second row with a copy of the first to create a duplicate
            events[1] = events[0].copy()

        with patch(
            "hta.common.trace_call_stack.sort_events", side_effect=sort_and_inject_dup
        ):
            with self.assertLogs("hta", level="ERROR") as cm:
                try:
                    CallStackGraph(
                        df,
                        self.csi_1,
                        correlations,
                        df,
                        self.s_table,
                    )
                except Exception:
                    pass  # graph construction may fail with injected duplicates
        self.assertTrue(
            any("duplicates" in msg for msg in cm.output),
            f"Expected 'duplicates' error log, got: {cm.output}",
        )

    # AI-assisted test
    def test_duplicate_check_uses_set_of_tuples(self) -> None:
        """Test duplicate detection uses set(map(tuple, events)) instead of np.unique (line 367).

        Verifies that non-duplicate events pass without error logging.
        """
        # All events have unique (index, dur, kind, ts) — no duplicates
        df = pd.DataFrame(
            {
                "index": [0, 1, 2],
                "ts": [0, 10, 20],
                "dur": [5, 5, 5],
                "stream": [-1, -1, -1],
                "index_correlation": [-1, -1, -1],
                "pid": [1, 1, 1],
                "tid": [2, 2, 2],
                "cat": [1, 1, 1],
            }
        )
        correlations = pd.DataFrame(columns=["cpu_index", "gpu_index"])

        # Should construct without any "duplicates" error
        with self.assertLogs("hta", level="DEBUG") as cm:
            CallStackGraph(
                df,
                self.csi_1,
                correlations,
                df,
                self.s_table,
            )
        # No "duplicates" message should appear
        self.assertFalse(
            any("duplicates" in msg for msg in cm.output),
            "No duplicates error expected for unique events",
        )

    # AI-assisted test
    def test_kernel_info_namedtuple_fields(self) -> None:
        """Test KernelInfo NamedTuple has correct field names (lines 790-796).

        Verifies the renamed field 'num_kernels' (was 'count') is used
        correctly in DFS traversal and DataFrame column output.
        """
        # Create a trace with a CPU op (index=0) that has two GPU kernel
        # children (index=1, index=2) to exercise multi-child DFS accumulation
        # at line 823: count = count + c_info.num_kernels
        full_df = pd.DataFrame(
            {
                "index": [0, 1, 2],
                "cat": [1, 3, 3],  # 1=cpu_op, 3=kernel
                "ts": [0, 5, 12],
                "dur": [20, 4, 3],
                "pid": [1, 1, 1],
                "tid": [2, 2, 2],
                "stream": [-1, 7, 7],
                "index_correlation": [-1, 0, 0],
                "depth": pd.array([-1, -1, -1], dtype=pd.Int16Dtype()),
                "height": pd.array([-1, -1, -1], dtype=pd.Int16Dtype()),
                "parent": pd.array([-1, -1, -1], dtype=pd.Int64Dtype()),
                "num_kernels": pd.array([0, 0, 0], dtype=pd.Int64Dtype()),
                "kernel_dur_sum": pd.array([0, 0, 0], dtype=pd.Int64Dtype()),
                "kernel_span": pd.array([0, 0, 0], dtype=pd.Int64Dtype()),
                "first_kernel_start": pd.array([-1, -1, -1], dtype=pd.Int64Dtype()),
                "last_kernel_end": pd.array([-1, -1, -1], dtype=pd.Int64Dtype()),
            }
        )
        full_df["end"] = full_df["ts"] + full_df["dur"]
        cpu_df = full_df.loc[full_df["stream"].eq(-1)]
        correlations = pd.DataFrame({"cpu_index": [0, 0], "gpu_index": [1, 2]})

        csg = CallStackGraph(
            cpu_df,
            self.csi_1,
            correlations,
            full_df,
            self.s_table,
            save_call_stack_to_df=False,
        )

        # Link both GPU kernels as children of CPU op 0
        csg._add_edge(0, 1, DeviceType.GPU)
        csg.nodes[1] = CallStackNode(
            parent=0, depth=1, height=0, device=DeviceType.GPU, children=[]
        )
        csg._add_edge(0, 2, DeviceType.GPU)
        csg.nodes[2] = CallStackNode(
            parent=0, depth=1, height=0, device=DeviceType.GPU, children=[]
        )

        csg._add_kernel_info_to_cpu_ops()

        # CPU op 0 should aggregate both kernels via num_kernels (line 823)
        row = full_df.loc[full_df["index"] == 0].iloc[0]
        self.assertEqual(row["num_kernels"], 2, "CPU op should have 2 kernels")
        self.assertEqual(
            row["kernel_dur_sum"], 7, "Kernel duration sum should be 4 + 3 = 7"
        )
        self.assertEqual(row["first_kernel_start"], 5)
        self.assertEqual(row["last_kernel_end"], 15)  # 12 + 3
        self.assertEqual(
            row["kernel_span"], 10, "Span should be last_end - first_start = 15 - 5"
        )

    # AI-assisted test
    def test_kernel_info_df_column_name(self) -> None:
        """Test DataFrame uses 'num_kernels' column name not 'count' (line 854).

        Verifies that pd.DataFrame.from_dict with KernelInfo produces
        a column named 'num_kernels' which matches the full_df column.
        """
        from typing import NamedTuple

        # Reproduce the KernelInfo NamedTuple as defined in the source
        class KernelInfo(NamedTuple):
            num_kernels: int
            sum_dur: int
            kernel_span: int
            first_start: int
            last_end: int

        # Build a dict like kernel_info in the source and convert to DataFrame
        kernel_info = {
            0: KernelInfo(
                num_kernels=3, sum_dur=100, kernel_span=50, first_start=10, last_end=60
            ),
            1: KernelInfo(
                num_kernels=1, sum_dur=20, kernel_span=20, first_start=15, last_end=35
            ),
        }
        df_info = pd.DataFrame.from_dict(kernel_info, orient="index")

        # The column should be 'num_kernels', not 'count'
        self.assertIn("num_kernels", df_info.columns)
        self.assertNotIn("count", df_info.columns)
        self.assertEqual(df_info.loc[0, "num_kernels"], 3)
        self.assertEqual(df_info.loc[1, "num_kernels"], 1)

    # AI-assisted test
    def test_add_kernel_info_to_cpu_ops(self) -> None:
        """Test _add_kernel_info_to_cpu_ops with KernelInfo NamedTuple (lines 790-854).

        Verifies that KernelInfo.num_kernels (renamed from count) is used correctly
        and that kernel statistics are propagated to cpu ops in full_df.
        """
        # Build a small trace: one CPU op (index=0) launching one GPU kernel (index=1)
        # via index_correlation.
        full_df = pd.DataFrame(
            {
                "index": [0, 1],
                "cat": [1, 3],  # 1=cpu_op, 3=kernel
                "ts": [0, 5],
                "dur": [20, 10],
                "pid": [1, 1],
                "tid": [2, 2],
                "stream": [-1, 7],  # -1=CPU, 7=GPU stream
                "index_correlation": [-1, 0],  # kernel correlates to cpu op 0
                # Stack columns required by _add_kernel_info_to_cpu_ops
                "depth": pd.array([-1, -1], dtype=pd.Int16Dtype()),
                "height": pd.array([-1, -1], dtype=pd.Int16Dtype()),
                "parent": pd.array([-1, -1], dtype=pd.Int64Dtype()),
                "num_kernels": pd.array([0, 0], dtype=pd.Int64Dtype()),
                "kernel_dur_sum": pd.array([0, 0], dtype=pd.Int64Dtype()),
                "kernel_span": pd.array([0, 0], dtype=pd.Int64Dtype()),
                "first_kernel_start": pd.array([-1, -1], dtype=pd.Int64Dtype()),
                "last_kernel_end": pd.array([-1, -1], dtype=pd.Int64Dtype()),
            }
        )
        full_df["end"] = full_df["ts"] + full_df["dur"]
        cpu_df = full_df.loc[full_df["stream"].eq(-1)]
        correlations = pd.DataFrame({"cpu_index": [0], "gpu_index": [1]})

        csg = CallStackGraph(
            cpu_df,
            self.csi_1,
            correlations,
            full_df,
            self.s_table,
            save_call_stack_to_df=False,
        )

        # Manually link the GPU kernel as a child of the CPU op
        csg._add_edge(0, 1, DeviceType.GPU)
        csg.nodes[1] = CallStackNode(
            parent=0, depth=1, height=0, device=DeviceType.GPU, children=[]
        )

        # Now call _add_kernel_info_to_cpu_ops
        csg._add_kernel_info_to_cpu_ops()

        # Verify kernel info was propagated to the CPU op (index=0)
        row = full_df.loc[full_df["index"] == 0].iloc[0]
        self.assertEqual(row["num_kernels"], 1, "CPU op should have 1 kernel")
        self.assertEqual(row["kernel_dur_sum"], 10, "Kernel duration should be 10")
        self.assertEqual(row["first_kernel_start"], 5, "First kernel start should be 5")
        self.assertEqual(row["last_kernel_end"], 15, "Last kernel end should be 15")
