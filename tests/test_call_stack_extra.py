# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import pandas as pd
from hta.common.call_stack import (
    CallStackGraph,
    CallStackIdentity,
    CallStackNode,
    compare_events,
    DeviceType,
    Event,
    EVENT_END,
    EVENT_START,
    infer_device_type,
    NON_EXISTENT_NODE_INDEX,
    NULL_NODE_INDEX,
)


def _make_cpu_df() -> pd.DataFrame:
    """Create a minimal CPU-thread trace dataframe with index_correlation column."""
    return pd.DataFrame(
        {
            "index": [1, 2, 3],
            "ts": [0, 5, 20],
            "dur": [30, 10, 5],
            "stream": [-1, -1, -1],
            "index_correlation": [-1, -1, -1],
            "pid": [100, 100, 100],
            "tid": [200, 200, 200],
        },
        index=pd.Index([1, 2, 3]),
    )


class TestInferDeviceType(unittest.TestCase):
    def test_gpu(self) -> None:
        df = pd.DataFrame({"stream": [1, 2, 3]})
        self.assertEqual(infer_device_type(df), DeviceType.GPU)

    def test_cpu(self) -> None:
        df = pd.DataFrame({"stream": [-1, -1]})
        self.assertEqual(infer_device_type(df), DeviceType.CPU)

    def test_unknown_when_empty(self) -> None:
        df = pd.DataFrame({"stream": []})
        self.assertEqual(infer_device_type(df), DeviceType.UNKNOWN)

    def test_unknown_when_mixed(self) -> None:
        df = pd.DataFrame({"stream": [1, -1]})
        self.assertEqual(infer_device_type(df), DeviceType.UNKNOWN)


class TestCompareEvents(unittest.TestCase):
    def test_same_idx_start_first(self) -> None:
        x = Event(idx=1, time=0, dur=10, type=EVENT_START)
        y = Event(idx=1, time=10, dur=10, type=EVENT_END)
        self.assertLess(compare_events(x, y), 0)

    def test_same_idx_end_after(self) -> None:
        x = Event(idx=1, time=10, dur=10, type=EVENT_END)
        y = Event(idx=1, time=0, dur=10, type=EVENT_START)
        self.assertGreater(compare_events(x, y), 0)

    def test_different_time(self) -> None:
        x = Event(idx=1, time=0, dur=5, type=EVENT_START)
        y = Event(idx=2, time=5, dur=5, type=EVENT_START)
        self.assertLess(compare_events(x, y), 0)


class TestCallStackIdentity(unittest.TestCase):
    def test_default_values(self) -> None:
        csi = CallStackIdentity()
        self.assertEqual(csi.rank, -1)
        self.assertEqual(csi.pid, -1)
        self.assertEqual(csi.tid, -1)

    def test_explicit_values(self) -> None:
        csi = CallStackIdentity(rank=0, pid=100, tid=200)
        self.assertEqual(csi.rank, 0)


class TestCallStackGraphBasics(unittest.TestCase):
    def setUp(self) -> None:
        self.csi = CallStackIdentity(rank=0, pid=100, tid=200)
        self.df = _make_cpu_df()
        self.csg = CallStackGraph(self.df, self.csi)

    def test_constructor_builds_nodes(self) -> None:
        self.assertEqual(self.csg.identity, self.csi)
        self.assertEqual(self.csg.device_type, DeviceType.CPU)
        self.assertGreater(len(self.csg.nodes), 0)

    def test_repr_contains_callstackgraph(self) -> None:
        self.assertIn("CallStackGraph", repr(self.csg))

    def test_get_nodes(self) -> None:
        nodes = self.csg.get_nodes()
        self.assertIsInstance(nodes, dict)
        self.assertIn(NULL_NODE_INDEX, nodes)

    def test_get_dataframe(self) -> None:
        self.assertIs(self.csg.get_dataframe(), self.df)

    def test_get_parent_existing(self) -> None:
        # Node 1 should be a child of NULL_NODE_INDEX (root)
        self.assertEqual(self.csg.get_parent(1), NULL_NODE_INDEX)

    def test_get_parent_missing(self) -> None:
        self.assertEqual(self.csg.get_parent(99999), NON_EXISTENT_NODE_INDEX)

    def test_get_children_existing(self) -> None:
        children = self.csg.get_children(1)
        self.assertIsInstance(children, list)

    def test_get_children_missing(self) -> None:
        self.assertEqual(self.csg.get_children(99999), [])

    def test_get_path_to_root_existing(self) -> None:
        path = self.csg.get_path_to_root(2)
        self.assertIn(2, path)
        # Should reach NULL_NODE_INDEX
        self.assertIn(NULL_NODE_INDEX, path)

    def test_get_path_to_root_missing(self) -> None:
        self.assertEqual(self.csg.get_path_to_root(99999), [])

    def test_get_paths_to_leaves_missing(self) -> None:
        self.assertEqual(self.csg.get_paths_to_leaves(99999), [])

    def test_get_paths_to_leaves_existing(self) -> None:
        paths = self.csg.get_paths_to_leaves(1)
        self.assertIsInstance(paths, list)

    def test_get_leaf_nodes(self) -> None:
        leaves = self.csg.get_leaf_nodes(1)
        self.assertIsInstance(leaves, list)

    def test_get_depth(self) -> None:
        depth = self.csg.get_depth()
        self.assertIsNotNone(depth)

    def test_dfs_traverse(self) -> None:
        visited_enter = []
        visited_exit = []

        def enter(idx: int, node: CallStackNode) -> None:
            visited_enter.append(idx)

        def exit_fn(idx: int, node: CallStackNode) -> None:
            visited_exit.append(idx)

        self.csg.dfs_traverse(enter, exit_fn)
        # Both should visit each node once
        self.assertEqual(len(visited_enter), len(visited_exit))
        self.assertGreater(len(visited_enter), 0)


class TestCallStackGraphErrors(unittest.TestCase):
    def test_missing_index_correlation_raises(self) -> None:
        df = pd.DataFrame(
            {
                "index": [1],
                "ts": [0],
                "dur": [10],
                "stream": [-1],
                "pid": [100],
                "tid": [200],
            },
            index=pd.Index([1]),
        )
        csi = CallStackIdentity(rank=0, pid=100, tid=200)
        with self.assertRaisesRegex(ValueError, "index_correlation"):
            CallStackGraph(df, csi)

    def test_gpu_short_circuits(self) -> None:
        # GPU device path skips graph construction
        df = pd.DataFrame(
            {
                "index": [1, 2],
                "ts": [0, 10],
                "dur": [5, 5],
                "stream": [1, 1],  # GPU
                "index_correlation": [-1, -1],
                "pid": [100, 100],
                "tid": [200, 200],
            },
            index=pd.Index([1, 2]),
        )
        csi = CallStackIdentity(rank=0, pid=100, tid=200)
        csg = CallStackGraph(df, csi)
        self.assertEqual(csg.device_type, DeviceType.GPU)
        # GPU short-circuits, so nodes dict stays empty
        self.assertEqual(len(csg.nodes), 0)


class TestCallStackNode(unittest.TestCase):
    def test_node_construction(self) -> None:
        node = CallStackNode(parent=0, depth=1, children=[2, 3])
        self.assertEqual(node.parent, 0)
        self.assertEqual(node.depth, 1)
        self.assertEqual(node.children, [2, 3])
