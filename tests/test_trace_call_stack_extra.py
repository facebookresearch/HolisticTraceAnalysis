# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import numpy as np
import pandas as pd
from hta.common.trace_call_stack import (
    _cmp_events_with_zero_duration,
    _less_than,
    CallStackGraph,
    CallStackIdentity,
    CallStackNode,
    CLOSE_END,
    is_events_sorted,
    NON_EXISTENT_NODE_INDEX,
    OPEN_END,
    sort_events,
)
from hta.common.trace_symbol_table import TraceSymbolTable
from hta.common.types import DeviceType


def _ev(idx: int, dur: int, kind: int, ts: int) -> np.ndarray:
    return np.array([idx, dur, kind, ts])


class TestCmpEventsWithZeroDuration(unittest.TestCase):
    def test_zero_inside_nonzero_close(self) -> None:
        x = _ev(1, 0, OPEN_END, 5)
        y = _ev(2, 10, CLOSE_END, 5)
        # zero event opens, non-zero closes: True (zero comes before)
        self.assertTrue(_cmp_events_with_zero_duration(x, y))

    def test_nonzero_open_zero(self) -> None:
        x = _ev(1, 10, OPEN_END, 5)
        y = _ev(2, 0, OPEN_END, 5)
        # x is non-zero, y is zero: True if x is opening
        self.assertTrue(_cmp_events_with_zero_duration(x, y))

    def test_two_zero_open_ends(self) -> None:
        x = _ev(1, 0, OPEN_END, 5)
        y = _ev(2, 0, OPEN_END, 5)
        # Both zero open: smaller index first
        self.assertTrue(_cmp_events_with_zero_duration(x, y))

    def test_two_zero_close_ends(self) -> None:
        x = _ev(2, 0, CLOSE_END, 5)
        y = _ev(1, 0, CLOSE_END, 5)
        # Both zero close: larger index first (so x with higher idx comes first)
        self.assertTrue(_cmp_events_with_zero_duration(x, y))

    def test_two_zero_one_open_one_close(self) -> None:
        x = _ev(1, 0, OPEN_END, 5)
        y = _ev(2, 0, CLOSE_END, 5)
        # One open one close: open first
        self.assertTrue(_cmp_events_with_zero_duration(x, y))


class TestLessThan(unittest.TestCase):
    def test_different_time(self) -> None:
        x = _ev(1, 10, OPEN_END, 5)
        y = _ev(2, 10, OPEN_END, 10)
        self.assertTrue(_less_than(x, y))

    def test_same_index_open_first(self) -> None:
        x = _ev(1, 10, OPEN_END, 5)
        y = _ev(1, 10, CLOSE_END, 15)
        # Same index: open first
        self.assertTrue(_less_than(x, y))

    def test_close_before_open_at_same_time(self) -> None:
        x = _ev(1, 10, CLOSE_END, 15)
        y = _ev(2, 10, OPEN_END, 15)
        self.assertTrue(_less_than(x, y))

    def test_two_open_ends_longer_first(self) -> None:
        x = _ev(1, 100, OPEN_END, 0)
        y = _ev(2, 50, OPEN_END, 0)
        # Longer duration first when both opening at same time
        self.assertTrue(_less_than(x, y))

    def test_two_close_ends_shorter_first(self) -> None:
        x = _ev(1, 50, CLOSE_END, 100)
        y = _ev(2, 100, CLOSE_END, 100)
        # Shorter duration first when both closing at same time
        self.assertTrue(_less_than(x, y))

    def test_zero_dur_path(self) -> None:
        x = _ev(1, 0, OPEN_END, 5)
        y = _ev(2, 10, CLOSE_END, 5)
        # Delegates to zero-duration comparator
        self.assertTrue(_less_than(x, y))


class TestSortAndIsSorted(unittest.TestCase):
    def test_is_events_sorted_true(self) -> None:
        a = np.array(
            [
                [1, 10, OPEN_END, 0],
                [1, 10, CLOSE_END, 10],
            ]
        )
        self.assertTrue(is_events_sorted(a))

    def test_is_events_sorted_false(self) -> None:
        a = np.array(
            [
                [1, 10, CLOSE_END, 10],
                [1, 10, OPEN_END, 0],
            ]
        )
        self.assertFalse(is_events_sorted(a))

    def test_sort_events(self) -> None:
        a = np.array(
            [
                [1, 10, CLOSE_END, 10],
                [1, 10, OPEN_END, 0],
            ]
        )
        sort_events(a)
        # After sort, OPEN should come first
        self.assertEqual(a[0][2], OPEN_END)


class TestCallStackIdentity(unittest.TestCase):
    def test_default_values(self) -> None:
        c = CallStackIdentity()
        self.assertEqual(c.rank, -1)
        self.assertEqual(c.pid, -1)
        self.assertEqual(c.tid, -1)

    def test_explicit(self) -> None:
        c = CallStackIdentity(rank=0, pid=1, tid=2)
        self.assertEqual(c.rank, 0)
        self.assertEqual(c.pid, 1)
        self.assertEqual(c.tid, 2)


class TestCallStackNode(unittest.TestCase):
    def test_defaults(self) -> None:
        n = CallStackNode()
        self.assertEqual(n.parent, -1)
        self.assertEqual(n.depth, -1)
        self.assertEqual(n.height, -1)
        self.assertEqual(n.device, DeviceType.CPU)
        self.assertEqual(n.children, [])

    def test_explicit(self) -> None:
        n = CallStackNode(
            parent=0, depth=2, height=3, device=DeviceType.GPU, children=[1, 2]
        )
        self.assertEqual(n.parent, 0)
        self.assertEqual(n.depth, 2)
        self.assertEqual(n.height, 3)
        self.assertEqual(n.device, DeviceType.GPU)
        self.assertEqual(n.children, [1, 2])


def _build_cpu_csg() -> CallStackGraph:
    """Helper to build a CallStackGraph from a minimal CPU thread DataFrame."""
    sym = TraceSymbolTable()
    sym.add_symbols(["op_a", "op_b", "op_c"])
    df = pd.DataFrame(
        {
            "index": [10, 11, 12],
            "ts": [0, 5, 100],
            "dur": [200, 50, 30],
            "stream": [-1, -1, -1],
            "pid": [1, 1, 1],
            "tid": [2, 2, 2],
            "name": [0, 1, 2],
            "cat": [0, 0, 0],
            "index_correlation": [-1, -1, -1],
        },
        index=pd.Index([10, 11, 12]),
    )
    full_df = df.copy()
    cpu_gpu_corr = pd.DataFrame({"cpu_index": [], "gpu_index": []})
    csi = CallStackIdentity(rank=0, pid=1, tid=2)
    return CallStackGraph(
        df=df,
        identity=csi,
        cpu_gpu_correlation=cpu_gpu_corr,
        full_df=full_df,
        symbol_table=sym,
        nodes=None,
        use_existing_stack_columns=False,
        save_call_stack_to_df=True,
    )


class TestCallStackGraphCpu(unittest.TestCase):
    def test_construct_repr_and_get_nodes(self) -> None:
        csg = _build_cpu_csg()
        self.assertIn("CallStackGraph", repr(csg))
        nodes = csg.get_nodes()
        # Root + 3 events
        self.assertGreaterEqual(len(nodes), 3)

    def test_get_parent_existing(self) -> None:
        csg = _build_cpu_csg()
        # Node 11 (op_b) starts at ts=5, fits inside op_a (ts=0..200)
        self.assertEqual(csg.get_parent(11), 10)

    def test_get_parent_missing(self) -> None:
        csg = _build_cpu_csg()
        self.assertEqual(csg.get_parent(99999), NON_EXISTENT_NODE_INDEX)

    def test_gpu_thread_returns_early(self) -> None:
        # GPU stream -> no graph constructed
        sym = TraceSymbolTable()
        sym.add_symbols(["k"])
        df = pd.DataFrame(
            {
                "index": [100],
                "ts": [0],
                "dur": [10],
                "stream": [1],
                "pid": [1],
                "tid": [2],
                "name": [0],
                "cat": [0],
                "index_correlation": [-1],
            },
            index=pd.Index([100]),
        )
        csi = CallStackIdentity(rank=0, pid=1, tid=2)
        csg = CallStackGraph(
            df=df,
            identity=csi,
            cpu_gpu_correlation=pd.DataFrame({"cpu_index": [], "gpu_index": []}),
            full_df=df,
            symbol_table=sym,
        )
        # Has device_type GPU and skipped construction
        self.assertEqual(csg.device_type, DeviceType.GPU)
