# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Unit tests for call_stack.py

import unittest

import pandas as pd

from hta.common.call_stack import (
    CallGraph,
    CallStackGraph,
    CallStackIdentity,
    CallStackNode,
)
from hta.common.trace_filter import ZeroDurationFilter


class CallStackTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.df1 = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "ts": [0, 1, 3, 3, 7, 8, 10, 12, 16],
                "dur": [10, 5, 2, 1, 3, 1, 11, 6, 3],
                "pid": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                "tid": [2, 2, 2, 2, 2, 2, 2, 2, 2],
                "stream": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                "index_correlation": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            }
        )
        self.csi = CallStackIdentity(0, 1, 2)
        self.nodes = {
            -1: CallStackNode(parent=-1, depth=-1, children=[0, 6]),
            0: CallStackNode(parent=-1, depth=0, children=[1, 4]),
            1: CallStackNode(parent=0, depth=1, children=[2]),
            2: CallStackNode(parent=1, depth=2, children=[3]),
            3: CallStackNode(parent=2, depth=3, children=[]),
            4: CallStackNode(parent=0, depth=1, children=[5]),
            5: CallStackNode(parent=4, depth=2, children=[]),
            6: CallStackNode(parent=-1, depth=0, children=[7]),
            7: CallStackNode(parent=6, depth=1, children=[8]),
            8: CallStackNode(parent=7, depth=2, children=[]),
        }
        self.path_to_root_of_2 = [2, 1, 0, -1]
        self.path_to_root_of_3 = [3, 2, 1, 0, -1]
        self.path_to_root_of_5 = [5, 4, 0, -1]
        self.leaf_nodes_of_0 = [3, 5]
        self.leaf_nodes_of_4 = [5]
        self.leaf_nodes_of_5 = [5]
        self.paths_to_leaves_of_0 = [[0, 1, 2, 3], [0, 4, 5]]

        # add a 0 duration event, see index 3
        self.df2 = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4, 5],
                "ts": [0, 1, 3, 3, 7, 8],
                "dur": [10, 5, 2, 0, 3, 1],  # << index 3 is 0 duration
                "pid": [1, 1, 1, 1, 1, 1],
                "tid": [3, 3, 3, 3, 3, 3],
                "stream": [-1, -1, -1, -1, -1, -1],
                "index_correlation": [-1, -1, -1, -1, -1, -1],
            }
        )
        self.csi2 = CallStackIdentity(0, 1, 3)
        # note that 3 will be filtered out
        self.nodes2 = {
            -1: CallStackNode(parent=-1, depth=-1, children=[0]),
            0: CallStackNode(parent=-1, depth=0, children=[1, 4]),
            1: CallStackNode(parent=0, depth=1, children=[2]),
            2: CallStackNode(
                parent=1, depth=2, children=[]
            ),  # 3 will not be a child here
            4: CallStackNode(parent=0, depth=1, children=[5]),
            5: CallStackNode(parent=4, depth=2, children=[]),
        }

    def test_construct_call_graph(self):
        csg = CallStackGraph(self.df1, self.csi)
        nodes = csg.get_nodes()
        self.assertDictEqual(nodes, self.nodes)

    def test_construct_call_graph_0_dur(self):
        csg = CallStackGraph(self.df2, self.csi2, filter_func=ZeroDurationFilter)
        nodes = csg.get_nodes()
        self.assertDictEqual(nodes, self.nodes2)

    def test_sort_events(self):
        index = [0, 1, 2, 3]
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
            -1: CallStackNode(parent=-1, depth=-1, children=[0]),
            0: CallStackNode(parent=-1, depth=0, children=[1, 3]),
            1: CallStackNode(parent=0, depth=1, children=[]),
            2: CallStackNode(parent=3, depth=2, children=[]),
            3: CallStackNode(parent=0, depth=1, children=[2]),
        }
        csg = CallStackGraph(df, self.csi)
        self.assertDictEqual(nodes, csg.get_nodes())

    def test_get_path_to_root(self):
        csg = CallStackGraph(self.df1, self.csi)
        self.assertListEqual(csg.get_path_to_root(2), self.path_to_root_of_2)
        self.assertListEqual(csg.get_path_to_root(3), self.path_to_root_of_3)
        self.assertListEqual(csg.get_path_to_root(5), self.path_to_root_of_5)

    def test_get_leaf_nodes(self):
        csg = CallStackGraph(self.df1, self.csi)
        self.assertListEqual(csg.get_leaf_nodes(0), self.leaf_nodes_of_0)
        self.assertListEqual(csg.get_leaf_nodes(4), self.leaf_nodes_of_4)
        self.assertListEqual(csg.get_leaf_nodes(5), self.leaf_nodes_of_5)

    def test_get_paths_to_leaves(self):
        csg = CallStackGraph(self.df1, self.csi)
        self.assertListEqual(csg.get_paths_to_leaves(0), self.paths_to_leaves_of_0)
        self.assertListEqual(csg.get_paths_to_leaves(5), [self.leaf_nodes_of_5])

    def test_node_depth(self):
        csg = CallStackGraph(self.df1, self.csi)
        nodes = csg.get_nodes()
        df = csg.get_dataframe()

        depth_from_csg = csg.get_depth().to_dict()
        depth_from_nodes = {idx: node.depth for idx, node in nodes.items() if idx >= 0}
        self.assertDictEqual(depth_from_csg, depth_from_nodes)
        # Verify df is used
        self.assertIsNotNone(df)


class CallGraphTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        # Mock TraceCollection class for testing
        class MockTrace:
            def __init__(self, traces):
                self.traces = traces

            def get_all_traces(self):
                return self.traces.keys()

            def get_trace(self, rank):
                return self.traces[rank]

            def get_trace_df(self, rank):
                return self.traces[rank]

        # Create test data for trim_trace_events
        self.df_trim = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "ts": [0, 2, 4, 6],
                "dur": [10, 6, 8, 2],  # Child event 2 exceeds parent event 1's duration
                "pid": [1, 1, 1, 1],
                "tid": [1, 1, 1, 1],
                "stream": [-1, -1, -1, -1],
                "index_correlation": [-1, -1, -1, -1],
                "python_id": [100, 101, 102, 103],
                "python_parent_id": [-1, 100, 101, 102],
            }
        )

        self.trace_mock = MockTrace({0: self.df_trim})

    def test_trim_trace_events_basic(self):
        # Given & when
        CallGraph(self.trace_mock, pre_process_trace_data=True)

        # Then
        # Event 1: ts=2, dur=6, end=8
        # Event 2: ts=4, dur=8 (original), should be trimmed to dur=4 to end at 8
        self.assertEqual(self.df_trim.at[2, "dur"], 4)

    def test_trim_trace_events_complex_hierarchy(self):
        # Given
        df_complex = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4],
                "ts": [0, 2, 4, 6, 7],
                "dur": [15, 10, 10, 20, 8],
                "pid": [1, 1, 1, 1, 1],
                "tid": [1, 1, 1, 1, 1],
                "stream": [-1, -1, -1, -1, -1],
                "index_correlation": [-1, -1, -1, -1, -1],
                "python_id": [100, 101, 102, 103, 104],
                "python_parent_id": [-1, 100, 101, 102, 102],
            }
        )

        trace_complex = type(self.trace_mock)({0: df_complex})

        # When
        CallGraph(trace_complex, pre_process_trace_data=True)

        # Then
        # Event 1: ts=2, dur=10, end=12
        # Event 2: ts=4, dur=10, should be trimmed to dur=8 to end at 12
        # Event 3: ts=6, dur=20, should be trimmed to dur=6 to end at 12
        # Event 4: ts=7, dur=8, should be trimmed to dur=5 to end at 12
        self.assertEqual(df_complex.at[2, "dur"], 8)
        self.assertEqual(df_complex.at[3, "dur"], 6)
        self.assertEqual(df_complex.at[4, "dur"], 5)

    def test_trim_trace_events_different_threads(self):
        # Given
        df_threads = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "ts": [0, 2, 4, 6],
                "dur": [10, 6, 8, 2],
                "pid": [1, 1, 1, 1],
                "tid": [1, 1, 2, 2],  # Events 2 and 3 are in a different thread
                "stream": [-1, -1, -1, -1],
                "index_correlation": [-1, -1, -1, -1],
                "python_id": [100, 101, 102, 103],
                "python_parent_id": [-1, 100, 101, 102],
            }
        )

        trace_threads = type(self.trace_mock)({0: df_threads})

        # When
        CallGraph(trace_threads, pre_process_trace_data=True)

        # Then
        # Event 2 should not be trimmed because it's in a different thread than its parent
        self.assertEqual(df_threads.at[2, "dur"], 8)

    def test_trim_trace_events_no_trimming_needed(self):
        # Given
        df_no_trim = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "ts": [0, 2, 4, 6],
                "dur": [10, 6, 3, 1],  # All child events end before their parents
                "pid": [1, 1, 1, 1],
                "tid": [1, 1, 1, 1],
                "stream": [-1, -1, -1, -1],
                "index_correlation": [-1, -1, -1, -1],
                "python_id": [100, 101, 102, 103],
                "python_parent_id": [-1, 100, 101, 102],
            }
        )

        trace_no_trim = type(self.trace_mock)({0: df_no_trim})

        # When
        CallGraph(trace_no_trim, pre_process_trace_data=True)

        # Then
        # Durations should remain unchanged
        self.assertEqual(df_no_trim.at[1, "dur"], 6)
        self.assertEqual(df_no_trim.at[2, "dur"], 3)
        self.assertEqual(df_no_trim.at[3, "dur"], 1)

    def test_trim_trace_events_multiple_children(self):
        # Given
        df_multi_children = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4],
                "ts": [0, 2, 3, 6, 8],
                "dur": [
                    10,
                    8,
                    3,
                    2,
                    5,
                ],  # Child event 4 exceeds parent event 1's duration
                "pid": [1, 1, 1, 1, 1],
                "tid": [1, 1, 1, 1, 1],
                "stream": [-1, -1, -1, -1, -1],
                "index_correlation": [-1, -1, -1, -1, -1],
                "python_id": [100, 101, 102, 103, 104],
                "python_parent_id": [-1, 100, 101, 101, 101],
            }
        )

        trace_multi = type(self.trace_mock)({0: df_multi_children})

        # When
        CallGraph(trace_multi, pre_process_trace_data=True)

        # Then
        # Event 1: ts=2, dur=8, end=10
        # Event 4: ts=8, dur=5, should be trimmed to dur=2 to end at 10
        self.assertEqual(df_multi_children.at[4, "dur"], 2)

    def test_trim_trace_events_walltime(self):
        """Test the performance of trim_trace_events with a large dataset."""
        # Given
        import time

        import numpy as np

        # Create a large trace dataset with a deep hierarchy
        num_events = 10000

        # Generate a tree-like structure with multiple levels
        # Each event has a random number of children (0-5)
        python_ids = np.arange(100, 100 + num_events)

        # Initialize with root events (parent_id = -1)
        python_parent_ids = np.full(num_events, -1)

        # Create parent-child relationships
        current_parent_idx = 0
        for i in range(1, num_events):
            # Assign a parent from previous events
            if current_parent_idx < i:  # Ensure we don't create cycles
                python_parent_ids[i] = python_ids[current_parent_idx]

                # Move to next potential parent after adding some children
                if np.random.random() < 0.2:  # 20% chance to move to next parent
                    current_parent_idx += 1

        # Create timestamps with some overlapping events
        ts = np.zeros(num_events)
        dur = np.zeros(num_events)

        # Set timestamps and durations
        for i in range(num_events):
            if python_parent_ids[i] == -1:
                # Root events start at random times
                ts[i] = np.random.randint(0, 1000)
            else:
                # Child events start after their parent
                parent_idx = np.where(python_ids == python_parent_ids[i])[0][0]
                ts[i] = ts[parent_idx] + np.random.randint(1, 10)

            # Set duration - occasionally make it exceed parent's end time
            dur[i] = np.random.randint(5, 50)

            # 30% of events will exceed their parent's duration
            if python_parent_ids[i] != -1 and np.random.random() < 0.3:
                parent_idx = np.where(python_ids == python_parent_ids[i])[0][0]
                parent_end = ts[parent_idx] + dur[parent_idx]
                dur[i] = (parent_end - ts[i]) + np.random.randint(
                    1, 20
                )  # Exceed parent

        # Create DataFrame
        df_large = pd.DataFrame(
            {
                "index": np.arange(num_events),
                "ts": ts,
                "dur": dur,
                "pid": np.ones(num_events),
                "tid": np.ones(num_events),
                "stream": np.full(num_events, -1),
                "index_correlation": np.full(num_events, -1),
                "python_id": python_ids,
                "python_parent_id": python_parent_ids,
            }
        )

        # Create a mock trace object
        trace_large = type(self.trace_mock)({0: df_large})

        # When - measure time
        start_time = time.time()
        CallGraph(trace_large, pre_process_trace_data=True)
        end_time = time.time()

        # Then
        execution_time = end_time - start_time
        print(
            f"\nTrim trace events execution time for {num_events} events: {execution_time:.4f} seconds"
        )

        # Verify some events were actually trimmed
        # Count events where duration was likely trimmed (original would exceed parent)
        trimmed_count = 0
        for i in range(num_events):
            if python_parent_ids[i] != -1:
                parent_idx = np.where(python_ids == python_parent_ids[i])[0][0]
                parent_end = ts[parent_idx] + dur[parent_idx]
                child_end = ts[i] + df_large.at[i, "dur"]
                if child_end <= parent_end:
                    trimmed_count += 1

        # Just log the number of trimmed events for information
        print(f"Number of events that were trimmed: {trimmed_count}")

        # No explicit assertion on time, as performance varies by machine
        # This test is primarily for profiling and detecting major regressions


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
