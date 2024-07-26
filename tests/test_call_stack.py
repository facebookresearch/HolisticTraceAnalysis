# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd

from hta.common.call_stack import CallStackGraph, CallStackIdentity, CallStackNode


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
        csg = CallStackGraph(self.df2, self.csi2, filter_query="dur > 0")
        nodes = csg.get_nodes()
        self.assertDictEqual(nodes, self.nodes2)

    def test_sort_events(self):
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
            -1: CallStackNode(parent=-1, depth=-1, children=[1]),
            1: CallStackNode(parent=-1, depth=0, children=[2, 4]),
            2: CallStackNode(parent=1, depth=1, children=[]),
            4: CallStackNode(parent=1, depth=1, children=[3]),
            3: CallStackNode(parent=4, depth=2, children=[]),
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
