# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd

from hta.common import singletrace
from hta.common.trace_collection import transform_correlation_to_index
from hta.common.trace_symbol_table import TraceSymbolTable


class CorrelationTestCase(unittest.TestCase):
    def test_something(self):
        # index, name, correlation, stream
        data = [
            [1, 22, 15, -1],
            [2, 12, 16, -1],
            [3, 10, 22, -1],
            [4, 10, -1, -1],
            [5, 45, 15, -1],  # event sync is on GPU but has stream = -1
            [6, 42, 16, -1],  # context sync is on GPU but has stream = -1
            [7, 71, 30, 7],
            [8, 12, -1, 7],
        ]
        df = pd.DataFrame(
            data,
            columns=["index", "name", "correlation", "stream"],
            index=[1, 2, 3, 4, 5, 6, 7, 8],
        )
        sym_index = {
            "Event Sync": 45,
            "Context Sync": 42,
        }
        mock_symbol_table = TraceSymbolTable()
        mock_symbol_table.sym_index = sym_index
        expected_index_correlation = [5, 6, 0, -1, 1, 2, 0, -1]
        trace = singletrace.create(None, None, df, mock_symbol_table)
        df2 = transform_correlation_to_index(trace)
        self.assertListEqual(
            expected_index_correlation, df2["index_correlation"].tolist()
        )
