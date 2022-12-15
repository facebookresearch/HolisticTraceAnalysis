# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd

from hta.common.trace import transform_correlation_to_index


class CorrelationTestCase(unittest.TestCase):
    def test_something(self):
        data = [
            [1, 15, -1],
            [2, 16, -1],
            [3, 22, -1],
            [4, -1, -1],
            [5, 15, 7],
            [6, 16, 7],
            [7, 30, 7],
            [8, -1, 7],
        ]
        df = pd.DataFrame(
            data,
            columns=["index", "correlation", "stream"],
            index=[1, 2, 3, 4, 5, 6, 7, 8],
        )
        expected_index_correlation = [5, 6, 0, -1, 1, 2, 0, -1]
        df2 = transform_correlation_to_index(df)
        self.assertListEqual(expected_index_correlation, df2["index_correlation"].tolist())
