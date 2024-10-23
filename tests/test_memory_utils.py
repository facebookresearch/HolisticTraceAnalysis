import unittest

import pandas as pd

from hta.utils.memory_utils import analyze_memory_usage


class TestMemoryUtility(unittest.TestCase):
    def test_analyze_memory_usage(self):
        n = 1024 * 1024
        data = {
            "A": [1] * n,
            "B": [2.3] * n,
            "C": ["7"] * n,
        }
        df = pd.DataFrame(data)

        result = analyze_memory_usage(df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(
            set(result.columns),
            {"Memory (MB)", "Count", "DType", "Memory Per Item (B)"},
        )
        self.assertListEqual(result["Memory (MB)"].tolist(), [8.0, 8.0, 58.0])
        self.assertListEqual(result["Count"].tolist(), [n, n, n])
        self.assertListEqual(result["DType"].tolist(), ["int64", "float64", "object"])
        self.assertListEqual(result["Memory Per Item (B)"].tolist(), [8, 8, 58])
