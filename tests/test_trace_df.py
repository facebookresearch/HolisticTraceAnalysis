import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List

import hta

import pandas as pd
from hta.common.trace import Trace
from hta.common.trace_df import get_iterations


class TestTraceDF(unittest.TestCase):
    base_data_dir = str(Path(hta.__file__).parent.parent.joinpath("tests/data"))
    trace_dir: str = os.path.join(base_data_dir, "trace_filter")
    htaTrace: Trace = Trace(trace_dir=trace_dir)
    htaTrace.parse_traces()

    def test_get_iterations(self) -> None:
        @dataclass
        class TC:
            df: pd.DataFrame
            expected_result: List[int]
            expected_type_error: bool

        test_cases: List[TC] = [
            TC(pd.DataFrame({"iteration": [1, 2, 3, 3, 4]}), [1, 2, 3, 4], False),
            TC(pd.DataFrame({"iteration": [-1, 1, 2, 3]}), [1, 2, 3], False),
            TC(pd.DataFrame({"iteration": [-1, -1, -1]}), [], False),
            TC(pd.DataFrame({"no_iteration_column": [1, 2, 3]}), [], True),
            TC(pd.DataFrame({"iteration": ["1", "2", "3"]}), [], True),
        ]

        for tc in test_cases:
            if tc.expected_type_error:
                with self.assertRaises(TypeError):
                    get_iterations(tc.df)
            else:
                result = get_iterations(tc.df)
                self.assertEqual(result, tc.expected_result)
