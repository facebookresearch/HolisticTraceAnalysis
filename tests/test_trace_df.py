import unittest
from dataclasses import dataclass
from typing import List

import pandas as pd
from hta.common.trace_df import find_op_occurrence, get_iterations


class TestTraceDF(unittest.TestCase):
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

    def test_find_op_occurrence(self) -> None:
        @dataclass
        class TC:
            df: pd.DataFrame
            op_name: str
            position: int
            expected_found: bool
            expected_index: int

        test_df = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4],
                "s_name": ["op1", "op2", "op1", "op3", "op1"],
                "ts": [100, 200, 300, 400, 500],
            }
        )

        empty_df = pd.DataFrame()

        test_cases: List[TC] = [
            TC(test_df, "op1", 1, True, 2),
            TC(test_df, "op1", -1, True, 4),
            TC(test_df, "op4", 0, False, -1),
            TC(empty_df, "op2", 0, False, -1),
        ]

        for tc in test_cases:
            found, event = find_op_occurrence(tc.df, tc.op_name, tc.position)
            self.assertEqual(found, tc.expected_found)
            if tc.expected_found:
                self.assertEqual(event["index"], tc.expected_index)
            else:
                self.assertTrue(event.empty)
