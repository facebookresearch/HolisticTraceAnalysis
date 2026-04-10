# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import namedtuple
from typing import List

import pandas as pd
from hta.utils.utils import get_symbol_column_names, shorten_name


class TestTracUtils(unittest.TestCase):
    # AI-assisted test
    def test_get_symbol_column_names_with_name_column(self) -> None:
        """Test line 212: 'name' column with string dtype is detected."""
        df = pd.DataFrame({"name": ["op1", "op2"], "other": [1, 2]})
        name_col, cat_col = get_symbol_column_names(df)
        self.assertEqual(name_col, "name")
        self.assertEqual(cat_col, "")

    # AI-assisted test
    def test_get_symbol_column_names_with_s_name_column(self) -> None:
        """Test line 212: fallback to 's_name' when 'name' is not string dtype."""
        df = pd.DataFrame({"name": [1, 2], "s_name": ["op1", "op2"]})
        name_col, cat_col = get_symbol_column_names(df)
        self.assertEqual(name_col, "s_name")
        self.assertEqual(cat_col, "")

    # AI-assisted test
    def test_get_symbol_column_names_with_cat_column(self) -> None:
        """Test line 216: 'cat' column with string dtype is detected."""
        df = pd.DataFrame({"cat": ["cpu_op", "kernel"], "name": ["a", "b"]})
        name_col, cat_col = get_symbol_column_names(df)
        self.assertEqual(name_col, "name")
        self.assertEqual(cat_col, "cat")

    # AI-assisted test
    def test_get_symbol_column_names_with_s_cat_column(self) -> None:
        """Test line 216: fallback to 's_cat' when 'cat' is not string dtype."""
        df = pd.DataFrame(
            {"cat": [1, 2], "s_cat": ["cpu_op", "kernel"], "name": ["a", "b"]}
        )
        name_col, cat_col = get_symbol_column_names(df)
        self.assertEqual(name_col, "name")
        self.assertEqual(cat_col, "s_cat")

    # AI-assisted test
    def test_get_symbol_column_names_no_matching_columns(self) -> None:
        """Test that empty strings are returned when no matching columns exist."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        name_col, cat_col = get_symbol_column_names(df)
        self.assertEqual(name_col, "")
        self.assertEqual(cat_col, "")

    def test_shorten_name(self) -> None:
        TC = namedtuple("TC", ["input", "expect_result"])
        test_cases: List[TC] = [
            TC("a", "a"),
            TC("aten::empty", "aten::empty"),
            TC("a(1)", "a"),
            TC("a<int>(b)", "a"),
            TC(
                "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<int>, at::detail::Array<char*, 3>)",
                "at::native::vectorized_elementwise_kernel",
            ),
            TC("Memcpy DtoD (Device -> Device)", "Memcpy DtoD (Device -> Device)"),
        ]

        for tc in test_cases:
            self.assertEqual(shorten_name(tc.input), tc.expect_result)
