# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import namedtuple
from typing import List

from hta.utils.utils import shorten_name


class TestTracUtils(unittest.TestCase):
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
