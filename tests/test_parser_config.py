import unittest
from typing import List, NamedTuple, Optional

from hta.configs.parser_config import AttributeSpec, AVAILABLE_ARGS, ParserConfig


class ParserConfigTestCase(unittest.TestCase):
    class _TCase(NamedTuple):
        input: Optional[List[AttributeSpec]]
        expect_result: List[AttributeSpec]

    @classmethod
    def _compare_attributes(
        cls, args1: List[AttributeSpec], args2: List[AttributeSpec]
    ) -> bool:
        set_args1 = {a.name for a in args1}
        set_args2 = {a.name for a in args2}
        return set_args1 == set_args2

    def test_constructor(self) -> None:
        t_cases = [
            self._TCase(None, ParserConfig.get_default_args()),
            self._TCase(
                ParserConfig.get_minimum_args(), ParserConfig.get_minimum_args()
            ),
        ]

        for i, tc in enumerate(t_cases):
            cfg = ParserConfig(tc.input)
            got = cfg.get_args()
            self.assertTrue(
                self._compare_attributes(tc.expect_result, got),
                f"case #{i}: input={tc.input} expect={tc.expect_result} got={got}",
            )

    def test_add_args(self) -> None:
        t_cases = [
            self._TCase(ParserConfig.ARGS_MINIMUM, ParserConfig.get_minimum_args()),
            self._TCase(
                ParserConfig.ARGS_INPUT_SHAPE,
                ParserConfig.ARGS_MINIMUM + ParserConfig.ARGS_INPUT_SHAPE,
            ),
        ]
        for i, tc in enumerate(t_cases):
            cfg = ParserConfig(ParserConfig.get_minimum_args())
            cfg.add_args(tc.input)
            got = cfg.get_args()
            self.assertTrue(
                self._compare_attributes(tc.expect_result, got),
                f"case #{i}: input={tc.input} expect={tc.expect_result} got={got}",
            )

    def test_global_cfg(self) -> None:
        custom_cfg = ParserConfig()
        custom_cfg.add_args([AVAILABLE_ARGS["kernel::queued"]])
        ParserConfig.set_default_cfg(custom_cfg)
        self.assertTrue(
            self._compare_attributes(
                ParserConfig.get_default_cfg().get_args(), custom_cfg.get_args()
            )
        )
