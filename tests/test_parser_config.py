import unittest
from typing import List, NamedTuple, Optional

from hta.configs.default_values import YamlVersion
from hta.configs.parser_config import (
    AttributeSpec,
    AVAILABLE_ARGS,
    ParserBackend,
    ParserConfig,
)
from hta.utils.test_utils import data_provider

_MODULE_NAME = "hta.configs.parser_config"


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

    @data_provider(
        lambda: [
            {
                "arg_name": "Kernel Queued",
                "expected_transformed": "kernel_queued",
            },
            {
                "arg_name": "Cuda-Kernel/Queued",
                "expected_transformed": "cuda_kernel_queued",
            },
        ]
    )
    def test_transform_arg_name(self, arg_name: str, expected_transformed: str) -> None:
        self.assertEqual(
            ParserConfig.transform_arg_name(arg_name), expected_transformed
        )

    def test_set_parse_all_args(self) -> None:
        cfg = ParserConfig()
        # Test default value is False
        cfg.set_parse_all_args(False)

        # Test setting to True
        cfg.set_parse_all_args(True)
        self.assertTrue(cfg.parse_all_args)

        # Test setting to False
        cfg.set_parse_all_args(False)
        self.assertFalse(cfg.parse_all_args)

    def test_set_global_parser_config_version(self) -> None:
        cfg = ParserConfig()
        version = YamlVersion(1, 0, 0)
        cfg.set_global_parser_config_version(version)
        self.assertEqual(cfg.version.get_version_str(), "1.0.0")
        self.assertEqual(
            ParserConfig.get_default_cfg().version.get_version_str(), "1.0.0"
        )

    def test_set_min_required_cols(self) -> None:
        cfg = ParserConfig()
        cols = ["a", "b", "c"]
        cfg.set_min_required_cols(cols)
        self.assertSetEqual(set(cfg.get_min_required_cols()), set(cols))

    def test_set_trace_memory(self) -> None:
        cfg = ParserConfig()
        self.assertFalse(cfg.trace_memory)
        cfg.set_trace_memory(True)
        self.assertTrue(cfg.trace_memory)
        cfg.set_trace_memory(False)
        self.assertFalse(cfg.trace_memory)

    def test_set_parser_backend(self) -> None:
        cfg = ParserConfig()
        for backend in ParserBackend:
            cfg.set_parser_backend(backend)
            self.assertEqual(cfg.parser_backend, backend)
