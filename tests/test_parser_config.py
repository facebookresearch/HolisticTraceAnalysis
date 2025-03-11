import unittest
from typing import Dict, List, NamedTuple, Optional

import pandas as pd

from hta.configs.default_values import ValueType, YamlVersion
from hta.configs.event_args_yaml_parser import parse_event_args_yaml
from hta.configs.parser_config import (
    AttributeSpec,
    AVAILABLE_ARGS,
    DEFAULT_PARSE_VERSION,
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
            {
                "arg_name": "Kernel.a(.Queued)__b_%",
                "expected_transformed": "kernel_a_b",
            },
            {
                "arg_name": "name",
                "expected_transformed": "arg_name",
            },
        ]
    )
    def test_transform_arg_name(self, arg_name: str, expected_transformed: str) -> None:
        self.assertEqual(
            ParserConfig.transform_arg_name(arg_name), expected_transformed
        )

    @data_provider(
        lambda: [
            {
                "cfg": ParserConfig(args=ParserConfig.get_minimum_args()),
                "args_selector": None,
                "expected_args": ParserConfig.get_minimum_args(),
            },
            {
                "cfg": ParserConfig(args=ParserConfig.get_minimum_args()),
                "args_selector": ["stream"],
                "expected_args": [
                    AttributeSpec(
                        "stream", "stream", ValueType.Int, -1, DEFAULT_PARSE_VERSION
                    )
                ],
            },
        ]
    )
    def test_args_selector(
        self,
        cfg: ParserConfig,
        args_selector: Optional[List[str]],
        expected_args: List[str],
    ) -> None:
        cfg.set_args_selector(args_selector)
        self.assertListEqual(cfg.get_args(), expected_args)

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

    @data_provider(
        lambda: [
            {
                "arg_name": "arg1",
                "arg_value": 1,
                "arg_spec": AttributeSpec(
                    "arg1", "arg1", ValueType.Int, 0, DEFAULT_PARSE_VERSION
                ),
            },
            {
                "arg_name": "arg2",
                "arg_value": 1.0,
                "arg_spec": AttributeSpec(
                    "arg2", "arg2", ValueType.Float, 0.0, DEFAULT_PARSE_VERSION
                ),
            },
            {
                "arg_name": "arg3",
                "arg_value": "1",
                "arg_spec": AttributeSpec(
                    "arg3", "arg3", ValueType.String, "", DEFAULT_PARSE_VERSION
                ),
            },
            {
                "arg_name": "arg4",
                "arg_value": {},
                "arg_spec": AttributeSpec(
                    "arg4", "arg4", ValueType.Object, None, DEFAULT_PARSE_VERSION
                ),
            },
        ]
    )
    def test_make_attribute_spec(
        self, arg_name: str, arg_value: object, arg_spec: AttributeSpec
    ) -> None:
        result_spec = ParserConfig.make_attribute_spec(arg_name, arg_value)
        self.assertEqual(result_spec, arg_spec)

    def test_infer_attribute_specs(self) -> None:
        cfg = ParserConfig()
        args = pd.Series(
            [
                {"arg1": 1, "arg2": 1.0},
                {"arg1": 2, "arg2": 2.0, "arg3": "abc"},
                {"arg4": {"key": "value"}},
                {"arg5": 3, "arg6": 3.0},
                None,
            ]
        )
        reference_specs: Dict[str, AttributeSpec] = {
            "ref::arg5": AttributeSpec(
                "arg5_ref", "arg5", ValueType.Int, 0, DEFAULT_PARSE_VERSION
            ),
            "ref::arg6": AttributeSpec(
                "arg6_ref", "arg6", ValueType.Float, 10.0, DEFAULT_PARSE_VERSION
            ),
        }
        expected_result = {
            "arg1": AttributeSpec(
                "arg1", "arg1", ValueType.Int, 0, DEFAULT_PARSE_VERSION
            ),
            "arg2": AttributeSpec(
                "arg2", "arg2", ValueType.Float, 0.0, DEFAULT_PARSE_VERSION
            ),
            "arg3": AttributeSpec(
                "arg3", "arg3", ValueType.String, "", DEFAULT_PARSE_VERSION
            ),
            "arg4": AttributeSpec(
                "arg4", "arg4", ValueType.Object, None, DEFAULT_PARSE_VERSION
            ),
            "arg5": reference_specs["ref::arg5"],
            "arg6": reference_specs["ref::arg6"],
        }
        result = cfg.infer_attribute_specs(args, reference_specs=reference_specs)
        self.assertDictEqual(result, expected_result)

    def test_enable_communication_args(self) -> None:
        cfg = ParserConfig.get_versioned_cfg(DEFAULT_PARSE_VERSION)
        cfg_1 = ParserConfig.enable_communication_args(cfg)
        for arg in parse_event_args_yaml(DEFAULT_PARSE_VERSION).ARGS_COMMUNICATION:
            self.assertIn(arg, cfg_1.get_args())

        # Run the following two steps purely for test coverage
        ParserConfig.show_available_args()
        ParserConfig.get_info_args()
