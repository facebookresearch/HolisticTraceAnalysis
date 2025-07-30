# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


from unittest import TestCase

from hta.configs.default_values import AttributeSpec, ValueType, YamlVersion
from hta.configs.event_args_yaml_parser import ARGS_INDEX_FUNC


class TestEventArgsYamlParser(TestCase):

    def test_ARGS_INDEX_FUNC_with_unavailable_args(self) -> None:
        # Given
        attribute_spec = AttributeSpec(
            name="key1",
            raw_name="Key 1",
            value_type=ValueType.Int,
            default_value=-1,
            min_supported_version=YamlVersion(1, 0, 0),
        )
        available_args = {
            "key1": attribute_spec,
            "key2": attribute_spec,
            "index::external_id": attribute_spec,  # only this one is available
        }

        # When
        result = ARGS_INDEX_FUNC(available_args)

        # Then
        self.assertEqual(result, [attribute_spec])
