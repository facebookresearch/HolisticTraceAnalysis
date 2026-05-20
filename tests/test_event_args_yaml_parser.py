# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import io
import textwrap
from contextlib import redirect_stdout
from unittest import TestCase
from unittest.mock import mock_open, patch

from hta.configs.default_values import AttributeSpec, ValueType, YamlVersion
from hta.configs.event_args_yaml_parser import (
    ARGS_INDEX_FUNC,
    main,
    parse_event_args_yaml,
    v1_0_0,
)

_MODULE = "hta.configs.event_args_yaml_parser"


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

    def test_parse_event_args_yaml_loads_local_file(self) -> None:
        """Cover the os.path.exists True branch (line 99) by mocking the
        filesystem to claim the yaml file exists and returning a minimal
        yaml content."""
        minimal_yaml = textwrap.dedent(
            """
        AVAILABLE_ARGS:
          cuda::stream:
            name: stream
            raw_name: stream
            value_type: Int
            default_value: -1
          correlation::cpu_gpu:
            name: correlation
            raw_name: correlation
            value_type: Int
            default_value: -1
          data::bytes:
            name: bytes
            raw_name: bytes
            value_type: Int
            default_value: -1
          data::bandwidth:
            name: bandwidth
            raw_name: bandwidth
            value_type: Float
            default_value: -1.0
          cuda_sync::stream:
            name: sync_stream
            raw_name: sync_stream
            value_type: Int
            default_value: -1
          cuda_sync::event:
            name: sync_event
            raw_name: sync_event
            value_type: Int
            default_value: -1
          cpu_op::input_dims:
            name: input_dims
            raw_name: input_dims
            value_type: String
            default_value: ""
          cpu_op::input_type:
            name: input_type
            raw_name: input_type
            value_type: String
            default_value: ""
          cpu_op::input_strides:
            name: input_strides
            raw_name: input_strides
            value_type: String
            default_value: ""
          cpu_op::kernel_backend:
            name: kernel_backend
            raw_name: kernel_backend
            value_type: String
            default_value: ""
          cpu_op::kernel_hash:
            name: kernel_hash
            raw_name: kernel_hash
            value_type: String
            default_value: ""
          index::external_id:
            name: external_id
            raw_name: external_id
            value_type: Int
            default_value: -1
          index::python_id:
            name: python_id
            raw_name: python_id
            value_type: Int
            default_value: -1
          index::python_parent_id:
            name: python_parent_id
            raw_name: python_parent_id
            value_type: Int
            default_value: -1
          info::labels:
            name: labels
            raw_name: labels
            value_type: String
            default_value: ""
          info::name:
            name: info_name
            raw_name: info_name
            value_type: String
            default_value: ""
          info::sort_index:
            name: sort_index
            raw_name: sort_index
            value_type: Int
            default_value: -1
          nccl::collective_name:
            name: collective_name
            raw_name: collective_name
            value_type: String
            default_value: ""
          nccl::in_msg_nelems:
            name: in_msg_nelems
            raw_name: in_msg_nelems
            value_type: Int
            default_value: -1
          nccl::out_msg_nelems:
            name: out_msg_nelems
            raw_name: out_msg_nelems
            value_type: Int
            default_value: -1
          nccl::dtype:
            name: dtype
            raw_name: dtype
            value_type: String
            default_value: ""
          nccl::group_size:
            name: group_size
            raw_name: group_size
            value_type: Int
            default_value: -1
          nccl::rank:
            name: rank
            raw_name: rank
            value_type: Int
            default_value: -1
          nccl::in_split_size:
            name: in_split_size
            raw_name: in_split_size
            value_type: String
            default_value: ""
          nccl::out_split_size:
            name: out_split_size
            raw_name: out_split_size
            value_type: String
            default_value: ""
        """
        )
        # Clear lru_cache so this version is not cached from a prior test
        parse_event_args_yaml.cache_clear()
        with (
            patch(f"{_MODULE}.os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=minimal_yaml)),
        ):
            result = parse_event_args_yaml(YamlVersion(9, 9, 9))
        self.assertIn("cuda::stream", result.AVAILABLE_ARGS)
        # Reset cache for other tests
        parse_event_args_yaml.cache_clear()

    def test_main_runs_without_error(self) -> None:
        """Cover the main() function (lines 134-137)."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        out = buf.getvalue()
        self.assertIn("Printed event args for version", out)
        self.assertIn(v1_0_0.get_version_str(), out)
