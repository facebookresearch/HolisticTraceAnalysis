# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple

import yaml

from hta.configs.default_values import AttributeSpec, EventArgs, ValueType


class YamlVersion(NamedTuple):
    major: int
    minor: int
    patch: int

    def get_version_str(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @staticmethod
    def from_string(version_str: str) -> "YamlVersion":
        pattern = r"^(\d+)\.(\d+)\.(\d+)$"
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid version string: {version_str}")
        major, minor, patch = map(int, match.groups())
        return YamlVersion(major, minor, patch)


# Yaml version will be mapped to the yaml files defined under the "event_args_formats" folder
v1_0_0: YamlVersion = YamlVersion(1, 0, 0)

DEFAULT_YAML_ARGS_FORMAT: str = """version: 1.0.0

AVAILABLE_ARGS:
  index::ev_idx:
    name: ev_idx
    raw_name: Ev Idx
    value_type: Int
    default_value: -1
  index::external_id:
    name: external_id
    raw_name: External id
    value_type: Int
    default_value: -1
  cpu_op::concrete_inputs:
    name: concrete_inputs
    raw_name: Concrete Inputs
    value_type: Object
    default_value: "[]"
  cpu_op::fwd_thread:
    name: fwd_thread_id
    raw_name: Fwd thread id
    value_type: Int
    default_value: -1
  cpu_op::input_dims:
    name: input_dims
    raw_name: Input Dims
    value_type: Object
    default_value: "-1"
  cpu_op::input_type:
    name: input_type
    raw_name: Input type
    value_type: Object
    default_value: "-1"
  cpu_op::input_strides:
    name: input_strides
    raw_name: Input Strides
    value_type: Object
    default_value: "-1"
  cpu_op::sequence_number:
    name: sequence
    raw_name: Sequence number
    value_type: Int
    default_value: -1
  cpu_op::kernel_backend:
    name: kernel_backend
    raw_name: kernel_backend
    value_type: String
    default_value: ""
  correlation::cbid:
    name: cbid
    raw_name: cbid
    value_type: Int
    default_value: -1
  correlation::cpu_gpu:
    name: correlation
    raw_name: correlation
    value_type: Int
    default_value: -1
  sm::blocks:
    name: blocks_per_sm
    raw_name: blocks per SM
    value_type: Object
    default_value: "[]"
  sm::occupancy:
    name: est_occupancy
    raw_name: est. achieved occupancy %
    value_type: Int
    default_value: -1
  sm::warps:
    name: warps_per_sm
    raw_name: warps per SM
    value_type: Float
    default_value: 0.0
  data::bytes:
    name: bytes
    raw_name: bytes
    value_type: Int
    default_value: -1
  data::bandwidth:
    name: memory_bw_gbps
    raw_name: memory bandwidth (GB/s)
    value_type: Float
    default_value: 0.0
  cuda::context:
    name: context
    raw_name: context
    value_type: Int
    default_value: -1
  cuda::device:
    name: device
    raw_name: device
    value_type: Int
    default_value: -1
  cuda::stream:
    name: stream
    raw_name: stream
    value_type: Int
    default_value: -1
  kernel::queued:
    name: queued
    raw_name: queued
    value_type: Int
    default_value: -1
  kernel::shared_memory:
    name: shared_memory
    raw_name: shared memory
    value_type: Int
    default_value: -1
  threads::block:
    name: block
    raw_name: block
    value_type: Object
    default_value: "[]"
  threads::grid:
    name: grid
    raw_name: grid
    value_type: Object
    default_value: "[]"
  threads::registers:
    name: registers_per_thread
    raw_name: registers per thread
    value_type: Int
    default_value: -1
  cuda_sync::stream:
    name: wait_on_stream
    raw_name: wait_on_stream
    value_type: Int
    default_value: -1
  cuda_sync::event:
    name: wait_on_cuda_event_record_corr_id
    raw_name: wait_on_cuda_event_record_corr_id
    value_type: Int
    default_value: -1
  info::labels:
    name: labels
    raw_name: labels
    value_type: String
    default_value: ""
  info::name:
    name: name
    raw_name: name
    value_type: Int
    default_value: -1
  info::op_count:
    name: op_count
    raw_name: Op count
    value_type: Int
    default_value: -1
  info::sort_index:
    name: sort_index
    raw_name: sort_index
    value_type: Int
    default_value: -1
  nccl::collective_name:
    name: collective_name
    raw_name: Collective name
    value_type: String
    default_value: ""
  nccl::in_msg_nelems:
    name: in_msg_nelems
    raw_name: In msg nelems
    value_type: Int
    default_value: 0
  nccl::out_msg_nelems:
    name: out_msg_nelems
    raw_name: Out msg nelems
    value_type: Int
    default_value: 0
  nccl::group_size:
    name: group_size
    raw_name: Group size
    value_type: Int
    default_value: 0
  nccl::dtype:
    name: msg_dtype
    raw_name: dtype
    value_type: String
    default_value: ""
  nccl::in_split_size:
    name: in_split_size
    raw_name: In split size
    value_type: Object
    default_value: "[]"
  nccl::out_split_size:
    name: out_split_size
    raw_name: Out split size
    value_type: Object
    default_value: "[]"
  nccl::process_group_name:
    name: process_group_name
    raw_name: Process Group Name
    value_type: String
    default_value: ""
  nccl::process_group_desc:
    name: process_group_desc
    raw_name: Process Group Description
    value_type: String
    default_value: ""
  nccl::process_group_ranks:
    name: process_group_ranks
    raw_name: Process Group Ranks
    value_type: Object
    default_value: "[]"
  nccl::rank:
    name: process_rank
    raw_name: Rank
    value_type: Int
    default_value: -1
"""


ARGS_INPUT_SHAPE_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: [
        available_args[k]
        for k in ["cpu_op::input_dims", "cpu_op::input_type", "cpu_op::input_strides"]
    ]
)
ARGS_BANDWIDTH_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: [
        available_args[k] for k in ["data::bytes", "data::bandwidth"]
    ]
)
ARGS_SYNC_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: [
        available_args[k] for k in ["cuda_sync::stream", "cuda_sync::event"]
    ]
)
ARGS_MINIMUM_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: [
        available_args[k] for k in ["cuda::stream", "correlation::cpu_gpu"]
    ]
)
ARGS_COMPLETE_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: [
        available_args[k] for k in available_args if not k.startswith("info")
    ]
)
ARGS_INFO_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: [
        available_args[k] for k in ["info::labels", "info::name", "info::sort_index"]
    ]
)
ARGS_COMMUNICATION_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: [
        available_args[k]
        for k in [
            "nccl::collective_name",
            "nccl::in_msg_nelems",
            "nccl::out_msg_nelems",
            "nccl::dtype",
            "nccl::group_size",
            "nccl::rank",
            "nccl::in_split_size",
            "nccl::out_split_size",
        ]
    ]
)
ARGS_DEFAULT_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: (
        ARGS_MINIMUM_FUNC(available_args)
        + ARGS_BANDWIDTH_FUNC(available_args)
        + ARGS_SYNC_FUNC(available_args)
        + ARGS_INPUT_SHAPE_FUNC(available_args)
        + [available_args["index::external_id"]]
    )
)


@lru_cache()
def parse_event_args_yaml(version: YamlVersion) -> EventArgs:
    pkg_path: Path = Path(__file__).parent
    yaml_file = f"event_args_{version.get_version_str()}.yaml"
    local_yaml_data_filepath = str(pkg_path.joinpath("event_args_formats", yaml_file))

    if Path(local_yaml_data_filepath).exists():
        with open(local_yaml_data_filepath, "r") as f:
            yaml_content = yaml.safe_load(f)
    else:
        yaml_content = yaml.safe_load(DEFAULT_YAML_ARGS_FORMAT)

    def parse_value_type(value: str) -> ValueType:
        return ValueType[value]

    available_args: Dict[str, AttributeSpec] = {
        k: AttributeSpec(
            name=value["name"],
            raw_name=value["raw_name"],
            value_type=parse_value_type(value["value_type"]),
            default_value=value["default_value"],
        )
        for k, value in yaml_content["AVAILABLE_ARGS"].items()
    }

    return EventArgs(
        AVAILABLE_ARGS=available_args,
        ARGS_INPUT_SHAPE=ARGS_INPUT_SHAPE_FUNC(available_args),
        ARGS_BANDWIDTH=ARGS_BANDWIDTH_FUNC(available_args),
        ARGS_SYNC=ARGS_SYNC_FUNC(available_args),
        ARGS_MINIMUM=ARGS_MINIMUM_FUNC(available_args),
        ARGS_COMPLETE=ARGS_COMPLETE_FUNC(available_args),
        ARGS_INFO=ARGS_INFO_FUNC(available_args),
        ARGS_COMMUNICATION=ARGS_COMMUNICATION_FUNC(available_args),
        ARGS_DEFAULT=ARGS_DEFAULT_FUNC(available_args),
    )


def main() -> None:
    version = v1_0_0
    event_args = parse_event_args_yaml(version)
    print(event_args)
    print(f"Printed event args for version {version.get_version_str()}")


if __name__ == "__main__":
    main()
