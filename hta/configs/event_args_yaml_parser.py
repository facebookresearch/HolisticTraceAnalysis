# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import importlib.resources
from enum import Enum
from functools import lru_cache
from typing import Callable, Dict, List

import yaml

from hta.configs.default_values import AttributeSpec, EventArgs, ValueType


class YamlVersion(Enum):
    v1_0_0 = "1.0.0"
    v2_0_0 = "2.0.0"


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
    local_yaml_data_filepath = str(
        importlib.resources.files(__package__).joinpath(
            f"event_args_{version.value}.yaml",
        )
    )
    with open(local_yaml_data_filepath, "r") as f:
        yaml_content = yaml.safe_load(f)

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
    event_args = parse_event_args_yaml(YamlVersion.v1_0_0)
    print(event_args)


if __name__ == "__main__":
    main()
