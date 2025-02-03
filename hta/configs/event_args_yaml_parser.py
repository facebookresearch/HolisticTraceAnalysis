# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List

import yaml

from hta.configs.default_values import AttributeSpec, EventArgs, ValueType, YamlVersion


# Yaml version will be mapped to the yaml files defined under the "event_args_formats" folder
v1_0_0: YamlVersion = YamlVersion(1, 0, 0)


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

ARGS_MEMORY_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: [
        available_args[k]
        for k in [
            "memory::total_reserved",
            "memory::total_allocated",
            "memory::bytes",
            "memory::addr",
            "memory::device_id",
            "memory::device_type",
            "memory::ev_idx",
        ]
    ]
)

ARGS_TRITON_KERNELS_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: [
        available_args[k] for k in ["cpu_op::kernel_backend", "cpu_op::kernel_hash"]
    ]
)

ARGS_DEFAULT_FUNC: Callable[[Dict[str, AttributeSpec]], List[AttributeSpec]] = (
    lambda available_args: (
        ARGS_MINIMUM_FUNC(available_args)
        + ARGS_BANDWIDTH_FUNC(available_args)
        + ARGS_SYNC_FUNC(available_args)
        + ARGS_INPUT_SHAPE_FUNC(available_args)
        + ARGS_MEMORY_FUNC(available_args)
        + [available_args["index::external_id"]]
    )
)


@lru_cache()
def parse_event_args_yaml(version: YamlVersion) -> EventArgs:
    pkg_path: Path = Path(__file__).parent
    yaml_file = f"event_args_{version.get_version_str()}.yaml"
    local_yaml_data_filepath = str(pkg_path.joinpath("event_args_formats", yaml_file))

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
            min_supported_version=(
                YamlVersion.from_string(value["min_supported_version"])
                if "min_supported_version" in value
                else version
            ),
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
        ARGS_TRITON_KERNELS=ARGS_TRITON_KERNELS_FUNC(available_args),
        ARGS_DEFAULT=ARGS_DEFAULT_FUNC(available_args),
    )


def main() -> None:
    version = v1_0_0
    event_args = parse_event_args_yaml(version)
    print(event_args)
    print(f"Printed event args for version {version.get_version_str()}")


if __name__ == "__main__":
    main()
