# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from enum import Enum
from typing import Dict, List, NamedTuple, Union

# Default Paths
DEFAULT_TRACE_DIR = "/tmp/trace"
DEFAULT_CONFIG_FILENAME: str = "trace_analyzer.json"

# Trace related
DF_SYMBOL_COLUMNS: List[str] = ["cat", "name"]

# Runtime configurations
IS_DEBUG_ENABLED: bool = True
# For non-compute-intensive tasks, do not use a large number of processes.
MAX_NUM_PROCESSES_SMALL: int = 4
MAX_NUM_PROCESSES: int = 32


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


class ValueType(Enum):
    """ValueType enumerates the possible data types for the attribute values."""

    Int = 1
    Float = 2
    String = 3
    Object = 4


class AttributeSpec(NamedTuple):
    """AttributeSpec specifies what an attribute looks like and how to parse it.

    An AttributeSpec instance has the following fields:
    + name: the column name used in the output dataframe.
    + raw_name: the key used in args dict object in the original json trace.
    + value_type: the expected data type for the values of the attribute.
    + default_value: what value will be used for missing attribute values.
    """

    name: str
    raw_name: str
    value_type: ValueType
    default_value: Union[int, float, str, object]
    # This property allows for addition of fields that do
    # not break backwards compatibility with other fields (minor version bumps).
    min_supported_version: YamlVersion

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AttributeSpec):
            return False
        return (
            self.name == other.name
            and self.raw_name == other.raw_name
            and self.value_type == other.value_type
            and self.default_value == other.default_value
        )


class EventArgs(NamedTuple):
    AVAILABLE_ARGS: Dict[str, AttributeSpec]
    ARGS_INPUT_SHAPE: List[AttributeSpec]
    ARGS_BANDWIDTH: List[AttributeSpec]
    ARGS_SYNC: List[AttributeSpec]
    ARGS_MINIMUM: List[AttributeSpec]
    ARGS_COMPLETE: List[AttributeSpec]
    ARGS_INFO: List[AttributeSpec]
    ARGS_COMMUNICATION: List[AttributeSpec]
    ARGS_TRITON_KERNELS: List[AttributeSpec]
    ARGS_DEFAULT: List[AttributeSpec]
