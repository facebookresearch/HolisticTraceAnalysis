import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import pandas as pd


class DeviceType(Enum):
    UNKNOWN = 0
    CPU = 1
    GPU = 2
    ALL = 3


def infer_device_type(df: pd.DataFrame) -> DeviceType:
    """Infer the device type based on trace data.

    Args:
        df (pd.DataFrame): A DataFrame slice consisting of trace events on a single thread or stream.

    Returns:
        DeviceType: the type of the device on which the trace events are collect.
    """
    if "stream" in df.columns and ("pid" not in df.columns or "tid" not in df.columns):
        if (df.stream.unique() > 0).all():
            return DeviceType.GPU
        elif (df.stream.unique() == -1).all():
            return DeviceType.CPU
    elif {"stream", "pid", "tid"}.issubset(set(df.columns)):
        if (df.stream.unique() > 0).all() or (
            (df.pid.unique() == 0).all() or (df.tid.unique() == 0).all()
        ):
            return DeviceType.GPU
        elif (df.stream.unique() == -1).all():
            return DeviceType.CPU
    return DeviceType.UNKNOWN


class MemcpyType(Enum):
    DTOD = 0
    DTOH = 1
    HTOD = 2
    UNKNOWN = 3


MEMCPY_TYPE_TO_STR: Dict[MemcpyType, str] = {
    MemcpyType.DTOD: "memcpy_dtod",
    MemcpyType.DTOH: "memcpy_dtoh",
    MemcpyType.HTOD: "memcpy_htod",
    MemcpyType.UNKNOWN: "memcpy_type_unknown",
}

# TODO Move these to a common constants file that can provide patterns and platform constants


@dataclass
class GroupingPattern:
    pattern: re.Pattern[str]
    inverse_match: bool = False
    group_name: str = ""

    def match(self, name: str) -> bool:
        m = self.pattern.match(name) is not None
        return m if not self.inverse_match else not m

    def __hash__(self) -> int:
        return hash((self.pattern.pattern, self.pattern.flags, self.inverse_match))


def to_grouping_pattern(
    pattern: Union[str, List[str], re.Pattern[str], GroupingPattern],
    group_name: Optional[str] = None,
    inverse_match: Optional[bool] = None,
) -> GroupingPattern:
    """Convert a pattern to a GroupingPattern object.

    Args:
        pattern: a pattern to be converted to a GroupingPattern object.
            When pattern is a GroupingPattern object, it is returned as is and
            the group_name and inverse_match are ignored.
        group_name: a name of the pattern.
        inverse_match: whether to invert the match result.
    """
    if isinstance(pattern, GroupingPattern):
        return pattern
    if isinstance(pattern, list):
        pattern = "|".join("(" + pattern + ")" for pattern in pattern)

    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    if isinstance(pattern, re.Pattern):
        pattern = GroupingPattern(
            pattern,
            inverse_match if inverse_match else False,
            group_name if group_name else "",
        )
        return pattern
    raise ValueError(f"Unsupported pattern type: {type(pattern)}")


ProfilerStepGroupingPattern = GroupingPattern(
    pattern=re.compile(r"^ProfilerStep#(\d+)"),
    inverse_match=False,
    group_name="Profiler Step",
)

MemoryKernelGroupingPattern = GroupingPattern(
    pattern=re.compile(r"(^(hip)?Memcpy)|(^(hip)?Memset)|(^dma)"),
    inverse_match=False,
    group_name="Memory",
)

MTIAMemoryKernelGroupingPattern = GroupingPattern(
    pattern=re.compile(r"(^dma)"),
    inverse_match=False,
    group_name="MTIA Memory",
)

CommunicateKernelGroupingPattern = GroupingPattern(
    pattern=re.compile(r"(^nccl)"),
    inverse_match=False,
    group_name="Communicate",
)

ComputeKernelGroupingPattern = GroupingPattern(
    pattern=re.compile(
        r"(^nccl)|(.*Memcpy)|(.*Memset)|(.*dma)|(^.*Sync)|(cuda.*LaunchKernel)|(^runFunction)"
    ),
    inverse_match=True,
    group_name="Compute",
)

KERNEL_CATEGORY_PATTERN = GroupingPattern(
    re.compile("(kernel)|(gpu_mem.+)|(mtia_ccp_events)"), False, "GPU Kernels"
)

CPU_OP_CATEGORY_PATTERN: GroupingPattern = GroupingPattern(
    re.compile("(cpu_op)|(user_annotation)"), False, "CPU Ops"
)

DEVICE_RUNTIME_CATEGORY_PATTERN = GroupingPattern(
    re.compile("(cuda_runtime)|(mtia_runtime)"), False, "Device Runtimes"
)

MEMORY_CATEGORY_PATTERN = GroupingPattern(
    re.compile("(gpu_mem.+)|(mtia_ccp_events)"), False, "Memory"
)

GPU_EVENT_CATEGORY_PATTERN: GroupingPattern = GroupingPattern(
    group_name="GPU Events",
    pattern=re.compile(r"(.*kernel)|(^gpu_mem)|(^mtia_ccp_events)|(^cuda_sync)"),
    inverse_match=False,
)

CPU_EVENTS_CATEGORY_PATTERN: GroupingPattern = GroupingPattern(
    group_name="CPU Events",
    pattern=re.compile(r"(cpu_op)|(user_annotation)|(^(.+)_runtime)"),
    inverse_match=False,
)

CPU_AND_GPU_EVENTS_CATEGORY_PATTERN: GroupingPattern = GroupingPattern(
    group_name="CPU/GPU Events",
    pattern=re.compile(
        r"(cpu_op)|(user_annotation)|(cuda_runtime)"
        + r"|(.*kernel)|(^gpu_mem)|(^mtia_ccp_events)|(^cuda_sync)"
    ),
    inverse_match=False,
)

PYTHON_FUNCTION_PATTERN: GroupingPattern = GroupingPattern(
    group_name="Python Functions",
    pattern=re.compile(r"(python_function)"),
    inverse_match=False,
)

ALL_CATEGORY_PATTERN: GroupingPattern = GroupingPattern(
    group_name="All",
    pattern=re.compile(r".*"),
    inverse_match=False,
)

CUDA_SYNC_PATTERN: GroupingPattern = GroupingPattern(
    group_name="CUDA Sync",
    pattern=re.compile(r"(cuda_sync)"),
    inverse_match=False,
)

common_grouping_patterns: Dict[str, GroupingPattern] = {
    "memory_kernels": MemoryKernelGroupingPattern,
    "compute_kernels": ComputeKernelGroupingPattern,
    "communicate_kernels": CommunicateKernelGroupingPattern,
}
