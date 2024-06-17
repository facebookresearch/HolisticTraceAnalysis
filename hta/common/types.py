from dataclasses import dataclass
from enum import Enum
from typing import Dict

import pandas as pd


@dataclass
class OperationOutcome:
    success: bool
    reason: str


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
