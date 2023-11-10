from enum import Enum

import numpy as np
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
    device_type: DeviceType = DeviceType.UNKNOWN
    if "stream" in df.columns:
        streams = df["stream"].unique()
        if len(streams) > 0:
            if np.all(np.greater(streams, 0)):
                device_type = DeviceType.GPU
            elif np.all(np.less(streams, 0)):
                device_type = DeviceType.CPU
    return device_type
