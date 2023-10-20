# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd
import psutil


class KernelType(Enum):
    COMMUNICATION = 0
    MEMORY = 1
    COMPUTATION = 2


class IdleTimeType(Enum):
    HOST_WAIT = 0
    KERNEL_WAIT = 1
    OTHER = 2


def normalize_path(path: str) -> str:
    """
    Convert a Linux path to Python path.

    Args:
        path (str) : a path acceptable by the OS.

    Returns:
        A path supported by Python.
    """
    if path.startswith("./"):
        path2 = path[2:]
        if len(path2) > 0:
            normalized_path = str(Path.cwd().joinpath(path2))
        else:
            normalized_path = str(Path.cwd())
    elif path.startswith("~/"):
        path2 = path[2:]
        if len(path2) > 0:
            normalized_path = str(Path.home().joinpath(path2))
        else:
            normalized_path = str(Path.home())
    else:
        normalized_path = path
    return normalized_path


def is_comm_kernel(name: str) -> bool:
    """
    Check if a given GPU kernel is a communication kernel.

    Args:
        name (str): name of the GPU kernel.

    Returns:
        A boolean indicating if the kernel is a communication kernel.
    """
    return "ncclKernel" in name


def is_memory_kernel(name: str) -> bool:
    """
    Check if a given GPU kernel is a memory kernel.

    Args:
        name (str): name of the GPU kernel.

    Returns:
        A boolean indicating if the kernel is an IO kernel.
    """
    return "Memcpy" in name or "Memset" in name


def get_kernel_type(name: str) -> str:
    if is_comm_kernel(name):
        return KernelType.COMMUNICATION.name
    elif is_memory_kernel(name):
        return KernelType.MEMORY.name
    else:
        return KernelType.COMPUTATION.name


def get_memory_kernel_type(name: str) -> str:
    """Memcpy Type is basically a prefix of the kernel name ~ Memcpy DtoH"""
    if name[:6] == "Memset":
        return "Memset"
    if name[:6] != "Memcpy":
        return "Memcpy Unknown"
    prefix_size = 11  # len("Memcpy DtoH")
    return name[:prefix_size]


def merge_kernel_intervals(kernel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all kernel intervals in the given dataframe such that there are no overlapping.
    """
    kernel_df.sort_values(by="ts", inplace=True)
    kernel_df["end"] = kernel_df["ts"] + kernel_df["dur"]
    # Operators within the same group need to be merged together to form a larger interval.
    kernel_df["group"] = (kernel_df["ts"] > kernel_df["end"].shift().cummax()).cumsum()
    kernel_df = (
        kernel_df.groupby("group", as_index=False)
        .agg({"ts": "min", "end": "max"})
        .drop(["group"], axis=1)
        .sort_values(by="ts")
    )
    return kernel_df


def shorten_name(name: str) -> str:
    """Shorten a long operator/kernel name.

    The CPU operator and CUDA kernel name in the trace can be too long to follow.
    This utility removes the functional arguments, template arguments, and return values
    to make the name easy to understand.
    """
    s: str = name.replace("->", "")
    stack: List[str] = []
    for c in s:
        if c == ">":  # match generic template arguments
            while len(stack) and stack[-1] != "<":
                stack.pop()

            if len(stack) > 0 and stack[-1] == "<":
                stack.pop()
        elif c == ")":  # match arguments or comments
            while len(stack) and stack[-1] != "(":
                stack.pop()
            if len(stack) > 0 and stack[-1] == "(":
                stack.pop()
        else:
            stack.append(c)
    return "".join(stack).split(" ")[-1]


def flatten_column_names(df: pd.DataFrame) -> None:
    """Flatten a DataFrame's a multiple index column names to a single string"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).rstrip("_") for col in df.columns]


def get_mp_pool_size(obj_size: int, num_objs: int) -> int:
    """
    Estimate the maximum pool size for multiprocessing

    Args:
        obj_size (int): the size of objects to be processed
        num_objs (int): the total number of objects to be processed

    Returns:
        int
            the recommend pool size
    """
    free_mem = psutil.virtual_memory().available
    # Leave 20% buffer for system and other processes
    max_np = int(0.8 * free_mem / obj_size)
    return min(max_np, num_objs, mp.cpu_count())
