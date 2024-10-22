# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd


def get_memory_usage_in_MB(df: pd.DataFrame) -> float:
    """Get the memory usage of a trace dataframe in megabytes (MB)."""
    memory_per_column = df.memory_usage(deep=True)
    total_memory = memory_per_column.sum()
    total_memory_mb = total_memory / (1024 * 1024)
    return total_memory_mb


def analyze_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the memory usage of a trace dataframe.

    Args:
        df (pd.DataFrame): The input dataframe to analyze.

    Returns:
        pd.DataFrame: A new dataframe containing the memory usage analysis.
    """
    _df = pd.DataFrame(
        {
            "Memory (MB)": df.memory_usage(deep=True),
            "Count": df.count(),
            "DType": df.dtypes,
        }
    )
    _df.dropna(inplace=True)
    _df["Memory (MB)"] = _df["Memory (MB)"] / (1024 * 1024)
    _df["Count"] = _df["Count"].astype(int)
    _df["Memory Per Item (B)"] = (
        _df["Memory (MB)"] * 1024 * 1024 / _df["Count"]
    ).astype(int)

    return _df.round(2)
