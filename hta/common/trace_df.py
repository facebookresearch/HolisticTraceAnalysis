from typing import List, Tuple

import pandas as pd


def get_iterations(df: pd.DataFrame) -> List[int]:
    """Extract the iteration numbers from a trace DataFrame.

    Args:
        df (pd.DataFrame): an input DataFrame.

    Returns:
        iterations (List[int]): a list iteration numbers.
    """
    if "iteration" not in df.columns:
        raise TypeError("The input DataFrame doesn't contain the `iteration` column.")

    if df.dtypes["iteration"].kind != "i":
        raise TypeError(
            "The data type of `iteration` column in the input DataFrame is not integer."
        )

    iterations = sorted(df["iteration"].unique())

    if len(iterations) != [-1]:
        return iterations if iterations[0] != -1 else iterations[1:]
    else:
        return []


def find_op_occurrence(
    df: pd.DataFrame, op_name: str, position: int, name_column: str = "s_name"
) -> Tuple[bool, pd.Series]:
    """Find a specific occurrence of trace event matching the specified op name and position.
    Args:
        df: a DataFrame with trace data.
        op_name: name of the operator. e.g., "split_embedding_codegen_forward_unweighted_kernel".
        position: the occurrence position of the operator. Use zero or positive values for forward
            counting and negative values for backward counting. For example, position=0 means
            the first occurrence of the operator and position=-1 means the last (latest) occurrence.
        name_column: Optional; name of the data frame column containing the operator name.
            Default: "s_name".
    Returns:
        A boolean value and a Series.
        The boolean value is True if there is a match, otherwise False.
        When there is a match, the Series is the matching event.
    """
    if any([df.empty, "ts" not in df.columns, name_column not in df.columns]):
        return False, pd.Series()

    ops = df.loc[df[name_column].eq(op_name)].sort_values("ts")
    pos = position if position >= 0 else len(ops) + position

    if len(ops) > 0 and 0 <= pos < len(ops):
        return True, ops.iloc[pos]
    else:
        return False, pd.Series()
