from typing import List

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
