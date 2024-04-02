from typing import List, Tuple

import pandas as pd

from hta.common.trace import TraceSymbolTable


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


def find_events_by_name_patterns_using_symbol_table(
    df: pd.DataFrame, list_name_patterns: List[str], symbol_table: TraceSymbolTable
) -> pd.Series:
    """Searches for events in the provided DataFrame using a list of name patterns, leveraging a symbol table for memory saving and fast match.
    Args:
        df: The DataFrame to search. It should contain a column 'name' with event names and 'index' with event indices.
        list_name_patterns: A list of regular expression pattern strings to search for in the event names.
        symbol_table: The TraceSymbolTable which maps the name and cat symbols to corresponding integer IDs.
    Returns:
        A Series object containing the indices of rows in the original DataFrame that match any of
        the name patterns in the list.
    """
    sym_index = pd.Series(symbol_table.get_sym_id_map())
    matched_ids = set()
    for module in list_name_patterns:
        matched_ids.update(sym_index.loc[sym_index.index.str.match(module)].values)
    indices: pd.Series = df.loc[df["name"].isin(matched_ids)]["index"]
    return indices


def find_events_by_name_patterns_using_decoded_names(
    df: pd.DataFrame, list_name_patterns: List[str], name_column: str = "s_name"
) -> pd.Series:
    """Searches for events in the provided DataFrame using a list of name patterns.
    Args:
        df (pd.DataFrame): The DataFrame to search. It should contain a column for event names specified by `name_column`.
        list_name_patterns (List[str]):  A list of regular expression pattern strings to search for in the event names.
        name_column (str, optional): The name of the column in `df` that contains the event names. Defaults to 's_name'.

    Returns:
        A Series object containing the indices of rows in the original DataFrame that match any of
            the name patterns in the list.
    """
    indices = set()
    for module in list_name_patterns:
        indices.update(df.loc[df[name_column].str.match(module)]["index"])
    return pd.Series(df.loc[list(indices)]["index"])
