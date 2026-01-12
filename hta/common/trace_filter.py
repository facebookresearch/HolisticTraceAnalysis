from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import pandas as pd
from hta.common.trace_symbol_table import TraceSymbolTable
from hta.configs.config import logger
from hta.utils.utils import get_symbol_column_names


class Filter(ABC):
    """An abstract base class for trace event filters.

    All concrete classes derived from Filter must implement the method __call__().
    The general user patter is like follows:
    1. Create a filter object.
        filter_func = ConcreateFilter(...)
    2. Apply the filter to an input DataFrame.
        df = filter_func(df)
    or an input DataFrame with the corresponding Trace Symbol Table.
        df = filter_func(df, symbol_table)
    """

    @abstractmethod
    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        """
        Args:
            df (pd.DataFrame): an input DataFrame.
            symbol_table (TraceSymbolTable, optional): a trace symbol table for encoding/decoding the name
                and cat columns of the input DataFrame.

        Returns:
            The output DataFrame.
        """
        raise NotImplementedError()


class IterationFilter(Filter):
    """A filter class for extracting the events of one or a list of iteration numbers.

    An iteration number is equivalent to a profiler step number or iteration id. For instance, An IterationFilter
     whose iterations attribute is [550,] will extract all trace events corresponding to the iteration 550.

    Attributes:
        iterations (int or List[int]): One or a list of iteration numbers for filtering the DataFrame.

    Examples:
        filter_func = IterationFilter(551)
        selected_df = filter_func(df)
        filter_func = IterationFilter([551, 552])
        selected_df = filter_func(df)
    """

    def __init__(self, iterations: Union[int, List[int]]) -> None:
        """
        Initialize an IterationFilter object with specified iterations.

        Args:
            iterations (int or List[int]): One or a list of integers representing iteration numbers.
        """
        if not isinstance(iterations, (int, list)):
            raise TypeError("Iterations must be an integer or a list of integers.")

        self.iterations: List[int] = (
            [iterations] if isinstance(iterations, int) else iterations
        )

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if "iteration" not in df.columns:
            logger.warning("The DataFrame has no column named 'iteration'.")
            return df

        return df.loc[df["iteration"].isin(self.iterations)]


class IterationIndexFilter(Filter):
    """
    A filter class for extracting events in specific iterations from a DataFrame.

    IterationFilter and IterationIndexFilter behave identically but interpret the argument to the constructor
    differently. An IterationFilter object selects events based on the values of the iteration column,
    while an IterationIndexFilter object selects the events based on the position of the iteration number in the
    available iteration numbers. For example, assume the iteration column in the DataFrame has unique values of
    [550, 551, 552], then both IterationIndexFilter(0) and IterationFilter(550) select the same set of events.

    Attributes:
        iteration_index (int, List[int]): A list of indices indicating which iterations to select.
    """

    def __init__(self, iteration_index: Union[int, List[int]]) -> None:
        """
        Initialize an IterationIndexFilter object with specified iteration indices.

        Args:
            iteration_index (int or List[int]): An integer or a list of integers representing iteration indices.
        """
        if not isinstance(iteration_index, (int, list)):
            raise TypeError("iteration_index must be an integer or a list of integers.")

        self.iteration_index: List[int] = (
            [iteration_index] if isinstance(iteration_index, int) else iteration_index
        )

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if "iteration" not in df.columns:
            logger.warning("The DataFrame has no column named 'iteration'.")
            return df

        iterations = sorted(df["iteration"].unique())
        if iterations == [-1]:
            return df

        if iterations[0] == -1:
            iterations.pop(0)

        filtered_iterations = [
            iteration
            for i, iteration in enumerate(iterations)
            if i in self.iteration_index
        ]
        if not filtered_iterations:
            logger.warning(f"Indices {self.iteration_index} select no iterations.")
            return pd.DataFrame()

        return df.loc[df["iteration"].isin(filtered_iterations)]


class FirstIterationFilter(IterationIndexFilter):
    """
    A filter class to select the first iteration from a DataFrame.

    This class extends IterationIndexFilter and is specialized in selecting the first iteration.
    """

    def __init__(self) -> None:
        super().__init__([0])


class RankFilter(Filter):
    """
    A trace event filter class to extract events corresponding to one or several given rank ids.

    Examples:
        filter_func = RankFilter(0)
        selected_df = filter_func(df)
        filter_func = RankFilter(list(range(16)))
        selected_df = filter_func(df)
    """

    def __init__(self, ranks: Union[int, List[int]]) -> None:
        """
        Initialize the RankFilter with specified rank ids.

        Args:
            ranks (Union[int, List[int]]): An integer or a list of integers representing rank ids.
        """
        if not isinstance(ranks, (int, list)):
            raise TypeError("ranks must be an integer or a list of integers.")

        self.ranks: List[int] = [ranks] if isinstance(ranks, int) else ranks

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if "rank" not in df.columns:
            logger.warning("DataFrame does not contain a 'rank' column.")
            return df

        return df.loc[df["rank"].isin(self.ranks)]


class TimeRangeFilter(Filter):
    """
    A trace event filter class that extracts events within a given time interval.

    This filter selects rows from a pandas DataFrame based on a specified time range.

    Examples:
        filter_func = TimeRangeFilter((start_time, end_time))
        selected_df = filter_func(df)
    """

    def __init__(self, time_range: Tuple[int, int]) -> None:
        """
        Initialize the TimeRangeFilter with a specified time range.

        Args:
            time_range (Tuple[int, int]): A tuple of two integers representing the start and end times.
        """
        if (
            not isinstance(time_range, tuple)
            or len(time_range) != 2
            or not all(isinstance(t, int) for t in time_range)
        ):
            raise ValueError("time_range must be a tuple of two integers.")

        if time_range[0] > time_range[1]:
            raise ValueError("Start time must be less than or equal to end time.")

        self.time_start: int = time_range[0]
        self.time_end: int = time_range[1]

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if "ts" not in df.columns:
            logger.warning("DataFrame does not contain a 'ts' column.")
            return df

        return df.loc[
            df["ts"].ge(self.time_start) & (df["ts"] + df["dur"]).le(self.time_end)
        ]


class NameStringColumnFilter(Filter):
    def __init__(self, name_pattern: str):
        self.name_pattern = name_pattern

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        name_column, _ = get_symbol_column_names(df)
        if name_column == "" or (
            name_column in df.columns and df[name_column].dtype != object
        ):
            logger.warning("The DataFrame doesn't have a name column of string type")
            return df

        return df.loc[df[name_column].str.match(self.name_pattern)]


class NameIdColumnFilter(Filter):
    def __init__(self, name_pattern: str) -> None:
        self.name_pattern = name_pattern
        self.name_column = "name"

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if symbol_table is None:
            logger.warning(
                "The argument `symbol_table` cannot be None in `NameIdColumnFilter`"
            )
            return df

        sym_index = pd.Series(symbol_table.sym_index)
        matched_ids = set(
            sym_index.loc[sym_index.index.str.match(self.name_pattern)].values
        )
        return df.loc[df[self.name_column].isin(matched_ids)]


class NameFilter(Filter):
    """
    A trace event filter class to select events based on a pattern in the name column.

    This filter matches rows where the values in the specified name column match a given pattern.
    If a symbol table is provided, it matches against the symbol indices.

    Attributes:
        name_pattern (str): The pattern to match in the name column.
        symbol_table (pd.TraceSymbolTable, optional): A TraceSymbolTable object.
        name_column (str): The name of the column to apply the pattern matching.
    """

    def __init__(
        self,
        name_pattern: str,
        symbol_table: Optional[TraceSymbolTable] = None,
        name_column: Optional[str] = None,
    ) -> None:
        self.name_pattern: str = name_pattern
        self.symbol_table: Optional[TraceSymbolTable] = symbol_table
        self.name_column: Optional[str] = name_column

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if df.empty:
            logger.warning("The DataFrame is empty.")
            return df

        if symbol_table is None and self.symbol_table is None:
            return NameStringColumnFilter(self.name_pattern)(df)
        else:
            if symbol_table:
                _symbol_table = symbol_table
            elif self.symbol_table:
                _symbol_table = self.symbol_table
            else:
                _symbol_table = TraceSymbolTable()

            return NameIdColumnFilter(self.name_pattern)(df, _symbol_table)


class QueryFilter:
    """
    A trace event filter class to select events matching a SQL condition
    """

    def __init__(self, query_str: str) -> None:
        self.filter_query = query_str

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        return df.query(self.filter_query)


# This filter matches rows where the duration is non zero.
ZeroDurationFilter = QueryFilter("dur > 0")


def _filter_gpu_kernels_with_cuda_sync(
    df: pd.DataFrame, symbol_table: TraceSymbolTable
):
    """Helper function that finds rows in the dataframe that are either
    GPU kernels or CUDA synchronization events."""

    # Device level Synchronization events are on stream = -1 but still
    # run on GPU
    event_sync_id = symbol_table.get_sym_id_map().get("Event Sync", -1)
    context_sync_id = symbol_table.get_sym_id_map().get("Context Sync", -1)
    return ((df["stream"] >= 0) & (df["correlation"] >= 0)) | df["name"].isin(
        [event_sync_id, context_sync_id]
    )


class GPUKernelFilter(Filter):
    """
    A trace event filter class that extracts GPU kernel events from a DataFrame.
    """

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if "stream" not in df.columns:
            logger.warning("DataFrame does not contain a 'stream' column.")
            return df

        if symbol_table is None:
            logger.warning(
                "GPUKernelFilter needs symbol table to identify GPU synchronization events"
            )
            return df.loc[(df["stream"] >= 0) & (df["correlation"] >= 0)]

        return df.loc[_filter_gpu_kernels_with_cuda_sync(df, symbol_table)]


class CPUOperatorFilter(Filter):
    """
    A trace event filter class that extracts CPU operator events from a DataFrame.
    """

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if "stream" not in df.columns:
            logger.warning("DataFrame does not contain a 'stream' column.")
            return df

        if symbol_table is None:
            logger.warning(
                "CPUOperatorFilter needs symbol table to exclude GPU synchronization events"
            )
            return df.loc[df["stream"] == -1]

        return df.loc[~_filter_gpu_kernels_with_cuda_sync(df, symbol_table)]


class CompositeFilter(Filter):
    """
    A composite trace filter class that applies a sequence of filters to a DataFrame.

    This class allows combining multiple filter objects and applying them sequentially to a DataFrame.

    Attributes:
        filters (List[Filter]): A list of filter objects to be applied.
    """

    def __init__(self, filters: List[Filter]) -> None:
        """
        Initialize a CompositeFilter object with a list of filter objects.

        Args:
            filters (List[Filter]): A list of filter objects.
        """
        for obj in filters:
            if not isinstance(obj, Filter):
                raise TypeError(
                    "All elements in 'filters' must be instances of Filter or its subclasses."
                )
            if obj is self:
                raise ValueError("CompositeFilter cannot contain itself as a filter.")

        self.filters: List[Filter] = filters

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        for f in self.filters:
            df = f(df, symbol_table)
        return df


class MemCopyEventFilter(Filter):
    """
    A trace event filter class to select memory copy events

    This filter matches rows where its category is "gpu_memcpy" and its name is the specified memory copy type.

    Attributes:
        memory_copy_type (str): The type of memory copy events to select.
    """

    def __init__(
        self,
        memory_copy_type: str,
        symbol_table: Optional[TraceSymbolTable] = None,
    ) -> None:
        self.memory_copy_type: str = memory_copy_type
        self.symbol_table: Optional[TraceSymbolTable] = symbol_table

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if df.empty:
            logger.warning("The DataFrame is empty.")
            return df
        if symbol_table:
            _symbol_table = symbol_table
        elif self.symbol_table:
            _symbol_table = self.symbol_table
        else:
            _symbol_table = TraceSymbolTable()

        if self.memory_copy_type in _symbol_table.sym_index:
            name_id = _symbol_table.sym_index[self.memory_copy_type]
            cat_id = _symbol_table.sym_index["gpu_memcpy"]
            return df.loc[df["name"].eq(name_id) & df["cat"].eq(cat_id)]
        else:
            return pd.DataFrame()
