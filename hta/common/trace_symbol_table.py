# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import multiprocessing as mp
import os
import queue
import re
from typing import Dict, Iterable, List

import pandas as pd
from hta.common.types import (
    CPU_EVENTS_CATEGORY_PATTERN,
    GroupingPattern,
    KERNEL_CATEGORY_PATTERN,
    KERNEL_LAUNCH_CATEGORY_PATTERN,
    MemoryKernelGroupingPattern,
    ProfilerStepGroupingPattern,
)

from hta.configs.default_values import MAX_NUM_PROCESSES_SMALL
from hta.utils.utils import shorten_name


class _SymbolCollector:
    """
    To support a multiprocessing version of symbol table update, we use _SymbolCollector to a shared queue
    to collect all the symbols and then use a single process to merge the results. A _SymbolCollector object
    implements a function object so that it allow the multiple processes to share the data.
    """

    def __init__(self, q: queue.Queue[str]) -> None:
        self.queue = q

    def __call__(self, symbols: Iterable[str]) -> None:
        for s in symbols:
            self.queue.put(s)


# We use a TraceSymbolTable to store the bidirectional symbol<-->id mapping for each Trace object.
# This table will be shared among all the ranks to encode/decode the symbols in their data frames.
class TraceSymbolTable:
    """
    TraceSymbolTable stores the bidirectional symbol<-->id mapping for all traces.
    Because of potential races caused by multiprocessing, we serialize updates to the
    TraceSymbolTable using a synchronize.lock. The table maintains both dictionary and
    pandas Series representations for efficient lookups and DataFrame operations.

    We assume all read operations to this table by a Trace object occur after it adds all its symbols.
    Therefore, there is no need to lock the read access.

    Attributes:
        sym_table (List[str]) : a list of symbols.
        sym_index (Dict[str, int]) : a map from symbol to ID.
        sym_index_series (pd.series): A Series representation for the symbol table.
        sym_table_series (pd.Series): A Series representation for the symbol index.
    """

    def __init__(self) -> None:
        self.sym_table: List[str] = []
        self.sym_index: Dict[str, int] = {}
        self._sym_index_series: pd.Series = pd.Series(self.sym_index, dtype="int64")
        self._sym_table_series: pd.Series = pd.Series(self.sym_table, dtype="string")

    def is_empty(self) -> bool:
        return len(self.sym_table) == 0

    def add_symbols(self, symbols: Iterable[str]) -> None:
        """Add new symbols to the table.

        Args:
            symbols: Iterable of symbol strings to add
        """
        for s in symbols:
            if s not in self.sym_index:
                idx = len(self.sym_table)
                self.sym_table.append(s)
                self.sym_index[s] = idx

    def add_symbols_mp(self, symbols_list: List[Iterable[str]]) -> None:
        """Add symbols using multiprocessing for parallel collection.

        Args:
            symbols_list: List of symbol iterables to process in parallel
        """
        m = mp.Manager()
        shared_queue: queue.Queue[str] = m.Queue()
        collector = _SymbolCollector(shared_queue)

        with mp.get_context("fork").Pool(
            min(MAX_NUM_PROCESSES_SMALL, mp.cpu_count(), len(symbols_list))
        ) as pool:
            pool.map(collector, symbols_list)
            pool.close()
            pool.join()

        all_symbols = []
        while not shared_queue.empty():
            all_symbols.append(shared_queue.get())
        self.add_symbols(all_symbols)

    def get_sym_id_map(self) -> Dict[str, int]:
        """Get the symbol to ID mapping dictionary.

        Returns:
            Dictionary mapping symbol strings to their integer IDs.
        """
        return self.sym_index

    def get_sym_index(self) -> Dict[str, int]:
        return self.sym_index

    def get_sym_table(self) -> List[str]:
        """Get the list of symbols.

        Returns:
            List of symbol strings in order of their IDs.
        """
        return self.sym_table

    def find_matches(self, patterns: List[str]) -> List[int]:
        """
        Get the indices in sym_table where any of the given patterns match.

        Args:
            patterns (List[str]): The list of patterns to match, can be regular expressions.

        Returns:
            List[int]: A list of indices where a pattern matches.
        """
        # Compile the patterns into a single regex pattern
        pattern = re.compile("|".join(re.escape(p) for p in patterns))

        # Find the indices of matches
        return [i for i, s in enumerate(self.sym_table) if pattern.search(s)]

    def find_matched_symbols(self, patterns: List[str]) -> List[str]:
        """
        Get symbols where any of the given patterns match.

        Args:
            patterns (List[str]): The list of patterns to match, can be regular expressions.

        Returns:
            List[str]: A list of symbols where the pattern matches
        """
        # Compile the patterns into a single regex pattern
        pattern = re.compile("|".join(re.escape(p) for p in patterns))

        # Find the names of matches
        return [s for s in self.sym_table if pattern.search(s)]

    def _update_symbol_series(self) -> None:
        if len(self.sym_table) != len(self._sym_table_series):
            self._sym_table_series = pd.Series(self.sym_table)
            self._sym_index_series = pd.Series(self.sym_index)

    def get_sym_index_series(self) -> pd.Series:
        self._update_symbol_series()
        return self._sym_index_series

    def get_sym_table_series(self) -> pd.Series:
        self._update_symbol_series()
        return self._sym_table_series

    def get_symbol_ids(self, symbol_pattern: GroupingPattern) -> Dict[str, int]:
        """Get the category ids of events in a trace dataframe."""
        s_map = self.get_sym_index_series()
        mask = s_map.index.str.match(symbol_pattern.pattern)
        if symbol_pattern.inverse_match:
            mask = ~mask
        return s_map[mask].to_dict()

    def get_symbol_names(self, symbol_ids: List[int]) -> Dict[int, str]:
        """Get the category ids of events in a trace dataframe."""
        s_table = self.get_sym_table_series()
        mask = s_table.index.isin(symbol_ids)
        return s_table[mask].to_dict()

    def _to_csv_file(self, file_path: str) -> None:
        """Save the symbol table to a CSV file"""
        pd.DataFrame(data=self.sym_table, columns=["symbol"]).to_csv(
            file_path, index=True
        )

    @staticmethod
    def from_csv_file(file_path: str) -> "TraceSymbolTable":
        """Read a TraceSymbolTable from a CSV file. The CSV file should contain at least a `symbol` column.

        Args:
            file_path (str): path to a csv file that contains the symbol table.

        Returns:
            TraceSymbolTable: the constructed TraceSymbolTable.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        df = pd.read_csv(file_path)
        if "symbol" not in df.columns:
            raise ValueError(f"{file_path}: expect a column'symbol' in the csv")
        symbol_table = TraceSymbolTable()
        symbol_table.add_symbols(df["symbol"].tolist())
        return symbol_table

    @staticmethod
    def combine_symbol_tables(tables: List["TraceSymbolTable"]) -> "TraceSymbolTable":
        """Combine multiple symbol tables into one symbol table."""
        result: TraceSymbolTable = TraceSymbolTable()
        for t in tables:
            result.add_symbols(t.get_sym_table())
        return result

    @staticmethod
    def clone(symbol_table: "TraceSymbolTable") -> "TraceSymbolTable":
        """Create a TraceSymbolTable by cloning <symbol_table>"""
        tst: TraceSymbolTable = TraceSymbolTable()
        tst.sym_table = symbol_table.sym_table.copy()
        tst.sym_index = symbol_table.sym_index.copy()
        return tst

    @staticmethod
    def create_from_df(df: pd.DataFrame) -> "TraceSymbolTable":
        """Create a symbol table from a DataFrame's cat and name columns.

        Creates a new symbol table containing all unique symbols found in the DataFrame's
        'name' and 'cat' columns. The symbols are added in the order they appear.

        Args:
            df (pd.DataFrame): an input DataFrame which has columns `name` and `cat`.

        Returns:
            TraceSymbolTable: a symbol table containing all unique `name` and `cat` symbols in df.

        Raise:
            ValueError: when `df` doesn't have the `name` and `cat` columns or they are not string type.
        """
        if (
            ("name" in df.columns)
            and (df["name"].dtype.kind == "O")
            and ("cat" in df.columns)
            and (df["cat"].dtype == "O")
        ):
            symbols = set(df["cat"].unique()).union(set(df["name"].unique()))
            symbol_table = TraceSymbolTable()
            symbol_table.add_symbols(symbols)
            return symbol_table
        raise ValueError(
            "Expect both name and cat columns of string types to be present in the dataframe"
        )

    @staticmethod
    def create_from_symbol_id_map(symbol_id_map: Dict[str, int]) -> "TraceSymbolTable":
        """Create a symbol table from the given symbol id map."""
        max_id = max(symbol_id_map.values())
        id_to_symbol_map = {v: k for (k, v) in symbol_id_map.items()}

        tst = TraceSymbolTable()
        tst.sym_table = [
            id_to_symbol_map.get(i, f"Undefined-{i}") for i in range(max_id + 1)
        ]
        tst.sym_index.update({s: i for i, s in enumerate(tst.sym_table)})

        return tst

    def encode_df(self, df: pd.DataFrame) -> None:
        """Encode the name and cat columns of a DataFrame with this symbol table."""
        for col in ["name", "cat"]:
            if col in df.columns and df[col].dtype.kind == "O":
                df[col] = df[col].apply(lambda sym: self.sym_index[sym])

    def decode_df(self, df: pd.DataFrame, create_new_columns: bool = True) -> None:
        """Decode the name and cat columns of a DataFrame with this symbol table."""
        for col in ["name", "cat"]:
            if col in df.columns and df[col].dtype.kind == "i":
                col_name = "s_" + col if create_new_columns else col
                df[col_name] = df[col].apply(lambda idx: self.sym_table[idx])

    def update_encoded_df(
        self, df: pd.DataFrame, old_symbol_table: "TraceSymbolTable"
    ) -> None:
        """Update the cat and name columns of df that was encoded using old_symbol_table."""
        new_map = self.get_sym_index()
        old_table = old_symbol_table.get_sym_table()
        for col in ["cat", "name"]:
            df[col] = df[col].apply(lambda idx: new_map[old_table[idx]])

    def add_symbols_to_trace_df(self, trace_df: pd.DataFrame, col: str) -> None:
        """
        Take a trace dataframe and expand symbols in one of its columns.
        Args:
            trace_df (pd.DataFrame): Dataframe for trace from one rank.
            col (str): column to expand symbols on.

        Returns:
            None
        """
        trace_df[col] = trace_df[col].apply(
            lambda i: self.sym_table[i] if (0 <= i < len(self.sym_table)) else ""
        )

    # Use a number lower than -1 as a sentinel for missing symbols
    NULL: int = -128

    def get_operator_or_cuda_runtime_mask(self, df: pd.DataFrame) -> pd.Series:
        """Returns a boolean mask you can use with pandas dataframes
        to filter events that are CUDA runtime events or operators."""
        cpu_op_id = self.sym_index.get("cpu_op")
        cuda_runtime_id = self.sym_index.get("cuda_driver", self.NULL)
        cuda_driver_id = self.sym_index.get("cuda_runtime", self.NULL)
        return (
            (df["cat"] == cpu_op_id)
            | (df["cat"] == cuda_runtime_id)
            | (df["cat"] == cuda_driver_id)
        )

    def _get_xpu_runtime_launch_events_mask(self, df: pd.DataFrame) -> pd.Series:
        """Returns a boolean mask you can use with pandas dataframes
        to filter events that are XPU runtime kernel and memory operations."""

        urEnqueueUSMFill_id = self.sym_index.get("urEnqueueUSMFill", self.NULL)
        urEnqueueUSMFill2D_id = self.sym_index.get("urEnqueueUSMFill2D", self.NULL)
        urEnqueueUSMMemcpy_id = self.sym_index.get("urEnqueueUSMMemcpy", self.NULL)
        urEnqueueUSMMemcpy2D_id = self.sym_index.get("urEnqueueUSMMemcpy2D", self.NULL)

        urEnqueueKernelLaunch_id = self.sym_index.get("urEnqueueKernelLaunch", self.NULL)
        urEnqueueKernelLaunchExp_id = self.sym_index.get("urEnqueueKernelLaunchExp", self.NULL)
        urEnqueueKernelLaunchCustomExp_id = self.sym_index.get(
            "urEnqueueKernelLaunchCustomExp", self.NULL
        )
        urEnqueueKernelLaunchWithArgsExp_id = self.sym_index.get(
            "urEnqueueKernelLaunchWithArgsExp", self.NULL
        )
        urEnqueueCooperativeKernelLaunchExp_id = self.sym_index.get(
            "urEnqueueCooperativeKernelLaunchExp", self.NULL
        )

        urEnqueueMemBufferFill_id = self.sym_index.get(
            "urEnqueueMemBufferFill", self.NULL
        )
        urEnqueueMemBufferRead_id = self.sym_index.get(
            "urEnqueueMemBufferRead", self.NULL
        )
        urEnqueueMemBufferWrite_id = self.sym_index.get(
            "urEnqueueMemBufferWrite", self.NULL
        )
        urEnqueueMemBufferCopy_id = self.sym_index.get(
            "urEnqueueMemBufferCopy", self.NULL
        )
        urUSMHostAlloc_id = self.sym_index.get("urUSMHostAlloc", self.NULL)
        urUSMSharedAlloc_id = self.sym_index.get("urUSMSharedAlloc", self.NULL)
        urUSMDeviceAlloc_id = self.sym_index.get("urUSMDeviceAlloc", self.NULL)

        name_mask = (
            (df["name"] == urEnqueueUSMFill_id)
            | (df["name"] == urEnqueueUSMFill2D_id)
            | (df["name"] == urEnqueueUSMMemcpy_id)
            | (df["name"] == urEnqueueUSMMemcpy2D_id)
            | (df["name"] == urEnqueueKernelLaunch_id)
            | (df["name"] == urEnqueueKernelLaunchExp_id)
            | (df["name"] == urEnqueueKernelLaunchCustomExp_id)
            | (df["name"] == urEnqueueKernelLaunchWithArgsExp_id)
            | (df["name"] == urEnqueueCooperativeKernelLaunchExp_id)
            | (df["name"] == urEnqueueMemBufferFill_id)
            | (df["name"] == urEnqueueMemBufferRead_id)
            | (df["name"] == urEnqueueMemBufferWrite_id)
            | (df["name"] == urEnqueueMemBufferCopy_id)
            | (df["name"] == urUSMHostAlloc_id)
            | (df["name"] == urUSMSharedAlloc_id)
            | (df["name"] == urUSMDeviceAlloc_id)
        )

        return name_mask

    def get_runtime_launch_events_mask(self, df: pd.DataFrame) -> pd.Series:
        """Returns a boolean mask you can use with pandas dataframes
        to filter events that are CUDA runtime kernel and memcpy launches."""
        cudaLaunchKernel_id = self.sym_index.get("cudaLaunchKernel", self.NULL)
        cudaLaunchKernelExC_id = self.sym_index.get("cudaLaunchKernelExC", self.NULL)
        cuLaunchKernel_id = self.sym_index.get("cuLaunchKernel", self.NULL)
        cuLaunchKernelEx_id = self.sym_index.get("cuLaunchKernelEx", self.NULL)
        cudaMemcpyAsync_id = self.sym_index.get("cudaMemcpyAsync", self.NULL)
        cudaMemsetAsync_id = self.sym_index.get("cudaMemsetAsync", self.NULL)
        mtiaLaunchKernel_id = self.sym_index.get(
            "runFunction - job_prep_and_submit_for_execution", self.NULL
        )
        rocmLaunchKernel_id = self.sym_index.get("hipLaunchKernel", self.NULL)
        rocmExtModuleLaunchKernel_id = self.sym_index.get(
            "hipExtModuleLaunchKernel", self.NULL
        )
        rocmMemsetAsync_id = self.sym_index.get("hipMemsetAsync", self.NULL)
        rocmMemcpyAsync_id = self.sym_index.get("hipMemcpyAsync", self.NULL)
        rocmMemcpyWithStream_id = self.sym_index.get("hipMemcpyWithStream", self.NULL)

        # Create a mask for each event type and combine with OR
        name_mask = (
            (df["name"] == cudaMemsetAsync_id)
            | (df["name"] == cudaMemcpyAsync_id)
            | (df["name"] == cudaLaunchKernel_id)
            | (df["name"] == cudaLaunchKernelExC_id)
            | (df["name"] == cuLaunchKernel_id)
            | (df["name"] == mtiaLaunchKernel_id)
            | (df["name"] == rocmLaunchKernel_id)
            | (df["name"] == rocmExtModuleLaunchKernel_id)
            | (df["name"] == rocmMemcpyAsync_id)
            | (df["name"] == rocmMemsetAsync_id)
            | (df["name"] == rocmMemcpyWithStream_id)
            | (df["name"] == cuLaunchKernelEx_id)
        )

        xpu_name_mask = self._get_xpu_runtime_launch_events_mask(df)

        # Add the index_correlation > 0 condition
        return (name_mask | xpu_name_mask) & (df["index_correlation"] > 0)

    def get_events_mask(self, df: pd.DataFrame, events: list[str] | None) -> pd.Series:
        """Returns a boolean mask you can use with pandas dataframes
        to filter for events that are in the given events list (regex supported)."""
        if events is None:
            return pd.Series(False, index=df.index)
        event_ids = self.find_matches(events)
        return df["name"].isin(event_ids)

    def get_cpu_event_cat_ids(self) -> List[int]:
        return list(self.get_symbol_ids(CPU_EVENTS_CATEGORY_PATTERN).values())

    def get_gpu_kernel_cat_ids(self) -> List[int]:
        return list(self.get_symbol_ids(KERNEL_CATEGORY_PATTERN).values())

    def get_profiler_step_ids(self) -> List[int]:
        return list(self.get_symbol_ids(ProfilerStepGroupingPattern).values())

    def get_memory_name_ids(self) -> List[int]:
        return list(self.get_symbol_ids(MemoryKernelGroupingPattern).values())

    def get_kernel_launch_ids(self) -> List[int]:
        return list(self.get_symbol_ids(KERNEL_LAUNCH_CATEGORY_PATTERN).values())


def decode_symbol_id_to_symbol_name(
    df: pd.DataFrame, symbol_table: TraceSymbolTable, use_shorten_name: bool
) -> None:
    """Decode symbol ids into symbol names and write the decoded data into s_name and s_cat columns."""
    s_tab: List[str] = symbol_table.sym_table
    if use_shorten_name:
        s_tab = [shorten_name(s) for s in s_tab]

    def get_sym(idx):
        return s_tab[idx] if idx >= 0 else ""

    if "name" in df.columns and df["name"].dtype.kind == "i":
        df["s_name"] = df["name"].apply(lambda idx: get_sym(idx))
    if "cat" in df.columns and df["cat"].dtype.kind == "i":
        df["s_cat"] = df["cat"].apply(lambda idx: get_sym(idx))
    if "user_annotation" in df.columns and df["user_annotation"].dtype.kind == "i":
        df["s_user_annotation"] = df["user_annotation"].apply(lambda idx: get_sym(idx))
