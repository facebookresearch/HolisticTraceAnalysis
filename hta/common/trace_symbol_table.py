# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import multiprocessing as mp

import queue
from typing import Any, Dict, Iterable, List

import pandas as pd

from hta.utils.utils import shorten_name


class _SymbolCollector:
    """
    To support a multiprocessing version of symbol table update, we use _SymbolCollector to a shared queue
    to collect all the symbols and then use a single process to merge the results. A _SymbolCollector object
    implements a function object so that it allow the multiple processes to share the data.
    """

    def __init__(self, q: queue.Queue[Any]):
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
    TraceSymbolTable using a synchronize.lock.

    We assume all read operations to this table by a Trace object occur after it adds all its symbols.
    Therefore, there is no need to lock the read access.

    Attributes:
        sym_table (List[str]) : a list of symbols.
        sym_index (Dict[str, int]) : a map from symbol to ID.
    """

    def __init__(self):
        self.sym_table: List[str] = []
        self.sym_index: Dict[str, int] = {}

    def add_symbols(self, symbols: Iterable[str]) -> None:
        for s in symbols:
            if s not in self.sym_index:
                idx = len(self.sym_table)
                self.sym_table.append(s)
                self.sym_index[s] = idx

    def add_symbols_mp(self, symbols_list: List[Iterable[str]]) -> None:
        m = mp.Manager()
        shared_queue: queue.Queue[Any] = m.Queue()
        collector = _SymbolCollector(shared_queue)

        with mp.get_context("spawn").Pool(
            min(mp.cpu_count(), len(symbols_list))
        ) as pool:
            pool.map(collector, symbols_list)
            pool.close()
            pool.join()

        all_symbols = []
        while not shared_queue.empty():
            all_symbols.append(shared_queue.get())
        self.add_symbols(all_symbols)

    def get_sym_id_map(self) -> Dict[str, int]:
        return self.sym_index

    def get_sym_table(self) -> List[str]:
        return self.sym_table

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

    @staticmethod
    def create_symbol_table_from_df(df: pd.DataFrame) -> TraceSymbolTable:
        """Create a symbol table from a DataFrame's cat and name columns.

        Args:
            df (pd.DataFrame): an input DataFrame.

        Returns:
            TraceSymbolTable: a symbol table containing all unique `name` and `cat` symbols in df.

        Raise:
            ValueError: when one of the follow two conditions happens:
                (1) `df` doesn't have the `name` and `cat` columns; or
                (2) they are not string type.
        """
        if (
            ("name" in df.columns)
            and (df.dtypes["name"] == "object")
            and ("cat" in df.columns)
            and (df.dtypes["cat"] == "object")
        ):
            symbols = set(df["cat"].unique()).union(set(df["name"].unique()))
            symbol_table = TraceSymbolTable()
            symbol_table.add_symbols(symbols)
            return symbol_table
        raise ValueError(
            "Expect both name and cat columns of string types to be present in the dataframe"
        )

    def is_cuda_runtime(self, trace_df: pd.DataFrame, idx: int) -> bool:
        """Check if an event is a CUDA runtime event"""
        return trace_df["cat"].loc[idx] == self.sym_index["cuda_runtime"] or (
            "cuda_driver" in self.sym_index.keys()
            and (trace_df["cat"].loc[idx] == self.sym_index["cuda_driver"])
        )

    def is_operator(self, trace_df: pd.DataFrame, idx: int) -> bool:
        """Check if an event is a CPU operator"""
        return trace_df["cat"].loc[idx] == self.sym_index["cpu_op"]

    def get_runtime_launch_events_query(self) -> str:
        """Returns a SQL query you can pass to trace dataframe query()
        to filter events that are CUDA runtime kernel and memcpy launches."""
        cudaLaunchKernel_id = self.sym_index.get("cudaLaunchKernel", -128)
        cudaLaunchKernelExC_id = self.sym_index.get("cudaLaunchKernelExC", -128)
        cuLaunchKernel_id = self.sym_index.get("cuLaunchKernel", -128)
        cudaMemcpyAsync_id = self.sym_index.get("cudaMemcpyAsync", -128)
        cudaMemsetAsync_id = self.sym_index.get("cudaMemsetAsync", -128)

        return (
            f"((name == {cudaMemsetAsync_id}) or (name == {cudaMemcpyAsync_id}) or "
            f" (name == {cudaLaunchKernel_id}) or (name == {cudaLaunchKernelExC_id})"
            f" or (name == {cuLaunchKernel_id})) and (index_correlation > 0)"
        )


def decode_symbol_id_to_symbol_name(
    df: pd.DataFrame, symbol_table: TraceSymbolTable, use_shorten_name: bool
) -> None:
    """Decode symbol ids into symbol names and write the decoded data into s_name and s_cat columns."""
    s_tab: List[str] = symbol_table.sym_table
    if use_shorten_name:
        s_tab = [shorten_name(s) for s in s_tab]
    if "name" in df.columns and df["name"].dtype.kind == "i":
        df["s_name"] = df["name"].apply(lambda idx: s_tab[idx])
    if "cat" in df.columns and df["cat"].dtype.kind == "i":
        df["s_cat"] = df["cat"].apply(lambda idx: s_tab[idx])
