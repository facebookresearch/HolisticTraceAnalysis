# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import tracemalloc
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from hta.common.trace_file import get_trace_files
from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR
from hta.configs.parser_config import AttributeSpec, ParserConfig, ValueType
from hta.utils.utils import get_mp_pool_size, normalize_path, shorten_name

MetaData = Dict[str, Any]
PHASE_COUNTER: str = "C"
PHASE_FLOW_START: str = "s"
PHASE_FLOW_END: str = "f"


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


def parse_trace_dict(trace_file_path: str) -> Dict[str, Any]:
    """
    Parse a raw trace file into a dictionary.

    Args:
        trace_file_path (str) : the path to a trace file.

    Returns:
        A dictionary representation of the trace.
    """
    t_start = time.perf_counter()
    trace_record: Dict[str, Any] = {}
    if trace_file_path.endswith(".gz"):
        with gzip.open(trace_file_path, "rb") as fh:
            trace_record = json.loads(fh.read())
    elif trace_file_path.endswith(".json"):
        with open(trace_file_path, "r") as fh2:
            trace_record = json.loads(fh2.read())
    else:
        raise ValueError(
            f"expect the value of trace_file ({trace_file_path}) ends with '.gz' or 'json'"
        )
    t_end = time.perf_counter()
    logger.info(f"Parsed {trace_file_path} time = {(t_end - t_start):.2f} seconds ")
    return trace_record


def compress_df(
    df: pd.DataFrame, cfg: Optional[ParserConfig] = None
) -> Tuple[pd.DataFrame, TraceSymbolTable]:
    """
    Compress a Dataframe to reduce its memory footprint.

    Args:
        df (pd.DataFrame): the input DataFrame
        cfg (Optional[ParserConfig]): an object to customize how to parse/compress the trace.

    Returns:
        Tuple[pd.DataFrame, TraceSymbolTable]
            The first item is the compressed dataframe.
            The second item is the local symbol table specific to this dataframe.
    """
    cfg = cfg or ParserConfig.get_default_cfg()

    # drop rows with null values
    df.dropna(axis=0, subset=["dur", "cat"], inplace=True)
    df.drop(df[df["cat"] == "Trace"].index, inplace=True)

    # drop columns
    columns_to_drop = {"ph", "id", "bp", "s"}.intersection(set(df.columns))
    df.drop(list(columns_to_drop), axis=1, inplace=True)

    # performance counters appear as args
    if "cuda_profiler_range" in df.cat.unique():
        counter_names = set.union(
            *[set(d.keys()) for d in df[df.cat == "cuda_profiler_range"]["args"].values]
        )
        # args_to_keep = args_to_keep.union(counter_names)
        cfg.add_args(
            [AttributeSpec(name, name, ValueType.Int, -1) for name in counter_names]
        )
        logger.info(f"counter_names={counter_names}")
        logger.info(f"args={cfg.get_args()}")

    args_to_keep = cfg.get_args()
    for arg in args_to_keep:
        df[arg.name] = df["args"].apply(
            lambda row: (
                row.get(arg.raw_name, arg.default_value)
                if isinstance(row, dict)
                else arg.default_value
            )
        )
    df.drop(["args"], axis=1, inplace=True)

    # create a local symbol table
    local_symbol_table = TraceSymbolTable()
    symbols = set(df["cat"].unique()).union(set(df["name"].unique()))
    local_symbol_table.add_symbols(symbols)

    sym_index = local_symbol_table.get_sym_id_map()
    for col in ["cat", "name"]:
        df[col] = df[col].apply(lambda s: sym_index[s])

    # data type downcast
    for col in df.columns:
        if df[col].dtype.kind == "i":
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    return df, local_symbol_table


def transform_correlation_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """Transform correlation to index_correlation and add a index_correlation column to df.

    The correlation in the trace is a reference ID which links a Cuda kernel launch
    and the cuda kernel. Because the correlation is not an index, using `correction` to find
    the corresponding Cuda launch event for a Cuda kernel requires searching the dataframe.

    The index_correlation column maps the correction to the actual index of the linked events
    so that finding the lined events will be easy. The values of `index_correlation` fall into
    three cases:
    index_correlation = -1 : the event does not link to another event
    index_correlation = 0  : the event is cuda related but the linked event is not shown in trace.
    index_correlation > 0  : the event links to another event whose index is index_correlation.

    The transform can be illustrated as follows:

    Given the following input DataFrame <df>:

    | index | stream | cat | s_cat  | correlation |
    ===============================================
    | 675   | 7      | 248 | Kernel | 278204204   |
    | 677   | -1     |   9 | Runtime| 278204204   |

    After calling `transform_correlation_to_index(df)`, <df> will become:

    | index | stream | cat | s_cat  | correlation |  index_correlation |
    ====================================================================
    | 675   | 7      | 248 | Kernel | 278204204   |       677          |
    | 677   | -1     |   9 | Runtime| 278204204   |       675          |

    This function changes the input DataFrame by adding a new column `index_correlation`.

    Example use: find the kernels of all Cuda Launch
    cuda_launch_events = df.loc[df['cat'].eq(9) & df['index_correlation'].gt(0)]
    cuda_kernel_indices = cuda_launch_events["index_correlation"]
    df.loc[cuda_kernel_indices]

    Args:
        df (pd.DataFrame): the input DataFrame

    Returns:
        pd.DataFrame: the transformed DataFrame with a index_correlation column.

    Affects:
        This function adds a index_correlation column to df.
    """
    if "correlation" not in df.columns:
        return df
    corr_df = df.loc[df["correlation"].ne(-1), ["index", "correlation", "stream"]]
    on_cpu = corr_df.loc[df["stream"].eq(-1)]
    on_gpu = corr_df.loc[df["stream"].ne(-1)]
    merged_cpu_idx = on_cpu.merge(on_gpu, on="correlation", how="inner")
    merged_gpu_idx = on_gpu.merge(on_cpu, on="correlation", how="inner")
    matched = pd.concat([merged_cpu_idx, merged_gpu_idx], axis=0)[
        ["index_x", "index_y"]
    ].set_index("index_x")

    corr_index_map: Dict[int, int] = matched["index_y"].to_dict()

    def _set_corr_index(row):
        idx = row.name
        if idx in corr_index_map:
            return corr_index_map[idx]
        elif df.loc[idx, "correlation"] == -1:
            return -1
        else:
            return 0

    df["index_correlation"] = df.apply(_set_corr_index, axis=1)
    df["index_correlation"] = pd.to_numeric(df["index_correlation"], downcast="integer")

    return df


def get_cpu_gpu_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Get the correlation between CPU cuda_launch ops and GPU kernels in a trace DataFrame.
    Args:
        df (pd.DataFrame): a DataFrame representation of the trace

    Returns:
        A DataFrame with two columns:
            gpu_index	cpu_index
        ===========================
        0	121898	    121900
        1	121906	    121908
    """
    kernel_indices = df[df["stream"].gt(0) & df["index_correlation"].gt(0)]["index"]
    cpu_gpu_correlation = (
        df.loc[kernel_indices][["index", "index_correlation"]]
        .copy()
        .rename(columns={"index": "gpu_index", "index_correlation": "cpu_index"})
        .reset_index(drop=True)
    )
    return cpu_gpu_correlation


def add_iteration(df: pd.DataFrame, symbol_table: TraceSymbolTable) -> pd.DataFrame:
    """
    Add an iteration column to the DataFrame <df>.

    This function extracts the trace iteration number from the `ProfilerStep` annotation
    in the name column and then apply the following logic to determine which iteration
    an event belongs:
    1. For events on the CPU devices, an event `e` will have the same iteration number
       as the event ProfilerStep#<iter> `s` if  `s.ts <= e.ts < s.ts + s.dur`.
    2. For events on the GPU devices, an event `e` will have the same iteration number
       as its correlated event (linked by index_correlation or correlation).
    3. For other events whose iteration number can not be determined with the above two
       steps, set the iteration number to -1.

    Args:
        df: a DataFrame representation of a trace.
        symbol_table: the TraceSymbolTable for the trace

    Returns:
        A DataFrame with the profiler steps information.

    Note:
        This function will change the input DataFrame by adding the iteration number column.
    """
    s_map = pd.Series(symbol_table.sym_index)
    s_tab = pd.Series(symbol_table.sym_table)
    profiler_step_ids = s_map[s_map.index.str.startswith("ProfilerStep")]
    profiler_step_ids.sort_index()

    def _extract_iter(profiler_step_name_id: int) -> int:
        """Convert a profiler_step_name_id to an iteration number.
        For example, 168 (string name: ProfilerStep#15) ==> 15
        """
        s = s_tab[profiler_step_name_id]
        m = re.match(r"ProfilerStep\s*#\s*(\d+)", s)
        return int(m.group(1))

    # Extract the profiler steps
    profiler_steps = df.loc[df["name"].isin(profiler_step_ids.values)]
    profiler_steps = profiler_steps[["ts", "dur", "name"]].copy()
    profiler_steps["s_name"] = profiler_steps["name"].apply(_extract_iter)
    profiler_steps["iter"] = profiler_steps["name"].apply(lambda idx: s_tab[idx])

    profiler_steps_array = profiler_steps.to_numpy()

    def _get_profiler_step(ts: int) -> int:
        """ "determine which profiler step a given timestamp <ts> falls into"""
        iter = -1
        for step in profiler_steps_array:
            if step[0] <= ts < step[0] + step[1]:
                iter = step[3]
        return iter

    # Update the trace iteration column
    df.loc[df["stream"].lt(0), "iteration"] = df["ts"].apply(_get_profiler_step)
    df.loc[df["stream"].gt(0), "iteration"] = df["index_correlation"].apply(
        lambda x: df.loc[x, "iteration"] if x > 0 else -1
    )
    df["iteration"] = pd.to_numeric(df["iteration"], downcast="integer")

    return profiler_steps


def parse_trace_dataframe(
    trace_file_path: str, cfg: Optional[ParserConfig] = None
) -> Tuple[MetaData, pd.DataFrame, TraceSymbolTable]:
    """parse a single trace file into a meat test_data dictionary and a dataframe of events.
    Args:
        trace_file_path (str): The path to a trace file. When the trace_file is a relative path.
            This method combines the object's trace_path with trace_file to get the full path of the trace file.
        cfg (ParserConfig, Optional): A ParserConfig object controls how to parse the trace file.
    Returns:
        Tuple[MetaData, pd.DataFrame, TraceSymbolTable]
            The first item is the trace's metadata;
            The second item is the dataframe representation of the trace's events.
            The third item is the symbol table to encode the symbols of the trace.


    Raises:
        OSError when the trace file doesn't exist or current process has no permission to access it.
        JSONDecodeError when the trace file is not a valid JSON document.
        ValueError when the trace_file doesn't end with ".gz" or "json".
    """
    t_start = time.perf_counter()
    trace_record = parse_trace_dict(trace_file_path)

    meta: Dict[str, Any] = {k: v for k, v in trace_record.items() if k != "traceEvents"}
    df: pd.DataFrame = pd.DataFrame()
    if "traceEvents" in trace_record:
        df = pd.DataFrame(trace_record["traceEvents"])

        # assign an index to each event
        df.reset_index(inplace=True)
        df["index"] = pd.to_numeric(df["index"], downcast="integer")

        # add fwd bwd links between CPU ops
        add_fwd_bwd_links(df)

        df, local_symbol_table = compress_df(df, cfg)
        transform_correlation_to_index(df)
        add_iteration(df, local_symbol_table)

    t_end = time.perf_counter()
    logger.debug(
        f"Parsed {trace_file_path} in {(t_end - t_start):.2f} seconds; current PID:{os. getpid()}"
    )
    return meta, df, local_symbol_table


def add_fwd_bwd_links(df: pd.DataFrame) -> None:
    df_fwdbwd = df.loc[df.cat.eq("fwdbwd")]
    # Check if fwdbwd events are present
    if df_fwdbwd.empty:
        return
    df_fwdbwd_start = df_fwdbwd.query("ph == 's'")[["ts", "id", "pid", "tid"]]
    df_fwdbwd_end = df_fwdbwd.query("ph == 'f' and bp == 'e'")[
        ["ts", "id", "pid", "tid"]
    ]

    df_cpu = df.loc[df.cat.eq("cpu_op")][["index", "ts"]]
    df_fwdbwd_start_events = df_fwdbwd_start.merge(df_cpu, how="inner", on="ts")
    df_fwdbwd_start_events.rename(columns={"index": "index_start"}, inplace=True)

    df_fwdbwd_end_events = df_fwdbwd_end.merge(df_cpu, how="inner", on="ts")
    df_fwdbwd_end_events.rename(columns={"index": "index_end"}, inplace=True)

    if df_fwdbwd_start_events.empty or df_fwdbwd_end_events.empty:
        return

    df_fwdbwd_start_events = pd.concat(
        g.sort_index().tail(1) for _, g in df_fwdbwd_start_events.groupby("id")
    )
    df_fwdbwd_end_events = pd.concat(
        g.sort_index().tail(1) for _, g in df_fwdbwd_end_events.groupby("id")
    )

    df_fwdbwd_merge = df_fwdbwd_start_events.merge(
        df_fwdbwd_end_events, how="inner", on="id", suffixes=("_start", "_end")
    )

    fwdbwd_start_to_end_map = {}
    fwdbwd_end_to_start_map = {}
    for _, entry in df_fwdbwd_merge.iterrows():
        start_tuple = (
            entry["ts_start"],
            entry["pid_start"],
            entry["tid_start"],
            entry["index_start"],
        )
        end_tuple = (
            entry["ts_end"],
            entry["pid_end"],
            entry["tid_end"],
            entry["index_end"],
        )

        fwdbwd_start_to_end_map[start_tuple] = entry["index_end"]
        fwdbwd_end_to_start_map[end_tuple] = entry["index_start"]

    def _set_fwdbwd_index(row):
        if row.get("cat") != "cpu_op":
            return -1
        key = (row.ts, row.pid, row.tid, row.get("index"))
        # print(key)
        if key in fwdbwd_start_to_end_map:
            return fwdbwd_start_to_end_map[key]
        elif key in fwdbwd_end_to_start_map:
            return fwdbwd_end_to_start_map[key]
        else:
            return -1

    def _set_fwd_or_bwd(row):
        if row.get("cat") != "cpu_op":
            return -1
        key = (row.ts, row.pid, row.tid, row.get("index"))
        if key in fwdbwd_start_to_end_map:
            return 0
        elif key in fwdbwd_end_to_start_map:
            return 1
        else:
            return -1

    df["fwdbwd_index"] = df.apply(_set_fwdbwd_index, axis=1)
    df["fwdbwd"] = df.apply(_set_fwd_or_bwd, axis=1)


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


class Trace:
    """
    A container for the traces collected for a distributed ML training job.

    An ML training job can have multiple trace collections. Each of those trace collections maps to
    one Trace object.


    Attributes:
        trace_path (str) : the path to the folder where the collected raw traces are stored. In other words,
            `trace_path = normalize_path(base_trace_dir)`.
        trace_files (Dict[int, str]) : a dictionary that maps the rank of a job's trainer to its trace file.
        traces (Dict[int, pd.DataFrame) : a dictionary that maps the rank of a job's trainer to its trace data.
        meta_data (Dict[int, MetaData) : a dictionary that maps the rank of a job's trainer to its meta_ata.
        symbol_table (TraceSymbolTable) : a symbol table used to encode the symbols in the trace.
        is_parsed (bool) : a flag indicting whether the trace is parsed or not.
    """

    def __init__(
        self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
    ) -> None:
        """
        The constructor of a Trace object.
        Args:
            trace_files: Dict[int, str] : a map from rank to trace file names. The trace file names can be either
                relative to the path `trace_path` or absolute file paths.
            trace_dir (str) : a path used to derive `trace_path = normalize_path(trace_dir)`.

        Raises:
            AssertionError
        """
        self.trace_path: str = normalize_path(trace_dir)
        logger.info(f"{self.trace_path}")
        self.trace_files: Dict[int, str] = (
            trace_files if trace_files is not None else get_trace_files(self.trace_path)
        )
        logger.debug(self.trace_files)
        self.traces: Dict[int, pd.DataFrame] = {}
        self.symbol_table = TraceSymbolTable()
        self.meta_data: Dict[int, MetaData] = {}
        self.min_ts: int = 0

        self._normalize_trace_filenames()
        assert self._validate_trace_files()
        logger.debug(f"trace_path={self.trace_path}")
        logger.debug(f"# trace_files={len(self.trace_files)}")
        if len(self.trace_files) > 0:
            rank = next(iter(self.trace_files))
            trace_file = self.trace_files[rank]
            logger.debug(f"trace_files[{rank}] = {trace_file}")
        self.is_parsed = False

    def load_traces(self, include_last_profiler_step: Optional[bool] = False) -> None:
        if self.is_parsed:
            logger.warning("Traces are already parsed and loaded!")
            return
        self.parse_traces()
        self.align_and_filter_trace(include_last_profiler_step)
        for rank, df in self.traces.items():
            df = self.traces[rank].set_index("index", drop=False)
            df.index.names = [None]
            self.traces[rank] = df
        self.is_parsed = True

    def parse_single_rank(self, rank: int) -> None:
        """
        Parse the trace for a given rank.

        Args:
            rank (int) : an integer representing the rank of a trainer
        """
        if rank in self.trace_files:
            trace_filepath = self.trace_files[rank]
            (
                self.meta_data[rank],
                self.traces[rank],
                local_symbol_table,
            ) = parse_trace_dataframe(trace_filepath)
            # update the global symbol table
            self.symbol_table.add_symbols(local_symbol_table.get_sym_table())
            # fix the encoding of the data frame
            local_table = local_symbol_table.get_sym_table()
            global_map = self.symbol_table.get_sym_id_map()
            for col in ["cat", "name"]:
                self.traces[rank][col] = self.traces[rank][col].apply(
                    lambda idx: global_map[local_table[idx]]
                )

    def parse_multiple_ranks(
        self, ranks: List[int], use_multiprocessing: bool = True
    ) -> None:
        """
        Parse the trace for a given rank.

        Args:
            ranks (List[int]) : a list of integers representing the ranks of multiple trainers.
            use_multiprocessing (bool) : whether the parser using multiprocessing or not.
        """
        logger.debug(
            f"entering {sys._getframe().f_code.co_name}(ranks={ranks}, use_multiprocessing={use_multiprocessing})"
        )
        t0 = time.perf_counter()
        trace_paths = [self.trace_files[rank] for rank in ranks]
        logger.debug(f"trace_path={trace_paths}")
        local_symbol_tables: Dict[int, TraceSymbolTable] = {}
        if not use_multiprocessing:
            for rank in ranks:
                logger.debug(f"parsing trace for rank-{rank}")
                result = parse_trace_dataframe(self.trace_files[rank])
                self.meta_data[rank], self.traces[rank], local_symbol_tables[rank] = (
                    result[0],
                    result[1],
                    result[2],
                )
                self.symbol_table.add_symbols(local_symbol_tables[rank].get_sym_table())
            logger.debug(f"finished parsing for all {len(ranks)} ranks")
        else:
            num_procs = min(mp.cpu_count(), len(ranks))
            if len(ranks) <= 8:
                num_procs = min(len(ranks), mp.cpu_count())
            else:
                tracemalloc.start()
                parse_trace_dataframe(trace_paths[0])
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                num_procs = get_mp_pool_size(peak, len(ranks))
            logger.debug(f"using {num_procs} processes for parsing.")

            with mp.get_context("fork").Pool(num_procs) as pool:
                results = pool.map(parse_trace_dataframe, trace_paths)
                pool.close()
                pool.join()
            logger.debug(f"finished parallel parsing using {num_procs} processes.")

            # collect the results
            for rank, result in zip(ranks, results):
                self.meta_data[rank], self.traces[rank], local_symbol_tables[rank] = (
                    result[0],
                    result[1],
                    result[2],
                )
                self.symbol_table.add_symbols(local_symbol_tables[rank].get_sym_table())

        # Now we update the IDs in the Dataframe using the global symbols table.
        global_map = self.symbol_table.get_sym_id_map()
        for rank in ranks:
            local_table = local_symbol_tables[rank].get_sym_table()
            for col in ["cat", "name"]:
                self.traces[rank][col] = self.traces[rank][col].apply(
                    lambda idx: global_map[local_table[idx]]
                )

        t1 = time.perf_counter()
        logger.debug(
            f"leaving {sys._getframe().f_code.co_name} duration={t1 - t0:.2f} seconds"
        )

    def parse_traces(
        self,
        max_ranks: int = -1,
        use_multiprocessing: bool = True,
    ) -> None:
        """
        Parse up to <max_rank> traces.

        Args:
            max_ranks (int): how many rank's traces to parse. Default value `-1` implies parsing all ranks.
            use_multiprocessing (bool) : whether the parser using multiprocessing or not.

        Effects:
            This function will parse the traces and save the parsed data into `self.traces`.
        """
        logger.debug(
            f"entering {sys._getframe().f_code.co_name}(max_ranks={max_ranks}, use_multiprocessing={use_multiprocessing})"
        )
        t0 = time.perf_counter()
        max_ranks = len(self.trace_files) if max_ranks == -1 else max_ranks
        ranks = sorted(self.trace_files.keys())[:max_ranks]
        logger.info(f"ranks={ranks}")

        if len(ranks) > 0:
            self.parse_multiple_ranks(ranks, use_multiprocessing and len(ranks) > 1)
            self.is_parsed = True
        else:
            logger.error("The list of ranks to be parsed is empty.")
            self.is_parsed = False
        t1 = time.perf_counter()
        logger.debug(
            f"leaving {sys._getframe().f_code.co_name} duration={t1 - t0:.2f} seconds"
        )

    def align_and_filter_trace(
        self, include_last_profiler_step: Optional[bool] = False
    ) -> None:
        """
        Align the starting time across multiple ranks and filter events that belong to incomplete iterations.
        """
        self._align_all_ranks()
        self._filter_irrelevant_gpu_kernels(include_last_profiler_step)

    def get_ranks(self) -> List[int]:
        """Get the list of (sorted) ranks included in this trace."""
        return sorted(self.traces.keys())

    def get_iterations(self, rank: Optional[int] = None) -> List[int]:
        """Get the list of iterations for a given rank.

        Args:
            rank (Optional[int]): a rank
                when rank is None, use the first item of the list returned by self.get_ranks().

        Returns:
            A list of iteration IDs for the given rank.
            Return an empty list when the rank is invalid or column "iteration" does not exists.
        """
        if rank is None:
            _ranks = self.get_ranks()
            rank = _ranks[0] if len(_ranks) > 0 else -1

        if rank in self.get_ranks():
            df = self.traces[rank]
            if "iteration" in df.columns:
                return sorted([i for i in df["iteration"].unique() if i >= 0])
        return []

    def get_trace(self, rank: int) -> pd.DataFrame:
        """
        Get the trace for a given rank.

        Args:
            rank (int) : the rank of the trainer whose trace is to be returned.

        Returns:
            The DataFrame for the given rank.

        Raises:
            ValueError when this Trace object doesn't have trace for the given rank.
        """
        if rank not in self.traces:
            logger.error(f"get_rank_trace - no trace for rank {rank}")
            raise ValueError
        return self.traces[rank]

    def get_all_traces(self) -> Dict[int, pd.DataFrame]:
        """
        Get the traces of all ranks.
        Returns:
            A dictionary with rank as key and its trace test_data as value.
        """
        return self.traces

    def get_raw_trace_for_one_rank(self, rank: int = 0) -> Dict[str, Any]:
        """
        Get raw trace content for one rank without filtering or compression to support writing back
        to a trace file.

        Args:
            rank (int) : the rank of the trainer whose trace is to be returned.

        Returns:
            The raw content of the trace file of the given rank.

        Raises:
            ValueError when this Trace object doesn't have trace for the given rank.
        """
        if rank not in self.trace_files:
            logger.error(f"get_rank_trace - no trace for rank {rank}")
            raise ValueError
        trace_filepath = self.trace_files[rank]
        return parse_trace_dict(trace_filepath)

    def write_raw_trace(self, output_file: str, trace_contents: Dict[str, Any]) -> None:
        with gzip.open(output_file, "wt") as fp:
            json.dump(trace_contents, fp, indent=2)

    def _normalize_trace_filenames(self) -> None:
        """
        Normalize the trace filenames so that a rank's trace file can be located by self.trace_files[rank] itself.
        """
        if not isinstance(self.trace_files, dict):
            logger.error(
                f"self.trace_files must be of type Dict[int, str]; got {type(self.trace_files)}"
            )
            raise ValueError(
                f"Expected trace_files to be Dict[int, str]; got {type(self.trace_files)}"
            )

        for rank in self.trace_files:
            filename = self.trace_files[rank]
            if not (filename.startswith(self.trace_path) or filename.startswith("/")):
                self.trace_files[rank] = os.path.join(self.trace_path, filename)

    def _validate_trace_files(self) -> bool:
        """
        Validate whether all trace files exist and are valid.

        Returns:
            success (bool) : True if all trace files in self.trace_files exist and are valid; False otherwise.
        """
        for rank, filepath in self.trace_files.items():
            if not (os.path.exists(filepath) and os.access(filepath, os.R_OK)):
                logger.error(
                    f"Trace file '{filepath}' doesn't exist or is not readable."
                )
                return False
            if not (filepath.endswith(".gz") or filepath.endswith(".json")):
                logger.error(
                    f"The postfix of trace file '{filepath}' is neither '.gz' or '.json'"
                )
                return False
        return True

    def _align_all_ranks(self) -> None:
        """
        Align dataframes for all ranks such that the earliest event starts at time 0.
        """
        self.min_ts = min(trace_df["ts"].min() for trace_df in self.traces.values())
        for rank, trace_df in self.traces.items():
            trace_df["ts"] = trace_df["ts"] - self.min_ts
            self.traces[rank] = trace_df

    def _filter_irrelevant_gpu_kernels(
        self, include_last_profiler_step: Optional[bool] = False
    ) -> None:
        """
        Filter out GPU kernels that are not launched by the CPU kernels in the traced iterations.
        """
        sym_index = self.symbol_table.get_sym_id_map()
        profiler_steps = [v for k, v in sym_index.items() if "ProfilerStep" in k]

        def filter_gpu_kernels_for_one_rank(trace_df: pd.DataFrame) -> pd.DataFrame:
            cpu_kernels = trace_df[trace_df["stream"].eq(-1)]
            gpu_kernels = trace_df[trace_df["stream"].ne(-1)]
            last_profiler_start = cpu_kernels[cpu_kernels["name"].isin(profiler_steps)][
                "ts"
            ].max()
            if include_last_profiler_step:
                cpu_kernels = cpu_kernels[cpu_kernels["ts"] <= last_profiler_start]
            else:
                cpu_kernels = cpu_kernels[cpu_kernels["ts"] < last_profiler_start]
            filtered_gpu_kernels = gpu_kernels.merge(
                cpu_kernels["correlation"], on="correlation", how="inner"
            )
            return pd.concat([filtered_gpu_kernels, cpu_kernels], axis=0)

        if not profiler_steps:
            logger.warning(
                "ProfilerStep not found in the trace. The analysis result may not be accurate."
            )
        elif len(profiler_steps) == 1:
            logger.warning(
                "There is only one iteration in the trace. The analysis result may not be accurate."
            )
        else:
            for rank, trace_df in self.traces.items():
                self.traces[rank] = filter_gpu_kernels_for_one_rank(trace_df)

    def decode_symbol_ids(self, use_shorten_name: bool = True) -> None:
        """Decode the name and cat column to show the original string names.

        Args:
            use_shorten_name (bool): shorten the long strings to make it easy to read.
                Default: True.
        """

        for rank in self.traces:
            decode_symbol_id_to_symbol_name(
                self.get_trace(rank), self.symbol_table, use_shorten_name
            )

    def convert_time_series_to_events(
        self, series: pd.DataFrame, counter_name: str, counter_col: str
    ) -> List[Dict[str, Any]]:
        """
        Takes time series data and convert it to Counter event format as per
        the Chrome Trace Format.
        See - https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.msg3086636uq

        @args
            series: is a dataframe with a minimum of pid, tid, ts.
                    (the id field is optional but use it to distinguish series with same name)
            counter_name (str): name of the counter.
            counter_col (str): name of the column used as the counter itself.

        Returns a list of json events that can be appended to the trace.
        """
        required_columns = ["pid", "ts", counter_col]
        if not set(required_columns).issubset(series.columns):
            logger.warning(
                "Time series dataframe does NOT contain required columns "
                f"{required_columns}, columns contained = {series.columns}"
            )
            return []

        events_df = series[required_columns].rename(columns={counter_col: "args"})
        events_df["ph"] = PHASE_COUNTER

        if "name" in series.columns:
            events_df["name"] = series["name"]
        else:
            events_df["name"] = counter_name
        if "id" in series.columns:
            events_df["id"] = series["id"]

        # add back ts delta
        events_df["ts"] = events_df["ts"] + self.min_ts

        def convert_to_args(value: int):
            return {counter_name: value}

        events_df.args = events_df.args.apply(convert_to_args)

        return events_df.to_dict("records")

    @staticmethod
    def flow_event(
        id: int,
        pid: int,
        tid: int,
        ts: int,
        is_start: bool,
        name: str,
        cat: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = {
            "ph": PHASE_FLOW_START if is_start else PHASE_FLOW_END,
            "id": id,
            "pid": pid,
            "tid": tid,
            "ts": ts,
            "cat": cat,
            "name": name,
        }
        if args is not None:
            res["args"] = args
        if not is_start:
            res["bp"] = "e"
        return res
