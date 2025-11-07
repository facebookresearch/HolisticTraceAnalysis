# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import re
import sys
import time
import tracemalloc
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

import pandas as pd

from hta.common.singletrace import Trace
from hta.common.trace_file import create_rank_to_trace_dict, get_trace_files
from hta.common.trace_filter import CPUOperatorFilter, GPUKernelFilter
from hta.common.trace_parser import parse_trace_dataframe, parse_trace_dict
from hta.common.trace_symbol_table import (
    decode_symbol_id_to_symbol_name,
    TraceSymbolTable,
)
from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR
from hta.configs.parser_config import ParserConfig
from hta.utils.utils import get_mp_pool_size, normalize_path

MetaData = Dict[str, Any]
PHASE_COUNTER: str = "C"
PHASE_FLOW_START: str = "s"
PHASE_FLOW_END: str = "f"


def trace_event_timestamp_to_unixtime_ns(
    trace_event_ts_us: float, trace_metadata: MetaData
) -> int:
    """
    Utility to convert a trace event timestamp (us) to unix time (ns).

    Returns: Unixtime (nanoseconds) of trace event (int)

    Raises: KeyError when baseTimeNanoseconds is not found in trace metadata.
    """
    if (
        base_time_nanoseconds := trace_metadata.get("baseTimeNanoseconds", None)
    ) is None:
        err_msg: str = (
            "'baseTimeNanoseconds' is not found in the trace metadata. Unable to convert trace event timestamp to unix time."
        )
        logger.warning(err_msg)
        raise KeyError(err_msg)
    # Convert to int to avoid loss of precision with float
    trace_event_ts_ns = int(trace_event_ts_us * 1e3)
    return trace_event_ts_ns + base_time_nanoseconds


def transform_correlation_to_index(trace: Trace) -> pd.DataFrame:
    """Transform correlation to index_correlation and add a index_correlation column to trace's df.

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

    Given the following input trace's DataFrame:

    | index | stream | cat | s_cat  | correlation |
    ===============================================
    | 675   | 7      | 248 | Kernel | 278204204   |
    | 677   | -1     |   9 | Runtime| 278204204   |

    After calling `transform_correlation_to_index(df)`, trace's df will become:

    | index | stream | cat | s_cat  | correlation |  index_correlation |
    ====================================================================
    | 675   | 7      | 248 | Kernel | 278204204   |       677          |
    | 677   | -1     |   9 | Runtime| 278204204   |       675          |

    This function changes the input trace's DataFrame by adding a new column `index_correlation`.

    Example use: find the kernels of all Cuda Launch
    cuda_launch_events = df.loc[df['cat'].eq(9) & df['index_correlation'].gt(0)]
    cuda_kernel_indices = cuda_launch_events["index_correlation"]
    df.loc[cuda_kernel_indices]

    Args:
        trace (Trace): the input Trace object, containing both the DataFrame and the symbol table.

    Returns:
        pd.DataFrame: the transformed DataFrame with a index_correlation column.

    Affects:
        This function adds a index_correlation column to trace's df.
    """
    df = trace.df

    if "correlation" not in df.columns:
        return df

    # Initialize the index_correlaion to the fallback value first
    df["index_correlation"] = np.minimum(df["correlation"], 0)
    corr_df = df.loc[
        df["correlation"].ne(-1), ["index", "correlation", "stream", "name"]
    ]

    on_cpu = CPUOperatorFilter()(corr_df, trace.symbol_table)
    on_gpu = GPUKernelFilter()(corr_df, trace.symbol_table)

    # We only need to merge once.
    # index_x --> index_y will be cpu to gpu mapping
    # index_y --> index_x will be gpu to cpu mapping
    merged = on_cpu.merge(on_gpu, on="correlation", how="inner")
    df.loc[merged["index_x"], "index_correlation"] = merged["index_y"].values
    df.loc[merged["index_y"], "index_correlation"] = merged["index_x"].values
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


def add_iteration(trace: Trace) -> pd.DataFrame:
    """
    Add an iteration column to the trace's DataFrame.

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
        trace (Trace): a Trace object containing both the DataFrame and the symbol table.

    Returns:
        A DataFrame with the profiler steps information.

    Note:
        This function will change the input trace's DataFrame by adding the iteration number column.
    """
    df = trace.df

    s_map = pd.Series(trace.symbol_table.sym_index)
    s_tab = pd.Series(trace.symbol_table.sym_table)
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
    # should catch host side
    df.loc[df["stream"].lt(0), "iteration"] = df["ts"].apply(_get_profiler_step)

    # get iteration for cpu ops that include stream data
    # triton ops potentially arrive with stream = 0
    if "cpu_op" in s_map.index:
        cpu_op_id = s_map.at["cpu_op"]
        # triton comes in with stream information but no index correlation
        df.loc[df["stream"].eq(0) & df["cat"].eq(cpu_op_id), "iteration"] = df[
            "ts"
        ].apply(_get_profiler_step)

    df.loc[df["stream"].gt(0), "iteration"] = df["index_correlation"].apply(
        lambda x: df.loc[x, "iteration"] if x > 0 else -1
    )
    df["iteration"] = pd.to_numeric(df["iteration"], downcast="integer")

    return profiler_steps


def parse_trace_file(trace_file_path: str, cfg: Optional[ParserConfig] = None) -> Trace:
    """parse a single trace file into a trace (Trace) object.
    Args:
        trace_file_path (str): The path to a trace file. When the trace_file is a relative path.
            This method combines the object's trace_path with trace_file to get the full path of the trace file.
        cfg (ParserConfig, Optional): A ParserConfig object controls how to parse the trace file.
    Returns:
        Trace object that contains:
            Trace's metadata.
            DataFrame representation of the trace's events.
            Symbol table to encode the symbols of the trace.

    Raises:
        OSError when the trace file doesn't exist or current process has no permission to access it.
        JSONDecodeError when the trace file is not a valid JSON document.
        ValueError when the trace_file doesn't end with ".gz" or "json".
        ValueError if parser config passes invalid parser backend.
    """
    if not (trace_file_path.endswith(".gz") or trace_file_path.endswith(".json")):
        raise ValueError(
            f"expect the value of trace_file ({trace_file_path}) ends with '.gz' or 'json'"
        )

    t_start = time.perf_counter()
    cfg = cfg or ParserConfig.get_default_cfg()

    trace: Trace = parse_trace_dataframe(trace_file_path, cfg)

    # add fwd bwd links between CPU ops
    add_fwd_bwd_links(trace.df)
    trace.df = transform_correlation_to_index(trace)
    add_iteration(trace)

    trace.df["end"] = trace.df["ts"] + trace.df["dur"]

    t_end = time.perf_counter()
    logger.warning(
        f"Overall parsing of {trace_file_path} in {(t_end - t_start):.2f} seconds; current PID:{os. getpid()}"
    )

    return trace


class _TraceFileParserWrapper:
    """A wrapper class for the parse_trace_file method."""

    def __init__(self, cfg: ParserConfig) -> None:
        self.cfg = cfg

    def __call__(self, trace_file: str) -> Trace:
        return parse_trace_file(trace_file, self.cfg)


def add_fwd_bwd_links(df: pd.DataFrame) -> None:
    t0 = time.perf_counter()
    if df.cat.eq("fwdbwd").sum() == 0:
        return

    # Initialize the fwdbwd columns to -1
    df["fwdbwd_index"] = -1
    df["fwdbwd"] = -1
    df["key"] = list(zip(df["ts"], df["tid"], df["pid"]))

    # Get the fwdbwd events. Only the "id" and "key" columns are needed for merging.
    df_fwdbwd = df.loc[df.cat.eq("fwdbwd")]
    df_fwdbwd_start = df_fwdbwd.query("ph == 's'")[["id", "key"]]
    df_fwdbwd_end = df_fwdbwd.query("ph == 'f' and bp == 'e'")[["id", "key"]]

    # The "index" column for the cpu event will be used when merging with the fwdbwd events.
    # The "key" column will be used for the merge.
    df_cpu = df.loc[df.cat.eq("cpu_op")][["index", "key"]]

    # Merge the fwdbwd events with the cpu events.
    # We will be using the index of last cpu event when multiple cpu events start from the same ts.
    df_fwdbwd_start_events = (
        df_fwdbwd_start.merge(df_cpu, how="inner", on="key")[["index", "id"]]
        .groupby("id")
        .max()
    )
    df_fwdbwd_end_events = (
        df_fwdbwd_end.merge(df_cpu, how="inner", on="key")[["index", "id"]]
        .groupby("id")
        .max()
    )
    if df_fwdbwd_start_events.empty or df_fwdbwd_end_events.empty:
        return

    # Merge the start and end events based on the "id" column.
    df_fwdbwd_merged = df_fwdbwd_start_events.merge(
        df_fwdbwd_end_events, how="inner", on="id", suffixes=("_start", "_end")
    )

    start_indices = df_fwdbwd_merged["index_start"]
    end_indices = df_fwdbwd_merged["index_end"]

    # Add the fwdbwd_index and fwdbwd columns to the dataframe.
    df.loc[start_indices, "fwdbwd_index"] = end_indices.values
    df.loc[end_indices, "fwdbwd_index"] = start_indices.values
    df.loc[start_indices, "fwdbwd"] = 0
    df.loc[end_indices, "fwdbwd"] = 1
    df.drop(columns=["key"], inplace=True)
    t1 = time.perf_counter()
    logger.debug(f"Time taken to add fwd_bwd links: {t1 - t0 :.2f} seconds")


class TraceCollection:
    """
    A container for the traces collected for a distributed ML training job.

    An ML training job can have multiple trace collections. Each of those trace collections maps to
    one TraceCollection object.


    Attributes:
        trace_path (str) : the path to the folder where the collected raw traces are stored. In other words,
            `trace_path = normalize_path(base_trace_dir)`.
        trace_files (Dict[int, str]) : a dictionary that maps the rank of a job's trainer to its trace file.
        traces (Dict[int, pd.DataFrame]) : a dictionary that maps the rank of a job's trainer to its trace data.
        meta_data (Dict[int, MetaData]) : a dictionary that maps the rank of a job's trainer to its meta_data.
        symbol_table (TraceSymbolTable) : a symbol table used to encode the symbols in the trace.
        is_parsed (bool) : a flag indicting whether the trace is parsed or not.
        parser_config (ParserConfig) : a configuration object for customizing the parser.
    """

    def __init__(
        self,
        trace_files: Optional[Union[List[str], Dict[int, str]]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        parser_config: Optional[ParserConfig] = None,
    ) -> None:
        """
        The constructor of a TraceCollection object.
        Args:
            trace_files: Optional[Union[List[str], Dict[int, str]]]: either a list of trace file names or a map from rank to trace file names.
                When a list is provided, HTA will infer the ranks by reading the trace file metadata.
                The trace file names can be either relative to the path `trace_path` or absolute file paths.
            trace_dir (str) : a path used to derive `trace_path = normalize_path(trace_dir)`.
            parser_config (ParserConfig) : a configuration object for customizing the
                parser. Default value is None.

        Raises:
            AssertionError
        """
        self.is_parsed: bool = False
        self.trace_path: str = normalize_path(trace_dir)
        self.parser_config: ParserConfig = (
            parser_config or ParserConfig.get_default_cfg()
        )

        logger.info(f"{self.trace_path}")
        self.trace_files: Dict[int, str]
        if trace_files is None:
            self.trace_files = get_trace_files(self.trace_path)
        elif isinstance(trace_files, dict):
            self.trace_files = trace_files
        elif isinstance(trace_files, list):
            ok, self.trace_files = create_rank_to_trace_dict(trace_files)
            if not ok:
                logger.warning("failed to create rank to trace map")
        else:
            logger.error(
                f"Unsupported type for trace_files = {trace_files}, should be list or dict"
            )
            return

        logger.debug(self.trace_files)
        self.traces: Dict[int, Trace] = {}
        self.symbol_table = TraceSymbolTable()
        self.meta_data: Dict[int, MetaData] = {}
        self.min_ts: int = 0

        self._normalize_trace_filenames()
        if not self._validate_trace_files():
            raise ValueError("Trace files validation failed.")
        logger.debug(f"trace_path={self.trace_path}")
        logger.debug(f"# trace_files={len(self.trace_files)}")
        if len(self.trace_files) > 0:
            rank = next(iter(self.trace_files))
            trace_file = self.trace_files[rank]
            logger.debug(f"trace_files[{rank}] = {trace_file}")

    def load_traces(
        self,
        include_last_profiler_step: Optional[bool] = False,
        use_multiprocessing: bool = True,
        use_memory_profiling: bool = True,
    ) -> None:
        if self.is_parsed:
            logger.warning("Traces are already parsed and loaded!")
            return
        self.parse_traces(
            use_multiprocessing=use_multiprocessing,
            use_memory_profiling=use_memory_profiling,
        )
        self.align_and_filter_trace(include_last_profiler_step)
        for trace in self.traces.values():
            df = trace.df
            df = df.set_index("index", drop=False)
            df.index.names = [None]
            trace.df = df
        self.is_parsed = True

    def parse_single_rank(self, rank: int) -> None:
        """
        Parse the trace for a given rank.

        Args:
            rank (int) : an integer representing the rank of a trainer
        """
        if rank in self.trace_files:
            trace_filepath = self.trace_files[rank]
            trace = parse_trace_file(trace_filepath, self.parser_config)
            # update the global symbol table
            local_symbol_table = trace.get_sym_table()
            self.symbol_table.add_symbols(local_symbol_table)
            # fix the encoding of the data frame
            global_map = self.symbol_table.get_sym_id_map()
            for col in ["cat", "name"]:
                trace.df[col] = trace.df[col].apply(
                    lambda idx: global_map[local_symbol_table[idx]]
                )
            self.traces[rank] = trace

    def parse_multiple_ranks(
        self,
        ranks: List[int],
        use_multiprocessing: bool = True,
        use_memory_profiling: bool = True,
    ) -> None:
        """
        Parse the trace for a given rank.

        Args:
            ranks (List[int]) : a list of integers representing the ranks of multiple trainers.
            use_multiprocessing (bool) : whether the parser using multiprocessing or not.
            use_memory_profiling (bool): whether the parser memory profiles or not.
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
                trace = parse_trace_file(self.trace_files[rank], self.parser_config)
                self.traces[rank] = trace
                self.symbol_table.add_symbols(trace.get_sym_table())
            logger.debug(f"finished parsing for all {len(ranks)} ranks")
        else:
            num_procs = min(mp.cpu_count(), len(ranks))
            if len(ranks) <= 8:
                num_procs = min(len(ranks), mp.cpu_count())
            elif use_memory_profiling:
                tracemalloc.start()
                parse_trace_file(trace_paths[0], self.parser_config)
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                num_procs = get_mp_pool_size(peak, len(ranks))
            logger.info(f"using {num_procs} processes for parsing.")

            _parser = _TraceFileParserWrapper(self.parser_config)
            with mp.get_context("fork").Pool(num_procs) as pool:
                trace_list = pool.map(_parser, trace_paths, chunksize=1)
            logger.debug(f"finished parallel parsing using {num_procs} processes.")

            # collect the results
            for rank, trace in zip(ranks, trace_list):
                self.traces[rank] = trace
                self.symbol_table.add_symbols(trace.get_sym_table())

        # Now we update the IDs in the Dataframe using the global symbols table.
        global_map = self.symbol_table.get_sym_id_map()
        for trace in self.traces.values():
            local_table = trace.get_sym_table()
            for col in ["cat", "name"]:
                trace.df[col] = trace.df[col].apply(
                    lambda idx: global_map[local_table[idx]]
                )

        t1 = time.perf_counter()
        logger.warning(
            f"leaving {sys._getframe().f_code.co_name} duration={t1 - t0:.2f} seconds"
        )

    def parse_traces(
        self,
        max_ranks: int = -1,
        use_multiprocessing: bool = True,
        use_memory_profiling: bool = True,
    ) -> None:
        """
        Parse up to <max_rank> traces.

        Args:
            max_ranks (int): how many rank's traces to parse. Default value `-1` implies parsing all ranks.
            use_multiprocessing (bool) : whether the parser using multiprocessing or not.
            use_memory_profiling (bool): whether the parser memory profiles or not.
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
            use_multiprocessing = use_multiprocessing and len(ranks) > 1
            self.parse_multiple_ranks(ranks, use_multiprocessing, use_memory_profiling)

            self.is_parsed = True
        else:
            logger.error("The list of ranks to be parsed is empty.")
            self.is_parsed = False
        t1 = time.perf_counter()
        logger.warning(
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

    def _get_first_rank(self, rank: Optional[int] = None) -> int:
        if rank is None:
            _ranks = self.get_ranks()
            rank = _ranks[0] if len(_ranks) > 0 else -1
        return rank

    def get_iterations(self, rank: Optional[int] = None) -> List[int]:
        """Get the list of iterations for a given rank.

        Args:
            rank (Optional[int]): a rank
                when rank is None, use the first item of the list returned by self.get_ranks().

        Returns:
            A list of iteration IDs for the given rank.
            Return an empty list when the rank is invalid or column "iteration" does not exists.
        """
        rank = self._get_first_rank(rank)

        if rank in self.get_ranks():
            df = self.get_trace_df(rank)
            if "iteration" in df.columns:
                return sorted([i for i in df["iteration"].unique() if i >= 0])
        return []

    def get_trace_df(self, rank: int) -> pd.DataFrame:
        """
        Get the trace's DataFrame for a given rank.

        Args:
            rank (int) : the rank of the trainer.

        Returns:
            The trace's DataFrame for the given rank.

        Raises:
            ValueError when this TraceCollection object doesn't have trace for the given rank.
        """
        if rank not in self.traces:
            logger.error(f"get_rank_trace - no trace for rank {rank}")
            raise ValueError

        return self.traces[rank].df

    def get_trace_duration(self, rank: Optional[int] = None) -> int:
        """Get the duration of specified rank.

        Args:
            rank (Optional[int]): a rank
                when rank is None, use the first item of the list returned by self.get_ranks().

        Returns: duration of trace (int)
        """
        rank = self._get_first_rank(rank)
        trace_df = self.get_trace_df(rank)

        return trace_df.ts.max() - trace_df.ts.min()

    def get_trace(self, rank: int) -> Trace:
        """
        Get the trace for a given rank.

        Args:
            rank (int) : the rank of the trainer whose trace is to be returned.

        Returns:
            The trace for the given rank.

        Raises:
            ValueError when this TraceCollection object doesn't have trace for the given rank.
        """
        if rank not in self.traces:
            logger.error(f"get_rank_trace - no trace for rank {rank}")
            raise ValueError
        return self.traces[rank]

    def get_all_traces(self) -> Dict[int, Trace]:
        """
        Get the traces of all ranks.
        Returns:
            A dictionary with rank as key and its trace as value.
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
            ValueError when this TraceCollection object doesn't have trace for the given rank.
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
        for _, filepath in self.trace_files.items():
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
        self.min_ts = min(trace.df["ts"].min() for trace in self.traces.values())
        for _, trace in self.traces.items():
            trace.df["ts"] = trace.df["ts"] - self.min_ts

    def _fix_mtia_memory_kernels(self, trace_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix the iteration and stream columns for memory kernels.

        In current MTIA traces, the iteration and stream columns are not set correctly for memory kernels.
        This function fixes the iteration and stream columns for memory kernels.

        Args:
            trace_df (pd.DataFrame): the DataFrame to be fixed.

        Returns:
            The DataFrame with fixed iteration and stream columns for memory kernels.

        Deprecated:
            This method is deprecated and will be removed in a future version.
        """
        warnings.warn(
            "_fix_mtia_memory_kernels is deprecated and will be removed in a future version. Your traces should no longer contain this fault to need a fix.",
            DeprecationWarning,
            stacklevel=2,
        )
        profiler_step_ids: List[int] = self.symbol_table.get_profiler_step_ids()
        memory_name_ids: List[int] = self.symbol_table.get_memory_name_ids()

        profiler_steps = trace_df[trace_df["name"].isin(profiler_step_ids)]
        memory_kernels = trace_df[trace_df["name"].isin(memory_name_ids)]

        fixed_indices = set()
        for step in profiler_steps.itertuples():
            within_time_range = memory_kernels["ts"].between(
                step.ts, step.ts + step.dur
            )
            memory_kernels_subset = memory_kernels[within_time_range]
            # Fix iteration column
            mask_iteration = memory_kernels_subset["iteration"] == -1
            trace_df.loc[memory_kernels_subset.index[mask_iteration], "iteration"] = (
                step.iteration
            )

            # Fix stream column
            mask_stream = memory_kernels_subset["stream"] == -1
            if mask_stream.any():
                # Convert stream column to int32 if it is a smaller value (assumes 64\bit max)
                if trace_df["stream"].dtype not in ("int32", "int64"):
                    trace_df["stream"] = trace_df["stream"].astype("int32")

                # Now assign tid values to stream
                trace_df.loc[memory_kernels_subset.index[mask_stream], "stream"] = (
                    trace_df.loc[
                        memory_kernels_subset.index[mask_stream], "tid"
                    ].astype("int32")
                )

            # Track modified indices
            fixed_indices.update(memory_kernels_subset.index)
        return trace_df.loc[list(fixed_indices)]

    def _filter_irrelevant_gpu_kernels(
        self, include_last_profiler_step: Optional[bool] = False
    ) -> None:
        """
        Filter out GPU kernels that are not launched by the CPU kernels in the traced iterations.
        """
        cpu_op_cat_ids: List[int] = self.symbol_table.get_cpu_event_cat_ids()
        gpu_kernel_cat_ids: List[int] = self.symbol_table.get_gpu_kernel_cat_ids()
        profiler_step_ids: List[int] = self.symbol_table.get_profiler_step_ids()
        device_type = self.get_device_type()

        def filter_gpu_kernels_with_cpu_correlation(
            trace_df: pd.DataFrame,
        ) -> pd.DataFrame:
            cpu_kernels = CPUOperatorFilter()(trace_df, self.symbol_table)
            if cpu_kernels.empty:
                return trace_df
            gpu_kernels = GPUKernelFilter()(trace_df, self.symbol_table)
            filtered_profiler_step_rows = cpu_kernels[
                cpu_kernels["name"].isin(profiler_step_ids)
            ]
            last_profiler_start = (
                filtered_profiler_step_rows["ts"].max()
                if not filtered_profiler_step_rows.empty
                else cpu_kernels["ts"].min()
            )
            last_profiler_end = (
                filtered_profiler_step_rows["end"].max()
                if not filtered_profiler_step_rows.empty
                else cpu_kernels["end"].max()
            )

            cpu_kernels = (
                cpu_kernels[cpu_kernels["ts"] <= last_profiler_end]
                if include_last_profiler_step
                else cpu_kernels[cpu_kernels["ts"] < last_profiler_start]
            )
            filtered_gpu_kernels = gpu_kernels.merge(
                cpu_kernels["correlation"], on="correlation", how="inner"
            )
            return pd.concat(
                [filtered_gpu_kernels, cpu_kernels], axis=0, ignore_index=True
            )

        def filter_mtia_kernels_for_one_rank(trace_df: pd.DataFrame) -> pd.DataFrame:
            cpu_events = trace_df[trace_df["cat"].isin(cpu_op_cat_ids)]
            profiler_steps = cpu_events[cpu_events["name"].isin(profiler_step_ids)]
            t_start = profiler_steps["ts"].min()
            t_end = profiler_steps["end"].max()
            cpu_kernels = cpu_events[
                cpu_events["ts"].ge(t_start) & cpu_events["end"].le(t_end)
            ]

            cpu_correlation_ids = set(cpu_kernels["correlation"].unique())
            cpu_correlation_ids.discard(-1)

            mtia_kernels = trace_df[
                trace_df["cat"].isin(gpu_kernel_cat_ids)
                & trace_df["correlation"].isin(cpu_correlation_ids)
            ]

            # Looks like new MTIA traces have memory kernels with stream values.
            memory_kernels = self._fix_mtia_memory_kernels(trace_df)

            indices = set(memory_kernels.index).union(
                set(cpu_kernels.index).union(set(mtia_kernels.index))
            )
            return trace_df.loc[list(indices)]

        if not profiler_step_ids:
            logger.warning(
                "ProfilerStep not found in the trace. The analysis result may not be accurate."
            )
            include_last_profiler_step = True
        elif len(profiler_step_ids) == 1:
            logger.warning(
                "There is only one iteration in the trace. The analysis result may not be accurate."
            )
            include_last_profiler_step = True
        for trace in self.traces.values():
            if device_type != "MTIA":
                trace.df = filter_gpu_kernels_with_cpu_correlation(trace.df)
            else:
                trace.df = filter_mtia_kernels_for_one_rank(trace.df)

    def decode_symbol_ids(self, use_shorten_name: bool = True) -> None:
        """Decode the name and cat column to show the original string names.

        Args:
            use_shorten_name (bool): shorten the long strings to make it easy to read.
                Default: True.
        """

        for rank in self.traces:
            decode_symbol_id_to_symbol_name(
                self.get_trace_df(rank), self.symbol_table, use_shorten_name
            )

    def get_device_type(self) -> str:
        """Get the device type of the trace.

        Returns:
            The device type parsed from the trace metadata. If the device type is not found, return "UNKNOWN".
        """
        rank = next(iter(self.traces))
        trace: Trace = self.traces[rank]
        device_type = trace.meta.get("device_type", "UNKNOWN")
        return device_type

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

    def get_trace_start_unixtime_ns(self, rank: int) -> int:
        """
        Get the start timestamp of the rank's trace in nanoseconds.
        Start timestamp is the timestamp of the earliest event in the trace.

        Returns: Unixtime (nanoseconds) of trace start (int)

        Raises: ValueError when this TraceCollection object doesn't have trace for the given rank.
        """
        if rank not in self.traces:
            err_msg: str = f"No trace found for rank {rank}"
            logger.warning(err_msg)
            raise ValueError(err_msg)

        trace: Trace = self.get_trace(rank)

        return trace_event_timestamp_to_unixtime_ns(self.min_ts, trace.meta)
