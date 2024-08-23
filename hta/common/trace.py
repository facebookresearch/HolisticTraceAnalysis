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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import pandas as pd

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


def transform_correlation_to_index(
    df: pd.DataFrame, symbol_table: TraceSymbolTable
) -> pd.DataFrame:
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
        symbol_table: the TraceSymbolTable for the trace

    Returns:
        pd.DataFrame: the transformed DataFrame with a index_correlation column.

    Affects:
        This function adds a index_correlation column to df.
    """

    if "correlation" not in df.columns:
        return df

    # Initialize the index_correlaion to the fallback value first
    df["index_correlation"] = np.minimum(df["correlation"], 0)
    corr_df = df.loc[
        df["correlation"].ne(-1), ["index", "correlation", "stream", "name"]
    ]

    on_cpu = CPUOperatorFilter()(corr_df, symbol_table)
    on_gpu = GPUKernelFilter()(corr_df, symbol_table)

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


def parse_trace_file(
    trace_file_path: str,
    cfg: Optional[ParserConfig] = None,
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
        ValueError if parser config passes invalid parser backend.
    """
    if not (trace_file_path.endswith(".gz") or trace_file_path.endswith(".json")):
        raise ValueError(
            f"expect the value of trace_file ({trace_file_path}) ends with '.gz' or 'json'"
        )

    t_start = time.perf_counter()
    cfg = cfg or ParserConfig.get_default_cfg()

    meta, df, local_symbol_table = parse_trace_dataframe(trace_file_path, cfg)

    # add fwd bwd links between CPU ops
    add_fwd_bwd_links(df)

    df = transform_correlation_to_index(df, local_symbol_table)

    add_iteration(df, local_symbol_table)
    df["end"] = df["ts"] + df["dur"]

    t_end = time.perf_counter()
    logger.warning(
        f"Overall parsing of {trace_file_path} in {(t_end - t_start):.2f} seconds; current PID:{os. getpid()}"
    )
    return meta, df, local_symbol_table


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
        trace_files: Union[List[str], Optional[Dict[int, str]]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
    ) -> None:
        """
        The constructor of a Trace object.
        Args:
            trace_files: Union[List[str], Dict[int, str] : either a list of trace file names or a map from rank to trace file names.
                When a list is provided, HTA will infer the ranks by reading the trace file metadata.
                The trace file names can be either relative to the path `trace_path` or absolute file paths.
            trace_dir (str) : a path used to derive `trace_path = normalize_path(trace_dir)`.

        Raises:
            AssertionError
        """
        self.trace_path: str = normalize_path(trace_dir)
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
            ) = parse_trace_file(trace_filepath)
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
                result = parse_trace_file(self.trace_files[rank])
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
            elif use_memory_profiling:
                tracemalloc.start()
                parse_trace_file(trace_paths[0])
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                num_procs = get_mp_pool_size(peak, len(ranks))
            logger.info(f"using {num_procs} processes for parsing.")

            with mp.get_context("fork").Pool(num_procs) as pool:
                results = pool.map(parse_trace_file, trace_paths)
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
            self.parse_multiple_ranks(
                ranks, use_multiprocessing and len(ranks) > 1, use_memory_profiling
            )

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
            cpu_kernels = CPUOperatorFilter()(trace_df, self.symbol_table)
            gpu_kernels = GPUKernelFilter()(trace_df, self.symbol_table)
            last_profiler_start = cpu_kernels[cpu_kernels["name"].isin(profiler_steps)][
                "ts"
            ].max()
            last_profiler_end = cpu_kernels[cpu_kernels["name"].isin(profiler_steps)][
                "end"
            ].max()

            cpu_kernels = (
                cpu_kernels[cpu_kernels["ts"] <= last_profiler_end]
                if include_last_profiler_step
                else cpu_kernels[cpu_kernels["ts"] < last_profiler_start]
            )
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
