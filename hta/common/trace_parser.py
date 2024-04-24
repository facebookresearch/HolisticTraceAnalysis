# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gzip
import json
import math
import os
import time
import tracemalloc
from typing import Any, Dict, Optional, Tuple

import numpy as np

import pandas as pd
from hta.common.trace_symbol_table import TraceSymbolTable

from hta.configs.config import logger
from hta.configs.parser_config import (
    AttributeSpec,
    ParserBackend,
    ParserConfig,
    ValueType,
)

# from memory_profiler import profile

MetaData = Dict[str, Any]
_TRACE_PARSING_BACKEND: Optional[ParserBackend] = None

IJSON_INSTRS = """ Install ijson with pip by using
  pip install ijson

Also please install yajl for optimal speed. You will need an installation of yajl (C++ library) for your system, followed by the python bindings.
  conda install anaconda::yajl  # or system install of yajl without conda
  pip install yajl-py==2.1.2

Check the backend with: python3 -c "import ijson; print(ijson.backend)" output = yajl2_c
For more details see https://pypi.org/project/ijson/#performance-tips, https://anaconda.org/anaconda/yajl, https://pypi.org/project/yajl/"""


def _auto_detect_parser_backend() -> ParserBackend:
    """Finds optimal parser backend and returns it"""
    try:
        import ijson
    except ModuleNotFoundError:
        logger.warning("Trace parsing can be sped up by using ijson." + IJSON_INSTRS)
        return ParserBackend.JSON

    if "yajl" not in ijson.backend:
        logger.warning(
            f"Current ijson backend {ijson.backend} does not use 'yajl'."
            "The ijson parser will be much slower as it uses python"
            "Reverting to simple json parser!\n"
            "Consider instructions-" + IJSON_INSTRS
        )
        return ParserBackend.JSON

    # Use the best ijson backend
    return ParserBackend.IJSON_BATCH_AND_COMPRESS


def set_default_trace_parsing_backend(parser_backend: ParserBackend):
    """Set the default trace parser backend"""
    global _TRACE_PARSING_BACKEND
    _TRACE_PARSING_BACKEND = parser_backend


def get_default_trace_parsing_backend() -> ParserBackend:
    """Get the default trace parser backend"""
    global _TRACE_PARSING_BACKEND
    if _TRACE_PARSING_BACKEND is None:
        # TODO: This will be updated in a future release
        # _TRACE_PARSING_BACKEND = _auto_detect_parser_backend()
        _TRACE_PARSING_BACKEND = ParserBackend.JSON
    return _TRACE_PARSING_BACKEND


# @profile
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
    logger.warning(f"Parsed {trace_file_path} time = {(t_end - t_start):.2f} seconds ")
    return trace_record


# @profile
def _parse_trace_events_ijson(trace_file_path: str) -> pd.DataFrame:
    """
    Parse the trace file using iterative json.

    Args:
        trace_file_path (str) : the path to a trace file.

    Returns:
        pd.DataFrame: parsed trace dataframe.
    """
    import ijson

    logger.info(f"Parsing using ijson (ijson backend = {ijson.backend})")

    t_start = time.perf_counter()
    with (
        gzip.open(trace_file_path, "rb")
        if trace_file_path.endswith(".gz")
        else open(trace_file_path, "rb")
    ) as fh:

        iterator = ijson.items(fh, "traceEvents.item", use_float=True)

        # Ignore python function tracer
        # TODO make this filter configuration in ParserConfig
        df = pd.DataFrame(e for e in iterator if e.get("cat") != "python_function")

    t_end = time.perf_counter()
    logger.warning(
        f"Parsed (ijson) {trace_file_path} time = {(t_end - t_start):.2f} seconds "
    )
    return df


def _parse_trace_events_ijson_batched(
    trace_file_path: str, cfg: ParserConfig, compress_on_fly: bool = False
) -> pd.DataFrame:
    """
    Parse the trace file using iterative json in batches.

    Args:
        trace_file_path (str): the path to a trace file.
        compress_on_fly (bool): parse "args" on the fly.

    Returns:
        pd.DataFrame: parsed trace dataframe.
    """
    import ijson

    logger.info(f"Parsing using ijson (ijson backend = {ijson.backend})")

    arg_name_map = {arg.raw_name: arg.name for arg in cfg.get_args()}
    args_to_keep = arg_name_map.keys()
    logger.debug(f"arg_name_map = {arg_name_map}")

    def trim_event(e):
        if "args" not in e:
            return e
        for arg, val in e["args"].items():
            if arg in args_to_keep:
                e[arg_name_map[arg]] = val
            elif e.get("cat", "") == "cuda_profiler_range":
                e[arg] = val
        e.pop("args", None)
        return e

    df = pd.DataFrame()

    t_start = time.perf_counter()
    with (
        gzip.open(trace_file_path, "rb")
        if trace_file_path.endswith(".gz")
        else open(trace_file_path, "rb")
    ) as fh:
        # TODO make events to skip configurable in ParserConfig
        iterator = (
            e
            for e in ijson.items(fh, "traceEvents.item", use_float=True)
            if e.get("cat") != "python_function"
        )
        if compress_on_fly:
            iterator = (trim_event(e) for e in iterator)

        # XXX Currently using 1000 as batch size.
        batch_size = 1000
        batch = []
        dfs = []

        # Iterate over filtered dictionaries and append to DataFrame in batches
        for item in iterator:
            batch.append(item)
            if len(batch) == batch_size:
                dfs.append(pd.DataFrame(batch))
                batch = []

        # Append remaining items if any
        if batch:
            dfs.append(pd.DataFrame(batch))

        df = pd.concat(dfs, ignore_index=True)

        # Fill args if not populated
        arg_default_map = {arg.name: arg.default_value for arg in cfg.get_args()}
        trace_args_cols = set(arg_default_map.keys()).intersection(set(df.columns))
        for arg_col in trace_args_cols:
            df[arg_col].fillna(arg_default_map[arg_col], inplace=True)

        missing_cols = set(arg_default_map.keys()).difference(set(df.columns))
        for arg_col in missing_cols:
            df[arg_col] = arg_default_map[arg_col]

    t_end = time.perf_counter()
    logger.warning(
        f"Parsed (ijson) {trace_file_path} time = {(t_end - t_start):.2f} seconds "
    )
    return df


def _compress_df(
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
    columns = set(df.columns)

    # performance counters appear as args
    if "args" in columns and "cuda_profiler_range" in df.cat.unique():
        counter_names = set.union(
            *[set(d.keys()) for d in df[df.cat == "cuda_profiler_range"]["args"].values]
        )
        # args_to_keep = args_to_keep.union(counter_names)
        cfg.add_args(
            [AttributeSpec(name, name, ValueType.Int, -1) for name in counter_names]
        )
        logger.info(f"counter_names={counter_names}")
        logger.info(f"args={cfg.get_args()}")

    if "args" in columns:
        args_to_keep = cfg.get_args()
        for arg in args_to_keep:
            df[arg.name] = df["args"].apply(
                lambda row, arg=arg: (
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


# @profile
def _parse_trace_dataframe_json(
    trace_file_path: str, cfg: ParserConfig
) -> Tuple[MetaData, pd.DataFrame, TraceSymbolTable]:
    """
    Parse the trace file into dataframe.

    Args:
        trace_file_path (str) : the path to a trace file.

    Returns:
        pd.DataFrame: parsed trace dataframe.
    """
    trace_record = parse_trace_dict(trace_file_path)
    meta: Dict[str, Any] = {k: v for k, v in trace_record.items() if k != "traceEvents"}
    df: pd.DataFrame = pd.DataFrame()
    local_symbol_table: TraceSymbolTable = TraceSymbolTable()
    if "traceEvents" in trace_record:
        df = pd.DataFrame(trace_record["traceEvents"])
        if df["ts"].dtype == np.dtype("float64"):
            logger.warning(
                f"Rounding down ns resolution events due to issue with events overlapping."
                f" ts dtype = {df['ts'].dtype}, dur dtype = {df['dur'].dtype}."
                f"Please see https://github.com/pytorch/pytorch/pull/122425"
            )
            # Don't floor directly, first find the end
            df["end"] = df["ts"] + df["dur"]

            df["ts"] = df[~df["ts"].isnull()]["ts"].apply(lambda x: math.ceil(x))
            df["end"] = df[~df["end"].isnull()]["end"].apply(lambda x: math.floor(x))
            df["dur"] = df["end"] - df["ts"]

        # assign an index to each event
        df.reset_index(inplace=True)
        df["index"] = pd.to_numeric(df["index"], downcast="integer")

        df, local_symbol_table = _compress_df(df, cfg)

    return meta, df, local_symbol_table


# @profile
def _parse_trace_dataframe_ijson(
    trace_file_path: str,
    cfg: ParserConfig,
    batched: bool = False,
    compress_on_fly: bool = False,
) -> Tuple[MetaData, pd.DataFrame, TraceSymbolTable]:
    """
    Parse the trace file using iterative json, supports multiple modes below.

    Args:
        trace_file_path (str): the path to a trace file.
        batched (bool): use batching
        compress_on_fly (bool): parse "args" on the fly.

    Returns:
        pd.DataFrame: parsed trace dataframe.
    """
    meta: Dict[str, Any] = {}
    # XXX handle adding metadata
    # k: v for k, v in trace_record.items() if k != "traceEvents"}

    if batched:
        df = _parse_trace_events_ijson_batched(trace_file_path, cfg, compress_on_fly)
    else:
        df = _parse_trace_events_ijson(trace_file_path)

    # assign an index to each event
    df.reset_index(inplace=True)
    df["index"] = pd.to_numeric(df["index"], downcast="integer")

    df, local_symbol_table = _compress_df(df, cfg)
    return meta, df, local_symbol_table


def parse_trace_dataframe(
    trace_file_path: str,
    cfg: ParserConfig,
) -> Tuple[MetaData, pd.DataFrame, TraceSymbolTable]:
    """Parse a single trace file into a meat test_data dictionary and a dataframe of events.
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
        JSONDecodeError when the trace file is not a valid JSON document.
        ValueError if parser config passes invalid parser backend.
    """
    trace_memory = cfg.trace_memory
    parser_backend: ParserBackend
    if cfg.parser_backend is None:
        parser_backend = get_default_trace_parsing_backend()
    else:
        parser_backend = cfg.parser_backend

    t_start = time.perf_counter()
    if trace_memory:
        tracemalloc.start()

    if parser_backend == ParserBackend.JSON:
        meta, df, local_symbol_table = _parse_trace_dataframe_json(trace_file_path, cfg)
    elif parser_backend == ParserBackend.IJSON:
        meta, df, local_symbol_table = _parse_trace_dataframe_ijson(
            trace_file_path, cfg
        )
    elif parser_backend == ParserBackend.IJSON_BATCHED:
        meta, df, local_symbol_table = _parse_trace_dataframe_ijson(
            trace_file_path,
            cfg,
            batched=True,
        )
    elif parser_backend == ParserBackend.IJSON_BATCH_AND_COMPRESS:
        meta, df, local_symbol_table = _parse_trace_dataframe_ijson(
            trace_file_path, cfg, batched=True, compress_on_fly=True
        )
    else:
        raise ValueError(f"unexpected or unsupported parser = {parser_backend}")

    t_end = time.perf_counter()
    logger.warning(
        f"Parsed {trace_file_path} backend={parser_backend} in {(t_end - t_start):.2f} seconds; current PID:{os. getpid()}"
    )
    if trace_memory:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.warning(
            f"Parser Memory usage peak = {(peak/1024/1024):.2f} MB, current = {(current/1024/1024):.2f} MB"
        )
    return meta, df, local_symbol_table
