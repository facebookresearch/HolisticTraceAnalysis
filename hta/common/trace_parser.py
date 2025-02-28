# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gzip
import io
import json
import math
import os
import time
import tracemalloc
from collections.abc import Generator
from typing import Any, Dict, Optional, Tuple

import hta.configs.env_options as hta_options

import numpy as np

import pandas as pd
from hta.common.trace_symbol_table import TraceSymbolTable

from hta.configs.config import logger
from hta.configs.default_values import ValueType
from hta.configs.parser_config import (
    AttributeSpec,
    DEFAULT_PARSE_VERSION,
    ParserBackend,
    ParserConfig,
)
from hta.utils.utils import normalize_gpu_stream_numbers

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


def _open_trace_file(trace_file_path: str) -> io.BufferedIOBase:
    return (
        gzip.open(trace_file_path, "rb")
        if trace_file_path.endswith(".gz")
        else open(trace_file_path, "rb")
    )


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
    with _open_trace_file(trace_file_path) as fh:

        generator = ijson.items(fh, "traceEvents.item", use_float=True)

        # Ignore python function tracer
        # TODO make this filter configuration in ParserConfig
        df = pd.DataFrame(e for e in generator if e.get("cat") != "python_function")

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
    with _open_trace_file(trace_file_path) as fh:
        # TODO make events to skip configurable in ParserConfig
        generator = (
            e
            for e in ijson.items(fh, "traceEvents.item", use_float=True)
            if e.get("cat") != "python_function"
        )
        if compress_on_fly:
            generator = (trim_event(e) for e in generator)

        # XXX Currently using 1000 as batch size.
        batch_size = 1000
        batch = []
        dfs = []

        # Iterate over filtered dictionaries and append to DataFrame in batches
        for item in generator:
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

    # Keep memory events (which are instant events)
    is_memory_event = (
        (df["ph"] == "i")
        & (df["name"] == "[memory]")
        & (df["cat"] == "cpu_instant_event")
    )
    df.loc[is_memory_event, "dur"] = 0
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
            [
                AttributeSpec(
                    name,
                    name,
                    ValueType.Int,
                    -1,
                    min_supported_version=DEFAULT_PARSE_VERSION,
                )
                for name in counter_names
            ]
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

    normalize_gpu_stream_numbers(df)

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


def round_down_time_stamps(df: pd.DataFrame) -> None:
    if df["ts"].dtype != np.dtype("float64"):
        return
    if hta_options.disable_ns_rounding():
        logger.warning("Rounding down ns resolution traces disabled")
        return

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
        round_down_time_stamps(df)

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
    # Parse trace metadata swiftly
    meta: MetaData = {}
    with _open_trace_file(trace_file_path) as fh:
        t_start = time.perf_counter_ns()
        meta = parse_metadata_ijson(fh)
        t_end = time.perf_counter_ns()
        logger.info(
            f"Parsed {trace_file_path} metadata in "
            f"{(t_end - t_start)/1000000:.2f} milli seconds"
        )

    if batched:
        df = _parse_trace_events_ijson_batched(trace_file_path, cfg, compress_on_fly)
    else:
        df = _parse_trace_events_ijson(trace_file_path)

    round_down_time_stamps(df)

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


# --- Trace metadata reader ---
def parse_metadata_ijson(fh: io.BufferedIOBase) -> MetaData:
    """
    Parses a trace file using the `ijson` library and extracts certain high-level key-value pairs.
    Args:
        fh (file-like object): The trace file to parse.
    Returns:
        A dictionary containing the extracted metadata.

    Note: We use the ijson low level API to read all dictionary
    elements in the json up to "traceEvents", at which point we just terminate.
    This leads to some blazingly fast metadata parsing!

    https://pypi.org/project/ijson/#toc-entry-4

    The ijson low-level parse API returns a series of (prefix, event, value). The
    prefix shows the path in the json object, while event can signify
    the start of a map, array or a key. Following is an excerpt of events for a trace file

    ```
     start_map None         # prefix = ''
     map_key schemaVersion
    schemaVersion number 1
     map_key distributedInfo
    distributedInfo start_map None
    distributedInfo map_key backend
    distributedInfo.backend string nccl
    distributedInfo map_key rank
    distributedInfo.rank number 0
    distributedInfo map_key world_size
    distributedInfo.world_size number 64
    distributedInfo end_map None
     map_key deviceProperties
    deviceProperties start_array None
    deviceProperties.item start_map None
    deviceProperties.item map_key id
    deviceProperties.item.id number 0
    ```
    """
    import ijson

    meta: MetaData = {}
    trace_parser = ijson.parse(fh)

    def handle_nested_map(
        generator: Generator, key: str, meta_so_far: MetaData
    ) -> MetaData:
        """
        Recursively handles events for a nested map like distributedInfo.
        """
        prefix, event, value = next(generator)
        logger.debug(" -> ", prefix, event, value, key, meta_so_far)
        if event == "end_map":
            return meta_so_far
        elif event == "map_key":
            key = value
            return handle_nested_map(generator, key, meta_so_far)
        elif event == "start_map" or event == "start_array":
            # Currently we do not supported nesting of maps/arrays in the nested map
            # skip to end of this section
            meta_so_far[key] = None

            end_prefix = prefix
            while not (prefix == end_prefix and event in ["end_map", "end_array"]):
                logger.debug("  skipping ", prefix, event, value)
                prefix, event, _ = next(trace_parser)
            return handle_nested_map(generator, "", meta_so_far)
        else:
            assert (
                key is not None
            ), f"map_key event was missed? (prefix, event, value) = ({prefix}, {event}, {value})"
            meta_so_far[key] = value
            return handle_nested_map(generator, "", meta_so_far)

    cur_key = None
    nested_map: MetaData = {}

    for prefix, event, value in trace_parser:
        logger.debug(prefix, event, value)
        if event == "map_key" and value == "traceEvents":
            # done ok bye!
            return meta

        # Handle a nested map like "distributedInfo"
        if prefix == cur_key and event == "start_map":
            nested_map = {}
            meta[cur_key] = handle_nested_map(trace_parser, "", nested_map)
            continue

        # Handle a nested array map like "deviceProperties"
        if prefix == cur_key and event == "start_array":
            meta[cur_key] = []
            # For deviceProperties this should be start_map or end_array
            _, _event_, _ = next(trace_parser)
            assert _event_ in [
                "start_map",
                "end_array",
            ], f"We only support an array with map elements like deviceProperties, (prefix, event, value) = ({prefix}, {event}, {value})"
            while _event_ != "end_array":
                nested_map = {}
                nested_map = handle_nested_map(trace_parser, "", nested_map)
                meta[cur_key].append(nested_map)
                _, _event_, _ = next(trace_parser)
            continue

        # Handle top level simple key values
        #  map_key schemaVersion
        # schemaVersion number 1
        if event == "map_key" and prefix == "":
            cur_key = value
        if prefix == cur_key:
            assert (
                cur_key is not None
            ), f"map_key event was missed?(prefix, event, value) = ({prefix}, {event}, {value})"
            meta[cur_key] = value

    return meta
