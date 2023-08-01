# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import logging
import sys
import time

from typing import List

import numpy as np
import pandas as pd

from hta.common.trace import Trace
from hta.configs.config import logger
from hta.utils.utils import normalize_path

from param_bench.train.compute.python.tools.execution_trace import ExecutionTrace

# PyTorch Events types that are correlated in the Execution Trace
EXECUTION_TRACE_SUPPORTED_EVENTS: List[str] = ["cpu_op", "user_annotation"]


def load_execution_trace(et_file: str) -> ExecutionTrace:
    """Loads Execution Trace from json file and parses it into an
    object representation. For large files this could take a lot of memory.

    Args:
        et_file (str): File path for the Execution Trace.

    Returns:
        ExecutionTrace object.
    """
    et_file_path = normalize_path(et_file)

    t_start = time.perf_counter()
    with gzip.open(et_file_path, "rb") if et_file.endswith("gz") else open(
        et_file_path, "r"
    ) as f:
        et = ExecutionTrace(json.load(f))
    t_end = time.perf_counter()

    logger.info(
        f"Parsed Execution Trace file {et_file}, "
        f"time = {(t_end - t_start):.2f} seconds "
    )
    return et


def _et_has_overlap(trace_df: pd.DataFrame, et: ExecutionTrace) -> bool:
    """Use record function IDs (rf_id) to find out if ET and Kineto trace
    have overlap

    Args:
        trace_df (pd.DataFrame): Trace dataframe for one rank.
        et (ExecutionTrace: Execution Trace object for the same rank.

    Returns:
        True if Kineto Trace and Execution Trace have overlap.
    """
    et_min_rf, et_max_rf = sys.maxsize, 0
    rf_ids = (
        node.rf_id
        for node in et.nodes.values()
        if node.rf_id is not None
        and ("execution_trace|thread" not in node.name)
        and ("execution_trace|process" not in node.name)
    )
    for rf_id in rf_ids:
        et_min_rf = min(rf_id, et_min_rf)
        et_max_rf = max(rf_id, et_max_rf)

    kt_min_rf, kt_max_rf = trace_df["External id"].min(), trace_df["External id"].max()

    has_overlap = kt_min_rf <= et_min_rf and kt_max_rf >= et_max_rf

    logging.info(f"Trace and ET have overlap = {has_overlap}")
    logging.info(
        f"Trace rf_ids ({kt_min_rf}, {kt_max_rf}),"
        f"ET rf_ids ({et_min_rf}, {et_max_rf})"
    )
    return has_overlap


def correlate_execution_trace(trace: Trace, rank: int, et: ExecutionTrace) -> None:
    """Correlate the trace from a specific rank with Execution Trace object.

    Args:
        trace (Trace): Trace object loaded using `TraceAnalysis(trace_dir=trace_dir)`
                        or other method.
        rank (int): Rank to correlate with.
        et (ExecutionTrace): An Execution Trace object to correlate with.

    Returns:
        None

    Outcome is the trace dataframe for specified rank will have a new column
    'et_node' that includes the correlated node index in Execution Trace.

    We use two different approaches depending if the PyTorch and ET trace
        1) Have overlap: correlation is done using record function ID.
        2) Do not have overlap: correlation is done by comparing the two
            trees using a graph edit distance similarity algorithm.

    Please note (2) is not supported yet and will come in future PRs.

    """
    trace_df = trace.get_trace(rank)

    if not _et_has_overlap(trace_df, et):
        logging.error(
            "Execution Trace and kineto trace do not overlap, this mode is not currently supported"
        )
        return

    # Mapping from rf_id to et node id
    rf_id_to_et_node_id = {node.rf_id: id for (id, node) in et.nodes.items()}

    # Only correlate specific events
    sym_index = trace.symbol_table.get_sym_id_map()
    sym_ids = [sym_index.get(cat) for cat in EXECUTION_TRACE_SUPPORTED_EVENTS]
    logger.info(f"Supported event type ('cat') symbols = {sym_ids}")

    row_indexer = trace_df["cat"].isin(sym_ids)
    trace_df.loc[row_indexer, "et_node"] = trace_df.apply(
        lambda row: rf_id_to_et_node_id.get(row["External id"], None), axis=1
    )
    return


def add_et_column(trace_df: pd.DataFrame, et: ExecutionTrace, column: str) -> None:
    """Add columns from Execution Trace nodes into the trace dataframe. Please
    run this after running correlate_execution_trace(...).
    Args:
        trace_df (pd.DataFrame): Dataframe for trace from one rank. Please
                                 run correlate_execution_trace() on the trace dataframe
                                 first so that the `et_node` is populated..
        et (ExecutionTrace): The Execution Trace object.
        column (stR): Column to add from the corresponding Execution Trace node.

    Returns:
        None

    """
    if "et_node" not in trace_df:
        logger.error("Please run correlate_execution_trace() first")
        return
    if column == "op_schema":

        def map_func(node_id):
            return et.nodes[node_id].op_schema

    elif column == "input_shapes":

        def map_func(node_id):
            return et.nodes[node_id].input_shapes

    elif column == "input_types":

        def map_func(node_id):
            return et.nodes[node_id].input_types

    elif column == "output_shapes":

        def map_func(node_id):
            return et.nodes[node_id].output_shapes

    elif column == "output_types":

        def map_func(node_id):
            return et.nodes[node_id].output_types

    elif column == "et_node_name":

        def map_func(node_id):
            return et.nodes[node_id].name

    else:
        logger.error(f"Unknown column {column}")
        return

    trace_df[column] = trace_df.apply(
        lambda row: map_func(row.et_node) if pd.notna(row.et_node) else np.nan, axis=1
    )
