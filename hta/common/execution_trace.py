# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import logging
import sys

import numpy as np

# from typing import Dict, List, NamedTuple, Optional
import pandas as pd

from hta.common.trace import Trace
from hta.utils.utils import normalize_path

from param.python.tools.execution_graph import ExecutionGraph


def load_execution_trace(et_file: str) -> ExecutionGraph:
    et_file_path = normalize_path(et_file)
    """Loads Execution Trace from json file and parses it."""
    with gzip.open(et_file_path, "rb") if et_file.endswith("gz") else open(
        et_file_path, "r"
    ) as f:
        et = ExecutionGraph(json.load(f))
    return et


def _et_has_overlap(trace_df: pd.DataFrame, et: ExecutionGraph) -> bool:
    """Use record function IDs (rf_id) to find out if ET and kineto trace
    have overlap"""
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


def correlate_execution_trace(trace: Trace, rank: int, et: ExecutionGraph) -> None:
    """Outcome is the trace dataframe for specified rank will have a new column
    'et_node' that includes the correlated node idx in Execution trace
    """
    trace_df = trace.get_trace(rank)

    if not _et_has_overlap(trace_df, et):
        logging.error(
            "Execution trace and kineto trace do not overlap, this mode is not currently supported"
        )
        return

    # Mapping from rf_id to et node id
    rf_id_to_et_node_id = {node.rf_id: id for (id, node) in et.nodes.items()}
    trace_df["et_node"] = trace_df.apply(
        lambda row: rf_id_to_et_node_id.get(row["External id"], None), axis=1
    )
    return


def add_et_column(trace_df: pd.DataFrame, et: ExecutionGraph, column: str) -> None:
    """Add columns from Execution trace nodes into the trace datafram"""
    if "et_node" not in trace_df:
        print("Please run correlate_execution_trace() first")
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

    else:
        print(f"Unknown column {column}")

    trace_df[column] = trace_df.apply(
        lambda row: map_func(row.et_node) if pd.notna(row.et_node) else np.nan, axis=1
    )
