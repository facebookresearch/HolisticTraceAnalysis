# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from time import perf_counter
from typing import List, Optional

import pandas as pd
import plotly.express as px

from hta.common.trace import Trace, TraceSymbolTable
from hta.configs.config import logger


def plot_timeline(title: str, events: pd.DataFrame, ranks: Optional[List[int]] = None) -> None:
    """
    Plot the timeline of events

    Args:
        title (str): a title for the timeline plot
        events (pd.DataFrame): a data contains the events to be plotted.
        ranks (List[int]) : a list of ranks whose events will be plotted.

    Return:
        None

    Raise:
        ValueError

    Notes:
        For the timeline to be properly drawn, the events DataFrame must contain the following columns:
        + calibrated_start_global
        + calibrated_end_global
        + task
        + label
    """
    t0 = perf_counter()
    must_have_columns: List[str] = [
        "calibrated_start_global",
        "calibrated_end_global",
        "task",
        "label",
    ]
    if not set(must_have_columns).issubset(set(events.columns)):
        raise ValueError(f"the events dataframe doesn't contain all required columns {must_have_columns}")

    if ranks is None:
        if "rank" not in events.columns:
            ranks = []
        else:
            ranks = sorted(events["rank"].unique())

    addition_columns: List[str] = [
        "rank",
        "stream",
        "iteration",
        "s_cat",
        "cat",
        "s_name",
        "name",
        "dur",
    ]
    hover_data = sorted(list(set(events.columns).intersection(set(addition_columns))))

    sorted_tasks = sorted(events["task"].unique())
    fig = px.timeline(
        events,
        x_start="calibrated_start_global",
        x_end="calibrated_end_global",
        y="task",
        hover_data=hover_data,
        category_orders={"task": sorted_tasks},
        color="label",
        color_discrete_sequence=px.colors.qualitative.D3,
        width=1600,
        height=200 + 120 * len(ranks),
        title=title,
    )
    fig.show()

    t1 = perf_counter()
    logger.debug(f"Plotted timeline in {t1 - t0:.2f} seconds")


def _get_unique_values(df: pd.DataFrame, col: str, exclude_values: List[int] = [-1]) -> List[int]:
    """Get the unique values for a given column <col> in the input Dataframe <df>

    Args:
        df (pd.DataFrame) : a DataFrame with trace events
        col (str) : a column name
        exclude_values (List[int]) : list of values to be excluded from the returned list

    Return:
        a list of unique values for df[col]
    """
    if col not in df:
        raise ValueError(f"{col} not in the DataFrame's columns: {df.columns}")
    unqiue_values = [e for e in df[col].unique().tolist() if e not in exclude_values]
    return sorted(unqiue_values)


def _simplify_name(name: str) -> str:
    """Simplify the name"""
    _patterns = [
        r"\(.+\)",
        r"<[^<]+>?",
        r"^void\s+",
        r"\s+",
        r"autograd::engine::evaluate_function",
    ]
    for pat in _patterns:
        name = re.sub(pat, "", name)
    return name if len(name) < 45 else name[:42] + "..."


def prepare_timeline_gpu_events(
    df: pd.DataFrame,
    symbol_table: TraceSymbolTable,
    ranks: Optional[List[int]] = None,
    iterations: Optional[List[int]] = None,
    streams: Optional[List[int]] = None,
    duration_threshold: int = 1000,
) -> pd.DataFrame:
    """
    Prepare GPU events for timeline analysis

    Args:
        df (pd.DataFrame) : an input DataFrame.
        symbol_table (TraceSymbolTable): input trace symbol table for mapping symbol id back to text.
        ranks (List[int]) : filter the input DataFrame with the given set of ranks; use all ranks if None.
        iterations (List[int]) : filter the input DataFrame with the given set of iteartions; use all iterations if None.
        streams (List[int]) : filter the input DataFrame with the given set of streams; use all streams if None.
        duration_threshold (int) : the minimum duration given for short kernels for them to be visible on the figure.
    Return:
        a DataFrame for selected GPU events for plot_timeline_px
    """
    t0 = perf_counter()
    required_columns: List[str] = ["iteration", "name", "cat", "rank", "stream", "ts"]
    if not set(required_columns).issubset(set(df.columns)):
        raise ValueError(f"columns {set(required_columns) - set(df.columns)} are not in input DataFrame")

    if ranks is None:
        ranks = _get_unique_values(df, "rank")

    if iterations is None:
        iterations = _get_unique_values(df, "iteration")

    if streams is None:
        streams = _get_unique_values(df, "stream")

    events = df.loc[(df["rank"].isin(ranks)) & (df["iteration"].isin(iterations)) & (df["stream"].isin(streams))].copy()

    sym_tab = symbol_table.get_sym_table()
    events["s_name"] = events["name"].apply(lambda i: sym_tab[i]).apply(_simplify_name)
    events["s_cat"] = events["cat"].apply(lambda i: sym_tab[i])
    events["calibrated_start_global"] = pd.to_datetime(events["ts"], unit="us")
    events["calibrated_end_global"] = pd.to_datetime(
        events["ts"] + events["dur"].apply(lambda x: x if x >= duration_threshold else duration_threshold),
        unit="us",
    )

    events["label"] = events["s_name"]
    events["task"] = "rank " + events["rank"].astype("str") + " stream " + events["stream"].astype("str")
    events = events.sort_values(by=["rank", "stream", "ts"])

    t1 = perf_counter()
    logger.debug(f"Preprocessed events data for timeline visualization in {t1 - t0:.2f} seconds")

    return events


def plot_timeline_gpu_kernels(
    title: str,
    df: pd.DataFrame,
    symbol_table: TraceSymbolTable,
    ranks: Optional[List[int]] = None,
    iterations: Optional[List[int]] = None,
    streams: Optional[List[int]] = None,
    duration_threshold: int = 1000,
) -> None:
    """
    Plot the timeline of selected GPU kernels in a DataFrame.

    Args:
        title (str): a title for the plot
        df: an input DataFrame.
        symbol_table: input trace symbol table for mapping symbol id back to text.
        ranks List[int]: filter the input DataFrame with the given set of ranks; use all ranks if None.
        iterations List[int]: filter the input DataFrame with the given set of iterations; use all iterations if None.
        streams List[int]: filter the input DataFrame with the given set of streams; use all streams if None.
        duration_threshold (int) : the minimum duration given for short kernels for them to be visible on the figure.
    """
    df_timeline = prepare_timeline_gpu_events(df, symbol_table, ranks, iterations, streams, duration_threshold)
    plot_timeline(title, df_timeline)


def plot_timeline_gpu_kernels_from_trace(
    title: str,
    trace_data: Trace,
    ranks: Optional[List[int]] = None,
    iterations: Optional[List[int]] = None,
    streams: Optional[List[int]] = None,
    duration_threshold: int = 1000,
) -> None:
    """
    Plot the timeline of selected GPU kernels in a Trace object.

    Args:
        title (str): a title for the timeline plot
        trace_data (Trace): a Trace object
        ranks (List[int]): filter the input DataFrame with the given set of ranks; use all ranks if None.
        iterations (List[int]): filter the input DataFrame with the given set of iterations; use all iterations if None.
        streams (List[int]): filter the input DataFrame with the given set of streams; use all streams if None.
        duration_threshold (int) : the minimum duration given for short kernels for them to be visible on the figure.
    """
    if ranks is not None:
        _ranks = list(set(trace_data.get_all_traces().keys()).intersection(set(ranks)))
    else:
        _ranks = list(trace_data.get_all_traces().keys())
    df = pd.concat(
        [trace_data.get_trace(r) for r in _ranks],
        axis=0,
        keys=_ranks,
        names=["rank", "idx"],
    ).reset_index()

    plot_timeline_gpu_kernels(
        title,
        df,
        trace_data.symbol_table,
        ranks,
        iterations,
        streams,
        duration_threshold,
    )
