# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Tuple

import pandas as pd
import plotly.express as px

from hta.analyzers.timeline import _get_unique_values, plot_timeline_gpu_kernels
from hta.common.trace import Trace, TraceSymbolTable


def extract_iteration_info(trace: Trace) -> pd.DataFrame:
    """Extract the iteration information from a trace.

    Args:
          trace (Trace) : a trace object

    Returns:
        a DataFrame that contains all the iteration information with rank as a column.
    """
    ranks = [k for k in trace.get_all_traces().keys()]
    sym_map = trace.symbol_table.get_sym_id_map()
    sym_tab = trace.symbol_table.get_sym_table()

    def get_iter_nbr(profiler_step_name_id: int) -> int:
        iter_re = re.compile(r"ProfilerStep\s*#\s*(\d+)")
        m = iter_re.match(sym_tab[profiler_step_name_id])
        if m:
            return int(m.group(1))
        return -1

    def _extract_one_rank(df) -> pd.DataFrame:
        profiler_step_name_ids = [
            sym_map[k] for k in sorted([k for k in sym_map.keys() if k.startswith("ProfilerStep")])
        ]
        profiler_step_index = df["name"].isin(profiler_step_name_ids)
        p_steps = df.loc[profiler_step_index, ["name", "ts", "dur"]].copy().rename(columns={"name": "profiler_step"})
        p_steps["iter"] = p_steps["profiler_step"].apply(get_iter_nbr)
        p_steps["end"] = p_steps["ts"] + p_steps["dur"]
        p_steps.set_index("iter", inplace=True)
        p_steps.drop(["profiler_step"], axis=1, inplace=True)
        return p_steps

    df_iterations = pd.concat(
        [_extract_one_rank(trace.get_trace(r)) for r in ranks],
        keys=ranks,
        names=["rank", "iter"],
    ).reset_index()
    return df_iterations


def _compute_normalized_start_time_of_significant_comm_kernels(
    df: pd.DataFrame,
    symbol_table: TraceSymbolTable,
    visualize: bool = False,
    min_normalized_duration: float = 0.01,
) -> Tuple[pd.DataFrame, str]:
    """
    Computes the normalized start time of significant comm_kernels.

    Args:
        df (pd.DataFrame) : an input DataFrame with traces for selected ranks and iterations.
        symbol_table (TraceSymbolTable) : the Trace Symbol Table for encoding/decoding the symbols in the trace DataFrame
        visualize (bool): a flag to enable/disable visualizing the intermediate results.
        min_normalized_duration (float) : a ratio for filtering out operators/kernels which are unlikely
            to cause a rank to be a straggler.

    Return:
        Tuple[df: PdDataFrame, metric_name: str]
            df is a DataFrame which contains
            metric_name is a column name which can be used for straggler detection

    Note:
        + We break down straggler identification into two steps:
            1. compute metric for straggler detection
            2. detect straggler from the metric
        + This function provides a reference implementation for computing a metric for straggler detection. A user
            can choose a different metric and implement the corresponding metric computing function.
        + The metric used in this function is the starting time of the last most dominating communication kernel. It
            is computed in the following steps:
            1. find all communication kernels which pass the duration test, i.e.,
                kernel duration > mean_iter_time * min_normalized_duration (the default value is 1%)
            2. pick the last kernel on each stream
            3. select the stream-kernel combination which has the largest average standard deviation across ranks
        + This metric is defined for each rank and for each iteration (excluding the last iteration
            if it should be ignored).

        + One implicit assumption underlying this function is that candidate GPU communication kernels for straggler
            detection will involve a blocking all-to-all synchronization and thus end at about the same time.
            Therefore, the later a kernel start, the bigger overall delay it likely causes.

        + With more model-specific information, a user can create a metric easier to compute and understand.
    """
    # find all communication kernels
    sym_table = symbol_table.get_sym_table()
    sym_map = symbol_table.get_sym_id_map()
    comm_op_ids = [sym_map[s] for s in sym_table if s.startswith("ncclKernel")]
    iterations = _get_unique_values(df, "iteration")
    ranks = _get_unique_values(df, "rank")
    df = df.loc[df["iteration"].isin(iterations)]

    # filter out short communication kernels with a duration threshold
    n_iters = len(iterations)
    t_min = df["ts"].min()
    mean_iter_time = ((df["ts"] + df["dur"]).max() - t_min) / float(n_iters)
    min_duration = mean_iter_time * min_normalized_duration
    long_comm_kernels = df.loc[
        (df["stream"] > 0) & (df["iteration"] > 0) & (df["dur"] >= min_duration) & (df["name"].isin(comm_op_ids))
    ]
    if visualize:
        plot_timeline_gpu_kernels(
            f"Timeline of Communication Kernels Longer Than {min_normalized_duration * 100:.2f}% of Iteration Time\n",
            long_comm_kernels,
            symbol_table,
            ranks=ranks,
            duration_threshold=2000,
        )

    # select the last long communication kernel for all combinations of (rank, stream, iteration, kernel_name)
    last_long_comm_kernels = long_comm_kernels.groupby(["rank", "stream", "iteration", "name"], as_index=False).last()

    # select the (stream, name) combination whose duration has the largest mean standard deviation across all ranks
    metric_name: str = "normalized_start_time"
    last_long_comm_kernels[metric_name] = (last_long_comm_kernels["ts"] - t_min) / mean_iter_time
    last_long_comm_kernels["duration"] = last_long_comm_kernels["dur"] / mean_iter_time
    duration_diff_across_ranks = last_long_comm_kernels.groupby(["stream", "iteration", "name"])["duration"].std()
    average_duration_diff = duration_diff_across_ranks.groupby(["stream", "name"]).mean()
    (stream, name) = average_duration_diff.idxmax()

    # Filter last_longer_comm_kernels
    candidate_metric_kernels = last_long_comm_kernels.loc[
        (last_long_comm_kernels["stream"].eq(stream)) & (last_long_comm_kernels["name"].eq(name))
    ]

    if visualize:
        plot_timeline_gpu_kernels(
            f"Timeline of Candidate Kernels (Iterations={iterations})",
            candidate_metric_kernels,
            symbol_table,
            ranks=ranks,
        )

    return candidate_metric_kernels, metric_name


def _get_top_k_stragglers_with_metric(
    metric_df: pd.DataFrame,
    metric_name: str,
    n_stragglers: int = 2,
    visualize: bool = False,
) -> pd.Series:
    """
    Get the top k potential straggler ranks.

    Args:
          metric_df (pd.DataFrame) : an input DataFrame which contains the performance metrics
            for determining whether a particular rank could be a straggler.
          metric_name (str) : a column name to select the metric from the DataFrame.
          n_stragglers (int) : how many candidate stragglers to use in this algorithm?
          visualize (bool) : a flag to enable/disable visualizing the results.

    Returns:
        A Series of integers with ranks as the index and the values indicating whether a rank is a potential straggler.
        If the value for a rank <r> is positive, then rank <r> is a potential straggler;
        otherwise it is treated as a non-straggler.

    Notes:
        + This function is built upon the metric derived for straggler detection and applies a candidate selection
            algorithm to select potential straggler candidates.
        + The algorithm implemented here is a k late start candidate which choose the top k candidates ranked by the
            metric values.
        + This function consider two scenarios:
            (1) metric_df only contains metric for one iteration;
            (2) metric_df contains metric for multiple iterations.
    """
    if n_stragglers <= 0:
        n_stragglers = 1
    n_iterations = metric_df["iteration"].nunique() if "iteration" in metric_df.columns else 1
    with_iteration = True if n_iterations > 1 else False
    if not with_iteration:
        metric = metric_df[["rank", metric_name]].copy()
        threshold = metric.sort_values(by=metric_name, ascending=False)[metric_name].tolist()[n_stragglers - 1]
        metric["is_straggler"] = metric[metric_name].apply(lambda x: 1 if x >= threshold else 0)
        data_columns = ["rank", metric_name]
        results = metric[["rank", "is_straggler"]].copy().set_index("rank")["is_straggler"]
    else:
        metric = metric_df[["rank", "iteration", metric_name]].copy()
        metric["is_straggler"] = 0
        for iteration, group in metric.groupby("iteration"):
            if iteration < 0:
                continue
            threshold = group.sort_values(by=metric_name, ascending=False)[metric_name].tolist()[n_stragglers - 1]
            metric.loc[group.index, "is_straggler"] = group[metric_name].apply(lambda x: 1 if x >= threshold else 0)
        data_columns = ["rank", "iteration", metric_name]
        results = metric.groupby("rank")["is_straggler"].sum()

    metric["is_straggler"] = metric["is_straggler"].apply(lambda x: "Yes" if x > 0 else "No")
    for col in data_columns:
        metric[col] = pd.to_numeric(metric[col])

    color_map = {"Yes": "red", "No": "blue"}
    if visualize:
        fig = px.bar(
            metric,
            x="rank",
            y=metric_name,
            facet_col="iteration" if with_iteration else None,
            color="is_straggler",
            color_discrete_map=color_map,
            hover_data=data_columns,
            width=300 * n_iterations,
            title="Potential Stragglers",
        )
        fig.show()

    return results


def find_stragglers_with_late_start_comm_kernels(
    df: pd.DataFrame,
    symbol_table: TraceSymbolTable,
    n_stragglers: int = 2,
    visualize: bool = False,
) -> pd.Series:
    """Identify potential stragglers from the traces.

    Args:
          df (pd.DataFrame) : an input DataFrame with traces for selected ranks and iterations.
          symbol_table (TraceSymbolTable) : the Trace Symbol Table for encoding/decoding the symbols in the trace DataFrame
          n_stragglers (int) : how many candidate stragglers to use in this algorithm?
          visualize (bool) : a flag to enable/disable visualizing the results.

    Returns:
        stragglers (pd.Series): a Series of which ranks are the index and values are the number of times
            that a rank is detected as a potential straggler. For example, given a rank r,
            if stragglers[r] > 0, then rank <r> is detected as a straggler for stragglers[r] times.
    """

    metric_df, metric_name = _compute_normalized_start_time_of_significant_comm_kernels(df, symbol_table, visualize)
    stragglers = _get_top_k_stragglers_with_metric(
        metric_df, metric_name, visualize=visualize, n_stragglers=n_stragglers
    )
    return stragglers
