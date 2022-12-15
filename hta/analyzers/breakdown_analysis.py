# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib_venn import venn2_unweighted, venn3_unweighted
from plotly.subplots import make_subplots

from hta.configs.config import logger
from hta.utils.utils import IdleTimeType, KernelType, get_kernel_type, merge_kernel_intervals

# import statement used without the "if TYPE_CHECKING" guard will cause a circular
# dependency with trace_analysis.py causing mypy to fail and should not be removed.
if TYPE_CHECKING:
    from hta.common.trace import Trace

# This configures the threshold under which we consider gaps between
# kernels to be due to realistic delays in launching back-back kernels on the GPU


class BreakdownAnalysis:
    def __init__(self):
        pass

    @classmethod
    def get_gpu_kernel_breakdown(
        cls,
        t: "Trace",
        visualize: bool = True,
        duration_ratio: float = 0.8,
        num_kernels: int = 10,
        include_memory_kernels: bool = False,
        image_renderer="notebook",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        GPU kernel breakdown implementation. See `get_gpu_kernel_breakdown` in `trace_analysis.py` for details.
        """
        sym_table = t.symbol_table.get_sym_table()

        all_kernel_df = pd.DataFrame(
            {
                "name": pd.Series(dtype="str"),
                "sum": pd.Series(dtype="int"),
                "max": pd.Series(dtype="int"),
                "min": pd.Series(dtype="int"),
                "std": pd.Series(dtype="float"),
                "mean": pd.Series(dtype="int"),
                "kernel_type": pd.Series(dtype="str"),
                "rank": pd.Series(dtype="int"),
            }
        )
        kernel_type_df = pd.DataFrame(
            {
                "kernel_type": pd.Series(dtype="str"),
                "sum": pd.Series(dtype="int"),
            }
        )

        kernel_type_to_analysis: List[str] = [
            KernelType.COMPUTATION.name,
            KernelType.COMMUNICATION.name,
        ]
        if include_memory_kernels:
            kernel_type_to_analysis.append(KernelType.MEMORY.name)

        kernel_per_rank: Dict[str, Dict] = defaultdict(dict)
        for rank, trace_df in t.traces.items():
            gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()
            gpu_kernels["kernel_type"] = gpu_kernels[["name"]].apply(
                lambda x: get_kernel_type(sym_table[x["name"]]), axis=1
            )
            gpu_kernels["name"] = gpu_kernels["name"].apply(lambda x: sym_table[x])

            # Create kernel type dataframe
            kernel_type_df = pd.concat(
                [
                    kernel_type_df,
                    cls._get_gpu_kernel_type_time(gpu_kernels, kernel_type_to_analysis),
                ],
                ignore_index=True,
            )

            # Create all kernel info dataframe
            for kernel_type in kernel_type_to_analysis:
                gpu_kernel_time = gpu_kernels[gpu_kernels["kernel_type"] == kernel_type]

                if kernel_type not in kernel_per_rank:
                    kernel_per_rank[kernel_type] = {}

                gpu_kernel_time = cls._aggr_gpu_kernel_time(
                    gpu_kernel_time,
                    duration_ratio=duration_ratio,
                    num_kernels=num_kernels,
                )

                kernel_per_rank[kernel_type][rank] = gpu_kernel_time

                gpu_kernel_time["kernel_type"] = kernel_type
                gpu_kernel_time["rank"] = int(rank)
                all_kernel_df = pd.concat([all_kernel_df, gpu_kernel_time], ignore_index=True)

        kernel_type_df = kernel_type_df.groupby(by=["kernel_type"])["sum"].agg(["sum"])
        kernel_type_df.reset_index(inplace=True)
        kernel_type_df.sort_values(by=["sum"], ignore_index=True, inplace=True, ascending=False)
        kernel_type_df["percentage"] = (kernel_type_df["sum"] / kernel_type_df["sum"].sum()) * 100
        kernel_type_df = kernel_type_df.round({"percentage": 1})

        all_kernel_df.sort_values(by=["kernel_type", "name", "rank"], ignore_index=True, inplace=True)
        all_kernel_df.rename(
            columns={
                "sum": "sum (ns)",
                "mean": "mean (ns)",
                "max": "max (ns)",
                "min": "min (ns)",
                "std": "stddev",
            },
            inplace=True,
        )

        if visualize:

            if len(kernel_type_to_analysis) == 2:
                comp_overlapping_comm = f"{KernelType.COMPUTATION.name} overlapping {KernelType.COMMUNICATION.name}"

                venn2_unweighted(
                    subsets=(
                        kernel_type_df[kernel_type_df["kernel_type"] == KernelType.COMPUTATION.name]["percentage"]
                        .reset_index(drop=True)
                        .get(0, 0),
                        kernel_type_df[kernel_type_df["kernel_type"] == KernelType.COMMUNICATION.name]["percentage"]
                        .reset_index(drop=True)
                        .get(0, 0),
                        kernel_type_df[kernel_type_df["kernel_type"] == comp_overlapping_comm]["percentage"]
                        .reset_index(drop=True)
                        .get(0, 0),
                    ),
                    set_labels=(
                        KernelType.COMPUTATION.name,
                        KernelType.COMMUNICATION.name,
                    ),
                    set_colors=("orange", "blue"),
                    alpha=0.5,
                )

                plt.title("Kernel Type Percentage (%)")
                plt.show()
            elif len(kernel_type_to_analysis) == 3:

                comp_overlapping_comm = f"{KernelType.COMPUTATION.name} overlapping {KernelType.COMMUNICATION.name}"
                comp_overlapping_mem = f"{KernelType.COMPUTATION.name} overlapping {KernelType.MEMORY.name}"
                comm_overlapping_mem = f"{KernelType.COMMUNICATION.name} overlapping {KernelType.MEMORY.name}"
                comp_overlapping_comm_overlapping_memory = f"{KernelType.COMPUTATION.name} overlapping {KernelType.COMMUNICATION.name} overlapping {KernelType.MEMORY.name}"

                venn3_unweighted(
                    subsets=(
                        kernel_type_df[kernel_type_df["kernel_type"] == KernelType.COMPUTATION.name]["percentage"]
                        .reset_index(drop=True)
                        .get(0, 0),
                        kernel_type_df[kernel_type_df["kernel_type"] == KernelType.COMMUNICATION.name]["percentage"]
                        .reset_index(drop=True)
                        .get(0, 0),
                        kernel_type_df[kernel_type_df["kernel_type"] == comp_overlapping_comm]["percentage"]
                        .reset_index(drop=True)
                        .get(0, 0),
                        kernel_type_df[kernel_type_df["kernel_type"] == KernelType.MEMORY.name]["percentage"]
                        .reset_index(drop=True)
                        .get(0, 0),
                        kernel_type_df[kernel_type_df["kernel_type"] == comp_overlapping_mem]["percentage"]
                        .reset_index(drop=True)
                        .get(0, 0),
                        kernel_type_df[kernel_type_df["kernel_type"] == comm_overlapping_mem]["percentage"]
                        .reset_index(drop=True)
                        .get(0, 0),
                        kernel_type_df[kernel_type_df["kernel_type"] == comp_overlapping_comm_overlapping_memory][
                            "percentage"
                        ]
                        .reset_index(drop=True)
                        .get(0, 0),
                    ),
                    set_labels=(
                        KernelType.COMPUTATION.name,
                        KernelType.COMMUNICATION.name,
                        KernelType.MEMORY.name,
                    ),
                    set_colors=("orange", "blue", "red"),
                    alpha=0.5,
                )

                plt.title("Kernel Type Percentage Across All Ranks")
                plt.show()

            for kernel in kernel_per_rank:
                specs = []
                for rank in kernel_per_rank[kernel]:
                    if rank % 2 == 0:
                        specs.append([{"type": "domain"}, {"type": "domain"}])
                fig = make_subplots(
                    rows=int((len(kernel_per_rank[kernel]) + 1) / 2),
                    cols=2,
                    specs=specs,
                )
                for rank in kernel_per_rank[kernel]:
                    fig.add_trace(
                        go.Pie(
                            labels=kernel_per_rank[kernel][rank]["name"],
                            values=kernel_per_rank[kernel][rank]["sum"],
                            title=f"Rank {rank}",
                        ),
                        int(rank / 2) + 1,
                        int(rank % 2) + 1,
                    )
                image_size_multiplier = 1 + len(t.traces.keys()) / 8
                fig.update_layout(
                    title_text=f'Kernel type "{kernel}" - kernel distribution on each rank',
                    showlegend=False,
                    height=1200 * image_size_multiplier,
                )
                fig.show(renderer=image_renderer)

                kernel_df = all_kernel_df[all_kernel_df["kernel_type"].eq(kernel)]

                kernel_name = kernel_df["name"].unique()
                for name in kernel_name:
                    if name != "others":
                        kernel_name_df = kernel_df[kernel_df["name"].eq(name)]
                        fig = px.bar(
                            kernel_name_df,
                            x="rank",
                            y="mean (ns)",
                            title=name,
                            labels={
                                "rank": "Rank",
                                "mean (ns)": "Mean Duration (ns)",
                            },
                            error_y=kernel_name_df["max (ns)"] - kernel_name_df["mean (ns)"],
                            error_y_minus=kernel_name_df["mean (ns)"] - kernel_name_df["min (ns)"],
                        )
                        fig.update_layout(
                            title_text=f'Kernel type "{kernel}" - {name}',
                            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
                        )
                        fig.show(renderer=image_renderer)

        return kernel_type_df, all_kernel_df

    @classmethod
    def _get_gpu_kernel_type_time(cls, gpu_kernels: pd.DataFrame, kernel_type_to_analysis: List[str]) -> pd.DataFrame:
        overlap_kernel_type_df = pd.DataFrame(
            {
                "status": pd.Series(dtype="str"),
                "time": pd.Series(dtype="int"),
            }
        )

        kernel_t_mapping: Dict[str, int] = defaultdict(int)
        for idx, kernel_type in enumerate(kernel_type_to_analysis):
            value = 1 << idx
            kernel_t_mapping[kernel_type] = value
            kernel_t_df = merge_kernel_intervals(gpu_kernels[gpu_kernels["kernel_type"].eq(kernel_type)].copy())

            overlap_kernel_type_df = (
                pd.concat(
                    [
                        overlap_kernel_type_df,
                        kernel_t_df.melt(var_name="status", value_name="time").replace({"ts": value, "end": -value}),
                    ]
                )
                .sort_values(by="time")
                .reset_index(drop=True)
            )

        overlap_kernel_type_df["running"] = overlap_kernel_type_df["status"].cumsum()
        overlap_kernel_type_df["next_time"] = overlap_kernel_type_df["time"].shift(-1)
        unique_running = overlap_kernel_type_df["running"].unique()
        running_mapping: Dict[int, str] = defaultdict(str)
        for u_running in unique_running:
            if u_running > 0:
                for k_t, v_t in kernel_t_mapping.items():
                    if u_running & v_t:
                        if u_running not in running_mapping:
                            running_mapping[u_running] = k_t
                        else:
                            running_mapping[u_running] = f"{running_mapping[u_running]} overlapping {k_t}"

        overlap_kernel_type_df["kernel_type"] = ""
        overlap_kernel_type_df = overlap_kernel_type_df[overlap_kernel_type_df["running"] > 0]
        for running in running_mapping:
            overlap_kernel_type_df.loc[overlap_kernel_type_df["running"].eq(running), "kernel_type"] = running_mapping[
                running
            ]
        overlap_kernel_type_df["dur"] = (overlap_kernel_type_df["next_time"] - overlap_kernel_type_df["time"]).astype(
            int
        )

        overlap_kernel_type_df = overlap_kernel_type_df.groupby(by=["kernel_type"])["dur"].agg(["sum"])
        overlap_kernel_type_df.reset_index(inplace=True)

        return overlap_kernel_type_df

    @classmethod
    def _aggr_gpu_kernel_time(
        cls,
        gpu_kernel_time: pd.DataFrame,
        duration_ratio: float = 0.8,
        num_kernels: int = 10,
    ) -> pd.DataFrame:
        gpu_kernel_time = gpu_kernel_time.groupby(by=["name"])["dur"].agg(["sum", "max", "min", "mean", "std"])
        gpu_kernel_time.reset_index(inplace=True)
        gpu_kernel_time = gpu_kernel_time.sort_values(by=["sum"], ascending=False, ignore_index=True)
        gpu_kernel_time["std"].fillna(0, inplace=True)

        # if there are more than num_kernels kernels, starting to aggregate kernels
        if gpu_kernel_time.shape[0] > num_kernels:
            gpu_kernel_time["cumsum"] = gpu_kernel_time["sum"].cumsum()
            quantiles = gpu_kernel_time["cumsum"].quantile(duration_ratio)
            gpu_kernel_time.loc[gpu_kernel_time["cumsum"] > quantiles, "name"] = "others"
            gpu_kernel_time.loc[gpu_kernel_time.index >= num_kernels, "name"] = "others"
            gpu_kernel_time = gpu_kernel_time.groupby(by=["name"])["sum"].agg(["sum", "max", "min", "mean", "std"])
            gpu_kernel_time.reset_index(inplace=True)
            gpu_kernel_time["std"].fillna(0, inplace=True)

        return gpu_kernel_time

    @classmethod
    def _get_idle_time_for_kernels(cls, kernels_df: pd.DataFrame) -> Tuple[int, int]:
        """
        Compute idle time for given set of GPU kernels :
          returns :
            idle time (ns) = kernel time - merged execution time of all kernels
            kernel time (ns) = defined as the time difference between end of the
                         last kernel and start of the first kernel.
            PS: we exclude the last profiler iteration while reading trace
            so total time is exclusive of that.
        """
        merged_kernels = merge_kernel_intervals(kernels_df)
        kernel_time = merged_kernels.iloc[-1]["end"] - merged_kernels.iloc[0]["ts"]
        # differences of end - ts are commutative
        kernel_run_time = merged_kernels.end.sum() - merged_kernels.ts.sum()
        return kernel_time - kernel_run_time, kernel_time

    @classmethod
    def get_temporal_breakdown(cls, t: "Trace", visualize: bool = True) -> pd.DataFrame:
        """
        Temporal breakdown implementation. See `get_temporal_breakdown` in `trace_analysis.py` for details.
        """
        sym_table = t.symbol_table.get_sym_table()

        def idle_time_per_rank(trace_df: pd.DataFrame) -> Tuple[int, int, int, int]:
            """returns idle_time (ns) , compute_time (ns), non_compute_time (ns), total_time (ns)"""
            gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()
            idle_time, kernel_time = cls._get_idle_time_for_kernels(gpu_kernels)

            gpu_kernels["kernel_type"] = gpu_kernels[["name"]].apply(
                lambda x: get_kernel_type(sym_table[x["name"]]), axis=1
            )

            # Isolate computation kernels and merge each one of them.
            comp_kernels = merge_kernel_intervals(
                gpu_kernels[gpu_kernels["kernel_type"].eq(KernelType.COMPUTATION.name)].copy()
            )
            compute_time = comp_kernels.end.sum() - comp_kernels.ts.sum()
            non_compute_time = kernel_time - compute_time - idle_time

            assert idle_time <= kernel_time
            assert compute_time <= kernel_time
            assert non_compute_time >= 0

            return idle_time, compute_time, non_compute_time, kernel_time

        result: Dict[str, List[float]] = defaultdict(list)
        for rank, trace_df in t.traces.items():
            result["rank"].append(rank)
            idle_time, compute_time, non_compute_time, kernel_time = idle_time_per_rank(trace_df)
            result["idle_time(ns)"].append(idle_time)
            result["compute_time(ns)"].append(compute_time)
            result["non_compute_time(ns)"].append(non_compute_time)
            result["kernel_time(ns)"].append(kernel_time)

        result_df = pd.DataFrame(result)
        result_df["idle_time"] = result_df["idle_time(ns)"] / result_df["kernel_time(ns)"]
        result_df["idle_time_pctg"] = round(100 * result_df["idle_time"], 2)
        result_df["compute_time"] = result_df["compute_time(ns)"] / result_df["kernel_time(ns)"]
        result_df["compute_time_pctg"] = round(100 * result_df["compute_time"], 2)
        result_df["non_compute_time"] = result_df["non_compute_time(ns)"] / result_df["kernel_time(ns)"]
        result_df["non_compute_time_pctg"] = round(100 * result_df["non_compute_time"], 2)

        if visualize:
            fig = px.bar(
                result_df,
                x="rank",
                y=["idle_time", "compute_time", "non_compute_time"],
                title="Temporal breakdown across ranks",
                labels={
                    "rank": "Rank",
                },
            )
            fig.update_layout(
                yaxis_tickformat=".2%",
                yaxis_title="Percentage",
                legend_title="Time Breakdown",
            )
            fig.show()

        return result_df[
            [
                "rank",
                "idle_time(ns)",
                "compute_time(ns)",
                "non_compute_time(ns)",
                "kernel_time(ns)",
                "idle_time_pctg",
                "compute_time_pctg",
                "non_compute_time_pctg",
            ]
        ]

    @classmethod
    def _analyze_idle_time_for_stream(
        cls,
        stream: int,
        gpu_kernels: pd.DataFrame,
        consecutive_kernel_delay: int,
        show_idle_interval_stats=False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Analyze a specific CUDA stream for idle time breakdown on it.

        stream (int): CUDA stream to consider.
        gpu_kernels: a dataframe of GPU kernels in a rank.

        returns
        1) dataframe with idle time breakdown.
        1) optional dataframe showing idle interval statistics.
        """
        logger.info(f"Processing stream {stream}")
        idle_interval_stats: Optional[pd.DataFrame] = None

        gpu_kernels_s = gpu_kernels[gpu_kernels.stream == stream].copy().sort_values(by="ts")

        gpu_kernels_s["end_ts"] = gpu_kernels_s.ts + gpu_kernels_s.dur
        gpu_kernels_s["prev_end_ts"] = gpu_kernels_s.end_ts.shift(1)
        gpu_kernels_s["idle_interval"] = gpu_kernels_s["ts"] - gpu_kernels_s["prev_end_ts"]

        # Default idle time category
        gpu_kernels_s["idle_category"] = IdleTimeType.OTHER.value

        """
        Host wait:
            If the current kernel's runtime started after previous kernel's end time
            this means the Host/CPU was not enqueuing kernels fast enough.
        CPU  Runtime 0             Runtime 1
        GPU     |--------Kernel 0      |-----------Kernel 1
        """
        is_host_wait = gpu_kernels_s["ts_runtime"] > gpu_kernels_s["prev_end_ts"]
        gpu_kernels_s.loc[is_host_wait, "idle_category"] = IdleTimeType.HOST_WAIT.value

        """
        Kernel wait:
            If the gap between kernels is below a threshold the idle time is
            likely due to the overhead for launching kernels.
        """
        is_kernel_kernel_delay = ~is_host_wait & (gpu_kernels_s["idle_interval"] < consecutive_kernel_delay)
        gpu_kernels_s.loc[is_kernel_kernel_delay, "idle_category"] = IdleTimeType.KERNEL_WAIT.value

        gpu_kernels_groupby = gpu_kernels_s.groupby("idle_category")
        if show_idle_interval_stats:
            logger.info(f"Computing descriptive statistics for idle time intervals on stream {stream}:")
            idle_interval_stats = gpu_kernels_groupby.idle_interval.describe()
            idle_interval_stats.insert(0, "stream", stream)

        result = pd.DataFrame(gpu_kernels_groupby.idle_interval.sum())
        total_idle_time = result.idle_interval.sum()
        result["stream"] = stream
        result["idle_time_ratio"] = result["idle_interval"] / total_idle_time
        result.rename(columns={"idle_interval": "idle_time"}, inplace=True)
        return result, idle_interval_stats

    @classmethod
    def get_idle_time_breakdown(
        cls,
        t: "Trace",
        consecutive_kernel_delay: int,
        rank: int = 0,
        streams: Optional[List[int]] = None,
        visualize: bool = True,
        visualize_pctg: bool = True,
        show_idle_interval_stats=False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Breakdown Idle time by host wait, kernel wait and other categories. See full description in trace_analysis.py

        consecutive_kernel_delay (int): configures the threshold under which we consider gaps between
           kernels to be due to realistic delays in launching back-back kernels on the GPU. Time is in ns.
        rank (int): the rank to analyze
        streams (List[int]): list of streams to provide analysis for.
            Defaults to all streams.
        visualize (bool): show the visualization chart or not (default = True).
        visualize_pctg (bool): show relative percentage across streams (default = True).
        show_idle_interval_stats (bool): prints statistics of the idle intervals like the min, max
           and median of idle intervals between kernels on a CUDA stream, also broken down by
           the idleness category (default = False).
        """
        trace_df: pd.DataFrame = t.get_trace(rank)
        gpu_kernels_pre = trace_df[trace_df["stream"].ne(-1)].copy().set_index("index_correlation")

        # correlate with the runtime event whenever possible
        gpu_kernels = gpu_kernels_pre.join(trace_df[["ts", "index"]], on="index_correlation", rsuffix="_runtime")

        if streams is None or len(streams) == 0:
            streams = list(gpu_kernels.stream.unique())

        result_list: List[pd.DataFrame] = []
        interval_stats_list: List[pd.DataFrame] = []

        for stream in streams:
            breakdown_df, idle_interval_df = cls._analyze_idle_time_for_stream(
                stream,
                gpu_kernels,
                consecutive_kernel_delay,
                show_idle_interval_stats,
            )
            result_list.append(breakdown_df)
            if idle_interval_df is not None:
                interval_stats_list.append(idle_interval_df)

        result_df = pd.concat(result_list)

        idle_category_name_map = {member.value: name.lower() for name, member in IdleTimeType.__members__.items()}
        result_df.rename(mapper=idle_category_name_map, axis=0, inplace=True)
        result_df.reset_index(inplace=True)

        if visualize:
            result_df["stream"] = result_df.stream.astype(str)
            ycol = "idle_time_ratio" if visualize_pctg else "idle_time"
            fig = px.bar(
                result_df,
                x="stream",
                y=ycol,
                color="idle_category",
                hover_data=["idle_time", "idle_time_ratio"],
                title=f"Idle time breakdown on rank {rank} per CUDA stream",
            )
            if visualize_pctg:
                fig.update_layout(
                    yaxis_tickformat=".2%",
                    yaxis_title="Percentage",
                    legend_title="Idle Time Breakdown",
                )
            else:
                fig.update_layout(yaxis_title="Idle time (ns)", legend_title="Idle Time Breakdown")
            fig.show()

        result_df["rank"] = rank
        interval_stats_df = pd.concat(interval_stats_list).round(2) if show_idle_interval_stats else None
        if interval_stats_df is not None:
            # add rank column to the starting
            interval_stats_df.insert(0, "rank", rank)
            interval_stats_df.rename(mapper=idle_category_name_map, axis=0, inplace=True)

        result_df = result_df[["rank", "stream", "idle_category", "idle_time", "idle_time_ratio"]].round(2)

        return result_df, interval_stats_df
