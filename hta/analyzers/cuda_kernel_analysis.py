# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import plotly.express as px

from hta.common.call_stack import CallGraph
from hta.common.trace import Trace
from hta.configs.config import logger


class CudaKernelAnalysis:
    def __init__(self):
        pass

    @classmethod
    def get_frequent_cuda_kernel_sequences(
        cls,
        t: "Trace",
        operator_name: str,
        output_dir: str,
        min_pattern_len: int = 3,
        rank: int = 0,
        top_k: int = 5,
        visualize: bool = True,
        compress_other_kernels: bool = True,
    ) -> pd.DataFrame:
        """
        Frequent CUDA kernel sequences implementation. See `get_frequent_cuda_kernel_sequences` in `trace_analysis.py`
        for details.
        """
        if not operator_name:
            logger.error("operator_name must be a non-empty string and a valid operator name in the trace file.")
            return pd.DataFrame()

        if not output_dir:
            logger.error("output_dir must be a non-empty string and a valid folder name.")
            return pd.DataFrame()

        # check output_dir is a valid path and a directory
        path = Path(output_dir)
        if not path.exists():
            logger.error(f"The path {str(path)} does not exist.")
            return pd.DataFrame()
        if not path.is_dir():
            logger.error(f"{output_dir} is not a directory.")
            return pd.DataFrame()

        sym_index = t.symbol_table.get_sym_id_map()
        cg = CallGraph(t, ranks=[rank])
        trace_df = t.get_trace(rank)

        cpu_kernels = trace_df[trace_df["stream"].eq(-1)].copy()
        gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()

        # get all the CPU operators which contain operator_name in their names
        candidate_root_idx = [idx for name, idx in sym_index.items() if operator_name in name]
        candidate_nodes = cpu_kernels[cpu_kernels["name"].isin(candidate_root_idx)]
        # to avoid double-counting when the same CPU operators appear multiple times in the call graph
        # hierarchy, only start the search from the shallowest CPU operators
        min_depth = candidate_nodes["depth"].min()
        root_nodes = candidate_nodes[candidate_nodes["depth"].eq(min_depth)][["index", "dur"]]

        pattern_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        # duration of the pattern: [GPU kernel duration, CPU op duration]
        pattern_durations: Dict[Tuple[int, ...], List[int]] = defaultdict(lambda: [0, 0])
        # record the indices of frequent patterns.
        pattern_occurrences: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)

        for cs in cg.call_stacks:
            for _, root_node, root_dur in root_nodes.itertuples():
                if root_node in cs.get_nodes():
                    # find all cuda kernels launched by the root_node by joining GPU kernels with leaf nodes.
                    cuda_kernels = gpu_kernels.merge(
                        cpu_kernels[cpu_kernels["index"].isin(cs.get_leaf_nodes(root_node))][["correlation"]],
                        on="correlation",
                        how="inner",
                    ).sort_values(by="ts")
                    # discard the pattern if it is too short
                    if len(cuda_kernels) < min_pattern_len:
                        continue
                    cpu_operator = cpu_kernels[cpu_kernels["index"].eq(root_node)]
                    pattern: Tuple[int, ...] = tuple(pd.concat([cpu_operator["name"], cuda_kernels["name"]]))
                    pattern_counts[pattern] += 1
                    # update the duration for GPU kernels in the pattern
                    pattern_durations[pattern][0] += cuda_kernels["dur"].sum()
                    # update the duration for CPU operators in the pattern
                    pattern_durations[pattern][1] += root_dur
                    for p in cs.get_paths_to_leaves(root_node):
                        pattern_occurrences[pattern].update(p)
                    pattern_occurrences[pattern].update(cuda_kernels["index"])

        if not pattern_counts:
            logger.error("operator_name not found in the trace file, or no frequent cuda kernel sequences found.")
            return pd.DataFrame()

        return cls._generate_frequent_pattern_results(
            t,
            pattern_counts,
            pattern_durations,
            pattern_occurrences,
            rank,
            top_k,
            visualize,
            output_dir,
            compress_other_kernels,
        )

    @classmethod
    def _generate_frequent_pattern_results(
        cls,
        t: "Trace",
        pattern_counts: Dict[Tuple[int, ...], int],
        pattern_durations: Dict[Tuple[int, ...], List[int]],
        pattern_occurrences: Dict[Tuple[int, ...], Set[int]],
        rank: int,
        top_k: int,
        visualize: bool,
        output_dir: str,
        compress_other_kernels: bool = True,
    ) -> pd.DataFrame:
        """
        Post-process the frequent patterns:
        - generate the final dataframe;
        - overlay the frequent pattern results on the raw trace file;
        - create inline visualization.

        Args:
            pattern_counts (Dict[Tuple[int, ...], int]): frequent patterns and their counts
            pattern_durations (Dict[Tuple[int, ...], List[int]]): frequent patterns and their total GPU kernel
                                                                  and CPU op duration
            pattern_occurrences (Dict[Tuple[int, ...], Set[int]]): frequent patterns and where they happen
            rank (int): rank number on which the analysis should be performed on
            top_k (int): top_k patterns in terms of frequency to be visualized and overlaid
            visualize (bool): whether to show the histogram of top_k frequent patterns inline
            output_dir (str): path to the folder containing the new trace file with overlaid top k
                              frequent patterns
            compress_other_kernels (bool): should the names and args for other kernels not belonging to
                                           any frequent patterns be compressed to save memory in the overlaid
                                           trace file
        Returns:
            patterns_df (pd.DataFrame): a dataframe with all patterns and their frequencies
        """
        output_file = os.path.join(str(Path(output_dir)), "overlaid_" + t.trace_files[rank].split("/")[-1])

        sym_table = t.symbol_table.get_sym_table()
        patterns_result: Dict[str, List[Union[int, str, Set[int]]]] = defaultdict(list)
        for pattern, count in pattern_counts.items():
            patterns_result["pattern"].append("|".join(sym_table[x] for x in pattern))
            patterns_result["count"].append(count)
            patterns_result["GPU kernel duration (ns)"].append(pattern_durations[pattern][0])
            patterns_result["CPU op duration (ns)"].append(pattern_durations[pattern][1])
            patterns_result["pattern_indices"].append(pattern_occurrences[pattern])

        patterns_df = pd.DataFrame(patterns_result).sort_values(
            by=["count", "pattern"], ascending=[False, True], ignore_index=True
        )
        if top_k > len(patterns_df):
            top_k = len(patterns_df)
            logger.info("The top_k argument value exceeds the number of patterns. Displaying all patterns.")

        overlaid_trace = cls._overlay_frequent_patterns_with_trace(t, patterns_df[:top_k], rank, compress_other_kernels)
        t.write_raw_trace(output_file, overlaid_trace)
        logger.info(f"Overlaid trace file for rank {rank} has been generated at {output_file}.")
        logger.info(
            'View the generated trace file using Chrome Tracing and search for "Patterns" to highlight the frequent patterns.'
        )

        if visualize:
            vis_df = patterns_df[:top_k].copy()
            # show the pattern in multiple lines in visualization
            vis_df["pattern"] = vis_df["pattern"].str.replace("|", "<br>", regex=False)
            fig = px.bar(
                vis_df,
                x=vis_df.index,
                y="count",
                hover_data=["pattern", "count"],
                title=f"Top {top_k} frequent patterns for rank {rank}",
                labels={"x": "Pattern Index", "count": "Pattern Count"},
            )
            fig.update_xaxes(type="category")
            fig.show()
        return patterns_df.drop("pattern_indices", axis=1)

    @classmethod
    def _overlay_frequent_patterns_with_trace(
        cls,
        t: "Trace",
        patterns_df: pd.DataFrame,
        rank: int = 0,
        compress_other_kernels: bool = True,
    ) -> Dict[str, Any]:
        """
        Overlay the identified frequent patterns on top of the trace file for visualization.

        Args:
            patterns_df (pd.DataFrame): a dataframe with all patterns and their frequencies
            rank (int): the rank number on which the analysis was performed on
            compress_other_kernels (bool): should the names and args for other kernels not belonging to
                                           any frequent patterns be compressed to save memory in the overlaid
                                           trace file
        Returns:
            raw_trace_content (Dict[str, Any]): the dictionary form of the overlaid trace file that can be dumped
        """
        raw_trace_content = t.get_raw_trace_for_one_rank(rank=rank)
        raw_trace_df = pd.DataFrame(raw_trace_content["traceEvents"]).reset_index()
        raw_trace_df["index"] = pd.to_numeric(raw_trace_df["index"], downcast="integer")
        sym_id_map = t.symbol_table.get_sym_id_map()

        # get a counter of patterns that each CPU operator/GPU kernel is in.
        top_patterns_df = (
            patterns_df.explode("pattern_indices")
            .groupby("pattern_indices", as_index=False)
            .apply(lambda g: pd.Series({"active_patterns": {r["pattern"]: r["count"] for _, r in g.iterrows()}}))
        )
        # join the pattern counter with the original events on their indices
        merged_df = raw_trace_df.merge(top_patterns_df, left_on="index", right_on="pattern_indices", how="left")

        def _add_patterns_to_args(row, compress_other_kernels):
            args = row["args"]
            if pd.isna(row["pattern_indices"]):
                # only drop complete events to ensure important information is retained
                return {} if compress_other_kernels and row["ph"] == "X" else args
            # add the frequent patterns to the args field
            if pd.isna(args):
                args = {}
            args["Patterns"] = row["active_patterns"]
            return args

        def _compress_kernel_names(row):
            # do not compress events that are part of a frequent pattern, or we don't know how to compress
            if row["args"] or row["name"] not in sym_id_map:
                return row["name"]
            # otherwise return the symbol id created during trace parsing of that event
            return str(sym_id_map[row["name"]])

        # add the frequent patterns information to the args field so that it is searchable
        merged_df["args"] = merged_df.apply(lambda r: _add_patterns_to_args(r, compress_other_kernels), axis=1)

        if compress_other_kernels:
            # compress kernel/operator names
            merged_df["name"] = merged_df.apply(_compress_kernel_names, axis=1)
            # add the symbol id -> event name mapping information to the PyTorch Profiler event for reference
            profiler_event = raw_trace_df[
                (raw_trace_df["name"].str.contains("PyTorch Profiler")) & (raw_trace_df["pid"].eq("Spans"))
            ]
            merged_df.loc[profiler_event.index, "args"] = [{str(i): name for name, i in sym_id_map.items()}]
        # drop unused columns before writing back to the overlaid trace file
        merged_df.drop(["index", "active_patterns", "pattern_indices"], axis=1, inplace=True)
        raw_trace_content["traceEvents"] = list(merged_df.apply(lambda row: row.dropna().to_dict(), axis=1))
        return raw_trace_content

    @classmethod
    def visualize_cuda_kernel_launch_stats(
        cls, rank: int, df: pd.DataFrame, runtime_cutoff: int, launch_delay_cutoff: int
    ) -> None:
        short_kernels = df[(df["cpu_duration"] <= runtime_cutoff) & (df["gpu_duration"] < df["cpu_duration"])]
        runtime_outliers = df[df["cpu_duration"] > runtime_cutoff]
        launch_delay_outliers = df[df["launch_delay"] > launch_delay_cutoff]
        fig1 = px.histogram(
            short_kernels,
            x=short_kernels["cpu_duration"],
            title="Short GPU kernels (gpu op duration < cpu op duration) on rank %d" % rank,
            labels={
                "cpu_duration": "Cuda Runtime Event Duration (us)",
            },
        )
        fig1.show()

        fig2 = px.histogram(
            runtime_outliers,
            x=runtime_outliers["cpu_duration"],
            nbins=300,
            title="Runtime Event Duration Outliers (duration > %s us)  on rank %d" % (runtime_cutoff, rank),
            labels={
                "cpu_duration": "Cuda Runtime Event Duration (us)",
            },
        )
        fig2.show()

        fig3 = px.histogram(
            launch_delay_outliers,
            x=launch_delay_outliers["launch_delay"],
            nbins=300,
            title="Launch Delay Outliers (duration > %s us) on rank %d" % (launch_delay_cutoff, rank),
            labels={
                "launch_delay": "Launch Delay Duration (us)",
            },
        )
        fig3.show()

    @classmethod
    def cuda_kernel_launch_stats(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
        runtime_cutoff: int = 50,
        launch_delay_cutoff: int = 100,
        include_memory_events: bool = True,
        visualize: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        """
        CUDA kernel launch statistics implementation. For details see `get_cuda_launch_kernel_info` in `trace_analysis.py`.
        """
        if ranks is None or ranks == []:
            ranks = [0]

        result_dict: Dict = {}
        sym_index = t.symbol_table.get_sym_id_map()

        for rank in ranks:
            # get trace for a rank
            trace_df: pd.DataFrame = t.get_trace(rank)

            # filter out events which have correlation value matching to
            # cudaLaunchKernel, cudaMemcpyAsync, cudaMemsetAsync
            cudaLaunchKernel_id = sym_index["cudaLaunchKernel"]
            cudaMemcpyAsync_id = sym_index["cudaMemcpyAsync"]
            cudaMemsetAsync_id = sym_index["cudaMemsetAsync"]

            # get correlation id's of cudaLaunchKernel events
            cudaLaunchKernel_correlation_series: pd.Series = trace_df[
                trace_df["name"] == cudaLaunchKernel_id
            ].correlation

            # whether to use memory events - cudaMemsetAsync and cudaMemcpyAsync.
            if include_memory_events:
                memory_event_correlation_series: pd.Series = trace_df[
                    (trace_df["name"] == cudaMemsetAsync_id) | (trace_df["name"] == cudaMemcpyAsync_id)
                ].correlation
                merged_series: pd.Series = pd.concat(
                    [
                        cudaLaunchKernel_correlation_series,
                        memory_event_correlation_series,
                    ]
                )
            else:
                merged_series = cudaLaunchKernel_correlation_series

            # filter cpu and gpu ops
            cpu_kernels = trace_df[trace_df["stream"].eq(-1)].copy()
            gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()

            # get correlation id, duration, timestamp and name of cpu and gpu events
            cpu_kernels_filtered = cpu_kernels[(cpu_kernels["correlation"]).isin(merged_series)][
                ["correlation", "dur", "name", "ts"]
            ]
            gpu_kernels_filtered = gpu_kernels[(gpu_kernels["correlation"]).isin(merged_series)][
                ["correlation", "dur", "name", "ts"]
            ]

            # join the dataframes created above on correlation. This is required to calculate the launch delay
            # for each correlation id.
            joined_df = pd.merge(
                cpu_kernels_filtered,
                gpu_kernels_filtered,
                how="inner",
                on="correlation",
            )
            joined_df["launch_delay"] = joined_df["ts_y"] - joined_df["ts_x"] + joined_df["dur_x"]

            # rename columns and select the required columns from the final dataframe
            renamed_df = joined_df.rename(columns={"dur_x": "cpu_duration", "dur_y": "gpu_duration"})
            events_df = renamed_df[["correlation", "cpu_duration", "gpu_duration", "launch_delay"]]

            if visualize:
                cls.visualize_cuda_kernel_launch_stats(rank, events_df, runtime_cutoff, launch_delay_cutoff)

            result_dict[rank] = events_df
        return result_dict
