# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import pandas as pd

from hta.common.trace import Trace
from hta.configs.config import logger
from hta.utils.utils import KernelType, get_kernel_type, get_memory_kernel_type


class TraceCounters:
    def __init__(self):
        pass

    @classmethod
    def _get_queue_length_time_series_for_rank(cls, t: "Trace", rank: int) -> pd.DataFrame:
        # get trace for a rank
        trace_df: pd.DataFrame = t.get_trace(rank)

        # cudaLaunchKernel, cudaMemcpyAsync, cudaMemsetAsync
        sym_index = t.symbol_table.get_sym_id_map()
        cudaLaunchKernel_id = sym_index["cudaLaunchKernel"]
        cudaMemcpyAsync_id = sym_index["cudaMemcpyAsync"]
        cudaMemsetAsync_id = sym_index["cudaMemsetAsync"]

        # CUDA Runtime events that may launch kernels
        # - filter events that have a correlated kernel event only.
        runtime_calls: pd.DataFrame = trace_df.query(
            "((name == @cudaMemsetAsync_id) or (name == @cudaMemcpyAsync_id) or "
            " (name == @cudaLaunchKernel_id))"
            "and (index_correlation > 0)"
        ).copy()
        runtime_calls.drop(["stream", "pid", "tid"], axis=1, inplace=True)
        runtime_calls["queue"] = 1

        # GPU kernel events
        gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()
        gpu_kernels["queue"] = -1

        # Concat the series of runtime launch events and GPU kernel events
        merged_df = (
            pd.concat(
                [
                    # use the pid, tid and cuda stream from the correlated GPU event.
                    runtime_calls.join(
                        gpu_kernels[["stream", "pid", "tid", "correlation"]].set_index("correlation"),
                        on="correlation",
                    ),
                    gpu_kernels,
                ]
            )
            .sort_values(by="ts")
            .set_index("index")
        )

        result_df_list = []
        for stream, stream_df in merged_df.groupby("stream"):
            logger.debug(f"Processing queue_length for rank {rank}, stream {stream}")
            stream_df["queue_length"] = stream_df["queue"].cumsum()
            result_df_list.append(stream_df)

        return pd.concat(result_df_list)[["ts", "pid", "tid", "stream", "queue_length"]]

    @classmethod
    def get_queue_length_time_series(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Returns a dictionary of rank -> time series for the queue length of a CUDA stream.

        Queue length is defined as the number of outstanding CUDA operations on a stream
        The value of the queue length is:
        1. Incremented when a CUDA runtime operation enqueues a kernel on a stream.
        3. Decremented when a CUDA kernel/memcopy operation executes on a stream.

        The dataframe returned contains time series points with columns
        - ts (timestamp), pid, tid (of corresponding GPU,stream), stream,
          and queue_length.
        Note that each row or time point shows a changes in the value of the time series.
        The value remains constant until the next time point. In essence, you can think
        of it like a step function that keeps changing.

        Args:
            t (Trace): Input trace data structure.
            ranks (list of int): ranks to perform this analysis for.
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        logger.info(
            "Please note that the time series only contains points "
            "when the value changes. Once a values is observed the time series "
            "stays constant until the next update."
        )

        return {rank: TraceCounters._get_queue_length_time_series_for_rank(t, rank) for rank in ranks}

    @classmethod
    def get_queue_length_summary(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Returns a dataframe with queue length statistics per CUDA stream and rank.
        We summarize queue length per stream and rank using-
            count, min, max, std-deviation, 25, 50th and 75th percentiles.
        The summary uses the pandas describe() function.

        Args:
            t (Trace): Input trace data structure.
            ranks (list of int): ranks to perform this analysis for.
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        results_list: List[pd.DataFrame] = []

        for rank, rank_df in TraceCounters.get_queue_length_time_series(t, ranks).items():
            rank_df["rank"] = rank
            result = rank_df[["rank", "stream", "queue_length"]].groupby(["rank", "stream"]).describe()
            results_list.append(result)
        return pd.concat(results_list)

    @classmethod
    def _get_memory_bw_time_series_for_rank(cls, t: "Trace", rank: int) -> pd.DataFrame:
        # get trace for a rank
        trace_df: pd.DataFrame = t.get_trace(rank)
        sym_table = t.symbol_table.get_sym_table()

        gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()
        gpu_kernels["kernel_type"] = gpu_kernels[["name"]].apply(
            lambda x: get_kernel_type(sym_table[x["name"]]), axis=1
        )

        memcpy_kernels = gpu_kernels[gpu_kernels.kernel_type == KernelType.MEMORY.name].copy()
        memcpy_kernels["name"] = memcpy_kernels[["name"]].apply(
            lambda x: get_memory_kernel_type(sym_table[x["name"]]), axis=1
        )

        membw_time_series_a = memcpy_kernels[["ts", "name", "pid", "memory_bw_gbps"]]
        membw_time_series_b = memcpy_kernels[["ts", "name", "dur", "pid", "memory_bw_gbps"]].copy()

        # The end events have timestamps = start timestamp + duration
        membw_time_series_b.ts = membw_time_series_b.ts + membw_time_series_b.dur
        membw_time_series_b.memory_bw_gbps = -membw_time_series_b.memory_bw_gbps

        membw_time_series = pd.concat(
            [
                membw_time_series_a,
                membw_time_series_b[["ts", "pid", "name", "memory_bw_gbps"]],
            ],
            ignore_index=True,
        ).sort_values(by="ts")

        result_df_list = []
        for _, membw_df in membw_time_series.groupby("name"):
            membw_df.memory_bw_gbps = membw_df.memory_bw_gbps.cumsum()
            result_df_list.append(membw_df)
        result_df = pd.concat(result_df_list)[["ts", "pid", "name", "memory_bw_gbps"]]
        result_df["tid"] = 0
        return result_df

    @classmethod
    def get_memory_bw_time_series(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Returns a dictionary of rank -> time series for the memory bandwidth.

        The dataframe returned contains time series points with columns
        - ts (timestamp), pid (of corresponding GPU), name of memory copy type
          and memory_bw_gbps - memory bandwidth in GB/sec
        Args:
            t (Trace): Input trace data structure.
            ranks (list of int): ranks to perform this analysis for.
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        logger.info(
            "Please note that the time series only contains points "
            "when the value changes. Once a values is observed the time series "
            "stays constant until the next update."
        )
        return {rank: TraceCounters._get_memory_bw_time_series_for_rank(t, rank) for rank in ranks}

    @classmethod
    def get_memory_bw_summary(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Returns a dataframe with memory copy bandwidth statistic per rank and
        memory/memset copy type.
        We summarize memory bandwidth by
            count, min, max, std-deviation, 25, 50th and 75th percentiles.
        The summary uses the pandas describe() function.
        Args:
            t (Trace): Input trace data structure.
            ranks (list of int): ranks to perform this analysis for.
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        results_list: List[pd.DataFrame] = []

        for rank, rank_df in TraceCounters.get_memory_bw_time_series(t, ranks).items():
            rank_df["rank"] = rank
            # Exclude the 0 points in time series
            rank_df = rank_df[rank_df.memory_bw_gbps > 0]

            result = rank_df[["rank", "name", "memory_bw_gbps"]].groupby(["rank", "name"]).describe()
            results_list.append(result)
        return pd.concat(results_list)
