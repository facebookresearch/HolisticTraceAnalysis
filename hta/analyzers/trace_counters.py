# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List, Optional

import pandas as pd
from hta.common.trace import Trace
from hta.configs.config import logger
from hta.utils.utils import get_kernel_type, get_memory_kernel_type, KernelType

# The order of a counter's points at one timestamp. The ends of intervals of
# positive length count first, then the starts of such intervals, then each
# zero-length interval, its start just before its end. Each level is then a
# sum over intervals that hold its timestamp under [start, end): it never
# adds an interval that ends at the timestamp to one that starts there. A
# zero-length interval keeps a point of its own and adds to no other
# zero-length interval.
_ORDER_END = 0
_ORDER_START = 1
_ORDER_POINT = 2


class TraceCounters:
    def __init__(self):
        pass

    @classmethod
    def _get_queue_length_time_series_for_rank(
        cls,
        t: "Trace",
        rank: int,
    ) -> Optional[pd.DataFrame]:
        """
        Returns an (optional) dataframe with time series for the queue length
        on a CUDA streams within requested rank.

        Queue length is defined as the number of outstanding CUDA operations on a stream
        The value of the queue length is:
        1. Incremented when a CUDA runtime operation enqueues a kernel on a stream.
        2. Decremented when a CUDA kernel/memcopy operation executes on a stream.

        Args:
            t (Trace): Input trace data structure.
            rank (int): rank to generate the time series for.

        Returns:
            Optional[pd.DataFrame]
                Returns an (optional) dataframe containing time series with the following
                columns: ts (timestamp), pid, tid (of corresponding GPU, stream), stream and
                queue_length.

                Note that each row or timestamp denotes a change in the value of the
                time series. The value remains constant until the next timestamp.
                In essence, it can be thought of as a step function.
        """
        # get trace for a rank
        trace_df: pd.DataFrame = t.get_trace(rank)

        # CUDA Runtime events that may launch kernels
        # - filter events that have a correlated kernel event only.
        runtime_calls: pd.DataFrame = trace_df[
            t.symbol_table.get_runtime_launch_events_mask(trace_df)
        ].copy()
        runtime_calls.drop(["stream", "pid", "tid"], axis=1, inplace=True)
        runtime_calls["queue"] = 1

        # GPU kernel events
        gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()
        gpu_kernels["queue"] = -1

        # use the pid, tid and cuda stream from the correlated GPU event. Each
        # launch also takes the start and the row index of its GPU event, so
        # that the launch and the start of one queue entry are a pair.
        runtime_calls_filt = runtime_calls.join(
            gpu_kernels[["stream", "pid", "tid", "correlation", "ts", "index"]]
            .rename(columns={"ts": "_gpu_ts", "index": "_pair"})
            .set_index("correlation"),
            on="correlation",
        )

        # filter GPU kernels with valid parent runtime events only, use a semi-join
        gpu_kernels_filt = gpu_kernels[
            gpu_kernels["correlation"].isin(runtime_calls["correlation"])
        ].copy()

        assert len(runtime_calls_filt) == len(gpu_kernels_filt)

        # A queue entry is the interval [launch, start of its GPU event).
        gpu_kernels_filt["_pair"] = gpu_kernels_filt["index"]
        launch_ts = gpu_kernels_filt["_pair"].map(
            runtime_calls_filt.groupby("_pair")["ts"].min()
        )
        runtime_calls_filt["_order"] = _ORDER_START
        runtime_calls_filt.loc[
            runtime_calls_filt["ts"].eq(runtime_calls_filt["_gpu_ts"]), "_order"
        ] = _ORDER_POINT
        runtime_calls_filt["_phase"] = 0
        gpu_kernels_filt["_order"] = _ORDER_END
        gpu_kernels_filt.loc[gpu_kernels_filt["ts"].eq(launch_ts), "_order"] = (
            _ORDER_POINT
        )
        gpu_kernels_filt["_phase"] = 1

        # At one timestamp, points count in the _ORDER_* order, then by pid,
        # tid and queue entry. Points equal in all of these except the entry
        # give equal rows, so the result does not depend on the row order of
        # the trace. A launch counts before the GPU event it enqueued.
        merged_df = cast(
            pd.DataFrame, pd.concat([runtime_calls_filt, gpu_kernels_filt])
        )
        merged_df = merged_df.sort_values(
            by=["ts", "_order", "pid", "tid", "_pair", "_phase"],
            kind="stable",
            na_position="first",
        ).set_index("index")

        result_df_list = []
        for stream, stream_df in merged_df.groupby("stream"):
            logger.debug(f"Processing queue_length for rank {rank}, stream {stream}")
            stream_df["queue_length"] = stream_df["queue"].cumsum()
            result_df_list.append(stream_df)

        return (
            pd.concat(result_df_list)[["ts", "pid", "tid", "stream", "queue_length"]]
            if len(result_df_list) > 0
            else None
        )

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
        2. Decremented when a CUDA kernel/memcopy operation executes on a stream.

        Args:
            t (Trace): Input trace data structure.
            rank (int): rank to perform this analysis for.

        Returns:
            Dict[int, pd.DataFrame]:
                A dictionary of rank -> time series with the queue length of each CUDA stream.
                Each dataframe contains a time series consisting of the following columns:
                ts (timestamp), pid, tid (of corresponding GPU, stream), stream and queue_length.

                Note that each row or timestamp shows a change in the value of the
                time series. The value remains constant until the next timestamp.
                In essence, it can be thought of as a step function.
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        logger.info(
            "Please note that the time series only contains points "
            "when the value changes. Once a values is observed the time series "
            "stays constant until the next update."
        )

        result = {
            rank: TraceCounters._get_queue_length_time_series_for_rank(t, rank)
            for rank in ranks
        }
        # pyrefly: ignore [bad-return]
        return dict(filter(lambda x: x[1] is not None, result.items()))

    @classmethod
    def get_queue_length_summary_from_time_series(
        cls,
        queue_length_dict: Dict[int, pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """
        Returns an (optional) dataframe with queue length statistics per CUDA stream and rank.
        This function takes the output from get_queue_length_time_series() directly.

        Args:
            Dict[int, pd.DataFrame]: A dictionary of rank -> time series with the queue length of each CUDA stream.
                This is the output of get_queue_length_time_series().

        Returns:
            Optional[pd.DataFrame]
                An (optional) dataframe containing the summary statistics of queue length per
                stream and rank.
        """
        results_list: List[pd.DataFrame] = []

        for rank, rank_df in queue_length_dict.items():
            rank_df["rank"] = rank
            result = (
                rank_df[["rank", "stream", "queue_length"]]
                .groupby(["rank", "stream"])
                .describe()
            )
            results_list.append(result)
        return pd.concat(results_list) if len(results_list) > 0 else None

    @classmethod
    def get_queue_length_summary(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Returns an (optional) dataframewith queue length statistics per CUDA stream and rank.

        Args:
            t (Trace): Input trace data structure.
            ranks (list of int): ranks to perform this analysis.

        Returns:
            Optional[pd.DataFrame]
                An (optional) dataframe containing the summary statistics of queue length per
                stream and rank.
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        return TraceCounters.get_queue_length_summary_from_time_series(
            TraceCounters.get_queue_length_time_series(t, ranks)
        )

    @classmethod
    def get_time_spent_blocked_on_full_queue(
        cls,
        t: "Trace",
        queue_length_dict: Dict[int, pd.DataFrame],
        max_queue_length: int,
    ) -> Optional[pd.DataFrame]:
        """

        Returns an (optional) dataframe with the time spent on the kernel launch queue full.
        This function takes the output from get_queue_length_time_series() and sums
        up the time spent on all streams where the queue is full (see max_queue_length)

        Args:
            t (Trace): Input trace data structure.
            queue_length_dict (Dict[int, pd.DataFrame]): A dictionary of rank -> time series with the queue length of each CUDA stream.
                This is the output of get_queue_length_time_series().
            max_queue_length (int): Max kernel launch queue length.

        Returns:
            Optional[pd.DataFrame]
                An (optional) dataframe containing the summary statistics blocked time per
                stream and rank
                The dataframe contains the columns- rank, stream, duration_at_max_queue_length,
                and relative_duration_at_max_queue_length.

                Relative duration at max queue length considers the total duration of a trace
                and normalizes the duration_at_max_queue_length.
        """

        result = {}

        for rank, ql in queue_length_dict.items():
            for stream in ql.stream.unique():
                df = ql.loc[ql.stream == stream].copy()
                df["dur"] = df.ts.shift(-1) - df.ts

                df = df.loc[ql.queue_length >= max_queue_length]
                if df.empty:
                    continue

                dur_at_max_queue_len = df.dur.sum()

                rel_dur_at_max_queue_len = (
                    dur_at_max_queue_len * 1.0 / t.get_trace_duration(rank)
                )
                logger.info(
                    f"Rank={rank}, stream={stream}, total dur at max_queue= {dur_at_max_queue_len} (ns)"
                    f" trace_duration = {t.get_trace_duration(rank)} (ns)"
                    f" rel_dur_at_max_queue_len = {rel_dur_at_max_queue_len} (ns)"
                )
                result[rank] = [stream, dur_at_max_queue_len, rel_dur_at_max_queue_len]

        result_df = pd.DataFrame.from_dict(
            result,
            orient="index",
            columns=[
                "stream",
                "duration_at_max_queue_length",
                "relative_duration_at_max_queue_length",
            ],
        )
        return result_df.reset_index(names="rank")

    @classmethod
    def _get_memory_bw_time_series_for_rank(
        cls, t: "Trace", rank: int
    ) -> Optional[pd.DataFrame]:
        """
        Returns time series for the memory bandwidth of memory copy and memory set operations
        for specified rank.

        Args:
            t (Trace): Input trace data structure.
            rank (int): rank to generate the time series for.

        Returns:
            Optional[pd.DataFrame]
                Returns an (optional) dataframe with time series for the memory bandwidth.
                The dataframe returned contains time series with columns:
                ts (timestamp), pid (of corresponding GPU), name of memory copy type
                and memory_bw_gbps (memory bandwidth in GB/sec).
        """
        # get trace for a rank
        trace_df: pd.DataFrame = t.get_trace(rank)
        sym_table = t.symbol_table.get_sym_table()

        gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()
        gpu_kernels["kernel_type"] = gpu_kernels[["name"]].apply(
            lambda x: get_kernel_type(sym_table[x["name"]]), axis=1
        )

        memcpy_kernels = gpu_kernels[
            gpu_kernels.kernel_type == KernelType.MEMORY.name
        ].copy()
        memcpy_kernels["name"] = memcpy_kernels[["name"]].apply(
            lambda x: get_memory_kernel_type(sym_table[x["name"]]), axis=1
        )

        copies = memcpy_kernels[
            ["ts", "dur", "name", "pid", "memory_bw_gbps"]
        ].reset_index(drop=True)
        is_point = copies["dur"].le(0)
        bandwidth = copies["memory_bw_gbps"]

        # Each copy adds its bandwidth at its start and removes it at its end
        # (start + duration).
        starts = copies[["ts", "pid", "name"]].assign(
            memory_bw_gbps=bandwidth,
            _order=_ORDER_START,
            _tie=bandwidth,
            _phase=0,
        )
        ends = copies[["ts", "pid", "name"]].assign(
            ts=copies["ts"] + copies["dur"],
            memory_bw_gbps=-bandwidth,
            _order=_ORDER_END,
            _tie=(-bandwidth).where(~is_point, bandwidth),
            _phase=1,
        )
        starts.loc[is_point, "_order"] = _ORDER_POINT
        ends.loc[is_point, "_order"] = _ORDER_POINT

        # At one timestamp, points count in the _ORDER_* order, then by step
        # (a zero-length copy's by its bandwidth, so its start and its end
        # stay a pair), then by pid and copy. Points equal in all of these
        # except the copy give equal rows, so the result does not depend on
        # the row order of the trace.
        membw_time_series = cast(
            pd.DataFrame,
            pd.concat(
                [
                    starts.assign(_copy=copies.index),
                    ends.assign(_copy=copies.index),
                ],
                ignore_index=True,
            ),
        )
        membw_time_series = membw_time_series.sort_values(
            by=["ts", "_order", "_tie", "pid", "_copy", "_phase"],
            kind="stable",
            na_position="first",
        )

        result_df_list = []
        for _, membw_df in membw_time_series.groupby("name"):
            membw_df.memory_bw_gbps = membw_df.memory_bw_gbps.cumsum()
            result_df_list.append(membw_df)

        if len(result_df_list) == 0:
            return None

        result_df = pd.concat(result_df_list)[["ts", "pid", "name", "memory_bw_gbps"]]
        return result_df

    @classmethod
    def get_memory_bw_time_series(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Returns a dictionary of rank -> time series for the memory bandwidth.

        Args:
            t (Trace): Input trace data structure.
            ranks (list of int): ranks to perform this analysis for.

        Returns:
            Dict[int, pd.DataFrame]
                Returns a dictionary of rank -> time series for the memory bandwidth.
                The dataframe returned contains time series along with the following columns:
                ts (timestamp), pid (of corresponding GPU), name of memory copy type
                and memory_bw_gbps (memory bandwidth in GB/sec).
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        logger.info(
            "Please note that the time series only contains points "
            "when the value changes. Once a values is observed the time series "
            "stays constant until the next update."
        )
        result = {
            rank: TraceCounters._get_memory_bw_time_series_for_rank(t, rank)
            for rank in ranks
        }
        # pyrefly: ignore [bad-return]
        return dict(filter(lambda x: x[1] is not None, result.items()))

    @classmethod
    def get_memory_bw_summary(
        cls,
        t: "Trace",
        ranks: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Returns an (optional) dataframe containing the summary statistics of memory ops. The
        tracked memory ops are MemcpyDtoH, MemcpyHtoD, MemcpyDtoD and MemSet.

        Args:
            t (Trace): Input trace data structure.
            ranks (list of int): ranks to perform this analysis for.

        Returns:
            Optional[pd.DataFrame]
                An (optional) dataframe containing the summary statistics of the following memory ops:
                MemcpyDtoH, MemcpyHtoD, MemcpyDtoD, MemSet.
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        results_list: List[pd.DataFrame] = []

        for rank, rank_df in TraceCounters.get_memory_bw_time_series(t, ranks).items():
            rank_df["rank"] = rank
            # Exclude the 0 points in time series
            rank_df = rank_df[rank_df.memory_bw_gbps > 0]

            result = (
                rank_df[["rank", "name", "memory_bw_gbps"]]
                .groupby(["rank", "name"])
                .describe()
            )
            results_list.append(result)
        return pd.concat(results_list) if len(results_list) > 0 else None
