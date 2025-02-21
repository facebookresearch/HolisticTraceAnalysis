# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from enum import auto, Flag
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from hta.analyzers.breakdown_analysis import BreakdownAnalysis
from hta.analyzers.communication_analysis import CommunicationAnalysis
from hta.analyzers.critical_path_analysis import CPGraph, CriticalPathAnalysis
from hta.analyzers.cuda_kernel_analysis import CudaKernelAnalysis
from hta.analyzers.cupti_counter_analysis import CuptiCounterAnalysis
from hta.analyzers.straggler import find_stragglers_with_late_start_comm_kernels
from hta.analyzers.straggler_analysis import StragglerAnalysis
from hta.analyzers.trace_counters import TraceCounters
from hta.analyzers.trace_memory_analysis import (
    classify_torchtitan_calls,
    MemoryAnalysis,
)
from hta.common.constants import CUDA_MAX_LAUNCH_QUEUE_PER_STREAM
from hta.common.trace import Trace
from hta.configs.config import logger
from hta.configs.default_values import DEFAULT_TRACE_DIR


class TimeSeriesTypes(Flag):
    QUEUE_LENGTH = auto()
    MEMCPY_BANDWIDTH = auto()


class TraceAnalysis:
    def __init__(
        self,
        trace_files: Optional[Dict[int, str]] = None,
        trace_dir: str = DEFAULT_TRACE_DIR,
        include_last_profiler_step: Optional[bool] = False,
        use_multiprocessing=True,
    ):
        self.t = Trace(trace_files, trace_dir)
        self.t.load_traces(
            include_last_profiler_step, use_multiprocessing=use_multiprocessing
        )
        assert self.t.is_parsed is True

    def get_comm_comp_overlap(self, visualize: bool = True) -> pd.DataFrame:
        r"""
        Compute the communication-computation overlap percentage for each rank.

        Args:
            visualize (bool): Set to True to display the graph. Default = True.

        Returns:
            pd.DataFrame
                A dataframe containing the communication computation overlap percentage for each rank.
        """
        return CommunicationAnalysis.get_comm_comp_overlap(self.t, visualize)

    def get_profiler_steps(self) -> List[int]:
        r"""
        Get the list of profiler steps.

        Args:
            None

        Returns:
            List[int]
                A list of profiler steps.
        """
        return StragglerAnalysis.get_profiler_steps(self.t)

    def get_potential_stragglers(
        self,
        profiler_steps: Optional[List[int]] = None,
        num_candidates: int = 2,
        visualize: bool = False,
        straggler_identification_impl: Callable[
            ..., pd.Series
        ] = find_stragglers_with_late_start_comm_kernels,
    ) -> List[int]:
        r"""
        Identify potential stragglers based on a pre-defined metric computed from the trace.

        Args:
            profiler_steps (List[int]) : A list of profiler steps used in the straggler analysis. If None, then use all
                                         iterations.
            num_candidates (int): Number of straggler candidates straggler analysis should look for
            visualize (bool): Set to True to visualize intermediate results
            straggler_identification_impl: A function or method which implements a straggler identification algorithm.
                                           The function takes a DataFrame, a TraceSymbolTable, num_candidates, and
                                           visualize as input and returns a Series with rank as the index and the
                                           value as an indicator whether a rank is a potential straggler.

        Returns:
            List[int]
                The list of ranks which can be potential stragglers.

        Notes:
            1. Find all communication kernels which pass the duration test, i.e., kernel duration > mean_iter_time * min_normalized_duration (the default value is 1%).
            2. Pick the last kernel for each combination of (iteration, rank, stream, kernel).
            3. For each combination of (iteration, stream, kernel), compute the standard deviation of the last kernel duration w.r.t. ranks.
            4. Compute the mean value of the last kernel duration standard deviation w.r.t. ranks across all available iterations. If there is only one iteration, the mean value would just be the std computed at 3.
            5. Select the (stream, kernel) combination which has the largest mean value computed at 4.
            6. Select the normalized start time for the (stream, kernel) combination as the candidate metric for detecting whether a rank will be a straggler.

            + This method forwards the analysis to the function hta.straggler.default_straggler_identification.
            + Because most distributed training jobs have one or multiple blocking synchronizations within an iteration, we define a straggler as the trainer which starts later than others and thus causes the largest slowdown of the overall training process.
            + Because empirical knowledge shows that all-to-all collective communication operators are most likely to have the largest impact on the training performance, this straggler analysis computes a default straggler metric as follows:
            + This metric is defined for each rank and for each iteration (excluding the last iteration if it should be ignored).
            + The metrics is passed into a straggler detection algorithm which picks the top k ranks ordered by the metric computed above for each iteration, where k is num_candidates.
            + For multiple iterations, it is possible that different iteration may return a different set of potential stragglers. The identify_potential_stragglers() will return the counts of how many times each rank is found to be a potential straggler.
        """
        return StragglerAnalysis.get_potential_stragglers(
            self.t,
            profiler_steps,
            num_candidates,
            visualize,
            straggler_identification_impl,
        )

    def get_gpu_kernel_breakdown(
        self,
        visualize: bool = True,
        duration_ratio: float = 0.8,
        num_kernels: int = 10,
        include_memory_kernels: bool = True,
        image_renderer: str = "",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        r"""
        Summarizes the time spent by each kernel and by kernel type. Outputs the following graphs:

        1. Pie chart indicating the percentage of time taken by each kernel type.
        2. Pie charts showing the most time consuming kernels for each rank for each kernel type.
        3. Bar graphs showing the average duration for the most time consuming kernels for each rank and each kernel type.

        Args:
            visualize (boolean): Set to True to display the graphs. Default = True.
            duration_ratio (float): Floating point value between 0 and 1 specifying the ratio of time taken
                                    by top COMM/COMP/MEMORY kernels. Default = 0.8.
            num_kernels (int): Maximum number of COMM/COMP/MEMORY kernels to show. Default = 10.
            include_memory_kernels (bool): Whether to include MEMORY kernels in the analysis. Default = True.
            image_renderer (str): Set to ``notebook`` when using jupyter and ``jupyterlab`` when using jupyter-lab.
                To see all available options execute: ``import plotly; plotly.io.renderers`` in a python shell.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]
                Returns two dataframes. The first dataframe shows the percentage of time spent by kernel type.
                The second dataframe shows the min, max, mean, standard deviation, total time taken by each
                kernel on each rank. This dataframe will be summarized based on values of ``duration_ratio``
                and ``num_kernels``. If both ``duration_ratio`` and ``num_kernels`` are specified,
                ``num_kernels`` takes precedence.
        """

        return BreakdownAnalysis.get_gpu_kernel_breakdown(
            self.t,
            visualize,
            duration_ratio,
            num_kernels,
            include_memory_kernels,
            image_renderer,
        )

    def get_gpu_kernels_with_user_annotations(
        self,
        rank: int,
        expand_names: bool = True,
        shortern_names: bool = True,
    ) -> Optional[pd.DataFrame]:
        r"""
        Provides a complete dataframe of GPU kernels and matches them to the corresponding user annotation i.e. user provided training phase. The output is a dataframe with all GPU kernel data alongside a "user_annotation" column.

        Args:
            rank (int): Specify rank to return GPU kernels for.
            expand_names (bool): Expand integer name value to full names. This will add
                the columns "s_name" and "s_user_annotation" to the dataframe.
            shortern_names (bool): When expand_names is True, this flag enables shortening
                large CUDA kernel names. This works by removing the '<' template parameters etc.

        Returns:
            pd.Dataframe:
                The returned dataframe has all trace columns along with "user_annotation",
                and optionally "s_user_annotation" column if expand_names=True.

        Note: This API is per rank, and does not have any visualization aspect.
        """
        return BreakdownAnalysis.get_gpu_kernels_with_user_annotations(
            self.t,
            rank,
            expand_names,
            shortern_names,
        )

    def get_gpu_user_annotation_breakdown(
        self,
        use_gpu_annotation: bool = True,
        visualize: bool = True,
        duration_ratio: float = 0.8,
        num_kernels: int = 1000,
        image_renderer: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        r"""
        Summarizes the time spent by each GPU user annotation. Outputs the following graphs:

        1. Pie charts showing the most time consuming user annotations for each rank.
        2. Bar graphs showing the average duration for the most time user annotations for each rank.

        Args:
            use_gpu_annotation (boolean): Use time on GPU for each user annotation, if false use the time on CPU instead. Default = True,
            visualize (boolean): Set to True to display the graphs. Default = True.
            duration_ratio (float): Floating point value between 0 and 1 specifying the ratio of time taken
                                    by top user annotations. Default = 0.8.
            num_kernels (int): Maximum number of user annotations to show. Default = 1000. Rest get grouped into "other".
            image_renderer (str): Set to ``notebook`` when using jupyter and ``jupyterlab`` when using jupyter-lab.
                To see all available options execute: ``import plotly; plotly.io.renderers`` in a python shell.

        Returns:
            Optional[pd.DataFrame]
                Returns a dataframe that shows the min, max, mean, standard deviation, total time taken by each
                user annotation on each rank. This dataframe will be summarized based on values of ``duration_ratio``
                and ``num_kernels``. If both ``duration_ratio`` and ``num_kernels`` are specified,
                ``num_kernels`` takes precedence.
                If user_annotations are not present on CPU or GPU (according to use_gpu_annotation flag), return None.
        """

        return BreakdownAnalysis.get_gpu_user_annotation_breakdown(
            self.t,
            use_gpu_annotation,
            visualize,
            duration_ratio,
            num_kernels,
            image_renderer,
        )

    def get_temporal_breakdown(self, visualize: bool = True) -> pd.DataFrame:
        r"""
        Compute the idle time, compute time and non-compute time for each rank. Time is measured in
        nanoseconds (ns). non-compute time is defined as the total time the GPU is not executing a
        compute operation such as data transfers, copying to/from memory and communication collectives.
        (In the strictest sense communication collectives do some compute but we classify it as a
        communication operation).

        Args:
            visualize (bool): Set to True to display the graphs. Default = True.

        Returns:
            pd.DataFrame
                A dataframe containing the raw value and percentage of idle time, compute time and non-compute
                time for each rank.
        """
        return BreakdownAnalysis.get_temporal_breakdown(self.t, visualize)

    def get_frequent_cuda_kernel_sequences(
        self,
        operator_name: str,
        output_dir: str,
        min_pattern_len: int = 3,
        rank: int = 0,
        top_k: int = 5,
        visualize: bool = False,
        compress_other_kernels: bool = True,
    ) -> pd.DataFrame:
        r"""
        Computes the most frequent CUDA kernel sequences originating from the CPU op with name
        ``operator_name``. Generates a dataframe summarizing the sequence of kernels, their frequency
        and total time taken. Additionally, writes a new trace file to ``output_dir`` with the
        top_k frequent patterns overlaid on top of the original trace file.

        Args:
            operator_name (str): Name of the operator from which the CUDA kernels are launched.
            output_dir (str): Output folder path containing the new trace file with overlaid top k
                              frequent patterns.
            min_pattern_len (int): Minimum length of the CUDA kernel sequences that should be identified. Default = 3.
            rank (int): Rank on which the analysis is performed. Default = 0.
            top_k (int): top_k patterns in terms of frequency to be visualized and overlaid. Default = 5.
            visualize (bool): Whether to show the histogram of top_k frequent patterns inline. Default = True.
            compress_other_kernels (bool): Should the names and args for other kernels not belonging to any
                                           frequent patterns be compressed to save memory in the overlaid
                                           trace file. Default = True.

        Returns:
            pd.DataFrame
                A dataframe with frequent cuda kernel sequences and their frequencies.
        """
        return CudaKernelAnalysis.get_frequent_cuda_kernel_sequences(
            self.t,
            operator_name,
            output_dir,
            min_pattern_len,
            rank,
            top_k,
            visualize,
            compress_other_kernels,
        )

    def get_cuda_kernel_launch_stats(
        self,
        ranks: Optional[List[int]] = None,
        runtime_cutoff: int = 50,
        launch_delay_cutoff: int = 100,
        include_memory_events: bool = True,
        visualize: bool = True,
    ) -> Dict[int, pd.DataFrame]:
        r"""
        For each event launched on the GPU there is a corresponding scheduling event on the CPU.
        These events are linked by a common correlation id. This feature calculates the duration
        of the CPU op, GPU kernel and the launch delay (difference between gpu kernel starting
        and cpu op ending for each correlation id on the specified rank(s). This function finds:

        1. GPU events with a shorter duration than the corresponding CPU events.
        2. CPU runtime events with a large duration i.e. outliers. The outliers are defined using
           the ``runtime_cutoff`` value.
        3. CPU events which have a large launch delay i.e. launch delay outliers. The launch delay
           outliers are defined using the ``launch_delay_cutoff`` value.

        Args:
            ranks (List[int]): List of ranks on which to run the analysis. Default = [0].
            runtime_cutoff (int): Duration in microseconds to determine outliers for cuda runtime events.
                Default = 50 microseconds.
            launch_delay_cutoff (int): Duration in microseconds to determine outliers for launch delay.
                Default value is 100 microseconds.
            include_memory_events (bool): Toggle to include cudaMemcpyAsync and cudaMemsetAsync events.
                Default = True.
            visualize (bool): Toggle to display the generated graphs. Default = True.

        Returns:
            Dict[int, pd.DataFrame]
                The function returns a dictionary of dataframes. The key corresponds to the rank and the value is
                a dataframe containing the cpu_duration, gpu_duration and launch_delay for each correlation id.
        """
        if ranks is None:
            ranks = [0]

        return CudaKernelAnalysis.cuda_kernel_launch_stats(
            self.t,
            ranks,
            runtime_cutoff,
            launch_delay_cutoff,
            include_memory_events,
            visualize,
        )

    def generate_trace_with_counters(
        self,
        time_series: Optional[TimeSeriesTypes] = None,
        ranks: Optional[List[int]] = None,
        output_suffix: str = "_with_counters",
        custom_time_series=None,
    ) -> None:
        r"""
        Adds a set of time series to the trace in order to aid debugging traces. Creates a new trace file
        for each requested rank with the a suffix '_with_counters.json'. The following time series are
        available in TimeSeriesTypes flag type.

        1. Queue length - adds a time series to the trace indicating the size of the queue at any given time on each CUDA stream.
        2. Memory copy bandwidth - adds a time series to the trace indicating the memory bandwidth used for device to host, host to device and device to device operations.

        Either or both of the above can be enabled.

        Args:
            time_series (Flag): Used to set the requested time series. Available values are
                TimeSeriesTypes.QUEUE_LENGTH and TimeSeriesTypes.MEMCPY_BANDWIDTH. By default
                both time series are added to the trace.
            ranks (List[int]): List of ranks to generate the counters for. Default = [0].
            output_suffix (str): Suffix to add to the trace file. Default = '_with_counters.json.gz'

        Returns:
            None
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        if time_series is None:
            time_series = (
                TimeSeriesTypes.QUEUE_LENGTH | TimeSeriesTypes.MEMCPY_BANDWIDTH
            )

        counter_events: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        if output_suffix == "":
            output_suffix = "_with_counters"

        def add_time_series(
            series_dict: Dict[int, pd.DataFrame], counter_name: str, counter_col: str
        ):
            nonlocal counter_events
            """accept a rank -> time series dict and append it"""
            for rank, series in series_dict.items():
                if "stream" in series.columns:
                    series.rename(columns={"stream": "id"}, inplace=True)
                ce = self.t.convert_time_series_to_events(
                    series, counter_name, counter_col
                )
                counter_events[rank].extend(ce)

        if TimeSeriesTypes.QUEUE_LENGTH in time_series:
            add_time_series(
                series_dict=TraceCounters.get_queue_length_time_series(self.t, ranks),
                counter_name="Queue Length",
                counter_col="queue_length",
            )
        if TimeSeriesTypes.MEMCPY_BANDWIDTH in time_series:
            add_time_series(
                series_dict=TraceCounters.get_memory_bw_time_series(self.t, ranks),
                counter_name="Memcpy BW",
                counter_col="memory_bw_gbps",
            )
        if custom_time_series:
            add_time_series(**custom_time_series)

        for rank, ev_list in counter_events.items():
            raw_trace_content = self.t.get_raw_trace_for_one_rank(rank=rank)
            raw_trace_content["traceEvents"].extend(ev_list)
            output_file = self.t.trace_files[rank].replace(
                ".json", f"{output_suffix}.json"
            )
            logger.info(f"Writing trace with counters for rank {rank} to {output_file}")
            self.t.write_raw_trace(output_file, raw_trace_content)

    def get_queue_length_summary(
        self,
        ranks: Optional[List[int]] = None,
    ) -> Optional[pd.DataFrame]:
        r"""
        Queue length is defined as the number of outstanding CUDA operations on a stream. This
        functions calculates the summary statistics for the queue length on each CUDA stream for
        the specified ranks.

        Args:
            ranks (List[int]): List of ranks for which to queue length summary is calculated. Default = [0].

        Returns:
            pd.DataFrame or None
                A dataframe summarizing the queue length statistics. The dataframe contains count,
                min, max, standard deviation, 25th, 50th and 75th percentiles.
                The function returns None when the dataframe is empty.
        """
        return TraceCounters.get_queue_length_summary(self.t, ranks)

    def get_queue_length_summary_from_time_series(
        self,
        queue_length_dict: Dict[int, pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        r"""
        Queue length is defined as the number of outstanding CUDA operations on a stream. This
        functions calculates the summary statistics for the queue length on each CUDA stream for
        the specified ranks. This function takes the output from get_queue_length_time_series() directly.

        Args:
            queue_length_dict (Dict[int, pd.DataFrame]): A dictionary of rank -> time series with the queue length of each CUDA stream.
                This is the output of get_queue_length_time_series().

        Returns:
            pd.DataFrame or None
                A dataframe summarizing the queue length statistics. The dataframe contains count,
                min, max, standard deviation, 25th, 50th and 75th percentiles.
                The function returns None when the dataframe is empty.
        """
        return TraceCounters.get_queue_length_summary_from_time_series(
            queue_length_dict
        )

    def get_queue_length_time_series(
        self,
        ranks: Optional[List[int]] = None,
    ) -> Dict[int, pd.DataFrame]:
        r"""
        Queue length is defined as the number of outstanding CUDA operations on a stream. This
        function calculates the time series for the queue length on each CUDA stream for the
        specified ranks.

        Args:
            ranks (List[int]): List of ranks for which the queue length time series is generated. Default = [0].

        Returns:
            Dict[int, pd.DataFrame]
                Returns a dictionary whose key is the rank and value is a dataframe of queue length
                counter events. The following fields are in each row of the dataframe: ts (timestamp), pid (process id),
                tid (thread id), stream, and queue length.
        """
        return TraceCounters.get_queue_length_time_series(self.t, ranks)

    def get_time_spent_blocked_on_full_queue(
        self,
        queue_length_dict: Dict[int, pd.DataFrame],
        max_queue_length: int = CUDA_MAX_LAUNCH_QUEUE_PER_STREAM,
    ) -> Optional[pd.DataFrame]:
        r"""
        The GPU kernels launch queue is finite. If the CPU fills up this queue the CPU
        will block till the GPU device launches kernels. We compute the time spent blocked
        on a full launch queue in this function.

        Returns an (optional) dataframe with the time spent on the kernel launch queue full.
        This function takes the output from get_queue_length_time_series() and sums
        up the time spent on all streams where the queue is full (see max_queue_length)

        Args:
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
        return TraceCounters.get_time_spent_blocked_on_full_queue(
            self.t, queue_length_dict, max_queue_length
        )

    def get_memory_bw_summary(
        self,
        ranks: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        r"""
        Summarizes the memory bandwidth statistics for memory copy and memset operations. This includes memory
        bandwidth for copies from Device to Host, Host to Device and Device to Device transfers. Note, this does
        not include memory bandwidth used by compute/communication kernels.

        Args:
            ranks (List[int]): List of ranks for which memory bandwidth is calculated. Default = [0].

        Returns:
            pd.DataFrame or None
                A dataframe containing the summary statistics. The dataframe includes count, min, max, standard deviation,
                25th, 50th and 75th percentiles of memory copy/memset operations.
                The function returns None when the dataframe is empty.
        """
        return TraceCounters.get_memory_bw_summary(self.t, ranks)

    def get_memory_bw_time_series(
        self,
        ranks: Optional[List[int]] = None,
    ) -> Dict[int, pd.DataFrame]:
        r"""
        Calculates the time series for memory copy bandwidth used by memcpy and memset operations in GB/s. The
        memory bandwidth is calculated for host to device, device to host and device to device copies. Note, this
        does not include memory bandwidth used by computation or communication kernels.

        Args:
            ranks (List[int]): List of ranks for which the memory bandwidth time series is generated. Default = [0].

        Returns:
            Dict[int, pd.DataFrame]
                Returns a dictionary whose key is the rank and value is a dataframe of memory bandwidth
                counter events. The following fields are in each row of the dataframe: ts (timestamp), pid (process id),
                tid (thread id), name (memcpy/memset), and memory bandwidth in GB/s.
        """
        return TraceCounters.get_memory_bw_time_series(self.t, ranks)

    def get_idle_time_breakdown(
        self,
        ranks: Optional[List[int]] = None,
        streams: Optional[List[int]] = None,
        visualize: bool = True,
        visualize_pctg: bool = True,
        show_idle_interval_stats=False,
        consecutive_kernel_delay: int = 30,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        r"""
        GPU is considered idle when no kernel is running on it. Idle time is broken down into 3 categories.

        1. Host wait time: a GPU or stream is idle because the CPU thread has not enqueued enough kernels to
           keep it occupied.
        2. Kernel wait time: This is the duration between kernels and is considered an overhead of
           launching multiple small kernels. We use the following heuristic to classify the duration as kernel wait:
           duration between consecutive kernels < ``consecutive_kernel_delay``.
        3. Other wait time: In this case the idle time is attributed to an unknown cause. For example, a compute
           kernel could be waiting for a CUDA event from a communication kernel to complete.

        Args:
            ranks (List[int]): List of ranks for which idle time breakdown is computed. Default = [0].
            streams (List[int]): List of streams to provide analysis for. Defaults to all streams.
            visualize (bool): Set to True to show the graph. Default = True.
            visualize_pctg (bool): Show relative percentage across streams. Default = True.
            show_idle_interval_stats (bool): Returns statistics of the idle intervals like the min, max
               and median of idle intervals between kernels on a CUDA stream, also broken down by
               the idleness category. Default = False.
            consecutive_kernel_delay (int): Configures the threshold under which we consider gaps between
                kernels to be due to realistic delays in launching back to back kernels on the GPU.
                Default = 30 nanoseconds.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]
                A tuple of dataframes. The first dataframe contains the idle time category and duration for each stream
                on each rank. The second dataframe contains the summary statistics (count, min, max, mean, standard deviation,
                25th, 50th, 75th percentile) for each idle category for each stream on each rank.
        """
        if ranks is None or len(ranks) == 0:
            ranks = [0]

        idle_time_df_list: List[pd.DataFrame] = []
        interval_df_list: List[pd.DataFrame] = []
        for rank in ranks:
            idle_time_r_df, interval_r_df = BreakdownAnalysis.get_idle_time_breakdown(
                self.t,
                consecutive_kernel_delay,
                rank,
                streams,
                visualize,
                visualize_pctg,
                show_idle_interval_stats,
            )
            idle_time_df_list.append(idle_time_r_df)
            if interval_r_df is not None:
                interval_df_list.append(interval_r_df)

        return (
            pd.concat(idle_time_df_list),
            pd.concat(interval_df_list) if show_idle_interval_stats else None,
        )

    def get_cupti_counter_data_with_operators(
        self,
        ranks: Optional[List[int]] = None,
    ) -> List[pd.DataFrame]:
        r"""Performance counters provide insights on how to speed up GPU
        kernels. The PyTorch Profiler has a lightweight API [CUPTI Range
        Profiler API](https://docs.nvidia.com/cupti/r_main.html#r_profiler)
        that enables users to monitor performance counters from the device.

        When the CUPTI Profiler mode is enabled then PyTorch will emit the
        performance counters and annotates them in the trace.
        * The events are logged under the `cuda_profiler_range` category.
        * Counter values are logged in the `args` section of the trace.

        This API can investigate performance measurements per kernel and
        associate them to operators that the kernel belongs to. A single kernel
        can map to multiple levels of operators (as operators can be nested).
        To represent this we basically provide a list column called `op_stack`.
        For further convenience we add the top and bottom level operator column
        as well.

        Args:
            ranks (List[int]): List of ranks on which to run the analysis. Default = [0].
        Returns:
            List[pd.DataFrame]
                A list of dataframes, one per rank, containing kernel name,
                op_stack (operator stack), top and bottom level op, and columns
                for individual performance counters.
        """
        return CuptiCounterAnalysis.get_counter_data_with_operators(self.t, ranks)

    def critical_path_analysis(
        self,
        rank: int,
        annotation: str,
        instance_id: Union[Optional[int], Tuple[int, int]],
    ) -> Tuple[CPGraph, bool]:
        r"""
        Perform critical path analysis for trace events within a rank.
        We further reduce the region of interest by selecting
        a trace annotation and instance id. This will
        limit the analysis to events within the time range of that annoation.
        This will include GPU kernels launched by the cpu operators in that
        time duration.
        For example, you can use this to limit the analysis to one iteration
        by passing annotation='ProfilerStep'. See notes for how to pick the iteration.

        Args:
            t (Trace): Input trace data structure.
            rank (int): rank to analyze for the critical path.
            annotation (str): a trace annotation to limit the analysis to,
                for example "ProfilerStep" would match all annotations that
                match this string (ProfilerStep#100, ProfilerStep#101 etc)
            instance_id: can be either of the following
                (int) - specify which instance of the annotation to consider.
                        Defaults to the first instance.
                (Tuple(int, int)) - considers a range of annotation instances start to end,
                        inclusive of both start and end instance.
        Returns:
            Tuple[CPGraph, bool]
                A tuple of CPGraph object and a success or fail boolean value.
                True indicates that the critical path analysis algorithm succeeded.

        CPGraph object that can be used to obtain statistics and further
        visualize the critical path. CPGraph is also a subinstance of a networkx.DiGraph.
        Run 'CPGraph?' for more info and APIs.

        Notes:

        1. Avoid using the first step / iteration in a trace as it usually
           has some missing events.
        2. The analysis requires CUDA synchronization events in the GPU trace,
           that were added in https://github.com/pytorch/pytorch/pull/105187
           Please see the documentation of this PR on how to enable CUDA sync events in the trace.
        """
        return CriticalPathAnalysis.critical_path_analysis(
            self.t, rank, annotation, instance_id
        )

    def overlay_critical_path_analysis(
        self,
        rank: int,
        critical_path_graph: CPGraph,
        output_dir: str,
        only_show_critical_events: bool = True,
        show_all_edges: bool = False,
    ) -> str:
        r"""
        Overlay the identified critical path on top of the trace file
        for visualization.

        Args:
            rank (int): rank to generate the time series for.
            critical_path_graph: Critical Path Graph object generated previously
            output_dir (str): Output directory to store overlaid trace.
            only_show_critical_events (bool): When set the output trace will only
                have operators and GPU kernels on the critical path. It will
                still retain the user annotations.
            show_all_edges (bool): When set this will add edge events for
                all types of edges in the critical path graph. This is useful
                for debugging the algorithm.

        Returns: the overlaid trace file path. The generated trace file will
        have a prefix of "overlaid_critical_path\_" in its name compared
        to the original trace file.

        Note: In case of kernel launches that are not on the critical path the graph
        still has a 0 weight edge between CUDA runtime and kernel. These 0 weight
        edges are not shown in the overlaid trace by default. Set the environment
        variable CRITICAL_PATH_SHOW_ZERO_WEIGHT_LAUNCH_EDGE=1 to enable adding this
        to the overlaid trace. Add this to your notebook
        `os.environ["CRITICAL_PATH_SHOW_ZERO_WEIGHT_LAUNCH_EDGE"] = 1`
        """
        return CriticalPathAnalysis.overlay_critical_path_analysis(
            self.t,
            rank,
            critical_path_graph,
            output_dir,
            only_show_critical_events,
            show_all_edges,
        )

    def get_memory_timeline(
        self, rank: Optional[int] = None, visualize: bool = True
    ) -> pd.DataFrame:
        """Get memory usage timeline

        This function analyzes memory allocation events in the trace to produce a
        timeline of memory usage including both allocated and reserved memory.

        Args:
            rank (Optional[int]): Analyze specific rank. If None, use first available rank.
            visualize (bool): Whether to display the memory timeline plot. Default=True.

        Returns:
            pd.DataFrame: DataFrame containing memory events with columns:
                - ts: timestamp
                - device_id: Device ID
                - device_type: Device type (1=CPU, 2=CUDA)
                - bytes_delta: Change in bytes
                - total_allocated: Total allocated memory
                - total_reserved: Total reserved memory
                - addr: Memory address
        """

        analyzer = MemoryAnalysis(self.t)
        return analyzer.get_memory_timeline(rank=rank, visualize=visualize)

    def get_memory_timeline_per_category(
        self,
        rank: Optional[int] = None,
        visualize: bool = True,
        classification_func=classify_torchtitan_calls,
    ):
        """Get memory usage timeline from user supplied categories.

        This function analyzes memory allocation events in the trace to produce a
        timeline of allocated memory for different user-customisable categories.

        Note: this function takes

        Args:
            rank (Optional[int]): Analyze specific rank. If None, use first available rank.
            visualize (bool): Whether to display the memory timeline plot. Default=True.
            classification_func (Callable[[MemoryEvent], str])

        Returns:
            pd.DataFrame: DataFrame containing memory events with columns:
                - ts: timestamp in nano-second
                - category: the category assigned to the event by the classification_func
                - stack_name: a string containing the name of all the parent events of the
                  allocation.
                - <category columns>: a column for each category with the current total allocated
                  for that category, or Nan if the timestep does not correspond to this data point.
        """

        analyzer = MemoryAnalysis(self.t)
        return analyzer.get_classified_memory_timelines(
            rank=rank, visualize=visualize, classification_func=classification_func
        )
