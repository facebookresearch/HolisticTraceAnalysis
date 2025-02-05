# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import time
from typing import Callable, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np

import pandas as pd
import plotly
import plotly.graph_objects as go

from hta.common.trace import Trace, TraceSymbolTable
from hta.configs.config import logger
from hta.utils.utils import shorten_name

from numba import float64, int64, njit, types
from numba.typed import List as TypedList

colorscheme = plotly.colors.qualitative.Pastel

# Define the tuple type for stack elements globally.
# Each element is a tuple: (index: int64, name_code: int64, ts: float64, end: float64)
STACK_TUPLE_TYPE = types.Tuple((int64, int64, float64, float64))


# --- JIT-compiled function with explicit type specifications ---
@njit
def _collect_stack_ids_for_process_group(indices, names, ts, dur, event_needs_stack):
    """
    Calculates one group of events (for a given (pid, tid)).

    Parameters:
      indices: np.array of event indices (int64).
      names: np.array of integer name codes (int64).
      ts: np.array of timestamps (float64).
      dur: np.array of durations (float64).
      event_needs_stack: np.array indicating if the stack should be saved for the event
        or not (bool).

    Returns:
      result_indices: TypedList of indices (int64) for events meeting the condition.
      result_stack_codes: TypedList where each element is a typed list (of int64) containing
                            the parent's name codes.
      result_stack_ids: TypedList where each element is a typed list (of int64) containing
                        the parent's event indices.
    """
    # Create an empty typed list for the stack, with the defined tuple type.
    stacks = TypedList.empty_list(STACK_TUPLE_TYPE)

    # Create an empty typed list for result_indices (each element is int64).
    result_indices = TypedList.empty_list(int64)

    # Initialize the outer lists for nested results (letting types be inferred on first append)
    result_stack_codes = TypedList()
    result_stack_ids = TypedList()

    n = len(indices)
    for i in range(n):
        t = ts[i]
        duration = dur[i]
        end = t + duration

        # Create a new empty stack for this iteration.
        new_stacks = TypedList.empty_list(STACK_TUPLE_TYPE)
        for st in stacks:
            # st is a tuple: (index, name_code, st_ts, st_end)
            if st[3] >= end and st[2] <= t:
                new_stacks.append(st)
        stacks = new_stacks

        # If the event meets the condition
        if event_needs_stack[i]:
            # Build a typed list of parent's name codes from the current stack.
            stack_codes = TypedList.empty_list(int64)
            for st in stacks:
                stack_codes.append(st[1])
            # Build a typed list of parent's indices (all but the last element of the stack).
            parent_indices = TypedList.empty_list(int64)
            for j in range(len(stacks) - 1):
                parent_indices.append(stacks[j][0])
            result_indices.append(indices[i])
            result_stack_codes.append(stack_codes)
            result_stack_ids.append(parent_indices)
        # Push the current event onto the stack.
        stacks.append((indices[i], names[i], t, end))
    return result_indices, result_stack_codes, result_stack_ids


StackCondition = Union[
    Callable[[pd.DataFrame], "pd.Series[bool]"], Callable[[pd.Series], bool]
]


def add_stack_to_event_df(
    trace_df: pd.DataFrame,
    symbol_table: TraceSymbolTable,
    condition: StackCondition,
    use_shorten_name: bool = True,
    stack_separator: str = ";",
):
    """
    Adds 'stack_ids' and 'stack_name' columns to the input DataFrame.

    These columns infer a stack trace of the event based on the start and end durations of the other events.

    Args:
        trace_df (pd.DataFrame): The input DataFrame containing event data.
        symbol_table (TraceSymbolTable): A table mapping integer codes to symbol names.
        condition (StackCondition): A function that determines whether an event should have its stack saved.
        use_shorten_name (bool, optional): Whether to shorten symbol names. Defaults to True.
        stack_separator (str, optional): A separator used to assemble the stack_name from the names of parent events. Defaults to ";".

    Returns:
        pd.DataFrame: The input DataFrame with 'stack_ids' and 'stack_name' columns added.

    TODO: move to utils if appropriate
    """
    if use_shorten_name:
        mapping = {i: shorten_name(s) for i, s in enumerate(symbol_table.sym_table)}
    else:
        mapping = {i: s for i, s in enumerate(symbol_table.sym_table)}

    saved_stacks = []  # Accumulate tuples: (index, stack_name, stack_ids)
    # Process each group (grouped by (pid, tid)).
    for group_id, group in trace_df.groupby(by=["pid", "tid"]):
        group.sort_values("ts", inplace=True)
        # Call the JIT-compiled function.
        try:
            event_meets_condition: "pd.Series[bool]" = condition(group)  # type: ignore
            assert len(event_meets_condition) == len(group)
            assert len(event_meets_condition) == event_meets_condition.size
        except:  # noqa: E722  # This is a user supplied function
            event_meets_condition = group.apply(condition, axis="columns")  # type: ignore
        res_indices, res_stack_codes, res_stack_ids = (
            _collect_stack_ids_for_process_group(
                group["index"].to_numpy(),
                group["name"].to_numpy(),
                group["ts"].to_numpy(dtype=np.float64),
                group["dur"].to_numpy(dtype=np.float64),
                event_meets_condition.to_numpy(dtype=bool),
            )
        )

        # Postprocess the results: convert the integer codes to string names.
        for i in range(len(res_indices)):
            event_index = res_indices[i]
            code_list = res_stack_codes[i]
            # Convert each integer code into its decoded symbol name.
            str_names = [mapping[code] for code in code_list]
            full_name = stack_separator.join(str_names)
            # Convert parent's indices list into a tuple.
            parent_ids = tuple(res_stack_ids[i])
            saved_stacks.append((event_index, full_name, parent_ids))

    # Merge the computed stacks into the original DataFrame.
    if saved_stacks:
        results_df = pd.DataFrame(
            saved_stacks, columns=["index", "stack_name", "stack_ids"]
        )
        results_df.set_index("index", inplace=True)
        trace_df[["stack_ids", "stack_name"]] = results_df[["stack_ids", "stack_name"]]
    else:
        trace_df["stack_name"] = np.nan
        trace_df["stack_ids"] = np.nan
    return trace_df


def _add_stack_to_event_df_python_reference(
    trace_df: pd.DataFrame, condition: StackCondition, stack_separator: str = ";"
):
    """Pure python reference implementation for add_stack_to_event_df"""
    saved_stacks = []
    for group_id, group in trace_df.groupby(by=["pid", "tid"]):
        stacks: List[Tuple] = []
        for ind, row in group.sort_values("ts").iterrows():
            end = row["ts"] + row["dur"]
            stacks = [(*i, s, e) for *i, s, e in stacks if e >= end and s <= row["ts"]]
            if condition(row):
                full_name = stack_separator.join(s[1] for s in stacks)
                stack_ids = tuple(s[0] for s in stacks[:-1])
                saved_stacks.append((row["index"], full_name, stack_ids))
            stacks.append((row["index"], row["s_name"], row["ts"], end))

    trace_df[["stack_ids", "stack_name"]] = pd.DataFrame(
        saved_stacks, columns=["index", "stack_name", "stack_ids"]
    ).set_index("index")[["stack_ids", "stack_name"]]
    return trace_df


class MemoryEvent(TypedDict):
    """A simple class to give approximate type hint for the internal functions in this class"""

    ts: int
    addr: int
    total_allocated: float
    total_reserved: float
    bytes_delta: float
    dur: float
    stack_name: str
    stack_class: str
    alloc_or_dealloc_ts: int
    device_id: int


class MemoryAnalysis:
    """Class for analyzing memory usage patterns in HTA traces"""

    def __init__(self, t: Trace):
        """Initialize with an HTA Trace object

        Args:
            t (Trace): HTA Trace object containing memory events
        """
        self.t = t

    def _get_rank(self, rank: Union[None, int]):
        if rank is None:
            ranks = sorted(self.t.get_all_traces().keys())
            if not ranks:
                raise ValueError("No ranks found in trace")
            rank = ranks[0]

        return self.t.get_trace(rank)

    def _process_memory_events(self, rank: Optional[int] = None) -> pd.DataFrame:
        """Process memory events from trace into a DataFrame

        Args:
            rank (Optional[int]): Process events for specific rank. If None, use first rank.

        Returns:
            pd.DataFrame containing memory events
        """
        trace_df = self._get_rank(rank)
        # Filter memory events using the column names from the default parser config
        memory_events = trace_df[
            (trace_df["total_allocated"] >= 0) | (trace_df["total_reserved"] >= 0)
        ].copy()

        if memory_events.empty:
            logger.warning("No memory events found in trace")
            return pd.DataFrame()

        return memory_events

    def get_memory_timeline(
        self,
        rank: Optional[int] = None,
        visualize: bool = True,
        device=None,
    ) -> pd.DataFrame:
        """Generate timeline of memory usage

        Args:
            rank (Optional[int]): Analyze specific rank. If None, use first rank.
            visualize (bool): Whether to display the plot. Default=True.

        Returns:
            pd.DataFrame: DataFrame containing memory timeline data
        """
        # Process events
        events_df = self._process_memory_events(rank)

        if events_df.empty:
            return pd.DataFrame()

        if visualize:
            # Create plot

            # Plot allocated memory
            events_df.sort_values("ts", inplace=True)
            for device, timeline in events_df.groupby("device_id"):
                fig = go.Figure()
                allocated_gb = timeline["total_allocated"] / (1024**3)
                reserved_gb = timeline["total_reserved"] / (1024**3)

                fig.add_trace(
                    go.Scatter(
                        x=timeline["ts"] / 1e6,  # Convert to milliseconds
                        y=allocated_gb,  # Convert to GB
                        name="Allocated Memory",
                        mode="lines",
                        line=dict(color=colorscheme[0]),
                    )
                )

                # Plot reserved memory
                fig.add_trace(
                    go.Scatter(
                        x=timeline["ts"] / 1e6,
                        y=reserved_gb,
                        name="Reserved Memory",
                        mode="lines",
                        line=dict(color=colorscheme[1]),
                    )
                )

                # Update layout
                fig.update_layout(
                    title=f"Memory Usage Timeline (device {device})",
                    xaxis_title="Time (ms)",
                    yaxis_title="Memory (GB)",
                    hovermode="x unified",
                )

                fig.show()

        return events_df

    def _add_stack_frames_to_memory_events(
        self,
        rank: Union[int, None] = None,
        stack_separator: str = ";",
        use_shorten_name: bool = True,
        force_recompute: bool = False,
        _use_reference_implementation: bool = False,
    ):
        """
        Adds 'stack_ids' and 'stack_name' columns for memory events.

        These columns infer a stack trace of the event based on the start and
        end durations of the other events.

        * 'stack_ids' is a tuple of parent event indices, ordered from oldest to newest.
        * 'stack_name' is a string combining the names of each parent (decoded from the 'name'
        column) with the stack_separator from oldest to newest.

        Only events on the same thread and process are considered parents.

        Args:
            rank: The rank of the trace that needs to be processed.
            stack_separator: A separator used to assemble the stack_name from the names of
            parent events.
            use_shorten_name: Whether to shorten symbol names (as in your decode function).
        """

        # Default condition if not provided.
        def is_memory(row):
            return row["total_allocated"] >= 0

        # Decode symbol ids (this will fill in s_name, s_cat, etc., if desired).
        self.t.decode_symbol_ids(use_shorten_name=use_shorten_name)
        trace_df = self._get_rank(rank)

        if not force_recompute and "stack_name" in trace_df.columns:
            logger.info("Previous stack_name found - skipping")
            return trace_df

        logger.info("Calculating stack_name - this process can be slow")

        # Build a mapping from integer code (from the 'name' column) to the decoded string.
        if _use_reference_implementation:
            _add_stack_to_event_df_python_reference(
                trace_df, condition=is_memory, stack_separator=stack_separator
            )
        else:
            add_stack_to_event_df(
                trace_df,
                self.t.symbol_table,
                condition=is_memory,
                use_shorten_name=use_shorten_name,
                stack_separator=stack_separator,
            )

        return trace_df

    def _test_stack_name_approaches(self, rank: Union[int, None] = None):
        """TODO: make this an actual test

        Checks that the numba and python implementation match.
        """
        print("Using fast stack build", sep=" ")
        start = time.time()
        fast_trace = self._add_stack_frames_to_memory_events(
            rank, force_recompute=True
        )[["stack_ids", "stack_name"]].copy()
        print(f" - done in {time.time() - start}s")
        print("Using python stack build", sep=" ")
        start = time.time()
        slow_trace = self._add_stack_frames_to_memory_events(
            rank, force_recompute=True, _use_reference_implementation=True
        )[["stack_ids", "stack_name"]].copy()
        print(f" - done in {time.time() - start}s")
        for n in ["stack_ids", "stack_name"]:
            assert (
                fast_trace.dropna()[n] == slow_trace.dropna()[n]
            ).any(), f"{n} did not have any match"
            assert (
                fast_trace.dropna()[n] == slow_trace.dropna()[n]
            ).all(), f"{n} did not all match"
        print("Stack names and IDs matched between python and numba implementations")
        return fast_trace, slow_trace

    def _add_alloc_or_dealloc_to_memory_events(
        self,
        rank: Union[int, None] = None,
    ):
        """Adds a column to memory events with the index of the corresponding allocation
        or deallocation.

        When an event is an allocation (i.e. it increases memory usage - bytes_delta > 0):
        - find events with the same address.
        - look for the next event operating on the same address with the same amount of memory.
        - store the index of that event.

        Conversely for deallocation events.

        Events for which a match is not found are left empty. This likely means the memory
        was allocated or deallocated outside of the profiling window.
        """

        trace_df = self._add_stack_frames_to_memory_events(rank)
        memory_events = trace_df.loc[trace_df.total_reserved >= 0]
        if "alloc_or_dealloc_id" in memory_events.columns:
            logger.info("Previous alloc and dealloc found - skipping")
            return memory_events
        logger.info("Calculating alloc and dealloc")
        mem_by_addr = memory_events.set_index("addr")

        def find_deallocation(mem_event: MemoryEvent):
            if mem_event["bytes_delta"] < 0 or mem_event["addr"] < 0:
                return
            same_addr = mem_by_addr.loc[mem_event["addr"]]
            if not isinstance(same_addr, pd.DataFrame):
                # There was a single event at that memory location - no matching events
                return

            same_addr = same_addr.loc[same_addr.device_id == mem_event["device_id"]]
            out = same_addr.loc[same_addr.ts > mem_event["ts"]]

            if len(out) > 0:
                out = out[-mem_event["bytes_delta"] == out["bytes_delta"]]
            if len(out) > 0:
                return int(out.iloc[out.ts.argmin()]["index"])
            else:
                return

        def find_allocation(mem_event: MemoryEvent):
            if mem_event["bytes_delta"] > 0 or mem_event["addr"] < 0:
                return
            same_addr = mem_by_addr.loc[mem_event["addr"]]
            if not isinstance(same_addr, pd.DataFrame):
                # There was a single event at that memory location - no matching events
                return
            same_addr = same_addr.loc[same_addr.device_id == mem_event["device_id"]]
            out = same_addr.loc[same_addr.ts < mem_event["ts"]]

            if len(out) > 0:
                out = out[-mem_event["bytes_delta"] == out["bytes_delta"]]
            if len(out) > 0:
                return int(out.iloc[out.ts.argmax()]["index"])
            else:
                return

        def find_mirror_event_time(mem_event: MemoryEvent):
            if mem_event["bytes_delta"] > 0:
                return find_deallocation(mem_event)
            elif mem_event["bytes_delta"] < 0:
                return find_allocation(mem_event)
            return

        trace_df.loc[memory_events.index, "alloc_or_dealloc_id"] = memory_events.apply(find_mirror_event_time, axis="columns")  # type: ignore
        return trace_df.loc[memory_events.index]

    def get_classified_memory_timelines(
        self,
        rank: Union[int, None] = None,
        classification_func: Union[Callable[[MemoryEvent], str], None] = None,
        visualize: bool = True,
    ):
        """
        Returns a dictionary of DataFrames, where each key represents a device ID and
        the corresponding value is a DataFrame containing categorized memory events.

        The output DataFrames have the following columns:
        * 'ts': time of the sample in nanoseconds
        * 'ms': time of the sample in milliseconds
        * 'stack_name': the stack of event inferred from which events started before and finished
            after.
        * 'stack_type': a string representing the category of the event, determined by the
            classification function
        * 'alloc_or_dealloc_id': the index of the allocation or deallocation event which is matched
            to the current memory event.
        * 'is_plotting_event_only': a boolean indicating whether the event is only for plotting
            purposes (i.e., it has zero delta and is used to create a square memory profile)
        * Categorized memory usage columns: one column for each unique stack type, containing
            the cumulative sum of 'bytes_delta' for events of that type.

        Events can be categorized using a built-in function which is tuned to give useful
        information on a Llama model, compiled with torch compile, and distributed over
        multiple GPUs with FSDP.

        Args:
            rank (Union[int, None], optional): The rank of the trace that needs to be processed.
                If None, use the first rank. Defaults to None.
            classification_func (Union[Callable[[MemoryEvent], str], None], optional):
                A function that takes a MemoryEvent and returns a string representing the
                category of the event. If None, the default classification function is used.
                Defaults to None.
            visualize (bool, optional): Whether to display the plot. Default is True.

        Returns:
            dict[int, pd.DataFrame], pd.DataFrame: A dictionary where each key is a device ID and the
            corresponding value is a DataFrame containing categorized memory events.

        """

        unknown_alloc_class = "allocated_before_profile"
        if classification_func is None:
            classification_func = classify_torchtitan_calls

        def _class_func(row):
            """Handle the special "allocated before profile" class"""
            if row["stack_name"] == unknown_alloc_class:
                return unknown_alloc_class
            return classification_func(row)

        memory_events = self._add_alloc_or_dealloc_to_memory_events(rank)
        device_timelines: Dict[int, pd.DataFrame] = {}
        first_memory_event_time = memory_events["ts"].min()
        device_dfs: Dict[int, pd.DataFrame] = {}
        for device, tmp in memory_events.groupby(by="device_id"):

            # Create an event for allocations which have happened before the profile
            pre_profile_allocs = tmp.iloc[tmp["ts"].argmin()].copy()
            pre_alloc_index = tmp.index.max() + 1
            pre_profile_allocs["total_allocated"] -= pre_profile_allocs["bytes_delta"]
            pre_profile_allocs["bytes_delta"] = pre_profile_allocs["total_allocated"]
            pre_profile_allocs["ts"] = first_memory_event_time - 1
            pre_profile_allocs["addr"] = -1
            pre_profile_allocs["stack_name"] = unknown_alloc_class
            pre_profile_allocs["ev_idx"] = -1
            pre_profile_allocs["external_id"] = -1
            pre_profile_allocs["index"] = pre_alloc_index
            tmp.loc[pre_alloc_index] = pre_profile_allocs
            tmp["stack_type"] = tmp.apply(_class_func, axis="columns")

            logger.info("indexing allocation types")
            # For the purpose of this plot deallocations need to match that of their allocation id
            mask = (tmp.bytes_delta < 0) & tmp.alloc_or_dealloc_id.isna()
            tmp.loc[mask, "stack_type"] = unknown_alloc_class
            mask = (tmp.bytes_delta < 0) & (tmp.alloc_or_dealloc_id > 0)
            tmp.loc[mask, "stack_type"] = tmp.loc[
                tmp.loc[mask, "alloc_or_dealloc_id"], "stack_type"
            ].values
            logger.info("Assembling timelines")
            timelines = pd.DataFrame()
            for i, g in tmp.groupby(by="stack_type"):
                if len(g) == 0:
                    continue
                # Add events with zero delta before each event to get nice square
                # memory profiles
                ref = g[["ts", "bytes_delta", "stack_name"]].copy()
                ref["ts"] -= 1
                ref["bytes_delta"] = 0
                ref["is_plotting_event_only"] = True
                g["is_plotting_event_only"] = False

                out = pd.concat(
                    [
                        g[
                            [
                                "ts",
                                "bytes_delta",
                                "stack_name",
                                "is_plotting_event_only",
                            ]
                        ],
                        ref,
                    ]
                ).sort_values("ts")
                out[i] = out["bytes_delta"].cumsum()
                out["category"] = i
                timelines = pd.concat([timelines, out])
            device_timelines[device] = timelines  # type: ignore
            device_dfs[device] = tmp  # type: ignore
        if visualize:
            for device, timelines in device_timelines.items():
                timelines["ms"] = timelines["ts"] / 1e6
                category_columns: pd.Series = (
                    timelines.sort_values("ts")
                    .apply(lambda col: col.first_valid_index(), axis="index")
                    .drop(
                        [
                            "bytes_delta",
                            "stack_name",
                            "category",
                            "ts",
                            "ms",
                            "is_plotting_event_only",
                        ]
                    )
                )
                column_order_by_appearance = pd.Series(
                    category_columns.index,
                    index=category_columns.values,
                )[
                    timelines.loc[category_columns.values, "ms"]
                    .groupby(level=0)
                    .min()
                    .sort_values()
                    .index
                ].values

                fig = (
                    timelines.sort_values("ms").ffill(axis="index").set_index("ms")
                    # .drop(["bytes_delta", "stack_name", "category", "ts"], axis="columns")
                    [column_order_by_appearance]
                    / (1024**3)
                ).plot.area(backend="plotly")
                fig.update_layout(
                    title=f"Memory breakdown - device {device}",
                    xaxis_title="Time (ms)",
                    yaxis_title="Memory (GB)",
                    legend_title="Memory type",
                )
                fig.show()
        return device_timelines, device_dfs


def classify_torchtitan_calls(mem_event: MemoryEvent):
    """Classifies memory events based on stack names

    Designed to give insight into a Llama 3 model implemented
    in torchtitan using FSDP.
    """
    stack_class = ""
    stack_name = mem_event["stack_name"]
    lower_name = stack_name.lower()
    if "backward" in lower_name:
        stack_class += "backward"
    elif "forward" in lower_name:
        stack_class += "forward"
    elif "optimizer" in lower_name:
        stack_class += "optimizer"
    else:
        stack_class += "unknown"

    if "loss" in lower_name:
        stack_class += ":loss"

    if re.search("optimizedmodule_([0-9]+)", lower_name):
        stack_class += ":layer"

    if "shard" in lower_name:
        stack_class += ":fsdp_parameters"
    # if "flash_attention" in lower_name:
    #     stack_class += ":flash_attention"
    if "forward" in lower_name and "checkpoint" in lower_name:
        stack_class += ":activation"
    if "embedding" in lower_name:
        stack_class += ":embedding"
    if (
        "rmsnorm_0" in lower_name
        or "linear_0" in lower_name
        and "transformer_0" in lower_name
    ):
        stack_class += ":output"

    if "backward" in lower_name and ("call" in lower_name or "aten" in lower_name):
        stack_class += ":details"

    if ":" not in stack_class:
        return stack_class + ":unknown"
    return stack_class
