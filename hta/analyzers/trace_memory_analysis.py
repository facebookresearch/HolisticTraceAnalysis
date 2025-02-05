import re
from typing import Callable, Optional, TypedDict

import pandas as pd
import plotly
import plotly.graph_objects as go

from hta.common.trace import Trace
from hta.configs.config import logger

colorscheme = plotly.colors.qualitative.Pastel


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

    def _get_rank(self, rank: None | int):
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
        self, rank: Optional[int] = None, visualize: bool = True
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
            fig = go.Figure()

            # Plot allocated memory
            events_df.sort_values("ts", inplace=True)
            gpu_device = events_df.device_id != -1
            allocated_gb = events_df.loc[gpu_device, "total_allocated"] / (1024**3)
            reserved_gb = events_df.loc[gpu_device, "total_reserved"] / (1024**3)

            fig.add_trace(
                go.Scatter(
                    x=events_df["ts"] / 1e6,  # Convert to milliseconds
                    y=allocated_gb,  # Convert to GB
                    name="Allocated Memory",
                    mode="lines",
                    line=dict(color=colorscheme[0]),
                )
            )

            # Plot reserved memory
            fig.add_trace(
                go.Scatter(
                    x=events_df["ts"] / 1e6,
                    y=reserved_gb,
                    name="Reserved Memory",
                    mode="lines",
                    line=dict(color=colorscheme[1]),
                )
            )

            # Update layout
            fig.update_layout(
                title="Memory Usage Timeline",
                xaxis_title="Time (ms)",
                yaxis_title="Memory (GB)",
                hovermode="x unified",
                width=1200,
                height=800,
            )

            fig.show()

        return events_df

    def _add_stack_frames_to_memory_events(
        self,
        rank: int| None=None,
        condition: None | Callable[[pd.Series], bool] = None,
        stack_separator: str = ";",
    ):
        """Adds 'stack_ids' and 'stack_name' columns for memory events.

        These columns capture infer a stack trace of the event based on the start and
        end durations of the other events.

        - 'stack_ids' is a tuple of index, ordered from oldest to newest parent event
        - 'stack_name' is a string combining the names of each parent with the
          stack_separator from oldest to newest.

        Only events on the same thread and process are considered parents.

        Args:
            rank: The rank of the trace that needs to be processed.
            condition: a callable to decide if the stack trace should be added. By default
                only memory events are processed.
            stack_separator: A separator used to assemble the stack_name.
        """

        def is_memory(row):
            return row["total_allocated"] >= 0

        if condition is None:
            condition = is_memory

        self.t.decode_symbol_ids()
        trace_df = self._get_rank(rank)
        # if rank in self._ranks_with_stacks:
        if "stack_name" in trace_df.columns:
            print("Previous stack_name found - skipping")
            return trace_df
        print("Calculating stack_name")

        saved_stacks = []
        for group_id, group in trace_df.groupby(by=["pid", "tid"]):
            stacks = []
            for ind, row in group.sort_values("ts").iterrows():
                end = row["ts"] + row["dur"]
                stacks = [
                    (*i, s, e) for *i, s, e in stacks if e >= end and s <= row["ts"]
                ]
                if condition(row):
                    full_name = stack_separator.join(s[1] for s in stacks)
                    stack_ids = tuple(s[0] for s in stacks[:-1])
                    saved_stacks.append((row["index"], full_name, stack_ids))
                stacks.append((row["index"], row["s_name"], row["ts"], end))

        trace_df[["stack_ids", "stack_name"]] = pd.DataFrame(
            saved_stacks, columns=["index", "stack_name", "stack_ids"]
        ).set_index("index")[["stack_ids", "stack_name"]]
        return trace_df

    def _add_alloc_or_dealloc_to_memory_events(self,         rank: int| None=None,
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
            print("Previous alloc and dealloc found - skipping")
            return memory_events
        print("Calculating alloc and dealloc")
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
        rank: int| None=None,
        classification_func: Callable[[MemoryEvent], str] | None = None,
        visualize: bool = True,
    ):

        unknown_alloc_class = "allocated_before_profile"
        if classification_func is None:
            classification_func = classify_torchtitan_calls

        def _class_func(row):
            """Handle the special "allocated before profile" class"""
            if row["stack_name"] == unknown_alloc_class:
                return unknown_alloc_class
            return classification_func(row)

        memory_events = self._add_alloc_or_dealloc_to_memory_events(rank)
        device_timelines = {}
        first_memory_event_time = memory_events["ts"].min()
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

                out = pd.concat(
                    [
                        g[["ts", "bytes_delta", "stack_name"]],
                        ref,
                    ]
                ).sort_values("ts")
                out[i] = out["bytes_delta"].cumsum()
                out["category"] = i
                timelines = pd.concat([timelines, out])
            device_timelines[device] = timelines
        if visualize:
            for device, timelines in device_timelines.items():
                timelines["ms"] = timelines["ts"] / 1e6
                fig = (
                    timelines.sort_values("ms")
                    .ffill(axis="index")
                    .set_index("ms")
                    .drop(["bytes_delta", "stack_name", "category", "ts"], axis="columns")
                    / (1024**3)
                ).plot.area(backend="plotly")
                fig.update_layout(
                    title=f"Memory breakdown - device {device}",
                    xaxis_title="Time (ms)",
                    yaxis_title="Memory (GB)",
                    legend_title="Memory type",
                    hovermode="x unified",
                )
                fig.show()
        return device_timelines


def classify_torchtitan_calls(mem_event: MemoryEvent):
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

    if m := re.search("optimizedmodule_([0-9]+)", lower_name):
        # stack_class += f":layer_{m.group(1)}"
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
