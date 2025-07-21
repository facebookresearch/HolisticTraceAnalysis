# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import time
from collections import defaultdict

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import hta.configs.env_options as hta_options

import networkx as nx
import pandas as pd
from hta.analyzers.trace_counters import TraceCounters

# Revert to old call stack as we have an issue with new one
# https://github.com/facebookresearch/HolisticTraceAnalysis/issues/113
# from hta.common.trace_call_graph import CallGraph, CallStackGraph, DeviceType
from hta.common.call_stack import CallGraph, CallStackGraph, DeviceType

from hta.common.trace import Trace
from hta.common.trace_symbol_table import decode_symbol_id_to_symbol_name
from hta.configs.config import logger
from hta.utils.utils import is_comm_kernel


PROFILE_TIMES = {}
CUDA_RUNTIME_EVENTS = frozenset(
    ["cudaHostAlloc", "cudaLaunchKernel", "cudaMemcpyAsync"]
)


# Enable per function timing
def timeit(func):
    global PROFILE_TIMES

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        PROFILE_TIMES[func.__name__] = total_time
        return result

    return timeit_wrapper


@dataclass
class CPNode:
    """A node in the critical path di-graph.
    This represents a point in time. It could be start
    or end of an operator or a kernel.
    """

    idx: int = -1  # Identifies this node in the constructed graph's node list.
    ev_idx: int = -1  # Identifies the corresponding event in the trace dataframe.
    ts: int = 0
    is_start: bool = False
    # Cache the is blocking calls in the object
    is_blocking: bool = False

    def __repr__(self) -> str:
        return (
            f"CPNode(event: {self.ev_idx}, node_id={self.idx}, "
            f"ts={self.ts}, is_start={self.is_start}, "
            f"is_blocking={self.is_blocking})"
        )


class CPEdgeType(Enum):
    OPERATOR_KERNEL = "critical_path_operator"  # Edge between start and end nodes for a single CPU operator or GPU kernel.
    DEPENDENCY = "critical_path_dependency"  # Edge between nested CPU operators.
    KERNEL_LAUNCH_DELAY = "critical_path_kernel_launch_delay"  # Edge between CPU launch and GPU kernel start.
    KERNEL_KERNEL_DELAY = "critical_path_kernel_kernel_delay"  # Edge between successive kernels on the same GPU.
    SYNC_DEPENDENCY = "critical_path_sync_dependency"  # Synchronization or control dependency between events.


DEFAULT_EDGE_TYPES_IN_VIZ: Set[CPEdgeType] = {
    CPEdgeType.DEPENDENCY,
    CPEdgeType.KERNEL_LAUNCH_DELAY,
    CPEdgeType.SYNC_DEPENDENCY,
}


@dataclass(frozen=True)
class CPEdge:
    """An edge in the critical path di-graph.

    This represents either one of:
    1) a span of time i.e. duration of an operator/kernel.
    2) a dependency among operators/kernels.
    3) a kernel launch or kernel-kernel delay.
    4) a synchronization delay.
    The weight represents time in the graph. Cases 1) and 3)
    above have non-zero weights.

    Once we initialize the edge we should not modify the data members below.
    """

    # begin and end node indices
    begin: int
    end: int
    type: CPEdgeType
    weight: int = 0


@dataclass(frozen=True)
class _CPGraphData:
    """Contains data members of CPGraph that we can save and
    restore from a file. This excludes the graph itself and the
    clipped dataframe that are saved separately.
    """

    node_list: List[CPNode]
    critical_path_nodes: List[int]
    critical_path_events_set: Set[int]
    critical_path_edges_set: Set[CPEdge]
    event_to_start_node_map: Dict[int, int]
    event_to_end_node_map: Dict[int, int]
    edge_to_event_map: Dict[Tuple[int, int], int]


class CPGraph(nx.DiGraph):
    """Critical path analysis graph representation for trace from one rank.
    This object constructs a graph that can be analyzed using networkx library.

    We maintain a mapping between node ids -> CPNode objects
    and use the integer as a node in the networkx graph datastructure.
    Edges are directly used as the type is hashable.

    Attributes:
        trace_df (pd.DataFrame): dataframe of trace events used to construct this graph.
        symbol_table (TraceSymbolTable): a symbol table used to encode the symbols in the trace.
        node_list (List[int]): list of critical path node objects, index in this list is always the node id..
        critical_path_nodes (List[int]): list of node ids on the critical path.
        critical_path_events_set (Set[int]): set of event ids corresponding to the critical path nodes.
        critical_path_edges_set (Set[CPEdge]): set of edge objects that are on the critical path.
    """

    BLOCKING_SYNC_CALLS = [
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        # CUDA event APIshttps://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        "cudaEventQuery",
        "cudaEventSynchronize",
        # Memory copies
        "cudaMemcpy",
        # In some cases even Async memcpy might synchronize
        # https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
        "cudaMemcpyAsync",
    ]

    @lru_cache()
    def _add_zero_weight_launch_edges(self) -> bool:
        return hta_options.critical_path_add_zero_weight_launch_edges()

    def __init__(
        self, t: Optional["Trace"], t_full: "Trace", rank: int, G=None
    ) -> None:
        """Initialize a critical path graph object.

        This can be done in two ways:
            1) Generate critical path analysis graph from scatch.
            2) Restore a serialized CPGraph object.

        For (2) the networkx.DiGraph object G is utilized, see restore_cpgraph() function in this file.

        Args:
            t (Trace): Clipped trace object focussing on region of interest.
            t_full (Trace): Full Trace object.
            rank (int): Rank to perform analysis on.
            G (networkx.DiGraph): An optional DiGraph object.
        """
        self.rank: int = rank
        self.t = t
        self.t_full = t_full
        self.full_trace_df: pd.DataFrame = self.t_full.get_trace(rank)
        self.symbol_table = t_full.symbol_table

        # init networkx DiGraph
        super(CPGraph, self).__init__(G)

        if t is None:
            return

        self.trace_df: pd.DataFrame = t.get_trace(rank)

        self.critical_path_nodes: List[int] = []
        self.critical_path_events_set: Set[int] = set()
        self.critical_path_edges_set: Set[CPEdge] = set()

        self.node_list: List[CPNode] = []

        # map from event id in trace dataframe -> CPGraph node_id
        self.event_to_start_node_map: Dict[int, int] = {}
        self.event_to_end_node_map: Dict[int, int] = {}

        # map from edge (u, v) -> event id in trace dataframe
        # this is the attributed event for an edge
        self.edge_to_event_map: Dict[Tuple[int, int], int] = {}

        self._construct_graph()

    def _add_node(self, node: CPNode) -> int:
        """Adds a node to the graph.
        Args: node (CPNode): node object
        Returns int as node index."""
        self.node_list.append(node)
        idx = node.idx = len(self.node_list) - 1
        self.add_node(idx)  # Call to networkx.DiGraph
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Adding critical path node = {node}")
        return idx

    def _add_edge(self, edge: CPEdge) -> None:
        """Adds a edge to the graph.
        Args: node (CPEdge): edge object"""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Adding critical path edge: {edge}")
        self.add_edge(edge.begin, edge.end, weight=edge.weight, object=edge)

    def _add_edge_helper(
        self,
        src: CPNode,
        dest: CPNode,
        type: CPEdgeType,
        zero_weight: bool = False,
    ) -> CPEdge:
        """Adds a edge between two nodes.

        Args:
            src (CPNode): node object for source.
            dest (CPNode): node object for destination.
            type: (CPEdgeType): type of edge.
            zero_weight (bool): if True, the edge is added with zero weight.

        Returns:
            CPEdge: The added edge.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Adding an edge between nodes {src.idx} -> {dest.idx}"
                f" type = {type}"
            )
        assert src.idx != dest.idx, f"Src node {src} == Dest node {dest}"
        weight = (
            0
            if (
                type in [CPEdgeType.DEPENDENCY, CPEdgeType.SYNC_DEPENDENCY]
                or zero_weight
            )
            else (dest.ts - src.ts)
        )

        e = CPEdge(begin=src.idx, end=dest.idx, type=type, weight=weight)
        self._add_edge(e)
        return e

    def _attribute_edge(self, e: CPEdge, src_parent: int) -> None:
        """Attribute an edge to nearest matching event by updating edge_to_event_map.

        Args:
            e (CPEdge): Edge to attribute
            src_parent (int): Parent event of the src node.

        The src_parent is required when we consider nested operators.
        See the explanation below for more details.
        """
        # Edge attribution is only applicable for edges representing
        # operator or kernel spans or delay spans
        if e.type not in {CPEdgeType.OPERATOR_KERNEL, CPEdgeType.KERNEL_KERNEL_DELAY}:
            return

        """ For nested events consider the following cases
        where the src and dest can each be either start or end nodes.

        N |            Start or End?  | Edge Attributed to
        - -------------------------------------------------
        1  Src, Dest = (Start, Start) |      Src
        2  Src, Dest = (Start, End)   |      Src = Dest
        3  Src, Dest = (End, End)     |      Dest
        4  Src, Dest = (End, Start)   |      Src.parent

        In cases 1 and 2 it is understandable that both cases
        the edge would actually reside in the start event. Case 2
        is especially referring to an operator that is at the bottom
        of the stack or is a GPU kernel.

        Case 3 we are unwinding the stack and in this case the edge will
        reside within Dest.
        Case 4 is intersting where we are in an intermediate phase within
        another operator like -

        |-------------------- Op A ----------------------|
            |--- Op B ---|  --------> |---- Op C ----|
                        end   ^edge  start
        The edge between Op B and Op C should be attributed to the parent
        operator A. Hence the exception here.
        """
        src, dest = self.node_list[e.begin], self.node_list[e.end]

        ev_idx = 0
        if e.type == CPEdgeType.KERNEL_KERNEL_DELAY:
            # arbitrary but assigning the delay to previous kernel
            ev_idx = src.ev_idx
        elif src.is_start:
            ev_idx = src.ev_idx  # Case 1 & 2
        elif not dest.is_start:
            ev_idx = dest.ev_idx  # Case 3
        else:
            ev_idx = src_parent  # Case 4
        self.edge_to_event_map[(src.idx, dest.idx)] = int(ev_idx)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Attributing edge between nodes {src.idx} -> {dest.idx}"
                f" to event id = {ev_idx}, event_name = {self._get_node_name(ev_idx)}"
            )

    @cached_property
    def _event_to_attributed_edges_map(self) -> Dict[int, List[CPEdge]]:
        """Caches a map from an event ID -> list of CPEdge objects
        attributed to an event"""
        d: Dict[int, List[CPEdge]] = defaultdict(list)
        for uv, ev_idx in self.edge_to_event_map.items():
            u, v = uv
            e = self.edges[u, v]["object"]
            d[ev_idx].append(e)
        return d

    def _get_node_name(self, ev_id: int) -> str:
        if ev_id < 0:
            return "ROOT"
        name_id = self.trace_df.name.loc[ev_id]
        return self.symbol_table.get_sym_table()[name_id]

    def get_nodes_for_event(
        self, ev_id: int
    ) -> Tuple[Optional[CPNode], Optional[CPNode]]:
        """Lookup corresponding nodes for an event id in the trace dataframe.

        Args:
            ev_id (int): index of the event in trace dataframe.

        Returns:
            Tuple[Optional[CPNode], Optional[CPNode]]
                Pair of CPNodes aka start and end CPNode for the event.
        """
        start_node = self.event_to_start_node_map.get(ev_id, -1)
        end_node = self.event_to_end_node_map.get(ev_id, -1)
        return (
            self.node_list[start_node] if start_node >= 0 else None,
            self.node_list[end_node] if end_node >= 0 else None,
        )

    def get_events_for_edge(self, edge: CPEdge) -> Tuple[int, int]:
        """Lookup corresponding event nodes for an edge.

        Args:
            edge (CPEdge): edge object that is part of the di-graph.

        Returns:
            Tuple[int, int]: Pair of event ids representing src and dest of the edge.
        """
        start_node, end_node = edge.begin, edge.end
        return int(self.node_list[start_node].ev_idx), int(
            self.node_list[end_node].ev_idx
        )

    def get_event_attribution_for_edge(self, edge: CPEdge) -> Optional[int]:
        """Helper to look up event attributed to an edge.

        Args:
            edge (CPEdge): edge object that is part of the di-graph

        Returns:
            Optional[int]: Event id attributed to this edge.
                Note only operator/kernel events have attribution.
                Returns None in other cases.
        """
        return self.edge_to_event_map.get((edge.begin, edge.end), None)

    def get_edges_attributed_to_event(self, ev_idx: int) -> List[CPEdge]:
        """Helper to look up edges attributed to a specific event.

        Args:
            ev_idx (int): Index of event to lookup attributed edges for.

        Returns:
            List[CPEdge]: List of edges attributed to this event in the graph.
        """
        return self._event_to_attributed_edges_map.get(ev_idx, [])

    def _construct_graph(self) -> None:
        if self._add_zero_weight_launch_edges():
            logger.info(
                "Adding zero weight launch edges to retain causality in subsequent simulations."
            )

        self._create_event_nodes()

        self._construct_graph_from_cuda_runtime_events()

        self._construct_graph_from_call_stacks()

        self._construct_graph_from_kernels()

    @timeit
    def _create_event_nodes(self) -> None:
        """Generates a start and end node for every event we would like
        to represent in our graph"""

        operator_or_runtime_events_mask = (
            self.symbol_table.get_operator_or_cuda_runtime_mask(self.trace_df)
        )
        gpu_events_mask = (self.trace_df["stream"] != -1) & (
            self.trace_df["index_correlation"] >= 0
        )

        # We only care about CPU op/runtime events and GPU events.
        events_mask = operator_or_runtime_events_mask | gpu_events_mask

        events_df = (self.trace_df[events_mask][["index", "ts", "dur", "name"]]).rename(
            columns={"index": "ev_idx"}
        )

        blocking_calls = {
            s
            for b in self.BLOCKING_SYNC_CALLS
            if (s := self.symbol_table.sym_index.get(b)) is not None
        }
        events_df["is_blocking_call"] = events_df.name.isin(blocking_calls)

        ops_df_start = events_df.copy()
        ops_df_end = events_df.copy()

        ops_df_start.drop(axis=1, columns=["dur"], inplace=True)
        ops_df_start["is_start"] = True

        ops_df_end["end"] = ops_df_end["ts"] + ops_df_end["dur"]
        ops_df_end.drop(axis=1, columns=["dur", "ts"], inplace=True)
        ops_df_end.rename(columns={"end": "ts"}, inplace=True)
        ops_df_end["is_start"] = False

        nodes_df = (
            pd.concat([ops_df_start, ops_df_end])
            .sort_values(by=["ts", "ev_idx"], axis=0)
            .reset_index(drop=True)
            .reset_index(names="idx")
        )

        # Create nodes
        _df = nodes_df
        self.node_list = [
            CPNode(*args)
            for args in zip(
                _df["idx"],
                _df["ev_idx"],
                _df["ts"],
                _df["is_start"],
                _df["is_blocking_call"],
            )
        ]

        _df = nodes_df[nodes_df.is_start]
        self.event_to_start_node_map = dict(zip(_df["ev_idx"], _df["idx"]))
        _df = nodes_df[~nodes_df.is_start]
        self.event_to_end_node_map = dict(zip(_df["ev_idx"], _df["idx"]))
        assert len(self.event_to_start_node_map) == len(self.event_to_end_node_map)

    def _has_nodes(self, ev_id: int) -> bool:
        """Checks if a node exists for an event id.

        Args:
            ev_id (int): event id to check for.

        Returns:
            bool: True if node exists for the event id.
        """
        start_node, end_node = self.get_nodes_for_event(ev_id)
        return start_node is not None and end_node is not None

    def _is_cuda_runtime_event_node(self, node: CPNode) -> bool:
        """Checks if a node is a locking event node.

        Args:
            node (CPNode): node object

        Returns:
            bool: True if node is a locking event node.
        """
        event_name = self._get_node_name(node.ev_idx)
        return event_name in CUDA_RUNTIME_EVENTS

    def _shares_cuda_runtime_lock(
        self, first_node: CPNode, second_node: CPNode
    ) -> bool:
        """
        Returns True if two consecutive nodes share a cuda runtime lock.

        Two consecutive nodes share a cuda runtime lock if they're both
        cuda runtime events AND they're not a pair of (end, start) nodes.

        A pair of (end, start) nodes would represent non-overlapping events
        as shown below.

        |---cudaLaunchKernel---|    |---cudaLaunchKernel---|
                               ^    ^
                            (end,  start)
        """
        return (
            self._is_cuda_runtime_event_node(first_node)
            and self._is_cuda_runtime_event_node(second_node)
            and (first_node.is_start or not second_node.is_start)
        )

    @timeit
    def _construct_graph_from_cuda_runtime_events(self) -> None:
        """Constructs the graph from CUDA runtime events.

        Specific events acquire the cuda runtime lock, such that only
        one of these events can make progress at a time. This is a
        type of cross-thread dependency that we need to model into our
        graph.

        For example, consider the following:

        Thread 0:               |(A)-- cudaHostAlloc --(B)|

        Thread 1:         |(C)-- cudaHostAlloc --(D)|

        Thread 2:     |(E)------------------------------ aten::clamp_min  ----------------------- (F)|
                             |(G)--- cudaLaunchKernel ------(H)|  |(I)-- cudaLaunchKernel --(J)|


        The cudaHostAlloc and cudaLaunchKernels all require acquiring the cuda
        runtime lock, so only one of them is able to make progress at a time.
        We assume that when overlapping events share a lock, we can track lock
        ownership by checking which event completed first. In the above instance,
        the order of completion events is (D) -> (B) -> (H) -> (J), so we assume that
        1. cudaHostAlloc on thread 1 acquired lock first
        2. cudaHostAlloc on thread 0 acquired lock next
        3. first cudaLaunchKernel on thread 2 acquired lock next
        4. second cudaLaunchKernel on thread 2 acquired lock last.

        The goal of this function is to create edges such that cross-thread events
        are blocked on the lock path. In other words, we want to create:
        1. lock paths, by tracing flow of completion events.
            - (D)->(B)->(H)
        2. a path from every start node to the lock path. In this case:
            - (C, A, G) -> (D)
            - (I) -> (J)

        The final result should contain all these edges.

        Note: This function is responsible for creating edges between events that share a
        cuda runtime lock. The edges between (E)->(G), (H)->(I), and (J)->(F) will be computed
        separately along with other call-stack edges in _construct_graph_from_call_stacks().
        """
        start_nodes = []
        last_node = None
        # Sort node_list by timestamp and then by event index to ensure consistent ordering
        # sorted_nodes = sorted(self.node_list, key=lambda node: (node.ts, node.ev_idx))
        for node in self.node_list:  # requires node_list to be sorted
            if not (
                self._is_cuda_runtime_event_node(node) and self._has_nodes(node.ev_idx)
            ):
                continue
            if node.is_start:
                start_nodes.append(node)
            else:  # end node
                while start_nodes:
                    previous_node = start_nodes.pop()
                    if previous_node.ev_idx == node.ev_idx:
                        e = self._add_edge_helper(
                            previous_node, node, CPEdgeType.OPERATOR_KERNEL
                        )
                        self._attribute_edge(e, node.ev_idx)
                    else:
                        self._add_edge_helper(
                            previous_node, node, CPEdgeType.DEPENDENCY
                        )
                if not last_node.is_start:
                    self._add_edge_helper(last_node, node, CPEdgeType.DEPENDENCY)

            last_node = node

    @timeit
    def _construct_graph_from_call_stacks(self) -> None:

        # For training jobs, backward pass usually happens on a separate thread.
        # But backward and forward pass events are typically non-overlapping.
        # To make sure our critical path correctly flows through backward
        # passes, if we detect a forward and backward pass thread, we'll remap
        # all the backward pass events to the forward pass thread.
        fwd_thread_tid = self._get_fwd_tid(self.trace_df)
        bwd_thread_tid = self._get_bwd_tid(self.trace_df)

        if fwd_thread_tid and bwd_thread_tid and fwd_thread_tid != bwd_thread_tid:
            logging.info("Merging backward pass thread into forward thread")
            remapped_tids = {self.rank: {bwd_thread_tid: fwd_thread_tid}}
        else:
            remapped_tids = None
        cg = CallGraph(self.t, ranks=[self.rank], remapped_tids=remapped_tids)

        cpu_call_stacks = (
            csg for csg in cg.call_stacks if csg.device_type == DeviceType.CPU
        )

        for csg in cpu_call_stacks:
            self._construct_graph_from_call_stack(csg)

    def _get_bwd_tid(self, trace_df: pd.DataFrame) -> int | None:
        """Get the thread id for the backward pass, or None is one cannot be identified.

        We identify the backward pass as the thread which contains "autograd" events. If
        there is not exactly one such thread, we fail to identify the backward pass, and
        return None.

        Args:
            trace_df (pd.DataFrame): Trace dataframe.

        Returns:
            Optional[int]: Thread id for the backward pass.
        """

        return self._get_tid_for_event(trace_df, "autograd")

    def _get_fwd_tid(self, trace_df: pd.DataFrame) -> int | None:
        """Get the thread id for the forward pass, or None is one cannot be identified.

        We identify the forward pass as the thread which contains
        "forward" events. If there is not exactly one such thread,
        we fail to identify the forward pass, and return None.

        Args:
            trace_df (pd.DataFrame): Trace dataframe.

        Returns:
            Optional[int]: Thread id for the forward pass.
        """
        return self._get_tid_for_event(trace_df, "forward")

    def _get_tid_for_event(self, trace_df: pd.DataFrame, ev_name: str) -> int | None:
        events = [sym for sym in self.symbol_table.sym_table if ev_name in sym]
        event_ids = [self.symbol_table.sym_index[candidate] for candidate in events]
        cpu_op_id = self.symbol_table.sym_index.get("cpu_op", -1)
        user_annotation_id = self.symbol_table.sym_index.get("user_annotation", -1)
        tids = set(
            trace_df.query(
                f"name in {event_ids} and (cat == {cpu_op_id} or cat == {user_annotation_id})"
            ).tid
        )
        if len(tids) == 1:
            return next(iter(tids))
        return None

    def _construct_graph_from_call_stack(self, csg: CallStackGraph) -> None:
        """Perform a depth first traversal of the Call Stack for CPU threads
        and generated CP node events.

        To enable nested operators we basically add edges between start/end
        nodes for events. For example say we have op A and op B and C nested
             |----------------------- Op A ----------------------|
                        |--- Op B ---|        |-- Op C--|
        Critical graph
             (OpA.b)--->(ObB.b)----->(OpB.e)->(OpC.b)-->(OpC.e)->(OpA.e)

        Args:
            csg (CallStackGraph): HTA CallStackGraph object for one CPU thread.

        Note: This will ignore edges between events sharing a cuda runtime
        lock, since that is handled in _construct_graph_from_cuda_runtime_events.
        """

        # Track the stack of last seen events
        last_node: Optional[CPNode] = None
        op_depth = 0
        last_ev_parent: Optional[int] = None

        def enter_func(ev_id, csnode):
            nonlocal last_node
            nonlocal op_depth
            nonlocal last_ev_parent
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "=" * csnode.depth
                    + f"Entering node {self._get_node_name(ev_id)}, id = {ev_id}"
                )

            start_node, end_node = self.get_nodes_for_event(ev_id)
            if start_node is None or end_node is None:
                return

            if last_node is not None and not self._shares_cuda_runtime_lock(
                last_node, start_node
            ):
                if op_depth == 0:  # Links consecutive top-level operators
                    edge_type = CPEdgeType.DEPENDENCY
                else:
                    edge_type = CPEdgeType.OPERATOR_KERNEL
                e = self._add_edge_helper(last_node, start_node, edge_type)
                self._attribute_edge(e, last_ev_parent)

            last_node = start_node
            last_ev_parent = csnode.parent
            op_depth += 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "=" * csnode.depth + f"Op depth = {op_depth} last_node={last_node}"
                )

        def exit_func(ev_id, csnode):
            nonlocal last_node
            nonlocal op_depth
            nonlocal last_ev_parent
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "=" * csnode.depth
                    + f"Exiting node {self._get_node_name(ev_id)}, id = {ev_id}"
                )

            start_node, end_node = self.get_nodes_for_event(ev_id)
            if start_node is None or end_node is None:
                return

            if last_node is not None and not self._shares_cuda_runtime_lock(
                last_node, end_node
            ):
                zero_weight = start_node.is_blocking
                if zero_weight and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Zeroing weight for synchronization runtime call "
                        f"id = {ev_id}"
                    )
                e = self._add_edge_helper(
                    last_node,
                    end_node,
                    CPEdgeType.OPERATOR_KERNEL,
                    zero_weight=zero_weight,
                )
                self._attribute_edge(e, last_ev_parent)

            last_node = end_node
            last_ev_parent = csnode.parent
            op_depth -= 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "=" * csnode.depth + f"Op depth = {op_depth} last_node={last_node}"
                )

        csg.dfs_traverse(enter_func, exit_func)

    def _get_cuda_runtime_calls_df(self, retain_index: bool = True) -> pd.DataFrame:
        """Returns a dataframe of CUDA launch runtime calls and associated CUDA stream

        The returned dataframe has additional columns
        * stream_kernel: for the CUDA stream the runtime event launched on
        * gpu_kernel: for the GPU ID the runtime event launched on
        * launch id: is a sequential number for each kernel/memx launch. This is useful
                     in CUDA event synchronization algorithms

        Args:
            retain_index - keep the original trace index

        Returns:
            pd.DataFrame: Dataframe of CUDA launch runtime calls
        """
        gpu_kernels = self.full_trace_df.query("stream != -1 and index_correlation > 0")

        runtime_calls = (
            (
                self.full_trace_df[
                    self.symbol_table.get_runtime_launch_events_mask(self.full_trace_df)
                ]
                .copy()
                .sort_values(by="ts", axis=0)
            )
            .merge(
                gpu_kernels[["stream", "index", "pid"]],
                left_on="index_correlation",
                right_on="index",
                suffixes=("", "_kernel"),
            )
            .set_index("index")
        ).rename(columns={"pid_kernel": "gpu_kernel"})

        # Give a sequential launch ID for every launch event
        runtime_calls.reset_index(inplace=True)
        runtime_calls["launch_id"] = runtime_calls.index
        if retain_index:
            runtime_calls.set_index("index", inplace=True)

        return runtime_calls

    def _get_cuda_event_to_stream_df(self) -> pd.DataFrame:
        """
        Each CUDA Event Record has a specific stream the event was recorded on.
        Synchronization events in the trace have columns shown below that track
        the correlation ID of the CUDA Event Record call, and the stream it was
        recorded on (wait_on_stream).

        Returns a dataframe with 3 columns
            correlation (index), event_stream, gpu

        This function will warn if we see a single event record to multiple
        (event_stream, gpu) combination.
        """
        temp = self.full_trace_df[self.full_trace_df.wait_on_stream > -1][
            ["wait_on_cuda_event_record_corr_id", "wait_on_stream", "pid"]
        ]
        cuda_record_stream_df = temp.drop_duplicates().rename(
            columns={
                "wait_on_cuda_event_record_corr_id": "correlation",
                "wait_on_stream": "event_stream",
                "pid": "gpu",
            }
        )
        # Sanity checking to do.
        return cuda_record_stream_df.set_index("correlation")

    @timeit
    def _get_cuda_event_record_df(self) -> Optional[pd.DataFrame]:
        """For Event based synchronization we need to track the last
        kernel/memcpy launched on a CPU thread just before the cudaEventRecord
        was called i.e. the CUDA event was recorded.

        This function returns a dataframe of cudaEventRecord events,
        and includes an additional column 'index_previous_launch'
        that specifies the closest CUDA kernel launch on the same CPU thread.
        """
        sym_index = self.symbol_table.get_sym_id_map()
        if "cudaEventRecord" not in sym_index:
            return None
        cudaEventRecord_id = sym_index.get("cudaEventRecord")

        # CUDA launch runtime calls
        runtime_calls = self._get_cuda_runtime_calls_df(retain_index=False)[
            [
                "pid",
                "tid",
                "ts",
                "index",
                "name",
                "correlation",
                "launch_id",
                "stream_kernel",
                "gpu_kernel",
            ]
        ].rename(columns={"stream_kernel": "stream", "gpu_kernel": "gpu"})

        # CUDA Event Record Event calls
        cuda_record_calls_ = (
            self.full_trace_df.query(f"name == {cudaEventRecord_id}")
            .copy()
            .sort_values(by="ts", axis=0)
        )

        # Each CUDA Event Record has a specific stream, gpu the event was recorded on.
        cuda_record_stream_df = self._get_cuda_event_to_stream_df()

        # Use the above to join by correlation, drop rows without a valid event_stream
        #  new columns added are : event_stream, gpu
        cuda_record_calls = (
            cuda_record_calls_.merge(
                cuda_record_stream_df,
                on="correlation",
                how="left",
                validate="one_to_one",
            )
            .dropna(subset=["event_stream"])[
                ["pid", "tid", "ts", "name", "correlation", "event_stream", "gpu"]
            ]
            .rename(columns={"event_stream": "stream"})
        )

        def find_previous_launch(gpu, stream):
            """Correlates the closes CUDA kernel launch to a CUDA Record Event"""
            comb = (
                pd.concat([runtime_calls, cuda_record_calls])
                .sort_values(by="ts", axis=0)
                .query(f"gpu == {gpu} and stream == {stream}")
                .copy()
            )
            comb.launch_id.fillna(-1, inplace=True)
            # previous_launch_id is max of launch ids seen uptill now
            comb.loc[:, "previous_launch_id"] = comb.launch_id.cummax(skipna=False)

            comb_launches = comb.loc[comb.name != cudaEventRecord_id].copy()
            comb_cuda_records = comb.loc[comb.name == cudaEventRecord_id].copy()
            comb_cuda_records.drop(axis=1, columns="launch_id", inplace=True)

            # Now join the previous_launch_id to actual kernel launch events.
            return pd.merge(
                comb_cuda_records,
                comb_launches[["launch_id", "index", "correlation"]],
                left_on="previous_launch_id",
                right_on="launch_id",
                how="left",
                suffixes=("", "_launch_event"),
                # multiple CUDA records can have same previous launch ID, but not vice versa
                validate="many_to_one",
            )

        gpu_streams = (
            cuda_record_calls[["gpu", "stream"]]
            .groupby(["gpu", "stream"])
            .groups.keys()
        )

        cuda_record_dfs = [
            find_previous_launch(gpu, stream) for (gpu, stream) in gpu_streams
        ]
        if len(cuda_record_dfs) == 0:
            return None

        cuda_record_calls = pd.concat(cuda_record_dfs, axis=0).sort_values(
            by="ts", axis=0
        )

        # Cleanup temporary columns
        # PS: you can comment the below if you need to debug any issue
        cuda_record_calls.drop(
            axis=1,
            columns=["launch_id", "previous_launch_id"],
            inplace=True,
        )
        cuda_record_calls.rename(
            columns={"index_launch_event": "index_previous_launch"}, inplace=True
        )
        cuda_record_calls.index_previous_launch.fillna(-1, inplace=True)

        return cuda_record_calls

    @timeit
    def _get_cuda_stream_wait_event_df(self) -> Optional[pd.DataFrame]:
        """For Event based synchronization we need to track the next
        kernel/memcpy launched on a CPU thread just after cudaStreamWaitEvent

        This function returns a dataframe of cudaStreamWaitEvent events,
        and includes an additional column 'index_next_launch'
        that specifies the closest CUDA kernel launch on the same CPU thread
        and associated CUDA stream.
        """
        sym_index = self.symbol_table.get_sym_id_map()
        if (
            "cudaStreamWaitEvent" not in sym_index
            or "Stream Wait Event" not in sym_index
        ):
            return None

        # CUDA launch runtime calls
        runtime_calls = self._get_cuda_runtime_calls_df()
        runtime_calls.drop(axis=1, columns=["stream"], inplace=True)
        runtime_calls.rename(columns={"stream_kernel": "stream"}, inplace=True)

        gpu_kernels = self.full_trace_df.query("stream != -1 and index_correlation > 0")

        # CUDA stream wait event runtime calls and associated CUDA stream
        cudaStreamWaitEvent_id = sym_index.get("cudaStreamWaitEvent")
        cuda_stream_wait_events = (
            (
                self.full_trace_df.query(
                    f"name == {cudaStreamWaitEvent_id} and index_correlation > 0"
                )[
                    [
                        "index",
                        "name",
                        "ts",
                        "pid",
                        "tid",
                        "correlation",
                        "index_correlation",
                    ]
                ]
                .copy()
                .sort_values(by="ts", axis=0)
            )
            .merge(
                gpu_kernels[["stream", "index"]],
                left_on="index_correlation",
                right_on="index",
                suffixes=("", "_kernel"),
            )
            .set_index("index")
        )

        def find_next_launch(pid, tid, stream):
            """Correlates the closes CUDA kernel launch to a CUDA Stream Wait Event"""
            # Combine CUDA runtime launch calls and cuda stream wait event calls
            # on the same pid, tid, stream
            comb = (
                pd.concat([runtime_calls, cuda_stream_wait_events])
                .sort_values(by="ts", axis=0)
                .query(f"pid == {pid} and tid == {tid} and stream == {stream}")
                .copy()
            )
            comb.launch_id.fillna(sys.maxsize, inplace=True)

            # Next launch ID is the next lowest launch ID in the sorted dataframe
            comb["next_launch_id"] = comb["launch_id"].iloc[::-1].cummin()
            comb.reset_index(inplace=True)

            comb_launches = comb.loc[comb.name != cudaStreamWaitEvent_id].copy()
            comb_stream_wait_events = comb.loc[
                comb.name == cudaStreamWaitEvent_id
            ].copy()
            comb_stream_wait_events.drop(axis=1, columns="launch_id", inplace=True)

            # Now join the next_launch_id to actual kernel launch events.
            return pd.merge(
                comb_stream_wait_events,
                comb_launches[["launch_id", "index", "correlation"]],
                left_on="next_launch_id",
                right_on="launch_id",
                how="left",
                suffixes=("", "_launch_event"),
            )

        pid_tid_streams = (
            cuda_stream_wait_events[["pid", "tid", "stream"]]
            .drop_duplicates()
            .to_dict("records")
        )

        cuda_stream_wait_events = pd.concat(
            [
                find_next_launch(r["pid"], r["tid"], r["stream"])
                for r in pid_tid_streams
            ],
            axis=0,
        ).sort_values(by="ts", axis=0)

        # Cleanup temporary columns
        cuda_stream_wait_events.drop(
            axis=1,
            columns=["launch_id", "next_launch_id", "correlation_launch_event"],
            inplace=True,
        )
        cuda_stream_wait_events.rename(
            columns={"index_launch_event": "index_next_launch"}, inplace=True
        )
        cuda_stream_wait_events.index_next_launch.fillna(-1, inplace=True)

        return cuda_stream_wait_events.set_index("index")

    def _check_event_sync_helper(self, row) -> bool:
        eid = row.index
        if not hasattr(row, "wait_on_cuda_event_record_corr_id"):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "CUDA Stream Wait event does not have correlation id of "
                    f"cudaEventRecord, name = {self._get_node_name(eid)}, "
                    f"correlation = {row.correlation}"
                )
            return False

        if (not hasattr(row, "index_previous_launch")) or (
            row.index_previous_launch == -1
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "CUDA Stream Wait event was not matched to a cudaRecordEvent"
                    f", name = {self._get_node_name(eid)}, correlation = "
                    f"{row.correlation}"
                )
            return False
        return True

    def _add_gpu_cpu_sync_edge(self, gpu_node: CPNode, runtime_eid: int) -> None:
        """Add an edge between gpu_node and the runtime event on CPU"""
        _, end_node = self.get_nodes_for_event(runtime_eid)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Adding a GPU->CPU sync edge between nodes {gpu_node} -> {end_node}"
            )
        self._add_edge_helper(gpu_node, end_node, CPEdgeType.SYNC_DEPENDENCY)

    def _add_kernel_launch_delay_edge(
        self, runtime_index: int, kernel_start_node: CPNode, zero_weight: bool = False
    ) -> bool:
        """Add a kernel launch delay edge from CUDA runtime launch event to a GPU kernel.

        Arguments:
            runtime_index(int): event index of the runtime
            kernel_start_node(CPNode): start node of the GPU kernel

        Returns:
            (bool) false if runtime event could not be looked up
        """
        runtime_start, _ = self.get_nodes_for_event(runtime_index)
        if runtime_start is None:
            return False
        self._add_edge_helper(
            runtime_start,
            kernel_start_node,
            CPEdgeType.KERNEL_LAUNCH_DELAY,
            zero_weight=zero_weight,
        )
        return True

    @timeit
    def _construct_graph_from_kernels(self) -> None:
        """Create nodes and edges for GPU kernels"""
        sym_id_map = self.symbol_table.get_sym_id_map()
        sync_cat = sym_id_map.get("cuda_sync", -1)
        context_sync = sym_id_map.get("Context Sync", -1)
        stream_sync = sym_id_map.get("Stream Sync", -1)
        event_sync = sym_id_map.get("Event Sync", -1)
        stream_wait_event = sym_id_map.get("Stream Wait Event", -1)

        # Note getting queue length on the clipped dataframe was showing errors,
        # it is worthwhile to consider the entire trace instead, hence use t_full
        q = TraceCounters._get_queue_length_time_series_for_rank(self.t_full, self.rank)

        gpu_kernels = (
            self.trace_df.query(
                f"(stream != -1 or (name == {event_sync} or name == {context_sync})) and index_correlation >= 0"
            )
            .join(q[["queue_length"]], on="index_correlation")
            .rename(columns={"queue_length": "queue_length_runtime"})
            .join(q[["queue_length"]], on="index")
        ).drop(columns=["s_cat", "s_name"], errors="ignore")

        # For "Wait on CUDA Event" syncs we look up all cudaRecord calls
        # and find the previous GPU kernel/memcpy launch, this is recorded
        # in the index_previous_launch column
        cuda_record_calls = self._get_cuda_event_record_df()
        cuda_stream_wait_events = self._get_cuda_stream_wait_event_df()

        if (
            "wait_on_cuda_event_record_corr_id" in self.trace_df
            and cuda_record_calls is not None
        ):
            # Join the event sync event with the index of the kernel/memcpy launch
            # that is was syncing on.
            gpu_kernels = (
                pd.merge(
                    gpu_kernels,
                    cuda_record_calls[["correlation", "index_previous_launch"]],
                    left_on="wait_on_cuda_event_record_corr_id",
                    right_on="correlation",
                    how="left",
                    suffixes=("", "_cuda_record"),
                )
                .fillna(-1)
                .astype(int)
            )
            # Note convert NAN to -1 and then turn all records to int

        # Sort kernels by start timestamp but use end time_stamp for sync events.
        # This handles the case where a stream sync event overlaps with an
        # actual kernel or memcpy -
        #    stream 7 ...   |---Stream Sync --------|
        #                       |--Memcpy HtoD--|
        gpu_kernels["end_ts"] = gpu_kernels["ts"] + gpu_kernels["dur"]
        gpu_kernels["sort_by"] = gpu_kernels["ts"]
        gpu_kernels.loc[gpu_kernels.cat == sync_cat, "sort_by"] = gpu_kernels[
            gpu_kernels.cat == sync_cat
        ]["end_ts"]
        gpu_kernels.sort_values(by="sort_by", axis=0, inplace=True)

        # Last node on a stream
        last_node: Dict[int, CPNode] = {}

        # Scheduled a GPU-GPU sync on a stream
        #  map from dest kernel event ID -> event ID of kernel to sync on
        kernel_sync: Dict[int, int] = {}

        def handle_cuda_sync(row):
            nonlocal last_node
            nonlocal kernel_sync
            eid = row.index
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"CUDA Sync event name = {self._get_node_name(eid)} corrid = {row.correlation}"
                )

            name = row.name

            # Handle event synchronizations
            if name == stream_wait_event or name == event_sync:
                if not self._check_event_sync_helper(row):
                    return

                # Note: use the full trace to find the src kernel
                src_kernel_index = self.full_trace_df.index_correlation[
                    row.index_previous_launch
                ]

                if name == stream_wait_event:
                    # Stream Wait event is indicating a dependency between
                    # the next GPU kernel on this stream and another GPU kernel

                    # Get the corresponding GPU event for this cudaStreamWaitEvent()
                    dest_kernel_launch_index = cuda_stream_wait_events.loc[
                        row.index_correlation
                    ]["index_next_launch"]

                    if dest_kernel_launch_index < 0:
                        # TODO make a warning?
                        return

                    # Use the next launch to find dest_kernel
                    dest_kernel_index = self.full_trace_df.index_correlation[
                        dest_kernel_launch_index
                    ]

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Scheduling a Stream Sync on stream {row.stream} "
                            f" dest kernel index {dest_kernel_index}, corr id = "
                            f"{self.full_trace_df.correlation.loc[dest_kernel_index]}\n "
                            f"on src kernel with index = {src_kernel_index}, corr id = "
                            f"{self.full_trace_df.correlation.loc[src_kernel_index]}"
                        )
                    kernel_sync[dest_kernel_index] = src_kernel_index
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Adding cudaEventSynchronize GPU->CPU edge between GPU kernel"
                            f" with index = {src_kernel_index}, corr id = "
                            f"{self.full_trace_df.correlation.loc[src_kernel_index]}"
                            f" to CPU event index {row.index_correlation}, corr id ="
                            f"{self.full_trace_df.correlation.loc[row.index_correlation]}"
                        )
                    _, gpu_end_node = self.get_nodes_for_event(src_kernel_index)
                    if (
                        gpu_end_node is not None
                    ):  # boundary case if previous was out of window
                        self._add_gpu_cpu_sync_edge(gpu_end_node, row.index_correlation)
                return

            assert name == stream_sync or name == context_sync

            # For Context Sync add a sync edge on the last event on all streams,
            # while for Stream Sync only add a sync edge on the specific stream.
            gpu_nodes_to_sync = (
                last_node.values()
                if name == context_sync
                else [last_node.get(row.stream)]
            )
            for gpu_node in gpu_nodes_to_sync:
                if gpu_node is not None:
                    self._add_gpu_cpu_sync_edge(gpu_node, row.index_correlation)

        # ---------------------------------------
        # Loop through all the kernels/sync events
        # ---------------------------------------

        for row in gpu_kernels.itertuples(index=False):
            # row = row_.astype(int, errors="ignore")

            if row.cat == sync_cat:
                handle_cuda_sync(row)
                continue

            eid, stream = row.index, row.stream
            queue_length, queue_length_runtime = (
                row.queue_length,
                row.queue_length_runtime,
            )
            runtime_index = row.index_correlation

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Adding GPU kernel node for eid = {eid}, stream = {stream}, "
                    f"name = {self._get_node_name(eid)}, correlation = {row.correlation}"
                )

            start_node, end_node = self.get_nodes_for_event(eid)
            e = self._add_edge_helper(start_node, end_node, CPEdgeType.OPERATOR_KERNEL)
            self._attribute_edge(e, -1)

            kernel_sync_index = kernel_sync.get(eid)
            kernel_sync_end: Optional[CPNode] = None

            edge_added = False
            launch_delay_added = False

            # Check if we need to sync between two kernels
            if kernel_sync_index is not None:
                _, kernel_sync_end = self.get_nodes_for_event(kernel_sync_index)
                if kernel_sync_end is None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Could not find source kernel sync index = {kernel_sync_index}, current kernel "
                            f" eid = {eid}, stream = {stream}, "
                            f"name = {self._get_node_name(eid)}, correlation = {row.correlation}"
                        )
                    kernel_sync_index = None
                else:
                    # note that the sync dependency has 0 weight
                    self._add_edge_helper(
                        kernel_sync_end, start_node, CPEdgeType.SYNC_DEPENDENCY
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Adding a GPU->GPU sync edge between nodes "
                            f"{kernel_sync_end} -> {start_node}"
                        )
                    edge_added = True
                # reset the kernel-kernel sync
                kernel_sync[eid] = None

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"queue_length_runtime = {queue_length_runtime}"
                    f" queue_length = {queue_length}"
                    f" last_node = {last_node}"
                    f" last_node.ts = {last_node[stream].ts if last_node.get(stream) is not None else -1}"
                    f" runtime.ts = {self.full_trace_df.ts.loc[runtime_index]}"
                )

            # Kernel Launch vs Kernel-Kernel delays
            if (
                # There were no outstanding kernels on this stream during the launch
                queue_length_runtime == 1
                and queue_length == 0
                # and the kernel was launched after previous kernel finished
                and (
                    last_node.get(stream) is None
                    or last_node[stream].ts < self.full_trace_df.ts.loc[runtime_index]
                )
                # and the kernel sync dependency if any finished earlier
                and (
                    kernel_sync_index is None
                    or kernel_sync_end.ts < self.full_trace_df.ts.loc[runtime_index]
                )
            ):
                success = self._add_kernel_launch_delay_edge(runtime_index, start_node)
                assert success, (
                    f"Could not find runtime index = {runtime_index}, current kernel "
                    f"stream = {stream}, "
                    f"name = {self._get_node_name(eid)}, correlation = {row.correlation}"
                )
                edge_added = True
                launch_delay_added = True
            elif (
                last_node.get(stream) is not None
                # and the kernel sync dependency if any finished earlier than last node
                and (
                    kernel_sync_index is None
                    or kernel_sync_end.ts < last_node[stream].ts
                )
            ):
                # If neither launch nor CUDA event sync occurs it is a kernel-kernel
                # delay
                e = self._add_edge_helper(
                    last_node[stream], start_node, CPEdgeType.KERNEL_KERNEL_DELAY
                )
                self._attribute_edge(e, -1)
                edge_added = True

            if not edge_added:
                # Neither of Sync, kernel-kernel or kernel launch edges were added
                logger.warning(
                    f"No edge was added - queue length is {queue_length_runtime}!= 1 but no "
                    f"last kernel on stream {stream}, current kernel: "
                    f"name = {self._get_node_name(eid)}, correlation = {row.correlation}"
                )

            # When we modify the CPGraph for  performance simulations, we do not want
            # GPU kernels to start  before their CPU launch counterparts.
            # To prevest this we always add a 0 weight edge for runtime launch -> kernel
            # the launch delay is not in critical path.
            if self._add_zero_weight_launch_edges() and not launch_delay_added:
                # Try adding this if runtime is found
                self._add_kernel_launch_delay_edge(
                    runtime_index, start_node, zero_weight=True
                )

            last_node[stream] = end_node

    def _show_digraph(self) -> None:
        """Prints the networkx digraph"""
        n = 0
        for n in self.nodes:
            node = self.node_list[n]
            logger.info(f"node id = {n}, node = {node}")
            logger.info("  neighbors = ", ",".join((str(n) for n in self.neighbors(n))))

    def critical_path(self) -> bool:
        """Calculates the critical path across nodes.

        Returns:
            (bool) True if critical path was calculated successfully

        Raises:
            ValueError: If the graph is not valid for critical path calculation
        """
        t0 = time.perf_counter()
        if not self._validate_graph():
            raise ValueError(
                "Graph is not valid, see prints above for help on debugging"
            )
        try:
            self.critical_path_nodes = nx.dag_longest_path(self, weight="weight")
        except nx.NetworkXUnfeasible as err:
            logger.error(f"Critical path algorithm failed due to {err}")
            return False
        assert len(self.critical_path_nodes) >= 2

        self.critical_path_events_set = {
            self.node_list[nid].ev_idx for nid in self.critical_path_nodes
        }

        # Reset critical_path_edges_set across invocations
        self.critical_path_edges_set = set()

        # Add edges connecting the nodes along the critical path
        niter = iter(self.critical_path_nodes)
        u = next(niter)

        while 1:
            try:
                v = next(niter)
                e = self.edges[u, v]["object"]
                self.critical_path_edges_set.add(e)
                u = v
            except StopIteration:
                break

        t1 = time.perf_counter()
        logger.info(f"calculating critical path took {t1 - t0:2f} seconds")
        assert len(self.critical_path_edges_set) == (len(self.critical_path_nodes) - 1)
        return True

    def _validate_graph(self) -> bool:
        """Validate whether the graph can be trusted for analysis"""
        # check for negative values and invalid sync edges
        negative_weights: bool = False
        sync_on_same_stream: bool = False

        # print helper
        def show_src_dest(src: CPNode, dest: CPNode) -> None:
            logger.error(
                f" Source node idx {src.ev_idx}, "
                f" node name = {self._get_node_name(src.ev_idx)}"
            )
            logger.error(
                f" Dest node idx {dest.ev_idx}, "
                f" node name = {self._get_node_name(dest.ev_idx)}"
            )

        for u, v in self.edges:
            e = self.edges[u, v]["object"]

            if (
                e.weight <= -1
                and not hta_options.critical_path_strict_negative_weight_check()
            ):
                # Nanosecond precision is causing some of the parent events
                # to end before child in stack. This is a separate issue that
                # needs fixing in the trace itself.
                # Please see https://github.com/pytorch/pytorch/pull/122425"
                logger.warning(f"Ignoring negative weight of -1 for {e}")
                self.edges[u, v]["weight"] = 0
            elif e.weight < -1:
                src, dest = self.node_list[u], self.node_list[v]
                logger.error(f"Found an edge with negative weight {e}")
                show_src_dest(src, dest)
                negative_weights = True

            if e.type == CPEdgeType.SYNC_DEPENDENCY:
                src, dest = self.node_list[u], self.node_list[v]
                stream_src = self.trace_df.stream.loc[src.ev_idx]
                stream_dest = self.trace_df.stream.loc[dest.ev_idx]

                if stream_src != -1 and stream_src == stream_dest:
                    logger.error(
                        f"Seeing a CUDA sync between kernels on same stream {e}"
                    )
                    show_src_dest(src, dest)
                    sync_on_same_stream = True

        if negative_weights:
            logger.error(
                "Negative weights means before-after relationships are not valid"
            )
            return False

        if sync_on_same_stream:
            logger.error(
                "Synchronization edges should not be between kernels on same stream"
            )
            return False

        # check for cycles
        has_cycles = not nx.is_directed_acyclic_graph(self)
        if has_cycles:
            logger.error("This graph has cycles, you can debug this by running -")
            logger.error(" import networkx as nx")
            logger.error(" C = sorted(nx.simple_cycles(cp_graph))")
            logger.error("and try len(C), C[0] ")
            return False

        return True

    def get_critical_path_breakdown(self) -> Optional[pd.DataFrame]:
        """Returns a breakdown of the critical path with each edge in the
        path attributed to an event in the trace.

        Returns: pd.Dataframe
            The output dataframe has one row per edge in the critical path.
            It includes the columns:
                event_id - Index of the event in original trace dataframe
                s_name - Shortened string name of the event.
                duration - Duration or weight of the edge in critical path
                type - String representing of type of edge (see CPEdgeType)
                cat, pid, tid, stream - Columns corresponding to similar values
                    in the original t/race dataframe
                bound_by - This column classifies the edges in the critical path
                    as bounded by "cpu_bound", "gpu_compute_bound",
                    "gpu_kernel_kernel_overhead" (gaps between kernels)
                    "gpu_kernel_launch_overhead" (delay between CPU to GPU launch)
        """
        if len(self.critical_path_nodes) == 0:
            return None

        trace_df = self.trace_df
        decode_symbol_id_to_symbol_name(
            trace_df, self.symbol_table, use_shorten_name=True
        )

        # Construct simple dataframe from edges on critical path
        def make_edge_record(e):
            return {
                "event_idx": self.get_event_attribution_for_edge(e),
                "duration": e.weight,
                "type": str(e.type.value),
            }

        edge_df = pd.DataFrame.from_records(
            make_edge_record(e) for e in self.critical_path_edges_set
        )

        edge_events_df = pd.merge(
            edge_df,
            trace_df[["s_name", "cat", "pid", "tid", "stream", "index"]],
            left_on="event_idx",
            right_on="index",
            how="left",
        )

        # Add column to classify boundedness
        edge_events_df["bound_by"] = edge_events_df.apply(bound_by, axis=1)
        return edge_events_df

    def summary(self) -> pd.core.series.Series:
        """Displays a summary or breakdown of the critical path into one of the following
        - cpu_bound
        - gpu_compute_bound
        - gpu_communication_bound (NCCL Collectives)
        - gpu_kernel_kernel_overhead (Gaps between GPU kernels)
        - gpu_kernel_launch_overhead (Delay launching kernels from CPU to GPU)

        Returns:
            A summary pandas series with bottleneck type -> % of duration on critical path.
        Also see get_critical_path_breakdown().
        """
        edf = self.get_critical_path_breakdown()
        summary = edf.groupby("bound_by").duration.sum() / edf.duration.sum() * 100
        print("Critical Path broken down by boundedness = (in % of duration)")
        return summary

    def show_critical_path(self) -> None:
        """List out the nodes in the critical path graph"""
        for n in self.critical_path_nodes:
            node = self.node_list[n]
            logger.info(
                f"Critical {node}, ev_id = {node.ev_idx} "
                f"ev_name = {self._get_node_name(node.ev_idx)}"
            )

    def save(self, out_dir: str) -> str:
        """Saves the critical path graph object to a zip file.
        Note that this save() operation can only be done after
        the graph has been constructed!

        Args:
             out_dir(str) is the directory used to first dump a
                collection of files used to save the state of CPGraph object.
                The directory is then compressed into a zipfile with the same name.
        Returns:
            trace_zip_file (str): path to the zip file containing the object.

        Note:
            Internally it saves the data into 3 files that are zipped up.
            1. Is the trace_data saved as a csv.
            2. The networkx graph is serialized using python pickle.
            2. Other object data is serialized using python pickle.
        """
        import pickle
        from zipfile import ZipFile

        # XXX check if graph has been constructed

        # Data members that can be saved as pickle
        pickle_obj = _CPGraphData(
            node_list=self.node_list,
            critical_path_nodes=self.critical_path_nodes,
            critical_path_events_set=self.critical_path_events_set,
            critical_path_edges_set=self.critical_path_edges_set,
            event_to_start_node_map=self.event_to_start_node_map,
            event_to_end_node_map=self.event_to_end_node_map,
            edge_to_event_map=self.edge_to_event_map,
        )

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        trace_csv_path = os.path.join(out_dir, "trace_data.csv")
        self.trace_df.to_csv(trace_csv_path, index=True, index_label="_index_")

        graph_pkl_path = os.path.join(out_dir, "cp_graph.pkl")
        # first convert the data in node link format
        d = nx.node_link_data(self)
        # we cannot use json as CPEdge needs to be serialized and de-serialized
        with open(graph_pkl_path, "wb") as f:
            pickle.dump(d, f)

        data_pkl_path = os.path.join(out_dir, "cp_data.pkl")
        with open(data_pkl_path, "wb") as f:
            pickle.dump(pickle_obj, f)

        zip_filename = f"{out_dir}.zip"
        with ZipFile(zip_filename, "w") as zipf:
            zipf.write(trace_csv_path)
            zipf.write(graph_pkl_path)
            zipf.write(data_pkl_path)

        logger.warning(f"Saved files to {zip_filename}")

        return zip_filename


def restore_cpgraph(zip_filename: str, t_full: "Trace", rank: int) -> CPGraph:
    """Restores the critical path graph object from a zip file.
    The graph will already be constructed in this case. You can
    however run critical_path() again and modify the graph etc.

    Returns:
        trace_zip_file (str): path to the zip file containing the object.
    Returns:
        CPGraph object restored from the file.
    """
    import pickle
    from zipfile import ZipFile

    with ZipFile(zip_filename, "r") as zipf:
        # namelist ex tmp/my_saved_cp_graph/trace_data.csv/trace_data.csv
        out_dir = "/".join(zipf.namelist()[0].split("/")[:-1])
        zipf.extractall(path="/tmp/")  # data will be extracted to /tmp/
    out_dir = os.path.join("/tmp", out_dir)
    logger.warning(f"Extracted zip archive to {out_dir}")

    graph_pkl_path = os.path.join(out_dir, "cp_graph.pkl")
    logger.warning(f"Restoring graph from {graph_pkl_path}")
    with open(graph_pkl_path, "rb") as f:
        pickled_graph = pickle.load(f)
    G = nx.node_link_graph(pickled_graph)

    # Use restored Graph to initialize CPGraph
    restored_instance = CPGraph(None, t_full, rank, G)

    logger.warning(f"Loading cp_graph from {out_dir}")
    trace_csv_path = os.path.join(out_dir, "trace_data.csv")

    # there is a default _index_ column we can use
    restored_instance.trace_df = pd.read_csv(trace_csv_path).set_index("_index_")

    data_pkl_path = os.path.join(out_dir, "cp_data.pkl")
    with open(data_pkl_path, "rb") as f:
        pickled_obj = pickle.load(f)

    restored_instance.node_list = pickled_obj.node_list
    restored_instance.critical_path_nodes = pickled_obj.critical_path_nodes
    restored_instance.critical_path_events_set = pickled_obj.critical_path_events_set
    restored_instance.critical_path_edges_set = pickled_obj.critical_path_edges_set
    restored_instance.event_to_start_node_map = pickled_obj.event_to_start_node_map
    restored_instance.event_to_end_node_map = pickled_obj.event_to_end_node_map
    restored_instance.edge_to_event_map = pickled_obj.edge_to_event_map

    return restored_instance


# TODO optimize using itertuple
def bound_by(row: Dict[str, Any]) -> str:
    """Function to classify the bounding resource for an edge on the critical path"""
    if row["type"] == "critical_path_kernel_kernel_delay":
        return "gpu_kernel_kernel_overhead"
    if row["type"] == "critical_path_kernel_launch_delay":
        return "gpu_kernel_launch_overhead"
    if row["type"] in ["critical_path_dependency", "critical_path_sync_dependency"]:
        return ""

    assert not pd.isna(row["s_name"]), f"name of edge is na : row = {row}"

    if row["stream"] < 0:
        return "cpu_bound"
    if is_comm_kernel(row["s_name"]):
        return "gpu_communication_bound"
    return "gpu_compute_bound"


class CriticalPathAnalysis:
    def __init__(self):
        # dict of critical path nodes, node id -> CPNode
        self.cp_nodes = {}

    @classmethod
    def critical_path_analysis(
        cls,
        t: "Trace",
        rank: int,
        annotation: str,
        instance_id: Union[Optional[int], Tuple[int, int]],
    ) -> Tuple[CPGraph, bool]:
        r"""
        Perform critical path analysis for trace events within a rank.
        We further reduce the region of interest by selecting
        a trace annotation and instance id. This will
        limit the analysis to events within the time range of that annoation.
        For example, you can use this to limit the analysis to one iteration
        by passing annotation='ProfilerStep'.

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

        Returns: Tuple[CPGraph, bool] a pair of CPGraph object and a success or
            fail boolean value. True indicates that the critical path analysis
            algorithm succeeded.

            CPGraph object that can be used to obtain statistics and further
            visualize the critical path.

            CPGraph is also a subinstance of a networkx.DiGraph.
            Run 'CPGraph?' for more info and APIs.
        """
        global PROFILE_TIMES
        t0 = time.perf_counter()
        trace_df: pd.DataFrame = t.get_trace(rank)
        sym_index = t.symbol_table.get_sym_id_map()

        if "cuda_sync" not in sym_index:
            logger.warning(
                "Trace does not contain CUDA Synchronization events "
                "so the results of analysis could be inaccurate."
            )
            logger.warning(
                "Please see this PR to learn how to enable CUDA sync "
                "events https://github.com/pytorch/pytorch/pull/105187"
            )

        annotation_id = sym_index.get(annotation, None)
        annotation_ids = [val for key, val in sym_index.items() if annotation in key]

        if len(annotation_ids) == 0:
            logger.error(f"Could not find annotation {annotation} in the trace.")
            return None

        if instance_id is None:
            instance_start, instance_end = 0, 0
        elif isinstance(instance_id, tuple):
            instance_start, instance_end = instance_id
        elif isinstance(instance_id, int):
            instance_start, instance_end = instance_id, instance_id
        else:
            logger.error("Unexpected input type instance_id")
            return

        logger.info(
            f"Looking up events under [{instance_start}, {instance_end}) "
            f"instance(s) of '{annotation}' annotation."
        )

        if annotation == "":
            # look up full trace
            start_ts = trace_df.ts.min()
            end_ts = trace_df.end.max()
        else:
            annotations = trace_df[trace_df.name.isin(annotation_ids)].copy()
            annotations["end_ts"] = annotations["ts"] + annotations["dur"]

            start_ts = annotations.ts[instance_start : instance_end + 1].min()
            end_ts = annotations.end_ts[instance_start : instance_end + 1].max()

        logger.info(f"Looking up events within the window ({start_ts}, {end_ts})")

        # Consider all events that start within the annotatated window.
        # and also fiter out 0 duration events as they mess up the
        # event stack generation
        cpu_kernels = trace_df[trace_df["stream"].eq(-1)]
        stream_wait_event_id = sym_index.get("Stream Wait Event", -200)
        a = cpu_kernels.query(f"(ts >= {start_ts} and ts <= {end_ts}) and (dur > 0)")

        # Only consider GPU kernels whose runtime events are in the correct
        # time window
        gpu_kernels = trace_df[trace_df["stream"].ne(-1)]
        cpu_kernels = cpu_kernels.copy().set_index("index_correlation")
        b = (
            gpu_kernels[["ts", "dur", "correlation", "name"]]
            .join(cpu_kernels[["ts", "dur"]], rsuffix="_runtime")
            .query(
                f"(ts_runtime >= {start_ts} and ts_runtime <= {end_ts} and dur_runtime > 0)"
                f" or (name == {stream_wait_event_id})"
            )
        )

        clipped_df = trace_df.loc[a.index.union(b.index)].copy()

        logger.info(f"Clipped dataframe has {len(clipped_df)} events")

        # XXX This is a bit hacky but CallGraph does not really support a way
        # to specify a dataframe, just supports passing Trace object
        t_copy = deepcopy(t)
        t_copy.traces[rank] = clipped_df
        t1 = time.perf_counter()
        logger.info(f"Preprocessing took {t1 - t0:.2f} seconds")

        cp_graph = CPGraph(t_copy, t, rank)
        t2 = time.perf_counter()
        logger.info(f"CPGraph construction took {t2 - t1:.2f} seconds")

        for func, total_time in PROFILE_TIMES.items():
            logger.info(f"  Function {func} Took {total_time:.4f} seconds")

        return cp_graph, cp_graph.critical_path()

    @staticmethod
    def _is_zero_weight_launch_edge(e: CPEdge) -> bool:
        return e.type == CPEdgeType.KERNEL_LAUNCH_DELAY and e.weight == 0

    @classmethod
    def overlay_critical_path_analysis(
        cls,
        t: "Trace",
        rank: int,
        critical_path_graph: CPGraph,
        output_dir: str,
        only_show_critical_events: bool,
        show_all_edges: bool,
        edge_types_to_viz: Optional[Set[CPEdgeType]] = None,
    ) -> str:
        r"""
        Overlay the identified critical path on top of the trace file
        for visualization.

        Args:
            t (Trace): Input trace data structure.
            rank (int): rank to generate the time series for.
            critical_path_graph: Critical Path Graph object generated previously.
            output_dir (str): Output directory to store overlaid trace.
            only_show_critical_events (bool): When set the output trace will only
                have operators and GPU kernels on the critical path. It will
                still retain the user annotations.
                Default value = True.
            show_all_edges (bool): When set this will add edge events for
                all types of edges in the critical path graph. This is useful
                for debugging the algorithm. The value will be forced to False
                if only_show_critical_events is True.
                Default value = False.
            edge_types_to_viz (Set): types of edges to add to the overlaid trace.
                By default we only include Kernel launch edges and CPU/GPU sync
                dependency edges.

        Returns: the overlaid_trace_file path.

        Note: In case of kernel launches that are not on the critical path the graph
        still has a 0 weight edge between CUDA runtime and kernel. These 0 weight
        edges are not shown in the overlaid trace by default. Set the environment
        variable CRITICAL_PATH_SHOW_ZERO_WEIGHT_LAUNCH_EDGE=1 to enable adding this
        to the overlaid trace. Add this to your notebook
        `os.environ["CRITICAL_PATH_SHOW_ZERO_WEIGHT_LAUNCH_EDGE"] = 1`
        """
        if only_show_critical_events:
            show_all_edges = False
        if edge_types_to_viz is None:
            edge_types_to_viz = DEFAULT_EDGE_TYPES_IN_VIZ

        path = Path(output_dir).expanduser()
        if not path.exists():
            os.makedirs(path)
        elif not path.is_dir():
            logger.error(f"{output_dir} is not a directory.")
            return ""

        output_file = os.path.join(
            str(path), "overlaid_critical_path_" + t.trace_files[rank].split("/")[-1]
        )
        overlaid_trace = t.get_raw_trace_for_one_rank(rank=rank)
        raw_events = overlaid_trace["traceEvents"]

        # Traverse events and mark them as critical
        for ev_idx, event in enumerate(raw_events):
            if ev_idx in critical_path_graph.critical_path_events_set:
                event["args"]["critical"] = 1

        flow_events = []
        flow_id = 0

        def get_flow_event(
            nid: int,
            event: Dict[str, Any],
            edge: CPEdge,
            flow_id: int,
            is_start: bool,
        ):
            # This helps with showing the arrows in chrome trace
            end_ts = event["ts"] + event["dur"]
            if event["args"].get("device", -1) >= 0:
                end_ts -= min(1, event["dur"])

            is_critical = e in critical_path_graph.critical_path_edges_set

            return Trace.flow_event(
                id=flow_id,
                pid=event["pid"],
                tid=event["tid"],
                ts=int(
                    event["ts"]
                    if critical_path_graph.node_list[nid].is_start
                    else end_ts
                ),
                is_start=is_start,
                name="critical_path",
                cat="critical_path",
                args={
                    "weight": int(edge.weight),
                    "critical": is_critical,
                    "type": str(edge.type.value),
                },
            )

        if show_all_edges:
            # Networkx provides iteration over the edges represented as (u, v)
            edges = (
                critical_path_graph.edges[u, v]["object"]
                for u, v in critical_path_graph.edges
            )
            if not hta_options.critical_path_show_zero_weight_launch_edges():
                edges = (
                    e
                    for e in edges
                    if not CriticalPathAnalysis._is_zero_weight_launch_edge(e)
                )
        else:
            edges = (
                e
                for e in critical_path_graph.critical_path_edges_set
                if e.type in edge_types_to_viz
            )

        for e in edges:
            u, v = e.begin, e.end
            start_ev_id, end_ev_id = critical_path_graph.get_events_for_edge(e)
            start_ev, end_ev = raw_events[start_ev_id], raw_events[end_ev_id]

            # XXX need to assert if raw event name and dataframe name are same
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Adding cp edge between {start_ev_id} and {end_ev_id}")

            flow_events.append(get_flow_event(u, start_ev, e, flow_id, is_start=True))
            flow_events.append(get_flow_event(v, end_ev, e, flow_id, is_start=False))
            flow_id += 1

        if only_show_critical_events:
            raw_events2 = [
                event
                for event in raw_events
                if (
                    event["ph"] != "X"
                    or event.get("cat", "") in ["user_annotation", "python_function"]
                    or ("args" in event and event["args"].get("critical", 0) == 1)
                )
            ]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Length of new raw events = {raw_events2}")
            overlaid_trace["traceEvents"] = raw_events2

        overlaid_trace["traceEvents"].extend(flow_events)

        t.write_raw_trace(output_file, overlaid_trace)

        return output_file
