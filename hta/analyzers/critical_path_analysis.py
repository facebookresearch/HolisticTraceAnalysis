# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import pandas as pd

from hta.analyzers.trace_counters import TraceCounters
from hta.common.call_stack import CallGraph, CallStackGraph, DeviceType

from hta.common.trace import Trace
from hta.configs.config import logger


@dataclass
class CPNode:
    """A node in the critical path di-graph.
    This represents a point in time. It could be start
    or end of an operator or a kernel.
    """

    idx: int = -1
    ev_idx: int = -1
    ts: int = 0
    is_start: bool = False

    def __repr__(self) -> str:
        return (
            f"CPNode(event: {self.ev_idx}, node_id={self.idx}, "
            f"ts={self.ts}, is_start={self.is_start})"
        )


class CPEdgeType(Enum):
    OPERATOR_KERNEL = "critical_path_operator"
    DEPENDENCY = "critical_path_dependency"
    KERNEL_LAUNCH_DELAY = "critical_path_kernel_launch_delay"
    KERNEL_KERNEL_DELAY = "critical_path_kernel_kernel_delay"
    SYNC_DEPENDENCY = "critical_path_sync_dependency"


@dataclass(frozen=True)
class CPEdge:
    """An edge in the critical path di-graph.
    This represents either a
    1) span of time i.e. duration of an operator/kernel.
    2) a dependency among operators/kernels.
    3) a kernel launch or kernel-kernel delay.
    3) synchronization delay.

    The weight represents time in the graph, cases 1) and 3)
    above have non-zero weights.
    Once we initialize the edge we should not modify the data members below.
    """

    # begin and end node indices
    begin: int
    end: int
    weight: int = 0
    type: CPEdgeType = CPEdgeType.OPERATOR_KERNEL


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

    def __init__(self, t: "Trace", t_full: "Trace", rank: int) -> None:
        self.cg = CallGraph(t, ranks=[rank])
        self.trace_df: pd.DataFrame = t.get_trace(rank)
        self.t_full: pd.DataFrame = t_full
        self.sym_table = t.symbol_table.get_sym_table()
        self.symbol_table = t.symbol_table
        self.critical_path_nodes: List[int] = []
        self.critical_path_events_set: Set[int] = set()
        self.critical_path_edges_set: Set[CPEdge] = set()

        self.t = t
        self.rank: int = rank
        self.node_list: List[CPNode] = []

        # map from event id in trace dataframe -> node_id pair
        self.event_to_node_map: Dict[int, Tuple[int, int]] = {}

        # init networkx DiGraph
        super(CPGraph, self).__init__()

        self._construct_graph()

    def _add_node(self, node: CPNode) -> int:
        """Adds a node to the graph.
        Args: node (CPNode): node object
        Returns int as node index."""
        self.node_list.append(node)
        idx = node.idx = len(self.node_list) - 1
        self.add_node(idx)  # Call to networkx.DiGraph
        logger.debug(f"Adding critical path node = {node}")
        return idx

    def _add_edge(self, edge: CPEdge) -> None:
        """Adds a edge to the graph.
        Args: node (CPEdge): edge object"""
        logger.debug(f"Adding critical path edge: {edge}")
        self.add_edge(edge.begin, edge.end, weight=edge.weight, object=edge)

    def _add_edge_helper(
        self, src: CPNode, dest: CPNode, type: CPEdgeType = CPEdgeType.OPERATOR_KERNEL
    ) -> None:
        """Adds a edge between two nodes
        Args: src, dest (CPNode): node objects for source and dest."""
        logger.debug(
            f"Adding an edge between nodes {src.idx} -> {dest.idx}" f" type = {type}"
        )
        assert src.idx != dest.idx, f"Src node {src} == Dest node {dest}"
        weight = (
            (dest.ts - src.ts)
            if type not in [CPEdgeType.DEPENDENCY, CPEdgeType.SYNC_DEPENDENCY]
            else 0
        )
        self._add_edge(CPEdge(begin=src.idx, end=dest.idx, weight=weight, type=type))

    def _add_event(self, ev_id: int) -> Tuple[CPNode, CPNode]:
        """Takes an event from the trace and generates two critical path nodes,
        one for start and one for the end.
        Args: ev_id (int): index of the event in trace dataframe.
        Returns: pair of CPNodes aka start and end CPNode
        """
        start_ts = self.trace_df["ts"].loc[ev_id]
        end_ts = start_ts + self.trace_df["dur"].loc[ev_id]
        nodes = [
            CPNode(ev_idx=ev_id, ts=start_ts, is_start=True),
            CPNode(ev_idx=ev_id, ts=end_ts, is_start=False),
        ]
        node_ids = [self._add_node(n) for n in nodes]
        assert len(node_ids) == 2
        self.event_to_node_map[ev_id] = (node_ids[0], node_ids[1])
        return (nodes[0], nodes[1])

    def _get_node_name(self, ev_id: int) -> str:
        if ev_id < 0:
            return "ROOT"
        name_id = self.trace_df.name.loc[ev_id]
        return self.sym_table[name_id]

    def get_nodes_for_event(
        self, ev_id: int
    ) -> Tuple[Optional[CPNode], Optional[CPNode]]:
        """Lookup corresponding nodes for an event id in the trace dataframe
        Args:
            ev_id (int): index of the event in trace dataframe.
        Returns:
            Tuple[Optional[CPNode], Optional[CPNode]]
                Pair of CPNodes aka start and end CPNode for the event.
        """
        start_node, end_node = self.event_to_node_map.get(ev_id, (-1, -1))
        return (
            self.node_list[start_node] if start_node >= 0 else None,
            self.node_list[end_node] if end_node >= 0 else None,
        )

    def get_events_for_edge(self, edge: CPEdge) -> Tuple[int, int]:
        """Lookup corresponding event nodes for an edge
        Args:
            edge (CPEdge): edge object that is part of the di-graph
        Returns:
            Tuple[int, int]
                Pair of event ids representing src and dest of the edge.
        """
        start_node, end_node = edge.begin, edge.end
        return (self.node_list[start_node].ev_idx, self.node_list[end_node].ev_idx)

    def _construct_graph(self) -> None:
        cpu_call_stacks = (
            csg for csg in self.cg.call_stacks if csg.device_type == DeviceType.CPU
        )
        for csg in cpu_call_stacks:
            self._construct_graph_from_call_stack(csg)
        self._construct_graph_from_kernels()

    def _construct_graph_from_call_stack(
        self, csg: CallStackGraph, link_operators: bool = True
    ) -> None:
        """Perform a depth first traversal of the Call Stack for CPU threads
        and generated CP node events.

            @args: csg (CallStackGraph): HTA CallStackGraph object for one CPU thread.
            @args link_operators (bool): If set add an automatic dependency edge
                between consecutive operators on a single thread.

        To enable nested operators we basicaly add edges between start/end
        nodes for events. For example say we have op A and op B and C nested
             |----------------------- Op A ----------------------|
                        |--- Op B ---|        |-- Op C--|
        Critical graph
             (OpA.b)--->(ObB.b)----->(OpB.e)->(OpC.b)-->(OpC.e)->(OpA.e)
        """

        # Track the stack of last seen events
        last_node: Optional[CPNode] = None
        last_highlevel_op: Optional[CPNode] = None
        op_depth = 0

        def is_op_or_runtime(ev_id):
            # ops to consider
            return ev_id > 0 and (
                self.symbol_table.is_operator(self.trace_df, ev_id)
                or self.symbol_table.is_cuda_runtime(self.trace_df, ev_id)
            )

        def enter_func(ev_id, csnode):
            nonlocal last_node
            nonlocal last_highlevel_op
            nonlocal op_depth
            logger.debug(
                "=" * csnode.depth
                + f"Entering node {self._get_node_name(ev_id)}, id = {ev_id}"
            )
            if not is_op_or_runtime(ev_id):
                return

            start_node, end_node = self._add_event(ev_id)

            if link_operators and op_depth == 0 and last_highlevel_op is not None:
                self._add_edge_helper(
                    last_highlevel_op, start_node, CPEdgeType.DEPENDENCY
                )

            op_depth += 1

            if last_node is not None:
                self._add_edge_helper(last_node, start_node)
            last_node = start_node

            logger.debug(
                "=" * csnode.depth + f"Op depth = {op_depth} last_node={last_node}"
            )

        def exit_func(ev_id, csnode):
            nonlocal last_node
            nonlocal last_highlevel_op
            nonlocal op_depth
            logger.debug(
                "=" * csnode.depth
                + f"Exiting node {self._get_node_name(ev_id)}, id = {ev_id}"
            )
            if not is_op_or_runtime(ev_id):
                return

            op_depth -= 1
            _, end_node = self.get_nodes_for_event(ev_id)

            if last_node is not None:
                self._add_edge_helper(last_node, end_node)

            if op_depth == 0:
                last_node = None
                last_highlevel_op = end_node
            else:
                last_node = end_node

            logger.debug(
                "=" * csnode.depth + f"Op depth = {op_depth} last_node={last_node}"
            )

        csg.dfs_traverse(enter_func, exit_func)

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

        # CUDA launch runtime calls
        runtime_calls: pd.DataFrame = (
            self.trace_df.query(self.symbol_table.get_runtime_launch_events_query())
            .copy()
            .sort_values(by="ts", axis=0)
        )

        cudaEventRecord_id = sym_index.get("cudaEventRecord")

        # CUDA Record Event calls
        cuda_record_calls = (
            self.trace_df.query(f"name == {cudaEventRecord_id}")
            .copy()
            .sort_values(by="ts", axis=0)
        )

        def _previous_launch(ts: int, pid: int, tid: int) -> Optional[int]:
            """Find the previous CUDA launch on same pid and tid"""
            df = runtime_calls.query(f"pid == {pid} and tid == {tid}")
            lower_neighbors = df[df["ts"] < ts]["ts"]
            return lower_neighbors.idxmax() if len(lower_neighbors) else None

        cuda_record_calls["index_previous_launch"] = cuda_record_calls.apply(
            lambda x: _previous_launch(x["ts"], x["pid"], x["tid"]), axis=1
        )

        return cuda_record_calls

    def _get_cuda_stream_wait_event_df(self) -> Optional[pd.DataFrame]:
        """For Event based synchronization we need to track the next
        kernel/memcpy launched on a CPU thread just juast after cudaStreamWaitEvent

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

        gpu_kernels = self.trace_df.query("stream != -1 and index_correlation > 0")

        # CUDA launch runtime calls and associated CUDA stream
        runtime_calls = (
            (
                self.trace_df.query(
                    self.symbol_table.get_runtime_launch_events_query()
                )[["index", "ts", "pid", "tid", "index_correlation"]]
                .copy()
                .sort_values(by="ts", axis=0)
            )
            .merge(
                gpu_kernels[["stream", "index"]],
                left_on="index_correlation",
                right_on="index",
                suffixes=["", "_kernel"],
            )
            .set_index("index")
        )

        # CUDA stream wait event runtime calls and associated CUDA stream
        cudaStreamWaitEvent_id = sym_index.get("cudaStreamWaitEvent")
        cuda_stream_wait_events = (
            self.trace_df.query(
                f"name == {cudaStreamWaitEvent_id} and index_correlation > 0"
            )[["index", "ts", "pid", "tid", "correlation", "index_correlation"]]
            .copy()
            .sort_values(by="ts", axis=0)
        ).merge(
            gpu_kernels[["stream", "index"]],
            left_on="index_correlation",
            right_on="index",
            suffixes=["", "_kernel"],
        )

        def _next_launch(ts: int, pid: int, tid: int, stream: int) -> int:
            """Find the next CUDA launch on same pid, tid and stream"""
            df = runtime_calls.query(
                f"pid == {pid} and tid == {tid} and stream == {stream}"
            )
            upper_neighbors = df[df["ts"] > ts]["ts"]
            return upper_neighbors.idxmin() if len(upper_neighbors) else -1

        cuda_stream_wait_events["index_next_launch"] = cuda_stream_wait_events.apply(
            lambda x: _next_launch(x["ts"], x["pid"], x["tid"], x["stream"]), axis=1
        )

        return cuda_stream_wait_events.set_index("index")

    def _check_stream_wait_event_helper(self, row) -> bool:
        eid = row["index"]
        if "wait_on_cuda_event_record_corr_id" not in row:
            logger.warning(
                "CUDA Stream Wait event does not have correlation id of "
                f"cudaEventRecord, name = {self._get_node_name(eid)}"
            )
            return False

        if "index_previous_launch" not in row or (row["index_previous_launch"] == -1):
            logger.warning(
                "CUDA Stream Wait event was not matched to a cudaRecordEvent"
                f", name = {self._get_node_name(eid)}"
            )
            return False
        return True

    def _add_gpu_cpu_sync_edge(self, gpu_node: CPNode, runtime_eid: int) -> None:
        """Add an edge between gpu_node and the runtime event on CPU"""
        _, end_node = self.get_nodes_for_event(runtime_eid)
        logger.debug(
            f"Adding a GPU->CPU sync edge between nodes {gpu_node} -> {end_node}"
        )
        self._add_edge_helper(gpu_node, end_node, type=CPEdgeType.SYNC_DEPENDENCY)

    def _construct_graph_from_kernels(self) -> None:
        """Create nodes and edges for GPU kernels"""
        # Note getting queue length on the clipped dataframe was showing errors,
        # it is worthwhile to consider the entire trace instead, hence use t_full
        q = TraceCounters._get_queue_length_time_series_for_rank(self.t_full, self.rank)

        gpu_kernels = (
            self.trace_df.query("stream != -1 and index_correlation >= 0")
            .join(q[["queue_length"]], on="index_correlation")
            .rename(columns={"queue_length": "queue_length_runtime"})
            .join(q[["queue_length"]], on="index")
        )

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
                    suffixes=["", "_cuda_record"],
                )
                .fillna(-1)
                .astype(int)
            )
            # Note convert NAN to -1 and then turn all records to int

        sym_id_map = self.symbol_table.get_sym_id_map()
        sync_cat = sym_id_map.get("cuda_sync", -1)
        context_sync = sym_id_map.get("Context Sync", -1)
        stream_sync = sym_id_map.get("Stream Sync", -1)
        stream_wait_event = sym_id_map.get("Stream Wait Event", -1)

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
            eid = row["index"]
            logger.debug(
                f"CUDA Sync event eid = {eid}, name = {self._get_node_name(eid)}"
            )

            name = row["name"]
            if name == stream_wait_event:
                # Stream Wait event is indicating a dependency between
                # the next GPU kernel on this stream and another GPU kernel

                if not self._check_stream_wait_event_helper(row):
                    return

                src_kernel_index = self.trace_df.index_correlation[
                    row["index_previous_launch"]
                ]

                # Get the corresponding CPU event for this StreamWaitEvent
                dest_kernel_launch_index = cuda_stream_wait_events.loc[
                    row["index_correlation"]
                ]["index_next_launch"]

                if dest_kernel_launch_index < 0:
                    # TODO make a warning?
                    return

                # Use the next lanuch to find dest_kernel
                dest_kernel_index = self.trace_df.index_correlation[
                    dest_kernel_launch_index
                ]

                logger.debug(
                    f"Scheduling a Stream Sync on stream {row['stream']} "
                    f" dest kernel index {dest_kernel_index}, corr id = "
                    f"{self.trace_df.correlation.loc[dest_kernel_index]}\n "
                    f"on src kernel with index = {src_kernel_index}, corr id = "
                    f"{self.trace_df.correlation.loc[src_kernel_index]}"
                )
                kernel_sync[dest_kernel_index] = src_kernel_index
                return

            assert name == stream_sync or name == context_sync

            # For Context Sync add a sync edge on the last event on all streams,
            # while for Stream Sync only add a sync edge on the specific stream.
            gpu_nodes_to_sync = (
                last_node.values()
                if name == context_sync
                else [last_node.get(row["stream"])]
            )
            for gpu_node in gpu_nodes_to_sync:
                if gpu_node is not None:
                    self._add_gpu_cpu_sync_edge(gpu_node, row["index_correlation"])

        for _, row_ in gpu_kernels.iterrows():
            row = row_.astype(int, errors="ignore")

            if row["cat"] == sync_cat:
                handle_cuda_sync(row)
                continue

            eid, stream = row["index"], row["stream"]
            queue_length, queue_length_runtime = (
                row["queue_length"],
                row["queue_length_runtime"],
            )
            runtime_index = row["index_correlation"]

            logger.debug(
                f"Adding GPU kernel node for eid = {eid}, stream = {stream}, "
                f"name = {self._get_node_name(eid)}, correlation = {row['correlation']}"
            )

            start_node, end_node = self._add_event(eid)
            self._add_edge_helper(start_node, end_node)

            kernel_sync_index = kernel_sync.get(eid)
            kernel_sync_end: Optional[CPNode] = None

            edge_added = False

            # We need to sync between two kernels
            if kernel_sync_index is not None:
                _, kernel_sync_end = self.get_nodes_for_event(kernel_sync_index)
                if kernel_sync_end is None:
                    logger.warning(
                        f"Could not find source kernel sync index = {kernel_sync_index}, current kernel "
                        f" eid = {eid}, stream = {stream}, "
                        f"name = {self._get_node_name(eid)}, correlation = {row['correlation']}"
                    )
                    kernel_sync[eid] = None
                    continue

                # assert kernel_sync_end is not None
                # note that the sync dependency has 0 weight
                self._add_edge_helper(
                    kernel_sync_end, start_node, type=CPEdgeType.SYNC_DEPENDENCY
                )
                logger.debug(
                    "Adding a GPU->GPU sync edge between nodes "
                    f"{kernel_sync_end} -> {start_node}"
                )
                # reset the kernel-kernel sync
                kernel_sync[eid] = None
                edge_added = True

            if (
                # There were no outstanding kernels on this stream during the launch
                queue_length_runtime == 1
                and queue_length == 0
                # and the kernel was launched after previous kernel finished
                and (
                    last_node.get(stream) is None
                    or last_node[stream].ts < self.trace_df.ts.loc[runtime_index]
                )
                # and the kernel sync dependency if any finished earlier
                and (
                    kernel_sync_index is None
                    or kernel_sync_end.ts < self.trace_df.ts.loc[runtime_index]
                )
            ):
                # Add launch delay edge if there were no outstanding kernels
                # when the CUDA runtime was launching it.
                runtime_start, _ = self.get_nodes_for_event(runtime_index)
                assert runtime_start is not None, (
                    f"Could not find runtime index = {runtime_index}, current kernel "
                    f"stream = {stream}, "
                    f"name = {self._get_node_name(eid)}, correlation = {row['correlation']}"
                )
                self._add_edge_helper(
                    runtime_start, start_node, type=CPEdgeType.KERNEL_LAUNCH_DELAY
                )
                edge_added = True
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
                self._add_edge_helper(
                    last_node[stream], start_node, type=CPEdgeType.KERNEL_KERNEL_DELAY
                )
                edge_added = True
            elif not edge_added:
                # Neither of Sync, kernel-kernel or kernel launch edges were added
                logger.error(
                    f"Final fall through Queue length is {queue_length_runtime}!= 1 but no "
                    f"last kernel on stream {stream}, current kernel = {row}"
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
        """Calculates the critical path across nodes"""
        try:
            self.critical_path_nodes = nx.dag_longest_path(self, weight="weight")
        except nx.NetworkXUnfeasible as err:
            logger.error(f"Critical path algorithm failed due to {err}")
            return False
        assert len(self.critical_path_nodes) >= 2

        self.critical_path_events_set = {
            self.node_list[nid].ev_idx for nid in self.critical_path_nodes
        }

        critical_path_nodes_set = set(self.critical_path_nodes)

        for u, v in self.edges:
            e = self.edges[u, v]["object"]
            if u in critical_path_nodes_set and v in critical_path_nodes_set:
                self.critical_path_edges_set.add(e)

        return True

    def show_critical_path(self) -> None:
        """List out the nodes in the critical path graph"""
        for n in self.critical_path_nodes:
            node = self.node_list[n]
            logger.info(
                f"Critical {node}, ev_id = {node.ev_idx} "
                f"ev_name = {self._get_node_name(node.ev_idx)}"
            )


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
                match this string (ProfilerStep100, ProfilerStep101 etc)
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
        annotation_ids = [
            val for key, val in sym_index.items() if annotation in key
        ]

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

        annotations = trace_df[trace_df.name.isin(annotation_ids)].copy()
        annotations["end_ts"] = annotations["ts"] + annotations["dur"]

        start_ts = annotations.ts[instance_start : instance_end + 1].min()
        end_ts = annotations.end_ts[instance_start : instance_end + 1].max()

        logger.info(f"Looking up events within the window ({start_ts}, {end_ts})")

        # Consider all events that start within the annotatated window.
        # and also fiter out 0 duration events as they mess up the
        # event stack generation
        cpu_kernels = trace_df[trace_df["stream"].eq(-1)]
        a = cpu_kernels.query(f"(ts >= {start_ts} and ts <= {end_ts}) and dur > 0")

        # Only consider GPU kernels whose runtime events are in the correct
        # time window
        gpu_kernels = trace_df[trace_df["stream"].ne(-1)]
        cpu_kernels = cpu_kernels.copy().set_index("index_correlation")
        b = (
            gpu_kernels[["ts", "dur", "correlation"]]
            .join(cpu_kernels["ts"], rsuffix="_runtime")
            .query(
                f"(ts_runtime >= {start_ts} and ts_runtime <= {end_ts}) and dur > 0"
            )
        )

        clipped_df = trace_df.loc[a.index.union(b.index)].copy()

        logger.info(f"Clipped dataframe has {len(clipped_df)} events")

        # XXX This is a bit hacky but CallGraph does not really support a way
        # to specify a dataframe, just supports passing Trace object
        t_copy = deepcopy(t)
        t_copy.traces[rank] = clipped_df

        cp_graph = CPGraph(t_copy, t, rank)

        return cp_graph, cp_graph.critical_path()

    @classmethod
    def overlay_critical_path_analysis(
        cls,
        t: "Trace",
        rank: int,
        critical_path_graph: CPGraph,
        output_dir: str,
        only_show_critical_events: bool,
        show_all_edges: bool,
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

        Returns: the overlaid_trace_file path.
        """
        if only_show_critical_events:
            show_all_edges = False

        path = Path(output_dir).expanduser()
        if not path.exists():
            logger.error(f"The path {str(path)} does not exist.")
            return ""
        if not path.is_dir():
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
                cat=str(edge.type.value),
                args={
                    "weight": int(edge.weight),
                    "critical": is_critical,
                },
            )

        if show_all_edges:
            # Networkx provides iteration over the edges represented as (u, v)
            edges = (
                critical_path_graph.edges[u, v]["object"]
                for u, v in critical_path_graph.edges
            )
        else:
            edges = (e for e in critical_path_graph.critical_path_edges_set)

        for e in edges:
            u, v = e.begin, e.end
            start_ev_id, end_ev_id = critical_path_graph.get_events_for_edge(e)
            start_ev, end_ev = raw_events[start_ev_id], raw_events[end_ev_id]

            # XXX need to assert if raw event name and dataframe name are same
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
            logger.debug("Length of new raw events = {raw_events2}")
            overlaid_trace["traceEvents"] = raw_events2

        overlaid_trace["traceEvents"].extend(flow_events)

        t.write_raw_trace(output_file, overlaid_trace)

        return output_file
