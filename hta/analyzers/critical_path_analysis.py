# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

import pandas as pd
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
            f"CPNode(event: {self.ev_idx}, nid={self.idx}, ts={self.ts}, "
            f"is_start={self.is_start})"
        )


class CPEdgeType(Enum):
    OPERATOR_KERNEL = 0
    DEPENDENCY = 1
    KERNEL_LAUNCH_DELAY = 2
    KERNEL_KERNEL_DELAY = 3
    SYNC_DELAY = 4


@dataclass
class CPEdge:
    """An edge in the critical path di-graph.
    This represents either a
    1) span of time i.e. duration of an operator/kernel.
    2) a dependency among operators/kernels.
    3) a kernel launch or kernel-kernel delay.
    3) synchronization delay.

    The weight represents time in the graph, cases 1) and 3)
    above have non-zero weights
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
        trace_df (pd.DataFrame) : dataframe of trace events used to construct this graph.
        symbol_table (TraceSymbolTable) : a symbol table used to encode the symbols in the trace.
        node_list (List[int]) : list of critical path node objects, index in this list is always the node id..
        critical_path_nodes (List[int]) : list of node ids on the critical path.
        critical_path_nodes_set (Set[int]) : set of node ids on the critical path.
        critical_path_events_set (Set[int]) : set of event ids corresponding to the critical path nodes.
    """

    def __init__(self, t: "Trace", rank: int) -> None:
        self.cg = CallGraph(t, ranks=[rank])
        self.trace_df: pd.DataFrame = t.get_trace(rank)
        self.sym_table = t.symbol_table.get_sym_table()
        self.symbol_table = t.symbol_table
        self.critical_path_nodes: List[int] = []
        self.critical_path_nodes_set: Set[int] = set()
        self.critical_path_events_set: Set[int] = set()

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

    def _add_edge_between(
        self, src: CPNode, dest: CPNode, type: CPEdgeType = CPEdgeType.OPERATOR_KERNEL
    ) -> None:
        """Adds a edge between two nodes
        Args: src, dest (CPNode): node objects for source and dest."""
        logger.debug(
            f"Adding an edge between nodes {src.idx} -> {dest.idx}" f" type = {type}"
        )
        assert src.idx != dest.idx, f"Src node {src} == Dest node {dest}"
        weight = (dest.ts - src.ts) if type == CPEdgeType.OPERATOR_KERNEL else 0
        self._add_edge(CPEdge(begin=src.idx, end=dest.idx, weight=weight, type=type))

    def _add_event(self, eid: int) -> Tuple[CPNode, CPNode]:
        """Takes an event from the trace and generates two critical path nodes,
        one for start and one for the end.
        Args: eid (int): index of the event in trace dataframe.
        Returns: pair of CPNodes aka start and end CPNode
        """
        start_ts = self.trace_df["ts"].loc[eid]
        end_ts = start_ts + self.trace_df["dur"].loc[eid]
        nodes = [
            CPNode(ev_idx=eid, ts=start_ts, is_start=True),
            CPNode(ev_idx=eid, ts=end_ts, is_start=False),
        ]
        node_ids = [self._add_node(n) for n in nodes]
        assert len(node_ids) == 2
        self.event_to_node_map[eid] = (node_ids[0], node_ids[1])
        return (nodes[0], nodes[1])

    def _get_node_name(self, eid: int):
        if eid < 0:
            return "ROOT"
        name_id = self.trace_df.name.loc[eid]
        return self.sym_table[name_id]

    def get_nodes_for_event(
        self, ev_id: int
    ) -> Tuple[Optional[CPNode], Optional[CPNode]]:
        """Lookup corresponding nodes for an event id in the trace dataframe
        Args:
            ev_id (int): index of the event in trace dataframe.
        Returns:
            Tuple[Optional[CPNode], Optional[CPNode]]
                Pair of CPNodes aka start and end CPNode
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

    # def _add_edge_for_depedency(
    #     self, source: int, sink: int
    # ) -> None:
    #     """Add an edge between two dependent operations"""
    #     if source == sink:
    #         return
    #     _, source_end_node = self.get_nodes_for_event(source)
    #     if source_end_node is None:
    #         logger.warning(
    #             f"Count not find critical path di-graph node for src event {source}"
    #         )
    #         return
    #     sink_begin_node, _ = self.get_nodes_for_event(sink)
    #     if sink_begin_node is None:
    #         logger.warning(
    #             f"Count not find critical path di-graph node for sink event {sink}"
    #         )
    #         return
    #     logger.info(
    #         f"Adding dependency edge between {source_end_node} -> {sink_begin_node}"
    #     )
    #     self._add_edge(
    #         CPEdge(
    #             begin=source_end_node.idx,
    #             end=sink_begin_node.idx,
    #             weight=0,
    #         )
    #     )

    def _construct_graph(self):
        cpu_call_stacks = (
            csg for csg in self.cg.call_stacks if csg.device_type == DeviceType.CPU
        )
        for csg in cpu_call_stacks:
            self._construct_graph_from_call_stack(csg)

    def _construct_graph_from_call_stack(
        self, csg: CallStackGraph, link_operators: bool = True
    ):
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
             (OpA.b)----(ObB.b)------(OpB.e)--(OpC.b)---(OpC.e)--(OpA.e)
        """

        # Track the stack of last seen events
        last_node: Optional[CPNode] = None
        last_highlevel_op: Optional[CPNode] = None
        op_depth = 0

        def is_op_or_runtime(eid):
            # ops to consider
            return eid > 0 and (
                self.symbol_table.is_operator(self.trace_df, eid)
                or self.symbol_table.is_runtime(self.trace_df, eid)
            )

        def enter_func(eid, csnode):
            nonlocal last_node
            nonlocal last_highlevel_op
            nonlocal op_depth
            logger.debug(
                "=" * csnode.depth
                + f"Entering node {self._get_node_name(eid)}, id = {eid}"
            )
            if not is_op_or_runtime(eid):
                return

            start_node, end_node = self._add_event(eid)

            if link_operators and op_depth == 0 and last_highlevel_op is not None:
                self._add_edge_between(
                    last_highlevel_op, start_node, CPEdgeType.DEPENDENCY
                )

            op_depth += 1

            if last_node is not None:
                self._add_edge_between(last_node, start_node)
            last_node = start_node

            logger.debug(
                "=" * csnode.depth + f"Op depth = {op_depth} last_node={last_node}"
            )

        def exit_func(eid, csnode):
            nonlocal last_node
            nonlocal last_highlevel_op
            nonlocal op_depth
            logger.debug(
                "=" * csnode.depth
                + f"Exiting node {self._get_node_name(eid)}, id = {eid}"
            )
            if not is_op_or_runtime(eid):
                return

            op_depth -= 1
            _, end_node = self.get_nodes_for_event(eid)

            if last_node is not None:
                self._add_edge_between(last_node, end_node)

            if op_depth == 0:
                last_node = None
                last_highlevel_op = end_node
            else:
                last_node = end_node

            logger.debug(
                "=" * csnode.depth + f"Op depth = {op_depth} last_node={last_node}"
            )

        csg.dfs_traverse(enter_func, exit_func)

    def _show_digraph(self) -> None:
        """Prints the networkx digraph"""
        n = 0
        for n in self.nodes:
            node = self.node_list[n]
            print(f"node id = {n}, node = {node}")
            print("  neighbors = ", ",".join((str(n) for n in self.neighbors(n))))

    def critical_path(self) -> bool:
        """Calculates the critical path across nodes"""
        try:
            self.critical_path_nodes = nx.dag_longest_path(self, weight="weight")
        except nx.NetworkXUnfeasible as err:
            logger.error(f"Critical path algorithm failed due to {err}")
            return False
        self.critical_path_nodes_set = set(self.critical_path_nodes)
        self.critical_path_events_set = {
            self.node_list[nid].ev_idx for nid in self.critical_path_nodes
        }
        return True

    def show_critical_path(self) -> None:
        """List out the nodes in the critical path graph"""
        for n in self.critical_path_nodes:
            node = self.node_list[n]
            print(
                f"Critical {node}, ev_id = {node.ev_idx} "
                f"ev_name = {self._get_node_name(node.ev_idx)}"
            )


class CriticalPathAnalysis:
    def __init__(self):
        # dict of critical path nodes, node id -> CPNode
        self.cp_nodes = {}

    @classmethod
    def critical_path_analysis_for_rank(
        cls,
        t: "Trace",
        rank: int,
        annotation: str,
        instance_id: Optional[int],
    ) -> Optional[CPGraph]:
        r"""
        Perform critical path analysis for trace events within a rank.
        We further reduce the region of interest by selecting
        a trace annotation and instance id. This will
        limit the analysis to events within the time range of that annoation.
        For example, you can use this to limit the analysis to one iteration
        by passing annotation='ProfilerStep500'.

        Args:
            t (Trace): Input trace data structure.
            rank (int): rank to analyze for the critical path.
            annotation (str): a trace annotation to limit the analysis to,
                such as "ProfilerStep1200"
            intance (int): optionally specify which instance of the annotation
                to consider. Defaults to the first instance.

        Returns:
            CPGraph object that can be used to obtain statistics and further
            visualize the critical path.

            CPGraph is also a subinstance of a networkx.DiGraph.
            Run 'CPGraph?' for more info and APIs.
        """
        trace_df: pd.DataFrame = t.get_trace(rank)
        sym_index = t.symbol_table.get_sym_id_map()
        sym_table = t.symbol_table.get_sym_table()

        annotation_id = sym_index.get(annotation, None)
        if annotation_id is None:
            logger.error(f"Could not find annotation {annotation} in the trace.")
            return None
        if instance_id is None:
            instance_id = 0

        logger.info(
            f"Looking up events under {instance_id} instance of '{annotation}' annotation"
        )

        annotation_span = (
            trace_df[trace_df.name == annotation_id].iloc[instance_id].to_dict()
        )

        start_ts = annotation_span["ts"]
        end_ts = annotation_span["ts"] + annotation_span["dur"]

        # Consider all events that start within the annotatated window.
        # and also fiter out 0 duration events as they mess up the
        # event stack generation
        clipped_df = trace_df.query(
            f"(ts >= {start_ts} and ts <= {end_ts}) and " " dur > 0"
        ).copy()
        logger.info(f"Clipped dataframe has {len(clipped_df)} events")

        # XXX This is a bit hacky but CallGraph does not really support a way
        # to specify a dataframe, just supports passing Trace object
        t_copy = deepcopy(t)
        t_copy.traces[rank] = clipped_df

        cp_graph = CPGraph(t_copy, rank)

        return cp_graph

    @classmethod
    def overlay_critical_path_analysis_for_rank(
        cls,
        t: "Trace",
        rank: int,
        critical_path_graph: CPGraph,
        output_dir: str,
        show_all_edges: bool = False,
    ) -> str:
        r"""
        Overlay the identified critical path on top of the trace file
        for visualization.

        Args:
            t (Trace): Input trace data structure.
            rank (int): rank to generate the time series for.
            critical_path_graph: Critical Path Graph object generated previously
            output_dir (str): Output director to store overlaid trace.
            show_all_edges (bool): When set this will add edge events for
                all types of edges in the critical path graph. This is useful
                for debugging the algorithm.

        Returns: the overlaid_trace_file path.
        """
        import os
        from pathlib import Path

        path = Path(output_dir).expanduser()
        if not path.exists():
            logger.error(f"The path {str(path)} does not exist.")
            return ""
        if not path.is_dir():
            logger.error(f"{output_dir} is not a directory.")
            return ""

        output_file = os.path.join(
            str(path), "overlaid_" + t.trace_files[rank].split("/")[-1]
        )
        overlaid_trace = t.get_raw_trace_for_one_rank(rank=rank)
        raw_events = overlaid_trace["traceEvents"]

        # Traverse events and mark them as critical
        for ev_idx, event in enumerate(raw_events):
            if ev_idx in critical_path_graph.critical_path_events_set:
                event["args"]["critical"] = 1

        flow_events = []
        flow_id = 0

        def get_edge_cat(edge_type: CPEdgeType) -> str:
            if edge_type == CPEdgeType.OPERATOR_KERNEL:
                return "critical_path_operator"
            if edge_type == CPEdgeType.DEPENDENCY:
                return "critical_path_dependency"
            return "critical_path_other"

        def get_flow_event(
            nid: int, event: Dict[str, Any], edge: CPEdge, flow_id: int, is_start: bool
        ):
            return Trace.flow_event(
                id=flow_id,
                pid=event["pid"],
                tid=event["tid"],
                ts=int(
                    event["ts"]
                    if critical_path_graph.node_list[nid].is_start
                    else event["ts"] + event["dur"]
                ),
                is_start=is_start,
                name="critical_path",
                cat=get_edge_cat(edge.type),
                args={"weight": int(edge.weight)},
            )

        # Networkx provides iteration over the edges represented as (u, v)
        for u, v in critical_path_graph.edges:
            e = critical_path_graph.edges[u, v]["object"]
            start_ev_id, end_ev_id = critical_path_graph.get_events_for_edge(e)
            start_ev, end_ev = raw_events[start_ev_id], raw_events[end_ev_id]

            if e.type == CPEdgeType.OPERATOR_KERNEL and not show_all_edges:
                continue

            # XXX need to assert if raw event name and dataframe name are same
            logger.debug(f"Adding cp edge between {start_ev_id} and {end_ev_id}")

            flow_events.append(get_flow_event(u, start_ev, e, flow_id, is_start=True))
            flow_events.append(get_flow_event(u, end_ev, e, flow_id, is_start=False))
            flow_id += 1

        raw_events.extend(flow_events)
        t.write_raw_trace(output_file, overlaid_trace)

        return output_file
