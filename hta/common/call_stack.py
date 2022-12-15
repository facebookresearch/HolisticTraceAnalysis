# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
from collections import namedtuple
from enum import Enum
from time import perf_counter
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from hta.common.trace import Trace

NON_EXISTENT_NODE_INDEX = -2
NULL_NODE_INDEX = -1
EVENT_START = 1
EVENT_END = -1


class DeviceType(Enum):
    UNKNOWN = 0
    CPU = 1
    GPU = 2


def infer_device_type(df: pd.DataFrame) -> DeviceType:
    """Infer the device type based on trace data.

    Args:
        df (pd.DataFrame): the filtered dataframe for a single thread/stream

    Returns:
        DeviceType: the type of device
    """
    streams = df["stream"].unique()
    device_type: DeviceType = DeviceType.UNKNOWN

    if len(streams) > 0:
        if np.all(np.greater(streams, 0)):
            device_type = DeviceType.GPU
        elif np.all(np.less(streams, 0)):
            device_type = DeviceType.CPU
    return device_type


"""
We break each trace record into two Event objects: the entity starts and the entity ends.
The Event objects are temporary objects for constructing the call stack.
When an entity starts, it is pushed into the call stack;
when an entity ends, it is popped off the call stack.
In an Event object, the index represents the entity, the type indicates whether the entity
starts or ends at the given time. The duration (<dur>) field is added to resolve cases
where two events may happen at the same time. The time, type, dur, and index are used to
sort the events in the same thread (call stack) using the sorting algorithm provided in
compare_events.
"""
Event = namedtuple("Event", ["idx", "time", "dur", "type"])


def compare_events(x: Event, y: Event) -> int:
    """Compare two events

    Args:
        x, y (Event): the two events to compare

    Returns:
        the ordering of two events
        < 0     x should go first than y
        = 0     same events
        > 0     y should go first than x

    Note:
        There are six cases:
        1. different time:
            Event(idx=1, time=0, dur=10, type=1)
            Event(idx=2, time=2, dur=5, type=1)
            Event(idx=2, time=7, dur=5, type=-1)
            Event(idx=1, time=10, dur=10, type=-1)
        2. same start time, different duration:
            Event(idx=1, time=0, dur=10, type=1)
            Event(idx=2, time=0, dur=5, type=1)
            Event(idx=2, time=5, dur=5, type=-1)
            Event(idx=1, time=10, dur=10, type=-1)
        3. Same end time, different duration:
            Event(idx=1, time=0, dur=10, type=1)
            Event(idx=1, time=10, dur=10, type=-1)
            Event(idx=2, time=10, dur=5, type=1)
            Event(idx=2, time=15, dur=5, type=-1)
        4. same time, one start event, one end event
            Event(idx=1, time=0, dur=10, type=1)
            Event(idx=1, time=10, dur=10, type=-1)
            Event(idx=2, time=10, dur=5, type=1)
            Event(idx=2, time=15, dur=5, type=-1)
        5. same time, same event type, same duration, different index
            Event(idx=1, time=0, dur=10, type=1)
            Event(idx=2, time=0, dur=10, type=1)
            Event(idx=2, time=10, dur=10, type=-1)
            Event(idx=1, time=10, dur=10, type=-1)
        6. same time, same event type, same duration, same index
            The ordering doesn't matter.
    """
    result = x.time - y.time
    if result == 0:
        if x.type == y.type:
            if x.type == EVENT_START:
                if x.dur == y.dur:
                    result = -1 if x.idx < y.idx else 1 if x.idx > y.idx else 0
                else:
                    result = 1 if x.dur < y.dur else -1
            else:
                if x.dur == y.dur:
                    result = 1 if x.idx < y.idx else -1 if x.idx < y.idx else 0
                else:
                    result = -1 if x.dur < y.dur else 1
        else:
            result = 1 if x.type == EVENT_START else -1
    return result


class CallStackNode(NamedTuple):
    """A CallStackNode object captures the connections between entities in the traces.

    Each CallStackNode maps to a unique trace entity, which is an abstraction for operators,
    kernels, functions, user annotations, module components, or any other entities in the traces.

    Each entity is represented by its index (i.e., ID) in the trace data representation.
    A CallStackNode does not store all the attributes of a trace entity. To get the entity's
    attributes, we use CallStackNode's index to query the trace data.

    We assume all entity indices are non-negative integers and use the sentinel value
    NULL_NODE_INDEX for a non-existent entity.

    Attributes:
        parent (int) : the index of the parent.
        depth (int) : the depth on the call stack
        children (List[int]) : the indices of the entities called by this entity of this node.
    """

    parent: int = NULL_NODE_INDEX
    depth: int = -1
    children: List[int] = []


class CallStackIdentity(NamedTuple):
    """A CallStackIdentity object keeps the identity data of a CallStackGraph object.

    Attributes:
        self.rank (int) : the trainer rank.
        self.pid (int) : the process ID of the traces used to construct the CallStack.
        self.tid (int) : the thread ID or stream ID of the trace used to construct the CallStack.
        self.device_type (DeviceType) : the type of the device on which the thread/stream is executed.
    """

    rank: int = -1
    pid: int = -1
    tid: int = -1


class CallStackGraph:
    """A CallStackGraph object tracks the call stacks constructed from the execution traces of
    a single CPU thread or GPU stream.

    Attributes:
        identity (CallStackIdentity) : the identity of this CallStackGraph object.
        df (pd.DataFrame) : the dataframe used to generate this CallStackGraph object.
        nodes (Dict[int, node]): a map from a trace entity's index to a CallStackNode object.
        device_type (DeviceType) : what type of device that the call stack resides.
        correlations (pd.Series) : a Series that maps a node index to the index of a correlated node.

    Notes:
    + Because the kernels on a GPU has only one level, we don't construct a call stack for GPU kernels.
    """

    def __init__(self, df: pd.DataFrame, identity: CallStackIdentity) -> None:
        """Construct an empty graph."""
        self.df = df
        self.identity: CallStackIdentity = identity
        self.device_type: DeviceType = infer_device_type(df)
        self.nodes: Dict[int, CallStackNode] = {}
        self.correlations: pd.Series = None
        self.depth: pd.Series = None
        self._construct_call_stack_graph(df)
        self._compute_depth()

    def __repr__(self):

        ret = "\n"
        for key, item in self.nodes.items():
            ret = ret + f"    {key}: {item}\n"

        return f"CallStackGraph({ret})"

    def _construct_call_stack_graph(self, df) -> None:
        """Construct the call stack from the trace.

        In this function, we assume:
        (1) the traces are from a single thread/stream and therefore
        (2) there is no overlap between the time intervals of the entities on the same level of the graph.

        We skip the call graph construction for GPU streams because the kernels on a single stream is just a list.
        """
        if "index_correlation" not in df.columns:
            raise ValueError("The input DataFrame doesn't have column 'index_correlation'")
        self.correlations = df["index_correlation"]

        if self.device_type == DeviceType.GPU:
            return

        self.nodes.clear()
        self.nodes[NULL_NODE_INDEX] = CallStackNode(NULL_NODE_INDEX, -1, [])
        events = []
        df = df[["index", "ts", "dur"]].copy()
        df["end"] = df["ts"] + df["dur"]
        for _, row in df.iterrows():
            events.append(Event(row["index"], row["ts"], row["dur"], EVENT_START))
            events.append(Event(row["index"], row["end"], row["dur"], EVENT_END))
        events.sort(key=functools.cmp_to_key(compare_events))

        stack: List[Event] = []
        for e in events:
            if e.type == EVENT_START:
                if len(stack) > 0:
                    parent_index = stack[-1].idx
                else:
                    parent_index = NULL_NODE_INDEX
                self._add_edge(parent_index, e.idx)
                stack.append(e)
            else:  # e.type == EVENT_END
                if len(stack) > 0:
                    stack.pop(-1)

    def _add_edge(self, parent_index: int, child_index: int) -> None:
        """Add an edge (parent->child) to the graph.
        Args:
            parent_index (int): the index of the parent node.
            child_index (int): the index of the child node.
        """
        if child_index in self.nodes:
            # Based on the single thread assumption, a child node should always come after the parent node.
            logging.error(f"node {child_index} has already existed.")
            return

        if parent_index not in self.nodes:
            # This should only occurs for the root node
            self.nodes[parent_index] = CallStackNode(NULL_NODE_INDEX, 0, [child_index])
        else:
            self.nodes[parent_index].children.append(child_index)

        # The parent node should always exist at this point.
        self.nodes[child_index] = CallStackNode(parent_index, self.nodes[parent_index].depth + 1, [])

    def get_nodes(self) -> Dict[int, CallStackNode]:
        """Return the nodes of this graph."""
        return self.nodes

    def get_parent(self, idx: int) -> int:
        """Return the parent of a given node <idx>""

        Args:
            idx (int): the index of a node.

        Returns:
            int: the index of the parent node; return -2 if node <idx> is not in the graph.
        """
        if idx in self.nodes:
            return self.nodes[idx].parent
        logging.error(f"node {idx} is not in current CallStackGraph {self.identity}")
        return NON_EXISTENT_NODE_INDEX

    def get_children(self, idx: int) -> List[int]:
        """Return the children of node <idx>"""
        if idx in self.nodes:
            return self.nodes[idx].children
        return []

    def get_path_to_root(self, idx: int) -> List[int]:
        """Get all the node indices along the path from the node <idx> to the root node

        Args:
            idx (int): the index of a given node.

        Returns:
            List[int]: the list of ancestors' indices.
        """
        if idx not in self.nodes:
            return []

        path = [idx]
        while idx >= 0:
            if idx in self.nodes:
                parent = self.nodes[idx].parent
                path.append(parent)
                idx = parent
            else:
                break
        return path

    def get_paths_to_leaves(self, idx: int) -> List[List[int]]:
        """Get all the paths from the node <idx> as the root to leaf nodes.

        Args:
            idx (int): the index of a given node.

        Returns:
            List[List[int]]: the list of paths from node <idx> to leaf nodes.
        """
        paths = []
        curr_path = []

        def _dfs(_idx: int) -> None:
            if _idx not in self.nodes:
                return

            curr_path.append(_idx)
            if not self.nodes[_idx].children:
                paths.append(list(curr_path))
            else:
                for child in self.nodes[_idx].children:
                    _dfs(child)
            curr_path.pop()

        _dfs(idx)
        return paths

    def get_leaf_nodes(self, idx: int) -> List[int]:
        """Get all leaf nodes on the sub graph with node <idx> as the root.

        Args:
            idx (int): the index of a given node.

        Returns:
            List[int]: the list of leaves nodes on the sub graph with node <idx> as the root.
        """
        return [path[-1] for path in self.get_paths_to_leaves(idx)]

    def get_dataframe(self) -> pd.DataFrame:
        """Get the trace dataframe for this stack"""
        return self.df

    def _compute_depth(self) -> None:
        """Add the depth information to the DataFrame"""
        if self.device_type == DeviceType.GPU:
            self.depth = pd.Series(
                data=np.full(self.correlations.size, -1),
                index=self.correlations.index,
                name="depth",
                copy=True,
            )
        else:  # self.device_type == DeviceType.CPU:
            self.depth = pd.Series(
                data={idx: node.depth for idx, node in self.nodes.items() if idx >= 0},
                name="depth",
                copy=True,
            )

    def get_depth(self) -> pd.Series:
        """Get the depth for all valid node

        Return:
            a Series with the node index as index and depth as the data
        """
        return self.depth


class CallGraph:
    """
    A CallGraph represents the entire set of traces with a set of CallStackGraph
    objects.

    The execution of a distributed training job can be abstracted as a hierarchical
    organization of CallStackGraph object, which abstracts the execution of a single
    thread/stream. The hierarchical structure is as follows:
    + distribute training job
      ++ trainer
         +++ process
           ++++ thread/stream
             ++++ a sequence of entity events - represented with a CallStackGraph object

    Because there are possible relationship links between two or more CallStackGraph objects,
    such as Cuda Kernel launches, AllToAll communications, etc., a CallStackGraph object
    includes all CallStackGraph objects in the trace and provides further further query and statistic APIs.

    Attributes:
        trace_data (Trace) : the trace data represented in a Trace object, which
            contains multiple DataFrame objects mapping the traces of each trainer.
        call_stacks (List[CallStackGraph]) : a list of per-thread CallStackGraph objects.
        mapping (pd.DataFrame) : the mapping from CallStackIdentity to CallStackGraph using a DataFrame
    """

    def __init__(self, trace: Trace, ranks: Optional[List[int]] = None) -> None:
        """Construct a CallGraph from a Trace object <trace_data>

        Args:
            trace (Trace): the trace data used to construct this CallGraph object.
            ranks (List[int]) : filter the traces using the given set of ranks. Using all ranks if None.
        Raises:
            ValueError: the trace data is invalid.
        """
        self.trace_data: Trace = trace
        self.mapping: pd.DataFrame = pd.DataFrame()
        self.call_stacks: List[CallStackGraph] = []

        _ranks = [k for k in trace.get_all_traces()] if ranks is None else ranks
        self._construct_call_graph(_ranks)

    def _construct_call_graph(self, ranks: List[int]) -> None:
        """
        Construct the call graph from the traces of a distributed training job.

        Args:
            ranks (List[int]) : a list ranks to select traces for construct the call stacks.
        """
        call_stack_ids: List[CallStackIdentity] = []
        t0 = perf_counter()
        # construct a call stack graph for each thread/stream
        for rank in ranks:
            df = self.trace_data.get_trace(rank)
            for pid, pid_group in df.groupby(by="pid"):
                for tid, tid_group in pid_group.groupby(by="tid"):
                    csi = CallStackIdentity(rank, pid, tid)
                    csg = CallStackGraph(tid_group, csi)
                    self.call_stacks.append(csg)
                    call_stack_ids.append(csi)

        t1 = perf_counter()
        logging.debug(f"Completed constructing call stack graph for in {t1 - t0:.3} seconds")

        # build a map from call stack meta data to call stack objects
        self.mapping = pd.DataFrame(
            {
                "rank": [csi.rank for csi in call_stack_ids],
                "pid": [csi.pid for csi in call_stack_ids],
                "tid": [csi.tid for csi in call_stack_ids],
                "csg_index": range(len(self.call_stacks)),
            }
        )

        # add depth information to the data frame
        for rank in ranks:
            call_stack_indices = self.mapping[self.mapping["rank"].eq(rank)]["csg_index"]
            depth = pd.concat([self.call_stacks[idx].get_depth() for idx in call_stack_indices])
            df = self.trace_data.get_trace(rank)
            df["depth"] = depth
        self.mapping.set_index(["rank", "pid", "tid"], inplace=True)
