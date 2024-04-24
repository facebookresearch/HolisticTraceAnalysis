from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from functools import cmp_to_key
from time import perf_counter
from typing import Callable, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from hta.common.trace_symbol_table import TraceSymbolTable
from hta.common.types import DeviceType, infer_device_type
from hta.configs.config import logger

NON_EXISTENT_NODE_INDEX = -2
NULL_NODE_INDEX = -1
EVENT_START = 1
EVENT_END = -1

_I_INDEX: int = 0
_I_DUR: int = 1
_I_KIND: int = 2
_I_TIME: int = 3
OPEN_END = -1
CLOSE_END = 1


def _cmp_events_with_zero_duration(x: np.ndarray, y: np.ndarray) -> bool:
    """Compare two events which at least one of them has zero duration

    Args:
        x (np.ndarray): An array with 4 elements (index, dur, kind, ts)
        y (np.ndarray): The second array with 4 elements (index, dur, type, ts)

    Raises:
        ValueError: When an unexpected case in encountered during the comparison.

    Returns:
        bool: True is x < y, otherwise False
    """
    result: bool = True

    if (
        x[_I_DUR] == 0 and y[_I_DUR] > 0
    ):  # a zero event is enclosed in a non-zero events.
        result = y[_I_KIND] == CLOSE_END
    elif x[_I_DUR] > 0 and y[_I_DUR] == 0:
        result = x[_I_KIND] == OPEN_END
    elif x[_I_DUR] == 0 and y[_I_DUR] == 0:  # two zero events
        if x[_I_KIND] == OPEN_END and y[_I_KIND] == OPEN_END:  # both are open ends
            result = x[_I_INDEX] < y[_I_INDEX]
        elif x[_I_KIND] == CLOSE_END and y[_I_KIND] == CLOSE_END:  # both are close ends
            result = x[_I_INDEX] > y[_I_INDEX]
        else:  # one open and one close
            result = x[_I_KIND] == OPEN_END
    else:
        raise ValueError(f"Unexpected case: {x} {y}")

    return result


def _less_than(x: np.ndarray, y: np.ndarray) -> bool:
    """Test if x < y where both x, y are 1d ndarray of shape (1, 4)

    Returns:
       bool: True if x < y, otherwise False.
    """
    if x[_I_TIME] != y[_I_TIME]:  # compare by time, earlier is smaller.
        return x[_I_TIME] < y[_I_TIME]

    # two ends of the same interval, starting end is smaller.
    if x[_I_INDEX] == y[_I_INDEX]:
        return x[_I_KIND] == OPEN_END

    if x[_I_DUR] == 0 or y[_I_DUR] == 0:
        return _cmp_events_with_zero_duration(x, y)

    if x[_I_KIND] == CLOSE_END and y[_I_KIND] == OPEN_END:  # x is closing, y is opening
        return True

    if x[_I_KIND] == OPEN_END and y[_I_KIND] == CLOSE_END:  # x is opening, y is closing
        return False

    # starting ends of two intervals, longer is smaller.
    if x[_I_KIND] == OPEN_END and y[_I_KIND] == OPEN_END:
        if x[_I_DUR] != y[_I_DUR]:
            return x[_I_DUR] > y[_I_DUR]

    # closing ends of two intervals, shorter is smaller.
    if x[_I_KIND] == CLOSE_END and y[_I_KIND] == CLOSE_END:
        if x[_I_DUR] != y[_I_DUR]:
            return x[_I_DUR] < y[_I_DUR]

    # same ends, same duration
    if x[_I_KIND] == OPEN_END:  # opening end, smaller index first
        return x[_I_INDEX] < y[_I_INDEX]
    else:  # closing end, larger index first
        return x[_I_INDEX] > y[_I_INDEX]


def sort_events(a: np.ndarray) -> None:
    """Sort an events array using the custom comparison function _less_than.

    Args:
        a (np.ndarray): an array of events.
    """

    def _less_than_cmp(x: np.ndarray, y: np.ndarray) -> int:
        return -1 if _less_than(x, y) else 1

    a[:] = sorted(a.tolist(), key=cmp_to_key(_less_than_cmp))


def is_events_sorted(arr: np.ndarray) -> bool:
    """Test if an events array is sorted.

    Args:
        arr (np.array): an events array

    Returns:
        bool: True if array arr is sorted; False otherwise.
    """
    is_sorted = True
    for i in range(0, arr.shape[0] - 1, 1):
        if not _less_than(arr[i], arr[i + 1]):
            logger.info(
                f"ordering violated: {arr[i]} < {arr[i + 1]} is {_less_than(arr[i], arr[i + 1])}"
            )
            is_sorted = False
    return is_sorted


class CallStackIdentity(NamedTuple):
    """Represents where the trace events for constructing the CallStackGraph object are collected.

    Attributes:
        rank (int) : the trainer rank.
        pid (int) : the process ID of the process used to construct the CallStack.
        tid (int) : the thread ID of the threads used to construct the CallStack.
    """

    rank: int = -1
    pid: int = -1
    tid: int = -1


@dataclass
class CallStackNode:
    """An object which captures the connections between entities in the traces.

    Attributes:
        parent (int) : the index of the parent.
            The CallStackNode with index <NULL_NODE_INDEX> always exists and is a dummy parent of all root nodes.
        depth (int) : the depth of the node on the call stack.
            The depth of the root node is -1.
            For any other node, its depth is its parent's depth + 1.
        height (int) : the height of the node on the call stack.
            For a GPU kernel, its height is 0.
            For a CPU operator, the height is max(1, max(children's height) + 1).
        device (DeviceType) : the type of the device on which the CallStackNode resides.
        children (List[int]) : the indices of the entities called by this entity of this node.
    """

    parent: int = NULL_NODE_INDEX
    depth: int = -1
    height: int = -1
    device: DeviceType = DeviceType.CPU
    children: List[int] = field(default_factory=lambda: [])


DFSCallback = Callable[[int, CallStackNode], None]


class CallStackGraph:
    """A CallStackGraph object tracks the call stacks constructed from the PyTorch traces of
    a single CPU thread.

    Attributes:
        identity (CallStackIdentity) : the identity of this CallStackGraph object.
        df (pd.DataFrame) : the dataframe used to construct this CallStackGraph object.
        nodes (Dict[int, node]): a map from a trace entity's index to a CallStackNode object.
        root_index (int): the index of the root node.
        device_type (DeviceType) : the type of device on which the call stack resides.
        correlations (pd.DataFrame) : a DataFrame that specifies the GPU to CPU correlation.
            The correlation DataFrame has two columns: cpu_index and gpu_index.
            With Cuda Call Graph, a cuda launch event can correspond to several kernels.
        full_df (pd.DataFrame): the whole DataFrame for a given rank.
            Because each CallStackGraph object is created from a slice of the full DataFrame, the full_df
            allows linking nodes across different CallStackGraph objects.

    Notes:
        Because the kernels on each GPU stream have one level only, there is no need to construct a call stack
        for the GPU streams.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        identity: CallStackIdentity,
        cpu_gpu_correlation: pd.DataFrame,
        full_df: pd.DataFrame,
        symbol_table: TraceSymbolTable,
        nodes: Optional[Dict[int, CallStackNode]] = None,
        use_existing_stack_columns: bool = True,
        save_call_stack_to_df: bool = True,
    ) -> None:
        """Construct a CallStackGraph object.

        Args:
            df (pd.DataFrame): A DataFrame slice representing trace events on a single thread.
            identity (CallStackIdentity): Identity of this CallStackGraph object.
            cpu_gpu_correlation (pd.DataFrame): A DataFrame with the cpu-gpu correlation information represented by
                the `cpu_index` and `gpu_index` columns.
            full_df (pd.DataFrame): The complete DataFrame which contains all relevant trace events.
            symbol_table (TraceSymbolTable): The symbol table which encode the string columns in the Trace DataFrames.
            nodes (Dict[int, CallStackNode]): Map from node index to CallStackNode object for one rank.
            use_existing_stack_columns (bool): A flag indicating whether constructing the CallGraphObjects from the
                existing stack columns available in the df. This feature is used to reconstruct the CallStackGraph data
                after loading a Trace object from its cache file.
            save_call_stack_to_df (bool): A flag indicating whether saving the CallStackGraph into full_df.

        Note:
            By passing a nodes map object to the CallStackGraph constructor, different call stacks can share the same
            node map so that we can link nodes across different call stacks.
        """
        self.df = df
        self.identity: CallStackIdentity = identity
        self.device_type: DeviceType = infer_device_type(df)
        if self.device_type == DeviceType.GPU:
            logger.error(
                f"CallStackGraph on GPU threads is not supported. CallStackIdentity={identity}"
            )
            return

        # Share trace event index to CallStackNode map when the map is passed into the constructor as an argument.
        self.nodes: Dict[int, CallStackNode] = nodes if nodes is not None else {}

        # Set the index of the root node to a negative integer so that it doesn't collide with the trace event indices.
        self.root_index: int = -abs(identity.tid)
        if self.root_index not in self.nodes:
            self.nodes[self.root_index] = CallStackNode(
                NULL_NODE_INDEX, -1, -1, self.device_type, []
            )

        self.correlations: pd.DataFrame = cpu_gpu_correlation

        self.full_df = full_df
        if "end" not in self.full_df.columns:
            self.full_df.loc[self.full_df.index, "end"] = (
                self.full_df["ts"] + self.full_df["dur"]
            )

        self.symbol_table = symbol_table

        self._num_errors = 0
        if use_existing_stack_columns and self._has_valid_stack_columns():
            self._construct_call_stack_graph_from_existing_df()
        else:
            self._construct_call_stack_graph(df)
            if save_call_stack_to_df:
                self.save_call_stack_to_dataframe(apply_whole_graph=False)

    def save_call_stack_to_dataframe(self, apply_whole_graph: bool = False) -> None:
        """Save the call stack graph information into the trace data frame.

        Args:
            apply_whole_graph (bool): a flag to control whether to apply this operation to the nodes of this
                CallStackGraph object or all CallStackGraph objects which share the same set of nodes.
        """
        self._compute_depth(apply_whole_graph=apply_whole_graph)
        self._compute_height(apply_whole_graph=apply_whole_graph)
        self._add_kernel_info_to_cpu_ops(apply_whole_graph=apply_whole_graph)
        self._save_call_stack_to_df()

    def __repr__(self) -> str:
        """Return a string representation of this CallStackGraph object"""

        s = "CallStackGraph\n"
        for key, item in self.nodes.items():
            s = s + f"    {key}: {item}\n"

        return s

    def _has_valid_stack_columns(self) -> bool:
        """Test if the DataFrame has the valid stack columns from which a CallStackGraph can be reconstructed.

        The DataFrame has the valid stack column when it satisfies the following three conditions:
        (1) The DataFrame self.full_df has columns "parent", "depth", and "height".
        (2) No null or na values for the above three columns in self.full_df[self.df.index].
        (3) The "parent" columns in self.full_df[self.df.index] must be either some non-negative integers or -1.
        """
        stack_columns = {"parent", "depth", "height"}
        # Test condition #1
        if not stack_columns.issubset(set(self.full_df.columns)):
            return False

        # Test condition #2
        if any(
            self.full_df.loc[self.df.index, col].isna().any() for col in stack_columns
        ):
            return False

        # Test condition #3
        parents = self.full_df.loc[self.df.index, "parent"].unique()
        if parents == [-1]:
            return False

        return True

    def _construct_call_stack_graph_from_existing_df(self) -> None:
        """Construct the call stack from the existing stack columns."""
        stack_columns = ["index", "parent", "depth", "height"]

        children: Dict[int, List[int]] = defaultdict(list)
        for idx, parent in self.full_df.parent.items():
            children[parent].append(idx)

        for idx, parent, depth, height in self.full_df.loc[self.df["index"]][
            stack_columns
        ].values:
            self.nodes[idx] = CallStackNode(
                parent=parent,
                depth=depth,
                height=height,
                device=self.device_type,
                children=children[idx],
            )

    def _construct_call_stack_graph(self, df: pd.DataFrame) -> None:
        """Construct the call stack graph from the trace.

        Args:
            df (pd.DataFrame) : a dataframe from which a call graph object is constructed.

        In this function, we assume:
        (1) the traces are from a single thread/stream and therefore
        (2) there is no overlap between the time intervals of the entities on the same level of the graph.

        We skip the call graph construction for GPU streams because the kernels on a single stream is just a list.
        """
        t0 = perf_counter()
        if "index_correlation" not in df.columns:
            raise ValueError(
                "The input DataFrame doesn't have column 'index_correlation'"
            )

        if self.device_type == DeviceType.GPU:
            return

        _df = df.loc[df["stream"].eq(-1)][["index", "ts", "dur"]].copy()
        _df["end"] = _df["ts"] + _df["dur"]
        # Convert the time series into a ndarray in which each row has four attributes:
        #   index, dur, kind, time
        #   type == -1 means event start; type == 1 means event end
        events: np.ndarray = (
            _df.melt(
                id_vars=["index", "dur"],
                value_vars=["ts", "end"],
                var_name="kind",
                value_name="time",
            )
            .replace({"ts": -1, "end": 1})
            .sort_values("time")
        ).to_numpy()

        t1 = perf_counter()
        sort_events(events)
        if not is_events_sorted(events):
            logger.fatal("BUG: the events array is not sorted.")
            raise SystemError("BUG: the events array is not sorted.")
        if len(np.unique(events, axis=0)) != len(events):
            logger.error("BUG: the sorted array contains duplicates")

        stack: List[int] = []
        for ev_idx, _ev_dur, ev_kind, _ev_ts in events:
            if ev_kind == -1:
                if len(stack) > 0:
                    parent_index = stack[-1]
                else:
                    parent_index = self.root_index
                self._add_edge(parent_index, ev_idx)
                stack.append(ev_idx)
            else:  # e.type == 1
                if len(stack) > 0:
                    stack.pop(-1)
        t2 = perf_counter()

        self._link_cpu_and_gpu()

        t3 = perf_counter()

        logger.debug(
            "completed call stack construction:\n"
            f"\tprepare {t1 - t0:.2f}seconds;  construct {t2 - t1:.2f} seconds; link kernel {t3 - t2:.2f} seconds"
        )

    def _add_edge(
        self, parent_index: int, child_index: int, device: DeviceType = DeviceType.CPU
    ) -> None:
        """Add an edge (parent->child) to the graph.

        Args:
            parent_index (int): the index of the parent node.
            child_index (int): the index of the child node.
        """
        if child_index in self.nodes:
            # Based on the single thread sequential execution assumption,
            # a child node should always be added come after its parent node.
            self._num_errors = self._num_errors + 1
            if self._num_errors <= 2:
                logger.error(
                    f"Error: edge={int(parent_index)}-{int(child_index)}; reason=node {int(child_index)} exists."
                )
                logger.error(f"Parent {int(parent_index)}: {self.nodes[parent_index]}")
                logger.error(f"Children {int(child_index)}: {self.nodes[child_index]}")
            return

        if parent_index not in self.nodes:
            # This should only occur for the root node
            self.nodes[parent_index] = CallStackNode(
                self.root_index, 0, -1, device, [child_index]
            )
        else:
            self.nodes[parent_index].children.append(child_index)

        # The parent node should always exist at this point.
        self.nodes[child_index] = CallStackNode(
            parent_index, self.nodes[parent_index].depth + 1, -1, device, []
        )

    def get_nodes(self) -> Dict[int, CallStackNode]:
        """Return the nodes of this call stack graph."""
        return self.nodes

    def update_parent_of_first_layer_nodes(self, new_parent_index: int) -> None:
        """Set the parent of the first layer nodes on call stack to node <new_root_index>.

        Args:
            new_parent_index: the index of the new parent node.

        Note: A call stack was constructed for each thread. However, the user annotations for backward
        phases are only available on the main thread (i.e., the forward thread) but the operators of the
        backward phases are executed on a separate threads. For better interpretation, we want to
        link the cpu ops on the backward thread to the backward annotation in the main thread.
        """
        if new_parent_index not in self.nodes:
            raise ValueError(f"{new_parent_index} not in the node map self.nodes")
        if new_parent_index not in self.full_df.index:
            raise ValueError(f"{new_parent_index} not in the self.full_df")
        ts, end = self.full_df.loc[self.full_df["index"].eq(new_parent_index)][
            ["ts", "end"]
        ].to_numpy()[0]

        indices = self.nodes[self.root_index].children
        guarded_indices = set(
            self.full_df.loc[
                self.full_df["index"].isin(indices)
                & self.full_df["ts"].ge(ts)
                & self.full_df["end"].le(end)
            ]["index"].to_list()
        )
        self._update_parent(list(guarded_indices), new_parent_index)

        if self.root_index not in self.nodes:
            # the old root was deleted because it doesn't have any child.
            self.root_index = self.get_root(new_parent_index)

    def _update_parent(self, node_indices: List[int], new_parent_index: int) -> None:
        """Set the parent for nodes included in the list <node_indices> to node <new_parent_index>.

        Args:
            node_indices (List[int]): a list node indices.
            new_parent_index (int): the index of the new parent.
        """
        if new_parent_index not in self.nodes:
            raise ValueError(f"{new_parent_index} not in the node map self.nodes")

        # filter out invalid node which are either
        # (1) not in self.nodes or
        # (2) already a child of the new parent
        valid_indices = set(node_indices).intersection(set(self.nodes)) - set(
            self.nodes[new_parent_index].children
        )
        old_parents = {self.nodes[n].parent for n in valid_indices}

        # attach nodes to new parents
        for idx in valid_indices:
            self.nodes[idx].parent = new_parent_index
        self.nodes[new_parent_index].children.extend(valid_indices)

        # detach nodes from previous parent
        for p in old_parents:
            children_of_p = set(self.nodes[p].children).intersection(valid_indices)
            for c in children_of_p:
                self.nodes[p].children.remove(c)
            if len(self.nodes[p].children) == 0:
                del self.nodes[p]

    def get_parent(self, idx: int) -> int:
        """Return the parent of a given node <idx>""

        Args:
            idx (int): the index of a node.

        Returns:
            the index of the parent node; return -2 if node <idx> is not in the graph.
        """
        if idx in self.nodes:
            return self.nodes[idx].parent

        logger.error(f"node {idx} is not in current CallStackGraph {self.identity}")
        return NON_EXISTENT_NODE_INDEX

    def get_children(self, idx: int) -> List[int]:
        """Return the children of node <idx>

        Args:
            idx: The index of a node.

        Returns:
            The list of indices of the specified node's children.
        """
        if idx in self.nodes:
            return self.nodes[idx].children
        return []

    def get_root(self, idx: int) -> int:
        """Get the root index of the subtree which contains node <idx>

        Args:
            idx (int): a node index

        Returns:
            int: the root index
        """
        if idx not in self.nodes:
            return NULL_NODE_INDEX

        root = self.nodes[idx].parent
        while root >= 0 and root in self.nodes:
            idx, root = root, self.nodes[root].parent
        return root

    def get_path_to_root(self, idx: int) -> List[int]:
        """Get all the node indices along the path from the node <idx> to the root node

        Args:
            idx (int): the index of a given node.

        Returns:
            List[int]: the list of ancestors' indices, including the node <idx> itself.
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

    def get_paths_to_leaves(
        self, idx: int, include_cuda_kernel: bool = False
    ) -> List[List[int]]:
        """Get all the paths from the node <idx> as the root to leaf nodes.

        Args:

            idx (int): the index of a given node.
            include_cuda_kernel (bool): a flag to control whether to include cuda_kernel in the path.

        Returns:
            List[List[int]]: the list of paths from node <idx> to leaf nodes.
        """
        paths: List[List[int]] = []
        curr_path: List[int] = []

        def _dfs(_idx: int) -> None:
            if _idx not in self.nodes:
                return

            if not include_cuda_kernel and self.nodes[idx].device != DeviceType.CPU:
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

    def get_leaf_nodes(self, idx: int, include_gpu_kernel: bool = False) -> List[int]:
        """Get all leaf nodes on the sub graph with node <idx> as the root.

        Args:
            idx (int): the index of a given node.
            include_gpu_kernel: a flag to control whether to include the GPU kernels as leaf nodes.

        Returns:
            List[int]: the list of leaves nodes on the sub graph with node <idx> as the root.
        """
        return [path[-1] for path in self.get_paths_to_leaves(idx, include_gpu_kernel)]

    def get_descendants(self, idx: int, include_gpu_kernel: bool = False) -> List[int]:
        """Get all leaf nodes on the sub graph with node <idx> as the root.

        Args:
            idx (int): the index of a given node.
            include_gpu_kernel (bool): should kernel nodes be included in the results.

        Returns:
            List[int]: the list of descendants nodes on the sub graph with node <idx> as the root.
        """
        descendants = []
        for path in self.get_paths_to_leaves(idx, include_gpu_kernel):
            descendants.extend(path)
        return list(set(descendants))

    def get_dataframe(self) -> pd.DataFrame:
        """Get the trace dataframe for this CallStackGraph object."""
        return self.df

    def _get_all_root_indices(self) -> List[int]:
        """Get all root indices of the CallGraph's CallStackGraph objects."""
        return [idx for idx in self.nodes if idx < 0 and idx != NON_EXISTENT_NODE_INDEX]

    def _compute_depth(
        self, root_index: Optional[int] = None, apply_whole_graph: bool = False
    ) -> None:
        """Compute the depth of each node in the subgraph rooted at <root_index>.

        Args:
            root_index (int): The index of the root node. If it is None, use the root of current stack.
            apply_whole_graph (bool): Whether to compute the whole graph.
                Because we share the same nodes between different graphs, we can compute the depth of the
                whole graph only once.

        A node's depth is the maximum length of the path from the node to the root.
        The depth of the root is 0.
        """

        def _bfs(_idx: int, parent_depth: int) -> None:
            """Compute the depth of node <idx> and its descendants"""
            if _idx in self.nodes:
                node = self.nodes[_idx]
                node.depth = parent_depth + 1
                for c in node.children:
                    _bfs(c, node.depth)
            else:
                logger.error(f"_compute_depth::_bfs: invalid node index {_idx}")

        if not apply_whole_graph:
            root_idx = self.root_index if root_index is None else root_index
            _bfs(root_idx, -2)
        else:
            for root_idx in self._get_all_root_indices():
                _bfs(root_idx, -2)

    def _compute_height(
        self, root_index: Optional[int] = None, apply_whole_graph: bool = False
    ) -> None:
        """Compute the height of all nodes.

        A node's height is the maximum length of the path from a GPU kernel to the node.
        The height of a GPU kernel is 0.
        """

        def _dfs(idx: int) -> int:
            """Compute the height of node <idx>

            Args:
                idx: the index of a node
            Returns:
                the height of the node <idx>
            """
            if idx in self.nodes:
                node = self.nodes[idx]
                if node.device == DeviceType.GPU:
                    node.height = 0
                else:
                    h = 1
                    for c in node.children:
                        h_c = _dfs(c) + 1
                        h = h_c if h_c > h else h
                    node.height = h
                return node.height
            else:
                logger.error(f"_compute_height::_dfs: invalid node index {idx}")
                return -1

        if not apply_whole_graph:
            root_idx = root_index if root_index else self.root_index
            _dfs(root_idx)
        else:
            for root_idx in self._get_all_root_indices():
                _dfs(root_idx)

    def _save_call_stack_to_df(self) -> None:
        """Save the call stack data into the data frame for quick search."""
        if self.device_type == DeviceType.GPU:
            return
        if "parent" not in self.full_df.columns:
            logger.warning("full_df doesn't have required column `parent`.")
            return
        t0 = perf_counter()
        parent = pd.Series(
            {idx: node.parent for idx, node in self.nodes.items() if idx >= 0},
            name="parent",
            dtype=pd.Int64Dtype(),
            copy=True,
        )

        depth = pd.Series(
            data={idx: node.depth for idx, node in self.nodes.items() if idx >= 0},
            name="depth",
            dtype=pd.Int16Dtype(),
            copy=True,
        )

        height = pd.Series(
            data={idx: node.height for idx, node in self.nodes.items() if idx >= 0},
            name="height",
            dtype=pd.Int16Dtype(),
            copy=True,
        )

        self.full_df.loc[parent.index, "parent"] = parent.astype(
            self.full_df.dtypes["height"]
        )
        self.full_df.loc[depth.index, "depth"] = depth.astype(
            self.full_df.dtypes["height"]
        )
        self.full_df.loc[height.index, "height"] = height.astype(
            self.full_df.dtypes["height"]
        )

        t1 = perf_counter()
        logger.debug(f"added call stack information in {t1 - t0:.2}s")

    def get_depth(self) -> pd.Series:
        """Get the depth for all valid nodes

        Return:
            a Series with the node index as index and depth as the data
        """
        if "depth" in self.full_df:
            return self.full_df.loc[self.df["index"]]["depth"]
        else:
            return pd.Series(
                data={idx: node.depth for idx, node in self.nodes.items() if idx >= 0},
                name="depth",
                dtype=pd.Int16Dtype(),
                copy=True,
            ).loc[self.df["index"]]

    def _link_cpu_and_gpu(self) -> None:
        """Add an edge from cuda_launch to gpu kernel"""
        correlations: List[List[int]] = self.correlations[
            self.correlations["cpu_index"].isin(self.df["index"])
        ][["cpu_index", "gpu_index"]].values.tolist()

        for cpu_index, gpu_index in correlations:
            self._add_edge(cpu_index, gpu_index, DeviceType.GPU)

    def _add_kernel_info_to_cpu_ops(
        self, root_index: Optional[int] = None, apply_whole_graph: bool = False
    ) -> None:
        """Add kernel information to cpu operators.

        Many AI model performance analyses are conducted at the operator level with kernel statistics.
        This method add the kernel information to the cpu operator side.
        """
        t0 = perf_counter()
        if "num_kernels" not in self.full_df.columns:
            logger.warning("full_df doesn't have stack columns such `num_kernels`.")
            return

        # extract nodes data
        ops: pd.DataFrame = self.full_df.loc[self.full_df["index"].isin(self.nodes)]
        gpu_kernels = ops.loc[ops.stream.ne(-1)][["ts", "dur", "end"]]
        s_start: Dict[int, int] = gpu_kernels["ts"]
        s_end: Dict[int, int] = gpu_kernels["end"]
        s_dur: Dict[int, int] = gpu_kernels["dur"]

        t1 = perf_counter()
        KernelInfo = namedtuple(
            "KernelInfo", "count sum_dur kernel_span first_start last_end"
        )
        kernel_info: Dict[int, KernelInfo] = {}

        t_max: int = self.full_df["ts"].max() * 2

        def _dfs(idx: int) -> KernelInfo:
            if idx not in self.nodes:
                return KernelInfo(0, 0, 0, t_max, -1)
            node: CallStackNode = self.nodes[idx]
            # node is a GPU kernel
            if node.device == DeviceType.GPU:
                if idx in s_start:
                    start = s_start[idx]
                    end = s_end[idx]
                    sum_dur = s_dur[idx]
                    kernel_info[idx] = KernelInfo(1, sum_dur, end - start, start, end)
                    return KernelInfo(1, sum_dur, end - start, start, end)
                else:
                    logger.error(f"unknown kernel (index={idx})")
                    return KernelInfo(0, 0, 0, t_max, -1)

            # node is a CPU op
            count, sum_dur, span, start, end = 0, 0, 0, t_max, -1

            for c in node.children:
                c_info = _dfs(c)
                count = count + c_info.count
                sum_dur = sum_dur + c_info.sum_dur
                start = min(start, c_info.first_start)
                end = max(end, c_info.last_end)
                span = end - start

            if idx >= 0:  # skip the root node
                kernel_info[idx] = KernelInfo(count, sum_dur, span, start, end)
            return KernelInfo(count, sum_dur, span, start, end)

        # traverse all nodes
        if not apply_whole_graph:
            root_index = self.root_index if root_index is None else root_index
            if (
                root_index not in self.nodes
                or len(self.nodes[root_index].children) == 0
            ):
                logger.warning(f"CallStackGraph {self.identity} is empty.")
                return
            _dfs(root_index)
        else:
            for root_idx in self._get_all_root_indices():
                _dfs(root_idx)
        t2 = perf_counter()

        # add information to the data frame
        df_info = pd.DataFrame.from_dict(kernel_info, orient="index")
        df_info = df_info.loc[df_info.index >= 0]
        if df_info.empty:
            logger.error("df_info is empty.")
            return
        self.full_df.loc[df_info.index, "num_kernels"] = df_info["count"].astype(
            self.full_df.dtypes["num_kernels"]
        )
        self.full_df.loc[df_info.index, "kernel_dur_sum"] = df_info["sum_dur"].astype(
            self.full_df.dtypes["kernel_dur_sum"]
        )
        self.full_df.loc[df_info.index, "kernel_span"] = df_info["kernel_span"].astype(
            self.full_df.dtypes["kernel_span"]
        )
        self.full_df.loc[df_info.index, "first_kernel_start"] = df_info[
            "first_start"
        ].astype(self.full_df.dtypes["first_kernel_start"])
        self.full_df.loc[df_info.index, "last_kernel_end"] = df_info["last_end"].astype(
            self.full_df.dtypes["last_kernel_end"]
        )

        t3 = perf_counter()
        logger.debug(
            f"Add kernel info: prepare {t1 - t0:.2}s; traverse in {t2 - t1:.2f}s; update df {t3 - t2:.2f}s"
        )

    def dfs_traverse(self, enter_func: DFSCallback, exit_func: DFSCallback) -> None:
        """Depth first traversal on a specific call stack.
        Call enter_func() and exit_func() on each callstack node.
        """
        self._dfs_traverse_node(self.root_index, enter_func, exit_func)

    def _dfs_traverse_node(
        self, node_id: int, enter_func: DFSCallback, exit_func: DFSCallback
    ) -> None:
        node = self.nodes[node_id]
        enter_func(node_id, node)

        all_nodes = self.nodes.keys()
        for child_nid in node.children:
            if child_nid in all_nodes:
                self._dfs_traverse_node(child_nid, enter_func, exit_func)

        exit_func(node_id, node)
