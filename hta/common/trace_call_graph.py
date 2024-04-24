from time import perf_counter
from typing import Dict, Generator, List, Optional

import pandas as pd

from hta.common.trace import get_cpu_gpu_correlation, Trace
from hta.common.trace_call_stack import CallStackGraph, CallStackIdentity, CallStackNode
from hta.common.trace_symbol_table import TraceSymbolTable
from hta.common.types import DeviceType, infer_device_type
from hta.configs.config import logger


class CallGraph:
    """
    A CallGraph represents the hierarchical structure of the trace events using a set of CallStackGraph
    objects. Each CallStackGraph object abstracts the execution of a single thread/stream.

    The hierarchical structure is as follows:
    + Distributed training job
    ++ Trainer
        +++ Process
        ++++ Thread/Stream
            ++++ A sequence of trace events represented with a CallStackGraph object

    A CallGraph object includes one or all CallStackGraph objects of the traces and supports further query and
    statistical APIs which utilize the relation links between two or more trace events, such as Cuda Kernel
    launches, AllToAll communications, etc.

    Attributes:
        trace_data (Trace) : A container consisting of a mapping from each trainer to a Trace DataFrame.
        ranks (List[int]) : A list of trainer IDs (i.e., ranks).
        rank_to_stacks (Dict[int, Dict[CallStackIdentity, CallStackGraph]]): a map from ranks to their
        CallStackGraph objects, which are represented as another map from CallStackIdentity to CallStackGraph objects.
        rank_to_nodes[int]: (Dict[int, Dict[int, CallStackNode]]): a map from ranks to their event indices to
            the events' corresponding CallStackNode objects.
        call_stacks (List[CallStackGraph]) : List of per-thread CallStackGraph objects.
        mapping (pd.DataFrame) : A mapping from CallStackIdentity to CallStackGraph using a DataFrame.

        The following attributes are internal to the CallGraph implementation.
        _cached_rank(int) : the current active rank
        _cached_nodes (Dict[int, CallStackNode]) : the CallStackGraph nodes corresponding to _cached_rank.
        _cached_df (pd.DataFame) : the trace DataFrame corresponding to _cached_rank.
        _cached_gpu_kernels (pd.DataFrame) : the GPU kernels corresponding to _cached_rank.
    """

    stack_columns = [
        "parent",
        "depth",
        "height",
        "first_kernel_start",
        "last_kernel_end",
        "num_kernels",
        "kernel_dur_sum",
        "kernel_span",
    ]

    def __init__(self, trace: Trace, ranks: Optional[List[int]] = None) -> None:
        """Construct a CallGraph object from a Trace object.

        Args:
            trace (Trace): The Trace object used to construct this CallGraph object.
            ranks (List[int]) : Only construct the CallGraph objects for the given set of ranks.
                When not provided, uses all available ranks in <trace>.
                Caution: this might be time-consuming.

        Raises:
            ValueError: If the trace data is invalid.
        """
        self.trace_data: Trace = trace
        _ranks: List[int] = self.trace_data.get_ranks()
        self.ranks: List[int] = [r for r in ranks if r in _ranks] if ranks else _ranks
        if (len(self.ranks)) == 0:
            raise ValueError("No rank was found for the trace.")

        self.rank_to_nodes: Dict[int, Dict[int, CallStackNode]] = {}
        self.rank_to_stacks: Dict[int, Dict[CallStackIdentity, CallStackGraph]] = {}
        self.call_stacks: List[CallStackGraph] = []
        self.mapping: pd.DataFrame = pd.DataFrame(
            columns=[
                "rank",
                "pid",
                "tid",
                "label",
                "stack_index",
                "stack_root",
                "count",
            ]
        )

        self._construct_call_graph()

        # Caching current rank's data
        self._cached_rank: int = self.ranks[0]
        self._cached_nodes: Dict[int, CallStackNode] = self.rank_to_nodes[
            self._cached_rank
        ]
        self._cached_df: pd.DataFrame = self.trace_data.get_trace(self._cached_rank)
        self._cached_gpu_kernels: pd.DataFrame = self._cached_df.loc[
            self._cached_df["stream"].ne(-1)
        ]

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        symbol_table: Optional[TraceSymbolTable] = None,
        rank: int = -1,
    ) -> "CallGraph":
        """Construct a CallGraph object from a DataFrame.

        Args:
            df (pd.DataFrame): a DataFrame containing a slice of the trace data for a single rank.
            symbol_table(TraceSymbolTable): a TraceSymbolTable object which encode the symbols in df.
            rank: a rank assigned to df.

        Returns:
            A CallGraph object.
        """
        t = Trace(trace_files={}, trace_dir="")
        t.symbol_table = (
            symbol_table
            if symbol_table
            else TraceSymbolTable.create_symbol_table_from_df(df)
        )
        t.traces[rank] = df.copy()
        t.is_parsed = True

        cg = CallGraph(t)
        return cg

    def _construct_call_graph(self) -> None:
        """
        Construct the call graph from the traces of a distributed training job.
        """
        # TODO:
        # - make it parallel when there are more than one ranks.
        logger.debug(f"Constructing Call Graph for ranks: {self.ranks}")
        for rank in self.ranks:
            t0 = perf_counter()
            self.rank_to_nodes[rank] = {}
            self.rank_to_stacks[rank] = {}
            df = self.trace_data.get_trace(rank)
            # add an "end" column for time interval based filtering
            if "end" not in df.columns:
                df["end"] = df["ts"] + df["dur"]
            self._build_call_stacks(df, self.trace_data.symbol_table, rank)
            t1 = perf_counter()
            logger.debug(
                f"constructed {len(self.rank_to_stacks[rank])} call stacks for rank {rank} in {t1-t0:.2f} seconds"
            )

    def _build_call_stacks(
        self,
        df: pd.DataFrame,
        symbol_table: TraceSymbolTable,
        rank: int = 0,
    ) -> None:
        """Construct the call stacks for a given rank.

        Args:
            df (pd.DataFrame): the rank's trace DataFrame
            symbol_table (TraceSymbolTable): the symbol table used to encode <df>
            rank (int) : the rank whose call stacks to be constructed.
        """
        df.loc[df.index, "depth"] = -1
        df.loc[df.index, "height"] = -1
        df.loc[df.index, "parent"] = -1
        df.loc[df.index, "num_kernels"] = 0
        df.loc[df.index, "kernel_dur_sum"] = 0
        df.loc[df.index, "kernel_span"] = 0
        df.loc[df.index, "first_kernel_start"] = -1
        df.loc[df.index, "last_kernel_end"] = -1

        s_map: pd.Series = pd.Series(self.trace_data.symbol_table.get_sym_id_map())
        s_tab: pd.Series = pd.Series(self.trace_data.symbol_table.get_sym_table())
        main_thread_indicators: pd.Series = s_map[
            s_map.index.str.startswith("ProfilerStep#")
        ]
        bwd_thread_indicators: pd.Series = s_map[s_map.index.str.contains("autograd::")]

        def _infer_stack_label(stack: CallStackGraph) -> str:
            name_ids = stack.df["name"].unique()
            if set(name_ids).intersection(set(main_thread_indicators.values)):
                label = "main"
            elif set(name_ids).intersection(set(bwd_thread_indicators.values)):
                label = "bwd"
            elif len(name_ids) > 0:
                label = s_tab[name_ids[0]]
            else:
                label = "empty"
            return label

        df_correlation = get_cpu_gpu_correlation(df)
        call_stacks: Dict[CallStackIdentity, CallStackGraph] = self.rank_to_stacks[rank]
        nodes: Dict[int, CallStackNode] = self.rank_to_nodes[rank]
        nodes.clear()

        for (pid, tid), df_thread in df.groupby(["pid", "tid"]):
            csi = CallStackIdentity(rank, pid, tid)
            device = infer_device_type(df_thread)

            if device == DeviceType.GPU:  # GPU stream - skip
                logger.debug(
                    f"skipping {df_thread.shape[0]} {device.name} events for (pid={csi.pid} tid={csi.tid})"
                )
                continue
            else:
                logger.debug(
                    f"processing {df_thread.shape[0]} {device.name} events for (pid={csi.pid} tid={csi.tid})"
                )
            t0 = perf_counter()
            csg = CallStackGraph(
                df_thread,
                csi,
                df_correlation,
                df,
                symbol_table,
                nodes,
                save_call_stack_to_df=False,
            )
            call_stacks[csi] = csg
            self.call_stacks.append(csg)

            self.mapping.loc[len(self.mapping)] = (
                csi.rank,
                csi.pid,
                csi.tid,
                _infer_stack_label(csg),
                len(self.call_stacks) - 1,
                csg.root_index,
                1,
            )
            t1 = perf_counter()
            logger.debug(
                f"Created CallStackGraph of {csg.identity}: num_events={csg.df.shape[0]}, num_nodes={len(csg.nodes)}"
                f"in {t1-t0:.2f} seconds"
            )

        logger.debug("connecting stacks of forward and backward threads")
        self._connect_stacks(rank)
        self._update_rank_stack_mapping(rank)

        # Save call stack information to the dataframe
        if len(self.call_stacks) > 0:
            csg = self.call_stacks[-1]
            csg.save_call_stack_to_dataframe(apply_whole_graph=True)

        self._normalize_stack_columns(df)

    @staticmethod
    def _normalize_stack_columns(df: pd.DataFrame) -> None:
        """Normalize the values of the stack columns."""
        if set(df.columns).issuperset(CallGraph.stack_columns):
            df["num_kernels"] = df["num_kernels"].fillna(-1)
            indices_no_kernel_child = df[df.num_kernels <= 0].index

            df.loc[indices_no_kernel_child, "kernel_dur_sum"] = 0
            df.loc[indices_no_kernel_child, "kernel_span"] = 0
            df.loc[indices_no_kernel_child, "first_kernel_start"] = -1
            df.loc[indices_no_kernel_child, "last_kernel_end"] = -1

            for col in ["depth", "height", "parent", "num_kernels"]:
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    def _connect_stacks(self, rank: int) -> None:
        """Connect the CallStackGraph of the main threads and the backward threads."""
        stacks: pd.DataFrame = self.mapping.loc[
            self.mapping["rank"].eq(rank) & self.mapping["label"].isin(["bwd", "main"])
        ].sort_values("label")

        if isinstance(stacks, pd.DataFrame) and stacks.shape[0] == 2:
            stack_indices: List[int] = stacks["stack_index"].to_list()
            bwd_stack: CallStackGraph = self.call_stacks[stack_indices[0]]
            main_stack: CallStackGraph = self.call_stacks[stack_indices[1]]
            self._link_main_and_bwd_stacks(main_stack, bwd_stack)

    def _update_rank_stack_mapping(self, rank: int) -> None:
        """Update the stack root and index after connect CallStackGraph objects."""
        m = self.mapping[self.mapping["rank"].eq(rank)].copy()
        m["stack_root"] = m["stack_index"].apply(
            lambda i: self.call_stacks[i].root_index
        )
        m["count"] = 1
        m["count"] = m.groupby("stack_root")["count"].cumsum()
        self.mapping.loc[m.index, "stack_root"] = m["stack_root"]
        self.mapping.loc[m.index, "count"] = m["count"]

    def _link_main_and_bwd_stacks(
        self,
        main_stack: CallStackGraph,
        bwd_stack: CallStackGraph,
        bwd_annotation_str: str = "## backward ##",
    ) -> None:
        """Link the main and backward stack and update stack node attributes

        Args:
            main_stack (CallStackGraph): the main stack
            bwd_stack (CallStackGraph): the backward stack
            bwd_annotation_str (str): the backward annotations string
        """

        def _get_backward_parents() -> List[int]:
            s_map: pd.Series = pd.Series(self.trace_data.symbol_table.get_sym_id_map())

            # Not all traces have a <bwd_annotation_str>. However, it is still possible to
            # attach the bwd stack to the main stack for each Profiler Step.
            for bwd_top_layer_annotation in [bwd_annotation_str, "ProfilerStep#"]:
                bwd_annotation_ids: pd.Series = s_map[
                    s_map.index.str.startswith(bwd_top_layer_annotation)
                ]
                bwd_annotation_indices = main_stack.df[
                    main_stack.df["name"].isin(bwd_annotation_ids.values)
                ]["index"].values.tolist()

                if len(bwd_annotation_indices) > 0:
                    return bwd_annotation_indices
            return []

        # Link backward operators to the backward annotation events in the same iteration
        for idx in _get_backward_parents():
            bwd_stack.update_parent_of_first_layer_nodes(idx)

    def get_call_stacks(
        self,
        rank: Optional[int] = None,
        pid: Optional[int] = None,
        tid: Optional[int] = None,
        stack_index: Optional[int] = None,
    ) -> Generator[CallStackGraph, None, None]:
        """
        Get the call stack indices for a given call stack identity from the mapping DataFrame.

        Args:
            rank (Optional[int]): a rank for selecting the call stacks.
            pid (Optional[int]): a pid for selecting the call stacks.
            tid (Optional[int]): a thread for selecting the call stacks.
            stack_index (Optional[int]): a call stack index for selecting a call stack.

        Return:
            A generator for iterate over the call stacks of this CallGraph object.
        """
        df = self.mapping
        if rank is not None:
            df = df.loc[df["rank"].eq(rank)]
        if pid is not None:
            df = df.loc[df["pid"].eq(pid)]
        if tid is not None:
            df = df.loc[df["tid"].eq(tid)]
        if stack_index is not None:
            df = df.loc[df["stack_index"].eq(stack_index)]
        indices: List[int] = df["stack_index"].to_list()
        for i in indices:
            yield self.call_stacks[i]

    def _update_cached_data(self, rank: int) -> None:
        """Set rank to the current active rank and update the cached data accordingly.

        Args:
            rank(int): a rank

        Effects:
            When the rank is valid, the cached data will be updated.
            When the valid is invalid, this method does nothing.
        """
        if rank in self.ranks and rank != self._cached_rank:
            self._cached_rank = rank
            self._cached_nodes = self.rank_to_nodes[self._cached_rank]
            self._cached_df = self.trace_data.get_trace(self._cached_rank)
            self._cached_gpu_kernels = self._cached_df.loc[
                self._cached_df["stream"].ne(-1)
            ]

    def get_node_attributes(self, index: int, rank: Optional[int] = None) -> pd.Series:
        """Get the attributes for a given node identified by <index>.

        Args:
            index (int): the index of a given event.
            rank (Optional[int]): the rank of the trace.
                When a valid rank is provided, the cached_data will be updated.
                When rank is None, the cached_data will be used.

        Returns:
            attributes(pd.Series): the matched row in the DataFrame self.trace_data.traces[rank]

        Raises:
            ValueError when the index is not in the DataFrame.
        """
        if rank and rank != self._cached_rank:
            self._update_cached_data(rank)
        if index in self._cached_df.index:
            return self._cached_df.loc[self._cached_df["index"].eq(index)].squeeze()
        raise ValueError(f"invalid node index - {index}")

    def get_csg_of_node(self, index: int, rank: Optional[int] = None) -> CallStackGraph:
        """Get the call stack with node <index> as the parent.

        Args:
            index (int): the index of a given event.
            rank (Optional[int]): the rank of the trace.
                When a valid rank is provided, the cached_data will be updated.
                When rank is None, the cached_data will be used.

        Returns:
            The CallStackGraph object to which the node <index> belongs.

        Raises:
            ValueError when the index is invalid.
        """
        node = self.get_node_attributes(index, rank)
        if node["stream"] > 0:  # GPU kernel -> no corresponding call stack
            node = self.get_node_attributes(node["parent"], rank)

        csg = next(
            self.get_call_stacks(
                rank=self._cached_rank, pid=node["pid"], tid=node["tid"]
            )
        )
        if csg:
            return csg

        raise ValueError(f"CallGraph::get_csg_of_node:: invalid node {index}")

    def get_stack_of_node(
        self, index: int, rank: Optional[int] = None, skip_ancestors: bool = False
    ) -> pd.DataFrame:
        """Get the stack with node <index> as the parent.

        Args:
            index (int): the index of a given event.
            rank (Optional[int]): the rank of the trace.
                When a valid rank is provided, the cached_data will be updated.
                When rank is None, the cached_data will be used.
            skip_ancestors (bool): whether to skip ancestor nodes in the subtree.

        Returns:
            A DataFrame that consists of the node <index>, its descendants,
            and its ancestors (when skip_ancestors == False).

        Raises:
            ValueError when the index is not in the DataFrame.
        """
        node = self.get_node_attributes(index, rank)
        is_cpu_op: bool = True
        if node["stream"] > 0:  # GPU kernel -> no corresponding call stack
            is_cpu_op = False
            node = self.get_node_attributes(node["parent"], rank)

        call_stack = next(
            self.get_call_stacks(
                rank=self._cached_rank, pid=node["pid"], tid=node["tid"]
            )
        )

        if call_stack:
            if is_cpu_op:
                descendants = call_stack.get_descendants(node["index"])
            else:
                descendants = [index]

            if skip_ancestors:
                valid_indices = [i for i in set(descendants) if i >= 0]
            else:
                ancestors = call_stack.get_path_to_root(node["index"])
                indices = set(ancestors).union(set(descendants))
                valid_indices = [i for i in indices if i >= 0]

            df = call_stack.full_df.loc[valid_indices].copy().sort_values("ts")
            return df
        else:
            logger.error(
                f"CallGraph::get_stack_of_node: could not locate call stack for node {index}"
            )
            raise ValueError(
                f"CallGraph::get_stack_of_node:: could not locate call stack for node {index}"
            )

    def get_gpu_kernels(self, rank: Optional[int] = None) -> pd.DataFrame:
        """Get the GPU kernels for a given rank.

        Args:
            rank (Optional[int]): a rank of the trace data to be searched.

        """
        if rank and rank != self._cached_rank:
            self._update_cached_data(rank)
        return self._cached_gpu_kernels
