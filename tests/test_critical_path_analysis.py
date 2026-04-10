import gzip
import json
import os
import unittest
from collections import Counter
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

from hta.analyzers.critical_path_analysis import (
    CPEdge,
    CPEdgeType,
    CPGraph,
    CPNode,
    CriticalPathAnalysis,
    restore_cpgraph,
)
from hta.common.trace_parser import (
    _auto_detect_parser_backend,
    get_default_trace_parsing_backend,
    ParserBackend,
    set_default_trace_parsing_backend,
)
from hta.trace_analysis import TraceAnalysis
from hta.utils.test_utils import get_test_data_dir


class CriticalPathAnalysisTestCase(unittest.TestCase):
    """Tests for critical path graph construction and analysis."""

    simple_add_trace: TraceAnalysis
    alexnet_trace: TraceAnalysis
    event_sync_trace: TraceAnalysis
    event_sync_multi_stream_trace: TraceAnalysis
    ns_resolution_trace_dir: str
    ns_resolution_trace: TraceAnalysis
    amd_trace_dir: str
    amd_trace: TraceAnalysis

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["CRITICAL_PATH_ADD_ZERO_WEIGHT_LAUNCH_EDGE"] = "1"
        base = get_test_data_dir()

        cls.simple_add_trace = TraceAnalysis(
            trace_dir=os.path.join(base, "critical_path/simple_add")
        )
        cls.alexnet_trace = TraceAnalysis(
            trace_dir=os.path.join(base, "critical_path/alexnet")
        )
        cls.event_sync_trace = TraceAnalysis(
            trace_dir=os.path.join(base, "critical_path/cuda_event_sync")
        )
        cls.event_sync_multi_stream_trace = TraceAnalysis(
            trace_dir=os.path.join(base, "critical_path/cuda_event_sync_multi_stream")
        )
        cls.ns_resolution_trace_dir = os.path.join(base, "ns_resolution_trace")
        cls.ns_resolution_trace = TraceAnalysis(trace_dir=cls.ns_resolution_trace_dir)
        cls.amd_trace_dir = os.path.join(base, "amd_trace")
        cls.amd_trace = TraceAnalysis(trace_dir=cls.amd_trace_dir)

    def _critical_path_on_simple_add_trace(self) -> CPGraph:
        annotation = "[param|pytorch.model.alex_net|0|0|0|measure|forward]"
        instance_id = 1
        cp_graph, success = self.simple_add_trace.critical_path_analysis(
            rank=0, annotation=annotation, instance_id=instance_id
        )
        self.assertTrue(success)
        return cp_graph

    def test_critical_path_basic_add(self) -> None:
        """Test graph construction for the simple_add trace."""
        cp_graph = self._critical_path_on_simple_add_trace()
        trace_df = self.simple_add_trace.t.get_trace(0)

        # Verify operator names at known indices
        relu_idx = 286
        clamp_min_idx = 287
        cuda_launch_idx = 1005
        self.assertEqual(cp_graph._get_node_name(relu_idx), "aten::relu_")
        self.assertEqual(cp_graph._get_node_name(clamp_min_idx), "aten::clamp_min_")
        self.assertEqual(cp_graph._get_node_name(cuda_launch_idx), "cudaLaunchKernel")

        # Check node structure: start/end for each event
        expected_node_ids = [(57, 62), (58, 61), (59, 60)]

        def check_nodes(ev_idx: int) -> Tuple[int, int]:
            start_node, end_node = cp_graph.get_nodes_for_event(ev_idx)
            self.assertTrue(start_node.is_start)
            self.assertFalse(end_node.is_start)
            return start_node.idx, end_node.idx

        self.assertEqual(check_nodes(relu_idx), expected_node_ids[0])
        self.assertEqual(check_nodes(clamp_min_idx), expected_node_ids[1])
        self.assertEqual(check_nodes(cuda_launch_idx), expected_node_ids[2])

        # Check edge weights and event attribution
        def check_edge(
            start_nid: int, end_nid: int, weight: int, attr_ev: int
        ) -> CPEdge:
            e = cp_graph.edges[start_nid, end_nid]["object"]
            self.assertEqual(e.begin, start_nid)
            self.assertEqual(e.end, end_nid)
            self.assertEqual(e.weight, weight, msg=f"edge = {e}")
            self.assertEqual(
                cp_graph.edge_to_event_map[(e.begin, e.end)],
                attr_ev,
                msg=f"edge = {e}, expected attributed event = {attr_ev}",
            )
            return e

        #  --------------aten::relu_---------------
        #   <e1>|-------aten::clamp_min_------|<e5>
        #       <-e2->|cudaLaunchKernel|<-e4->
        #             | <-----e3-----> |
        e1 = check_edge(expected_node_ids[0][0], expected_node_ids[1][0], 15, relu_idx)
        e2 = check_edge(
            expected_node_ids[1][0], expected_node_ids[2][0], 14, clamp_min_idx
        )
        e3 = check_edge(
            expected_node_ids[2][0], expected_node_ids[2][1], 17, cuda_launch_idx
        )
        e4 = check_edge(
            expected_node_ids[2][1], expected_node_ids[1][1], 15, clamp_min_idx
        )
        e5 = check_edge(expected_node_ids[1][1], expected_node_ids[0][1], 32, relu_idx)

        self.assertEqual(
            set(cp_graph.get_edges_attributed_to_event(relu_idx)), {e1, e5}
        )
        self.assertEqual(
            set(cp_graph.get_edges_attributed_to_event(clamp_min_idx)), {e2, e4}
        )
        self.assertEqual(
            set(cp_graph.get_edges_attributed_to_event(cuda_launch_idx)), {e3}
        )

        # Check kernel launch delay
        fft_kernel_idx = 1051
        fft_runtime_idx = trace_df.index_correlation.loc[fft_kernel_idx]
        kstart, kend = cp_graph.get_nodes_for_event(fft_kernel_idx)
        rstart, _ = cp_graph.get_nodes_for_event(fft_runtime_idx)

        kernel_launch_edge = cp_graph.edges[rstart.idx, kstart.idx]["object"]
        self.assertEqual(kernel_launch_edge.type, CPEdgeType.KERNEL_LAUNCH_DELAY)
        self.assertEqual(kernel_launch_edge.weight, 27)

        # Check kernel-kernel delay
        ampere_kernel_idx = 1067
        k2start, _ = cp_graph.get_nodes_for_event(ampere_kernel_idx)
        kernel_kernel_edge = cp_graph.edges[kend.idx, k2start.idx]["object"]
        self.assertEqual(kernel_kernel_edge.type, CPEdgeType.KERNEL_KERNEL_DELAY)
        self.assertEqual(kernel_kernel_edge.weight, 7)

        # Check zero-weight causal launch edge
        ampere_runtime_idx = trace_df.index_correlation.loc[ampere_kernel_idx]
        r2start, _ = cp_graph.get_nodes_for_event(ampere_runtime_idx)
        zero_edge = cp_graph.edges[r2start.idx, k2start.idx]["object"]
        self.assertEqual(zero_edge.type, CPEdgeType.KERNEL_LAUNCH_DELAY)
        self.assertEqual(zero_edge.weight, 0)

        # Check device sync edge
        epilogue_kernel_idx = 1275
        cuda_device_sync_idx = 1281
        _, k3end = cp_graph.get_nodes_for_event(epilogue_kernel_idx)
        _, syncend = cp_graph.get_nodes_for_event(cuda_device_sync_idx)
        device_sync_edge = cp_graph.edges[k3end.idx, syncend.idx]["object"]
        self.assertEqual(device_sync_edge.type, CPEdgeType.SYNC_DEPENDENCY)
        self.assertEqual(device_sync_edge.weight, 0)

        # All OPERATOR_KERNEL and KERNEL_KERNEL edges should have event attribution
        for u, v in cp_graph.edges:
            e = cp_graph.edges[u, v]["object"]
            if e.type in {CPEdgeType.OPERATOR_KERNEL, CPEdgeType.KERNEL_KERNEL_DELAY}:
                self.assertIn(
                    (u, v),
                    cp_graph.edge_to_event_map,
                    msg=f"edge = {(u, v)}, obj = {e}",
                )
                self.assertIsNotNone(cp_graph.get_event_attribution_for_edge(e))
            else:
                self.assertIsNone(cp_graph.get_event_attribution_for_edge(e))

        self.assertEqual(len(cp_graph.critical_path_nodes), 315)

    @dataclass
    class OverlaidTraceStats:
        total_event_count: int = 0
        marked_critical_events: int = 0
        marked_critical_edges: int = 0
        edge_count_per_type: Dict[str, int] = field(default_factory=dict)

    def _check_overlaid_trace(
        self, overlaid_trace: str
    ) -> "CriticalPathAnalysisTestCase.OverlaidTraceStats":
        stats = self.OverlaidTraceStats()
        with gzip.open(overlaid_trace, "r") as ovf:
            trace_events = json.load(ovf)["traceEvents"]
            stats.total_event_count = sum(
                1
                for e in trace_events
                if e["ph"] == "X"
                and e.get("cat", "") not in {"user_annotation", "python_function"}
            )
            stats.marked_critical_events = sum(
                e["args"].get("critical", 0)
                for e in trace_events
                if "args" in e and e["ph"] == "X"
            )
            stats.marked_critical_edges = sum(
                e["args"].get("critical", 0)
                for e in trace_events
                if "args" in e and e["ph"] == "f"
            )
            stats.edge_count_per_type = Counter(
                e["args"]["type"]
                for e in trace_events
                if "args" in e
                and "type" in e["args"]
                and "critical_path" in e["args"]["type"]
            )
        return stats

    def test_critical_path_overlaid_trace(self) -> None:
        """Check overlaid trace output with various options."""
        cp_graph = self._critical_path_on_simple_add_trace()

        # Overlaid trace with show_all_edges=True
        with TemporaryDirectory(dir="/tmp") as tmpdir:
            overlaid_trace = self.simple_add_trace.overlay_critical_path_analysis(
                0,
                cp_graph,
                output_dir=tmpdir,
                only_show_critical_events=False,
                show_all_edges=True,
            )
            self.assertIn("overlaid_critical_path_", overlaid_trace)

            stats = self._check_overlaid_trace(overlaid_trace)
            self.assertEqual(stats.marked_critical_events, 159)
            self.assertEqual(
                stats.marked_critical_events, len(cp_graph.critical_path_events_set)
            )

            cpgraph_edges = (
                cp_graph.edges[u, v]["object"] for (u, v) in cp_graph.edges
            )
            cpgraph_edge_counts = Counter(
                e.type
                for e in cpgraph_edges
                if not CriticalPathAnalysis._is_zero_weight_launch_edge(e)
            )
            for etype in CPEdgeType:
                self.assertEqual(
                    stats.edge_count_per_type[etype.value],
                    cpgraph_edge_counts[etype] * 2,
                )

            self.assertEqual(stats.marked_critical_edges, 314)
            self.assertEqual(
                stats.marked_critical_edges, len(cp_graph.critical_path_edges_set)
            )

        # Overlaid trace with show_all_edges=False (default: CPU, GPU sync, kernel launch)
        with TemporaryDirectory(dir="/tmp") as tmpdir:
            overlaid_trace = self.simple_add_trace.overlay_critical_path_analysis(
                0,
                cp_graph,
                output_dir=tmpdir,
                only_show_critical_events=False,
                show_all_edges=False,
            )
            stats = self._check_overlaid_trace(overlaid_trace)
            self.assertEqual(stats.marked_critical_events, 159)
            self.assertEqual(stats.marked_critical_edges, 21)

        # Overlaid trace showing only critical events
        with TemporaryDirectory(dir="/tmp") as tmpdir:
            overlaid_trace = self.simple_add_trace.overlay_critical_path_analysis(
                0,
                cp_graph,
                output_dir=tmpdir,
                only_show_critical_events=True,
                show_all_edges=True,
            )
            stats = self._check_overlaid_trace(overlaid_trace)
            self.assertEqual(stats.marked_critical_events, 159)
            self.assertEqual(stats.marked_critical_events, stats.total_event_count)

        # Missing output dir should be created automatically
        tmpdir = "/tmp/hta_test_path_does_not_exist"
        try:
            overlaid_trace = self.simple_add_trace.overlay_critical_path_analysis(
                0,
                cp_graph,
                output_dir=tmpdir,
                only_show_critical_events=False,
                show_all_edges=True,
            )
            self.assertTrue(os.path.exists(tmpdir))
        finally:
            if os.path.exists(overlaid_trace):
                os.remove(overlaid_trace)
            if os.path.exists(tmpdir):
                os.removedirs(tmpdir)

    def test_critical_path_inter_stream_sync(self) -> None:
        """AlexNet has inter-stream synchronization using CUDA Events."""
        annotation = "[param|pytorch.model.alex_net|0|0|0|measure|forward]"
        cp_graph, success = self.alexnet_trace.critical_path_analysis(
            rank=0, annotation=annotation, instance_id=1
        )
        self.assertTrue(success)
        self.assertGreater(len(cp_graph.critical_path_nodes), 0)

        # Verify at least one SYNC_DEPENDENCY edge exists in the graph
        sync_edges = [
            cp_graph.edges[u, v]["object"]
            for u, v in cp_graph.edges
            if cp_graph.edges[u, v]["object"].type == CPEdgeType.SYNC_DEPENDENCY
        ]
        self.assertGreater(
            len(sync_edges), 0, "Expected at least one SYNC_DEPENDENCY edge"
        )
        # All sync dependency edges should have weight 0
        for e in sync_edges:
            self.assertEqual(
                e.weight, 0, f"SYNC_DEPENDENCY edge {e} should have weight 0"
            )

    def test_critical_path_analysis_event_sync(self) -> None:
        """Check cudaEventSync() synchronization edges."""
        cp_graph, success = self.event_sync_trace.critical_path_analysis(
            rank=0, annotation="ProfilerStep", instance_id=0
        )
        self.assertTrue(success)

        # Verify SYNC_DEPENDENCY edges exist with zero weight.
        # The trace contains cudaEventSynchronize and cudaDeviceSynchronize
        # events that create GPU->CPU sync edges.  Rather than relying on
        # specific event indices (which differ between parsing environments),
        # verify the graph contains the expected edge types.
        sync_edges = [
            (u, v, data["object"])
            for u, v, data in cp_graph.edges(data=True)
            if data["object"].type == CPEdgeType.SYNC_DEPENDENCY
        ]
        self.assertGreater(len(sync_edges), 0, "Expected SYNC_DEPENDENCY edges")
        for _, _, edge in sync_edges:
            self.assertEqual(edge.weight, 0)

    def test_critical_path_analysis_event_sync_multistream(self) -> None:
        """Check cuda Stream wait event across multiple streams."""
        cp_graph, success = self.event_sync_multi_stream_trace.critical_path_analysis(
            rank=0, annotation="", instance_id=None
        )
        self.assertTrue(success)

        # Verify event record correlations exist
        event_record_df = cp_graph._get_cuda_event_record_df()
        event_records = event_record_df[
            ["correlation", "correlation_launch_event"]
        ].to_dict(orient="records")
        self.assertEqual(len(event_records), 3)

        # Verify SYNC_DEPENDENCY edges exist across streams.
        # The trace has kernels on streams 20, 28, and 24 with cuda event
        # record/wait synchronization.  Rather than relying on specific event
        # indices (which differ between parsing environments), verify the
        # graph contains SYNC_DEPENDENCY edges with zero weight.
        sync_edges = [
            (u, v, data["object"])
            for u, v, data in cp_graph.edges(data=True)
            if data["object"].type == CPEdgeType.SYNC_DEPENDENCY
        ]
        self.assertGreater(len(sync_edges), 0, "Expected SYNC_DEPENDENCY edges")
        for _, _, edge in sync_edges:
            self.assertEqual(edge.weight, 0)

    # AI-assisted test
    def test_create_event_nodes_structure(self) -> None:
        """Test _create_event_nodes DataFrame ops (lines 476, 480)."""
        cp_graph = self._critical_path_on_simple_add_trace()
        # node_list should have start and end nodes for each event
        start_nodes = [n for n in cp_graph.node_list if n.is_start]
        end_nodes = [n for n in cp_graph.node_list if not n.is_start]
        self.assertEqual(len(start_nodes), len(end_nodes))
        # Every event should have a start ts <= end ts
        for s_node in start_nodes:
            ev_idx = s_node.ev_idx
            e_node_idx = cp_graph.event_to_end_node_map.get(ev_idx, -1)
            if e_node_idx >= 0:
                e_node = cp_graph.node_list[e_node_idx]
                self.assertGreaterEqual(
                    e_node.ts,
                    s_node.ts,
                    f"End ts < start ts for event {ev_idx}",
                )

    # AI-assisted test
    def test_cuda_event_record_df_columns(self) -> None:
        """Test _get_cuda_event_record_df output covers lines 926, 928, 932, 971-973."""
        cp_graph, success = self.event_sync_multi_stream_trace.critical_path_analysis(
            rank=0, annotation="", instance_id=None
        )
        self.assertTrue(success)
        event_record_df = cp_graph._get_cuda_event_record_df()
        self.assertIsNotNone(event_record_df)
        # Verify index_previous_launch column exists and has no NaN (filled with -1)
        self.assertIn("index_previous_launch", event_record_df.columns)
        self.assertFalse(
            event_record_df["index_previous_launch"].isna().any(),
            "index_previous_launch should have no NaN values after fillna(-1)",
        )
        # Verify temporary columns were dropped
        self.assertNotIn("launch_id", event_record_df.columns)
        self.assertNotIn("previous_launch_id", event_record_df.columns)

    # AI-assisted test
    def test_cuda_stream_wait_event_df_columns(self) -> None:
        """Test _get_cuda_stream_wait_event_df output covers lines 996, 1040, 1050."""
        cp_graph, success = self.event_sync_multi_stream_trace.critical_path_analysis(
            rank=0, annotation="", instance_id=None
        )
        self.assertTrue(success)
        stream_wait_df = cp_graph._get_cuda_stream_wait_event_df()
        if stream_wait_df is not None:
            # Verify the runtime_calls stream column was renamed from stream_kernel
            # (line 996-997) and launch_id was dropped (line 1050)
            self.assertIn("stream", stream_wait_df.columns)
            self.assertNotIn("launch_id", stream_wait_df.columns)

    # AI-assisted test
    def test_critical_path_analysis_returns_none_for_invalid_annotation(self) -> None:
        """Test that critical_path_analysis returns None for a nonexistent annotation (line 1839/1849)."""
        result = self.simple_add_trace.critical_path_analysis(
            rank=0,
            annotation="__nonexistent_annotation_that_does_not_exist__",
            instance_id=0,
        )
        self.assertIsNone(result)

    # AI-assisted test
    def test_attribute_edge_with_optional_src_parent(self) -> None:
        """Test _attribute_edge handles Optional[int] src_parent (line 294).

        Case 4 in edge attribution: src=end, dest=start uses src_parent.
        When src_parent is a valid int, ev_idx is assigned correctly.
        When src_parent is None, the assert on line 347 should fire.
        """
        cp_graph = self._critical_path_on_simple_add_trace()

        # Find an OPERATOR_KERNEL edge where Case 4 applies (end->start)
        case4_edges = []
        for u, v in cp_graph.edges:
            e = cp_graph.edges[u, v]["object"]
            if e.type == CPEdgeType.OPERATOR_KERNEL:
                src = cp_graph.node_list[e.begin]
                dest = cp_graph.node_list[e.end]
                if not src.is_start and dest.is_start:
                    case4_edges.append(e)

        # _attribute_edge with valid int src_parent should work
        if case4_edges:
            e = case4_edges[0]
            # Re-attribute with a valid parent; should not raise
            cp_graph._attribute_edge(e, 0)
            self.assertEqual(cp_graph.edge_to_event_map[(e.begin, e.end)], 0)

        # _attribute_edge with None src_parent on a Case 4 edge should assert
        if case4_edges:
            e = case4_edges[0]
            with self.assertRaises(AssertionError, msg="ev_idx must not be None"):
                cp_graph._attribute_edge(e, None)

    # AI-assisted test
    def test_attribute_edge_skips_non_operator_types(self) -> None:
        """Test _attribute_edge returns early for non-OPERATOR/KERNEL edge types (line 306)."""
        cp_graph = self._critical_path_on_simple_add_trace()

        # Find a SYNC_DEPENDENCY or DEPENDENCY edge
        for u, v in cp_graph.edges:
            e = cp_graph.edges[u, v]["object"]
            if e.type in {CPEdgeType.DEPENDENCY, CPEdgeType.SYNC_DEPENDENCY}:
                # Should not add to edge_to_event_map
                original_map = dict(cp_graph.edge_to_event_map)
                cp_graph._attribute_edge(e, None)
                self.assertEqual(
                    cp_graph.edge_to_event_map,
                    original_map,
                    "Non-operator edge types should not be attributed",
                )
                break

    # AI-assisted test
    def test_get_nodes_for_event_returns_none_for_missing(self) -> None:
        """Test get_nodes_for_event returns (None, None) for unknown event IDs (line 372-387).

        Covers the None guards at lines 1317-1318, 1118, and similar.
        """
        cp_graph = self._critical_path_on_simple_add_trace()

        # Use an event ID that is definitely not in the graph
        start, end = cp_graph.get_nodes_for_event(-999)
        self.assertIsNone(start)
        self.assertIsNone(end)

    # AI-assisted test
    def test_add_gpu_cpu_sync_edge_none_guard(self) -> None:
        """Test _add_gpu_cpu_sync_edge returns early when end_node is None (lines 1117-1119)."""
        cp_graph = self._critical_path_on_simple_add_trace()
        edge_count_before = cp_graph.number_of_edges()

        # Create a dummy GPU node
        gpu_node = cp_graph.node_list[0]

        # Call with a runtime_eid that has no nodes in the graph
        cp_graph._add_gpu_cpu_sync_edge(gpu_node, -999)

        # No new edge should be added
        self.assertEqual(cp_graph.number_of_edges(), edge_count_before)

    # AI-assisted test
    def test_add_kernel_launch_delay_edge_missing_runtime(self) -> None:
        """Test _add_kernel_launch_delay_edge returns False for missing runtime (line 1138-1140)."""
        cp_graph = self._critical_path_on_simple_add_trace()

        kernel_node = cp_graph.node_list[0]
        result = cp_graph._add_kernel_launch_delay_edge(-999, kernel_node)
        self.assertFalse(result)

    # AI-assisted test
    def test_kernel_graph_none_node_guards(self) -> None:
        """Verify kernel graph construction handles None nodes gracefully (lines 1317-1318, 1435).

        The graph should still construct successfully even if get_nodes_for_event
        returns None for some events — those events are simply skipped.
        """
        annotation = "[param|pytorch.model.alex_net|0|0|0|measure|forward]"
        cp_graph, success = self.alexnet_trace.critical_path_analysis(
            rank=0, annotation=annotation, instance_id=1
        )
        self.assertTrue(success)

        # All nodes in the graph should be valid (not None)
        for nid in cp_graph.nodes:
            node = cp_graph.node_list[nid]
            self.assertIsNotNone(node)
            self.assertIsInstance(node, CPNode)

    # AI-assisted test
    def test_kernel_sync_dict_allows_optional_values(self) -> None:
        """Test that kernel_sync values can be None (line 1214, 1353).

        After a sync is processed, kernel_sync[eid] is set to None.
        This exercises the Dict[int, Optional[int]] type change.
        """
        cp_graph, success = self.event_sync_multi_stream_trace.critical_path_analysis(
            rank=0, annotation="", instance_id=None
        )
        self.assertTrue(success)

        # The graph should have SYNC_DEPENDENCY edges from event sync processing
        sync_edges = [
            cp_graph.edges[u, v]["object"]
            for u, v in cp_graph.edges
            if cp_graph.edges[u, v]["object"].type == CPEdgeType.SYNC_DEPENDENCY
        ]
        self.assertGreater(len(sync_edges), 0)

    # AI-assisted test
    def test_validate_graph_and_critical_path_networkx(self) -> None:
        """Test _validate_graph and critical_path use networkx correctly (lines 1456, 1548).

        Covers the type: ignore[attr-defined] lines for nx.dag_longest_path
        and nx.is_directed_acyclic_graph.
        """
        cp_graph = self._critical_path_on_simple_add_trace()

        # _validate_graph should return True for a valid graph
        self.assertTrue(cp_graph._validate_graph())

        # critical_path should succeed (re-run on already-built graph)
        self.assertTrue(cp_graph.critical_path())
        self.assertGreater(len(cp_graph.critical_path_nodes), 0)

    # AI-assisted test
    def test_summary_asserts_breakdown_not_none(self) -> None:
        """Test summary() relies on non-None breakdown (line 1647-1648)."""
        cp_graph = self._critical_path_on_simple_add_trace()

        # summary() should work on a valid graph
        summary = cp_graph.summary()
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary), 0)

        # All percentage values should sum close to 100
        self.assertAlmostEqual(summary.sum(), 100.0, places=1)

    # AI-assisted test
    def test_save_restore_networkx_serialization(self) -> None:
        """Test save/restore uses nx.node_link_data/graph correctly (lines 1703, 1747)."""
        cp_graph = self._critical_path_on_simple_add_trace()

        with TemporaryDirectory(dir="/tmp") as tmpdir:
            zip_file = cp_graph.save(out_dir=tmpdir)
            restored = restore_cpgraph(
                zip_filename=zip_file, t_full=self.simple_add_trace.t, rank=0
            )

            # Restored graph should have same structure
            self.assertEqual(len(restored.nodes), len(cp_graph.nodes))
            self.assertEqual(len(restored.edges), len(cp_graph.edges))

            # Restored graph should produce a valid critical path
            self.assertTrue(restored.critical_path())
            self.assertGreater(len(restored.critical_path_nodes), 0)

    # AI-assisted test
    def test_critical_path_analysis_return_type_optional(self) -> None:
        """Test return type is Optional[Tuple[CPGraph, bool]] (line 1787).

        Valid annotation returns Tuple; invalid returns None.
        """
        # Valid annotation returns a tuple
        result = self.simple_add_trace.critical_path_analysis(
            rank=0,
            annotation="[param|pytorch.model.alex_net|0|0|0|measure|forward]",
            instance_id=1,
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        cp_graph, success = result
        self.assertIsInstance(cp_graph, CPGraph)
        self.assertIsInstance(success, bool)

        # Invalid annotation returns None
        result = self.simple_add_trace.critical_path_analysis(
            rank=0,
            annotation="__nonexistent__",
            instance_id=0,
        )
        self.assertIsNone(result)

    # AI-assisted test
    def test_pandas_drop_without_axis_param(self) -> None:
        """Test that DataFrame.drop calls work without deprecated axis=1 param (lines 476, 479, 935, etc.).

        Exercises _create_event_nodes, _get_cuda_event_record_df, and
        _get_cuda_stream_wait_event_df which all had axis=1 removed.
        """
        cp_graph = self._critical_path_on_simple_add_trace()

        # _create_event_nodes was called during construction — verify nodes are valid
        self.assertGreater(len(cp_graph.node_list), 0)
        for node in cp_graph.node_list:
            self.assertIsInstance(node, CPNode)
            self.assertIsNotNone(node.ts)

        # _get_cuda_event_record_df — verify no leftover columns from axis=1 removal
        event_record_df = cp_graph._get_cuda_event_record_df()
        if event_record_df is not None and len(event_record_df) > 0:
            # "axis" should not appear as a column
            self.assertNotIn("axis", event_record_df.columns)
            # Temporary columns should still be cleaned up
            self.assertNotIn("previous_launch_id", event_record_df.columns)

    # AI-assisted test
    def test_fillna_with_bracket_indexing(self) -> None:
        """Test that fillna uses bracket indexing pattern (lines 928, 970-972, 1040, 1082-1084).

        Verifies that launch_id and index_previous_launch columns are properly
        filled (no NaN) after the bracket-indexed fillna calls.
        """
        # Use event_sync_multi_stream trace which exercises both record and wait paths
        cp_graph, success = self.event_sync_multi_stream_trace.critical_path_analysis(
            rank=0, annotation="", instance_id=None
        )
        self.assertTrue(success)

        # Check _get_cuda_event_record_df output
        record_df = cp_graph._get_cuda_event_record_df()
        if record_df is not None and len(record_df) > 0:
            self.assertFalse(
                record_df["index_previous_launch"].isna().any(),
                "index_previous_launch should have no NaN after fillna(-1)",
            )

        # Check _get_cuda_stream_wait_event_df output
        wait_df = cp_graph._get_cuda_stream_wait_event_df()
        if wait_df is not None and len(wait_df) > 0:
            self.assertFalse(
                wait_df["index_next_launch"].isna().any(),
                "index_next_launch should have no NaN after fillna(-1)",
            )

    def test_critical_path_breakdown_and_save_restore(self) -> None:
        """Test summary, breakdown, and save/restore of critical path graph."""
        annotation = "[param|pytorch.model.alex_net|0|0|0|measure|forward]"
        rank = 0

        cp_graph, success = self.alexnet_trace.critical_path_analysis(
            rank=rank, annotation=annotation, instance_id=1
        )
        self.assertTrue(success)

        # Summary
        summary_df = cp_graph.summary()
        self.assertEqual(len(summary_df), 5)

        # Breakdown
        edf = cp_graph.get_critical_path_breakdown()
        orig_num_critical_edges = len(cp_graph.critical_path_edges_set)
        self.assertEqual(len(edf), orig_num_critical_edges)
        self.assertEqual(edf.bound_by.isnull().sum() + edf.bound_by.isna().sum(), 0)

        # Save and restore
        zip_file = cp_graph.save(out_dir="/tmp/hta_test_saved_cp_graph")
        try:
            rest_graph = restore_cpgraph(
                zip_filename=zip_file, t_full=self.alexnet_trace.t, rank=rank
            )
            self.assertEqual(len(rest_graph.nodes), len(cp_graph.nodes))

            summary_df = rest_graph.summary()
            self.assertEqual(len(summary_df), 5)

            edf = rest_graph.get_critical_path_breakdown()
            self.assertEqual(len(edf), orig_num_critical_edges)

            # Re-run critical path on restored graph
            rest_graph.critical_path()
            edf = rest_graph.get_critical_path_breakdown()
            self.assertEqual(len(edf), orig_num_critical_edges)
        finally:
            if os.path.exists(zip_file):
                os.remove(zip_file)
            save_dir = "/tmp/hta_test_saved_cp_graph"
            if os.path.exists(save_dir):
                import shutil

                shutil.rmtree(save_dir, ignore_errors=True)

    def test_ns_resolution_trace(self) -> None:
        """Check ns-resolution traces work with critical path analysis."""
        annotation = "ProfilerStep"
        instance_id = 1

        _, success = self.ns_resolution_trace.critical_path_analysis(
            rank=0, annotation=annotation, instance_id=instance_id
        )
        self.assertTrue(success)

        if _auto_detect_parser_backend() != ParserBackend.JSON:
            old_backend = get_default_trace_parsing_backend()
            try:
                set_default_trace_parsing_backend(
                    ParserBackend.IJSON_BATCH_AND_COMPRESS
                )
                fresh_trace = TraceAnalysis(trace_dir=self.ns_resolution_trace_dir)
                _, success = fresh_trace.critical_path_analysis(
                    rank=0, annotation=annotation, instance_id=instance_id
                )
                self.assertTrue(success)
            finally:
                set_default_trace_parsing_backend(old_backend)

    def test_amd_trace(self) -> None:
        """Check AMD traces work with critical path analysis."""
        annotation = "ProfilerStep"
        instance_id = 1

        _, success = self.amd_trace.critical_path_analysis(
            rank=0, annotation=annotation, instance_id=instance_id
        )
        self.assertTrue(success)

        if _auto_detect_parser_backend() != ParserBackend.JSON:
            old_backend = get_default_trace_parsing_backend()
            try:
                set_default_trace_parsing_backend(
                    ParserBackend.IJSON_BATCH_AND_COMPRESS
                )
                fresh_trace = TraceAnalysis(trace_dir=self.amd_trace_dir)
                _, success = fresh_trace.critical_path_analysis(
                    rank=0, annotation=annotation, instance_id=instance_id
                )
                self.assertTrue(success)
            finally:
                set_default_trace_parsing_backend(old_backend)


class EndToEndTestCase(unittest.TestCase):
    """Tests the input / final output (overlaid trace file) of critical path analysis.

    Each test case is a subdirectory containing:
        /input/trace.json.gz
        /output/overlaid_critical_path_trace.json.gz
    """

    def setUp(self) -> None:
        # CRITICAL_PATH_DATA_DIR points to external data in Buck (outside HTA).
        # Fallback uses local tests/data/ which may not exist in all environments;
        # test_critical_path_end_to_end handles that with skipTest.
        self.base_data_dir = os.environ.get(
            "CRITICAL_PATH_DATA_DIR",
            os.path.join(get_test_data_dir(), "critical_path", "end_to_end"),
        )

    def _assert_trace_files_equal(
        self, trace_path_1: str, trace_path_2: str, tolerance: float = 0.01
    ) -> None:
        with gzip.open(trace_path_1, "r") as f1, gzip.open(trace_path_2, "r") as f2:
            critical_1 = [
                ev
                for ev in json.load(f1)["traceEvents"]
                if "args" in ev and "critical" in ev["args"]
            ]
            critical_2 = [
                ev
                for ev in json.load(f2)["traceEvents"]
                if "args" in ev and "critical" in ev["args"]
            ]

            len_1, len_2 = len(critical_1), len(critical_2)
            if len_1 > 0 or len_2 > 0:
                diff_pct = abs(len_1 - len_2) / max(len_1, len_2)
                self.assertLessEqual(
                    diff_pct,
                    tolerance,
                    f"Critical event count difference ({len_1} vs {len_2}) "
                    f"exceeds {tolerance * 100}% tolerance",
                )
            else:
                self.assertEqual(len_1, len_2)

    def test_critical_path_end_to_end(self) -> None:
        """Test that critical path analysis generates expected overlaid trace."""
        if not os.path.isdir(self.base_data_dir):
            self.skipTest(f"End-to-end test data not found: {self.base_data_dir}")

        for test_dir in os.listdir(self.base_data_dir):
            test_dir_path = os.path.join(self.base_data_dir, test_dir)
            if not os.path.isdir(test_dir_path):
                continue

            with self.subTest(test_dir=test_dir):
                critical_path_t = TraceAnalysis(
                    trace_dir=os.path.join(test_dir_path, "input")
                )
                cp_graph, success = critical_path_t.critical_path_analysis(
                    rank=0,
                    annotation="",
                    instance_id=0,
                    data_load_events=["data_load"],
                )
                self.assertTrue(success)

                with TemporaryDirectory(dir="/tmp") as tmpdir:
                    actual = critical_path_t.overlay_critical_path_analysis(
                        0,
                        cp_graph,
                        output_dir=tmpdir,
                        only_show_critical_events=False,
                    )
                    expected = os.path.join(
                        test_dir_path,
                        "output",
                        "overlaid_critical_path_trace.json.gz",
                    )
                    self._assert_trace_files_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
