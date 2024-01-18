import gzip
import json
import os
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

from hta.analyzers.critical_path_analysis import CPEdge, CPEdgeType
from hta.trace_analysis import TraceAnalysis


class CriticalPathAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.base_data_dir = str(Path(__file__).parent.parent.joinpath("tests/data"))

    def test_critical_path_analysis(self):
        critical_path_trace_dir: str = os.path.join(
            self.base_data_dir, "critical_path/simple_add"
        )
        critical_path_t = TraceAnalysis(trace_dir=critical_path_trace_dir)

        annotation = "[param|pytorch.model.alex_net|0|0|0|measure|forward]"
        instance_id = 1
        cp_graph, success = critical_path_t.critical_path_analysis(
            rank=0, annotation=annotation, instance_id=instance_id
        )
        self.assertTrue(success)

        trace_df = critical_path_t.t.get_trace(0)
        sym_table = critical_path_t.t.symbol_table.get_sym_table()

        def get_node_name(nid):
            if nid < 0:
                return "ROOT"
            trace_entry = trace_df.loc[nid].to_dict()
            return sym_table[int(trace_entry["name"])]

        # Check the graph construction for the aten::relu_ operator
        # There are 3 stacked operators/runtime events here;
        #  aten::relu_-------------
        #    aten::clamp_min_----
        #      cudaLaunchKernel
        # quick sanity check that we are looking at right events

        relu_idx = 286
        clamp_min_idx = 287
        cuda_launch_idx = 1005
        self.assertEqual(get_node_name(relu_idx), "aten::relu_")
        self.assertEqual(get_node_name(clamp_min_idx), "aten::clamp_min_")
        self.assertEqual(get_node_name(cuda_launch_idx), "cudaLaunchKernel")

        expected_node_ids = [(32, 33), (34, 35), (36, 37)]

        def check_nodes(ev_idx: int) -> Tuple[int, int]:
            start_node, end_node = cp_graph.get_nodes_for_event(ev_idx)
            self.assertTrue(start_node.is_start)
            self.assertFalse(end_node.is_start)
            return start_node.idx, end_node.idx

        self.assertEqual(check_nodes(relu_idx), expected_node_ids[0])
        self.assertEqual(check_nodes(clamp_min_idx), expected_node_ids[1])
        self.assertEqual(check_nodes(cuda_launch_idx), expected_node_ids[2])

        def check_edge(start_nid: int, end_nid: int, weight: int, attr_ev: int) -> None:
            """Arga = start node id, end node id, weight of edge, ev id to attirbute to"""
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

        # expected_node_ids[...][0] is the start node, and [...][1] is the end node.
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

        # Make sure edges show up when we reverse look up attributed edges from event id.
        #  --------------aten::relu_---------------
        #   <e1>|-------aten::clamp_min_------|<e5>
        #       <-e2->|cudaLaunchKernel|<-e4->
        #             | <-----e3-----> |

        self.assertEqual(
            set(cp_graph.get_edges_attributed_to_event(relu_idx)), {e1, e5}
        )
        self.assertEqual(
            set(cp_graph.get_edges_attributed_to_event(clamp_min_idx)), {e2, e4}
        )
        self.assertEqual(
            set(cp_graph.get_edges_attributed_to_event(cuda_launch_idx)), {e3}
        )

        # Check kernel launch and kernel-kernel delays
        # fft kernel correlation ID 5597
        fft_kernel_idx = 1051
        fft_runtime_idx = trace_df.index_correlation.loc[fft_kernel_idx]
        self.assertEqual(
            get_node_name(fft_kernel_idx),
            "void fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int,"
            " int, int, int, cudnn::reduced_divisor, bool, int2, int, int)",
        )
        kstart, kend = cp_graph.get_nodes_for_event(fft_kernel_idx)
        rstart, _ = cp_graph.get_nodes_for_event(fft_runtime_idx)

        kernel_launch_edge = cp_graph.edges[rstart.idx, kstart.idx]["object"]
        self.assertEqual(
            kernel_launch_edge,
            CPEdge(
                begin=rstart.idx,
                end=kstart.idx,
                weight=27,
                type=CPEdgeType.KERNEL_LAUNCH_DELAY,
            ),
        )

        # next kernel is ampere_sgemm correlation ID 5604
        ampere_kernel_idx = 1067
        k2start, _ = cp_graph.get_nodes_for_event(ampere_kernel_idx)
        kernel_kernel_edge = cp_graph.edges[kend.idx, k2start.idx]["object"]
        self.assertEqual(
            kernel_kernel_edge,
            CPEdge(
                begin=kend.idx,
                end=k2start.idx,
                weight=7,
                type=CPEdgeType.KERNEL_KERNEL_DELAY,
            ),
        )

        # Check device sync event
        epilogue_kernel_idx = 1275
        cuda_device_sync_idx = 1281

        _, k3end = cp_graph.get_nodes_for_event(epilogue_kernel_idx)
        _, syncend = cp_graph.get_nodes_for_event(cuda_device_sync_idx)
        device_sync_edge = cp_graph.edges[k3end.idx, syncend.idx]["object"]
        self.assertEqual(
            device_sync_edge,
            CPEdge(
                begin=k3end.idx,
                end=syncend.idx,
                weight=0,
                type=CPEdgeType.SYNC_DEPENDENCY,
            ),
        )

        # Check that all edges have event attribution
        for (u, v) in cp_graph.edges:
            e = cp_graph.edges[u, v]["object"]
            if e.type == CPEdgeType.OPERATOR_KERNEL:
                self.assertTrue(
                    (u, v) in cp_graph.edge_to_event_map,
                    msg=f"edge = {(u,v)}, obj = {e}",
                )
                self.assertTrue(cp_graph.get_event_attribution_for_edge(e))
            else:
                self.assertEqual(cp_graph.get_event_attribution_for_edge(e), None)

        # Make sure critical path is as expected
        self.assertEqual(len(cp_graph.critical_path_nodes), 315)

        # check overlaid trace matches up correctly
        with TemporaryDirectory(dir="/tmp") as tmpdir:
            overlaid_trace = critical_path_t.overlay_critical_path_analysis(
                0,
                cp_graph,
                output_dir=tmpdir,
                only_show_critical_events=False,
                show_all_edges=True,
            )
            self.assertTrue("overlaid_critical_path_" in overlaid_trace)

            with gzip.open(overlaid_trace, "r") as ovf:
                trace_events = json.load(ovf)["traceEvents"]
                marked_critical_events = sum(
                    e["args"].get("critical", 0)
                    for e in trace_events
                    if "args" in e and e["ph"] == "X"
                )
                self.assertEqual(marked_critical_events, 159)
                self.assertEqual(
                    marked_critical_events, len(cp_graph.critical_path_events_set)
                )

                trace_edge_counts = Counter(
                    e["cat"]
                    for e in trace_events
                    if "critical_path" in e.get("cat", "")
                )
                cpgraph_edge_counts = Counter(
                    cp_graph.edges[u, v]["object"].type for (u, v) in cp_graph.edges
                )

                for etype in CPEdgeType:
                    self.assertEqual(
                        trace_edge_counts[etype.value],
                        cpgraph_edge_counts[etype] * 2,
                    )
                marked_critical_edges = sum(
                    e["args"].get("critical", 0)
                    for e in trace_events
                    if "args" in e and e["ph"] == "f"
                )
                self.assertEqual(marked_critical_edges, 314)
                self.assertEqual(
                    marked_critical_edges,
                    len(cp_graph.critical_path_edges_set),
                )

        with TemporaryDirectory(dir="/tmp") as tmpdir:
            overlaid_trace = critical_path_t.overlay_critical_path_analysis(
                0,
                cp_graph,
                output_dir=tmpdir,
                only_show_critical_events=True,
                show_all_edges=True,  # this should be overriden to false
            )
            self.assertTrue("overlaid_critical_path_" in overlaid_trace)
            with gzip.open(overlaid_trace, "r") as ovf:
                trace_events = json.load(ovf)["traceEvents"]
                events_to_check = [
                    e
                    for e in trace_events
                    if e["ph"] == "X"
                    and e.get("cat", "") not in ["user_annotation", "python_function"]
                ]
                num_events_to_check = len(events_to_check)
                marked_critical_events = sum(
                    e["args"].get("critical", 0) for e in events_to_check if "args" in e
                )
                self.assertEqual(marked_critical_events, 159)
                # Only critical events are written out to the trace
                self.assertEqual(marked_critical_events, num_events_to_check)

                trace_edge_counts = Counter(
                    "critical" if e["args"].get("critical", 0) else "non_critical"
                    for e in trace_events
                    if "critical_path" in e.get("cat", "")
                )
                # Only critical edges are shown
                self.assertEqual(
                    trace_edge_counts["critical"], sum(trace_edge_counts.values())
                )

        # AlexNet has inter-stream synchronization using CUDA Events
        critical_path_trace_dir2: str = os.path.join(
            self.base_data_dir, "critical_path/alexnet"
        )
        critical_path_t = TraceAnalysis(trace_dir=critical_path_trace_dir2)

        trace_df = critical_path_t.t.get_trace(0)
        sym_table = critical_path_t.t.symbol_table.get_sym_table()

        cp_graph, success = critical_path_t.critical_path_analysis(
            rank=0, annotation=annotation, instance_id=instance_id
        )
        self.assertTrue(success)

        # Make sure critical path is as expected
        self.assertEqual(len(cp_graph.critical_path_nodes), 149)

        # Check GPU->GPU sync edge between kernel on stream 20 -> stream 7
        # In the trace in tests/data/critical_path/alexnet look for correlation
        # IDs 5606 and 5629
        fft_src_kernel_idx = 1109
        self.assertEqual(
            get_node_name(fft_src_kernel_idx),
            "void fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, "
            "int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)",
        )
        _, fft_kernel_end = cp_graph.get_nodes_for_event(fft_src_kernel_idx)

        elwise_dest_kernel_idx = 1161
        elwise_kernel_start, _ = cp_graph.get_nodes_for_event(elwise_dest_kernel_idx)

        gpu_gpu_sync_edge = cp_graph.edges[fft_kernel_end.idx, elwise_kernel_start.idx][
            "object"
        ]
        self.assertEqual(
            gpu_gpu_sync_edge,
            CPEdge(
                begin=fft_kernel_end.idx,
                end=elwise_kernel_start.idx,
                weight=0,
                type=CPEdgeType.SYNC_DEPENDENCY,
            ),
        )


if __name__ == "__main__":
    unittest.main()
