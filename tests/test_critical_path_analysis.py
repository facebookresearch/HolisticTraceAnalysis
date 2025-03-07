import gzip
import json
import os
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import hta.configs.env_options as hta_options
from hta.analyzers.critical_path_analysis import (
    CPEdge,
    CPEdgeType,
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


class CriticalPathAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        os.environ["CRITICAL_PATH_ADD_ZERO_WEIGHT_LAUNCH_EDGE"] = "1"
        self.base_data_dir = str(Path(__file__).parent.parent.joinpath("tests/data"))
        critical_path_trace_dir: str = os.path.join(
            self.base_data_dir, "critical_path/simple_add"
        )
        self.simple_add_trace = TraceAnalysis(trace_dir=critical_path_trace_dir)
        critical_path_trace_dir2: str = os.path.join(
            self.base_data_dir, "critical_path/alexnet"
        )
        self.alexnet_trace = TraceAnalysis(trace_dir=critical_path_trace_dir2)
        critical_path_trace_dir3: str = os.path.join(
            self.base_data_dir, "critical_path/cuda_event_sync"
        )
        self.event_sync_trace = TraceAnalysis(trace_dir=critical_path_trace_dir3)
        critical_path_trace_dir4: str = os.path.join(
            self.base_data_dir, "critical_path/cuda_event_sync_multi_stream"
        )
        self.event_sync_multi_stream_trace = TraceAnalysis(
            trace_dir=critical_path_trace_dir4
        )
        critical_path_trace_dir5: str = os.path.join(
            self.base_data_dir, "ns_resolution_trace"
        )
        self.ns_resolution_trace_dir = critical_path_trace_dir5
        self.ns_resolution_trace = TraceAnalysis(trace_dir=critical_path_trace_dir5)
        self.amd_trace_dir: str = os.path.join(self.base_data_dir, "amd_trace")
        self.amd_trace = TraceAnalysis(trace_dir=self.amd_trace_dir)

    def test_critical_path_analysis(self):
        critical_path_t = self.simple_add_trace

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

        expected_node_ids = [(57, 62), (58, 61), (59, 60)]

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

        # also check for 0 duration causal launch edge
        ampere_runtime_idx = trace_df.index_correlation.loc[ampere_kernel_idx]
        r2start, _ = cp_graph.get_nodes_for_event(ampere_runtime_idx)
        zero_weight_kernel_launch_edge = cp_graph.edges[r2start.idx, k2start.idx][
            "object"
        ]
        self.assertEqual(
            zero_weight_kernel_launch_edge,
            CPEdge(
                begin=r2start.idx,
                end=k2start.idx,
                weight=0,
                type=CPEdgeType.KERNEL_LAUNCH_DELAY,
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
        for u, v in cp_graph.edges:
            e = cp_graph.edges[u, v]["object"]
            if e.type in {CPEdgeType.OPERATOR_KERNEL, CPEdgeType.KERNEL_KERNEL_DELAY}:
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

                cpgraph_edges = (
                    cp_graph.edges[u, v]["object"] for (u, v) in cp_graph.edges
                )
                if not hta_options.critical_path_show_zero_weight_launch_edges():
                    cpgraph_edges = filter(
                        lambda e: not CriticalPathAnalysis._is_zero_weight_launch_edge(
                            e
                        ),
                        cpgraph_edges,
                    )
                cpgraph_edge_counts = Counter(e.type for e in cpgraph_edges)

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

        # is it resilient to missing overlaid path?
        tmpdir = "/tmp/path_does_not_exist"
        overlaid_trace = critical_path_t.overlay_critical_path_analysis(
            0,
            cp_graph,
            output_dir=tmpdir,
            only_show_critical_events=False,
            show_all_edges=True,
        )
        self.assertTrue(os.path.exists(tmpdir))
        os.remove(overlaid_trace)
        os.removedirs(tmpdir)

        # AlexNet has inter-stream synchronization using CUDA Events
        critical_path_t = self.alexnet_trace

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

    def test_critical_path_analysis_event_sync(self):
        """Checks cudaEventSync() synchronization edges"""
        critical_path_t = self.event_sync_trace

        annotation = "ProfilerStep"
        instance_id = 0
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

        cuda_kernel_idx = 33
        cuda_event_sync_idx = 41
        cuda_event_query_idx = 45
        cuda_device_sync_idx = 51

        self.assertEqual(
            get_node_name(cuda_kernel_idx),
            "at::cuda::(anonymous namespace)::spin_kernel(long)",
        )
        self.assertEqual(get_node_name(cuda_event_sync_idx), "cudaEventSynchronize")
        self.assertEqual(get_node_name(cuda_event_query_idx), "cudaEventQuery")
        self.assertEqual(get_node_name(cuda_device_sync_idx), "cudaDeviceSynchronize")

        # There are two GPU -> CPU dependencies in this trace
        # both start at a CUDA kernel that precedes the CUDA event and ends in trace.
        _, cuda_kernel_end = cp_graph.get_nodes_for_event(cuda_kernel_idx)
        _, cuda_event_sync_end = cp_graph.get_nodes_for_event(cuda_event_sync_idx)
        _, cuda_event_query_end = cp_graph.get_nodes_for_event(cuda_event_query_idx)
        _, cuda_device_sync_end = cp_graph.get_nodes_for_event(cuda_device_sync_idx)

        def check_sync_edge(start_node_idx: int, end_node_idx: int) -> None:
            gpu_cpu_sync_edge = cp_graph.edges[start_node_idx, end_node_idx]["object"]
            self.assertEqual(
                gpu_cpu_sync_edge,
                CPEdge(
                    begin=start_node_idx,
                    end=end_node_idx,
                    weight=0,
                    type=CPEdgeType.SYNC_DEPENDENCY,
                ),
            )

        check_sync_edge(cuda_kernel_end.idx, cuda_event_sync_end.idx)
        check_sync_edge(cuda_kernel_end.idx, cuda_device_sync_end.idx)

    def test_critical_path_analysis_event_sync_multistream(self):
        """Checks cuda Stream wait event across multiple stream"""
        critical_path_t = self.event_sync_multi_stream_trace

        annotation = ""
        instance_id = None
        cp_graph, success = critical_path_t.critical_path_analysis(
            rank=0, annotation=annotation, instance_id=instance_id
        )
        self.assertTrue(success)

        # The trace contains the following
        # 1. GPU kernel 1 (correlation = 27)  stream = 20
        # 2. GPU kernel 2 (correlation = 57)  stream = 28
        # 3. Record cuda event on stream 20
        # 4. Wait event on stream 20
        # 5. GPU kernel 3 (correlation = zz)  stream = 24

        # For step (3) we want the algorithm to indicate previous launch to
        # be the last kernel on stream 20 and not stream 24.
        correlation_kernel1 = 27
        correlation_event_record = 1385

        event_record_df = cp_graph._get_cuda_event_record_df()
        event_records = event_record_df[
            ["correlation", "correlation_launch_event"]
        ].to_dict(orient="records")

        self.assertEqual(len(event_records), 3)
        self.assertEqual(event_records[2]["correlation"], correlation_event_record)
        # The cudaEventRecord should sync back to GPU kernel 1 and not kernel 2
        self.assertEqual(
            event_records[2]["correlation_launch_event"], correlation_kernel1
        )

        # Check that sync edge is added
        kernel1_idx = 24  # ampere_sgemm_128x64_nn
        kernel3_idx = 84  # Memset (Device)
        self.assertEqual(
            cp_graph._get_node_name(kernel1_idx),
            "ampere_sgemm_128x64_nn",
        )
        self.assertEqual(
            cp_graph._get_node_name(kernel3_idx),
            "Memset (Device)",
        )
        _, kernel1_end = cp_graph.get_nodes_for_event(kernel1_idx)
        kernel3_start, _ = cp_graph.get_nodes_for_event(kernel3_idx)

        inter_kernel_sync_edge = cp_graph.edges[kernel1_end.idx, kernel3_start.idx][
            "object"
        ]
        self.assertEqual(
            inter_kernel_sync_edge,
            CPEdge(
                begin=kernel1_end.idx,
                end=kernel3_start.idx,
                weight=0,
                type=CPEdgeType.SYNC_DEPENDENCY,
            ),
        )

    def test_critical_path_breakdown_and_save_restore(self):
        annotation = "[param|pytorch.model.alex_net|0|0|0|measure|forward]"
        instance_id = 1
        rank = 0

        critical_path_t = self.alexnet_trace
        cp_graph, success = critical_path_t.critical_path_analysis(
            rank=rank, annotation=annotation, instance_id=instance_id
        )
        self.assertTrue(success)

        # Call the summary function
        summary_df = cp_graph.summary()
        self.assertEqual(len(summary_df), 5)

        # Check full path breakdown
        edf = cp_graph.get_critical_path_breakdown()
        self.assertEqual(len(edf), len(cp_graph.critical_path_edges_set))
        orig_num_critical_edges = len(cp_graph.critical_path_edges_set)

        # Check the boundby column is populated
        self.assertEqual(edf.bound_by.isnull().sum() + edf.bound_by.isna().sum(), 0)

        # Check Save and Restore functionality
        zip_file = cp_graph.save(out_dir="/tmp/my_saved_cp_graph")

        rest_graph = restore_cpgraph(
            zip_filename=zip_file, t_full=critical_path_t.t, rank=rank
        )
        self.assertEqual(len(rest_graph.nodes), len(cp_graph.nodes))

        # check restored cp_graph
        summary_df = rest_graph.summary()
        self.assertEqual(len(summary_df), 5)

        edf = rest_graph.get_critical_path_breakdown()
        self.assertEqual(len(edf), orig_num_critical_edges)
        self.assertEqual(
            len(rest_graph.critical_path_edges_set), orig_num_critical_edges
        )

        # run critical path algorithm again
        rest_graph.critical_path()

        edf = rest_graph.get_critical_path_breakdown()
        self.assertEqual(len(edf), orig_num_critical_edges)
        self.assertEqual(
            len(rest_graph.critical_path_edges_set), orig_num_critical_edges
        )

    def test_ns_resolution_trace(self):
        """New Kineto feature enables sub microsecond timstamp and duration,
        check that these traces are compatible with Critical Path Analysis"""
        annotation = "ProfilerStep"
        instance_id = 1

        def test():
            cp_graph, success = critical_path_t.critical_path_analysis(
                rank=0, annotation=annotation, instance_id=instance_id
            )
            self.assertTrue(success)

        critical_path_t = self.ns_resolution_trace
        test()

        if _auto_detect_parser_backend() != ParserBackend.JSON:
            old_backend = get_default_trace_parsing_backend()
            set_default_trace_parsing_backend(ParserBackend.IJSON_BATCH_AND_COMPRESS)

            critical_path_t = TraceAnalysis(trace_dir=self.ns_resolution_trace_dir)
            test()

            set_default_trace_parsing_backend(old_backend)

    def test_amd_trace(self):
        """Check that AMD traces are compatible with Critical Path Analysis"""
        annotation = "ProfilerStep"
        instance_id = 1

        def test():
            cp_graph, success = critical_path_t.critical_path_analysis(
                rank=0, annotation=annotation, instance_id=instance_id
            )
            self.assertTrue(success)

        critical_path_t = self.amd_trace
        test()

        if _auto_detect_parser_backend() != ParserBackend.JSON:
            old_backend = get_default_trace_parsing_backend()
            set_default_trace_parsing_backend(ParserBackend.IJSON_BATCH_AND_COMPRESS)

            critical_path_t = TraceAnalysis(trace_dir=self.amd_trace_dir)
            test()

            set_default_trace_parsing_backend(old_backend)


if __name__ == "__main__":
    unittest.main()
