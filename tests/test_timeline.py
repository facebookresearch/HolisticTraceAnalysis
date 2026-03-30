import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional
from unittest.mock import Mock, patch

import hta
import pandas as pd
from hta.common.timeline import (
    _simplify_name,
    align_module_with_kernels,
    AnnotationStreamID,
    get_fig_name_from_fig_title,
    get_fig_title_from_fig_name,
    plot_events_timeline,
    PlotFormat,
    prepare_timeline_events,
    REQUIRED_COLUMNS_FOR_CPU_GPU_ALIGNMENT,
    Timeline,
    TimelinePlotSetting,
)
from hta.common.trace import Trace
from hta.common.trace_call_graph import CallGraph
from hta.common.trace_filter import CPUOperatorFilter, GPUKernelFilter
from hta.common.trace_symbol_table import TraceSymbolTable

_MODULE = "hta.common.timeline"


class TestTimelineAnalysis(unittest.TestCase):
    base_data_dir = str(Path(hta.__file__).parent.parent.joinpath("tests/data"))
    trace_path: str = os.path.join(base_data_dir, "timeline_analysis")
    t = Trace(trace_dir=trace_path)
    t.parse_traces()
    t.decode_symbol_ids(use_shorten_name=False)
    cg = CallGraph(t, ranks=[0])

    def setUp(self) -> None:
        self.trace_path: str = TestTimelineAnalysis.trace_path
        self.t = TestTimelineAnalysis.t
        self.df = self.t.get_trace(0)

    @patch(f"{_MODULE}.px.timeline")
    def test_plot_timeline(self, mock_timeline: Mock) -> None:
        @dataclass
        class TC:
            in_df: pd.DataFrame
            figure_name: str
            expected_task_label_re: str

        @dataclass
        class DataItem:
            x: int
            name: str

        def fake_traces(
            in_df: pd.DataFrame, trace_type: str, n_ranks: int, n_traces
        ) -> pd.DataFrame:
            result_df = in_df.copy()
            if trace_type == "multi_ranks":
                result_df["rank"] = in_df["index"].apply(lambda i: i % n_ranks)
            elif trace_type == "multi_traces":
                result_df["trace"] = in_df["index"].apply(
                    lambda i: i % n_traces * n_ranks
                )
                result_df["rank"] = in_df["index"].apply(lambda i: i % n_ranks)
            return result_df

        def fake_multi_traces(in_df: pd.DataFrame, n_traces: int) -> pd.DataFrame:
            result_df = in_df.copy()
            result_df["trace"] = in_df["index"].apply(lambda i: i % n_traces)
            return result_df

        test_cases: List[TC] = [
            TC(self.df, "one_rank", r"(cpu level)|(gpu stream)"),
            TC(GPUKernelFilter()(self.df), "gpu", r"gpu stream"),
            TC(CPUOperatorFilter()(self.df), "cpu", r"cpu level"),
            TC(fake_traces(self.df, "multi_ranks", 3, 1), "multi_ranks", r"rank \d+"),
            TC(
                fake_traces(self.df, "multi_traces", 2, 2),
                "multi_traces",
                r"trace \w+ rank \d+",
            ),
        ]
        output_dir = self.trace_path
        plot_setting = TimelinePlotSetting(task_height=50, plot_format=PlotFormat.File)

        mock_figure = Mock()
        mock_figure.show.side_effects = [None for i in range(len(test_cases))]
        mock_figure.update_layout.side_effects = [None for i in range(len(test_cases))]
        mock_figure.write_html.side_effects = [None for i in range(len(test_cases))]
        mock_figure.write_image.side_effects = [None for i in range(len(test_cases))]
        mock_figure.data = [Mock(x=i, name=str(i)) for i in range(10)]
        mock_timeline.return_value = mock_figure

        for i, tc in enumerate(test_cases):
            events = tc.in_df
            timeline_events = prepare_timeline_events(
                events, symbol_table=self.t.symbol_table, setting=plot_setting
            )

            self.assertTrue(
                all(
                    timeline_events[plot_setting.task_column].str.match(
                        tc.expected_task_label_re
                    )
                )
            )

            output_file = os.path.join(output_dir, tc.figure_name + ".html")

            _ = plot_events_timeline(
                title=get_fig_title_from_fig_name(tc.figure_name),
                events=timeline_events,
                setting=plot_setting,
                output_image_path=output_file,
            )
            self.assertEqual(i + 1, mock_timeline.call_count)

    def test_align_module_with_kernels(self) -> None:
        @dataclass
        class _TCase:
            in_df: pd.DataFrame
            module_list: List[str]
            sym_table: Optional[TraceSymbolTable]
            include_original_cpu_events: bool
            expected_type_error: bool
            expected_num_annotations: int
            expected_total_num_events: int

        test_cases = [
            _TCase(
                self.df,
                ["## forward ##"],
                self.t.symbol_table,
                False,
                False,
                2,
                1206,
            ),
            _TCase(self.df, [r"## forward ##"], None, False, False, 2, 1206),
            _TCase(
                self.df[["index", "name", "stream"]].copy(),
                [r"## forward ##"],
                None,
                False,
                True,
                -1,
                -1,
            ),
            _TCase(
                self.df[list(REQUIRED_COLUMNS_FOR_CPU_GPU_ALIGNMENT)].copy(),
                [r"## forward ##"],
                None,
                False,
                True,
                -1,
                -1,
            ),
        ]

        for tc in test_cases:
            if tc.expected_type_error:
                with self.assertRaises(ValueError):
                    align_module_with_kernels(
                        tc.in_df,
                        tc.module_list,
                        sym_table=tc.sym_table,
                        include_original_cpu_events=tc.include_original_cpu_events,
                    )
            else:
                aligned_events = align_module_with_kernels(
                    tc.in_df,
                    tc.module_list,
                    sym_table=tc.sym_table,
                    include_original_cpu_events=tc.include_original_cpu_events,
                )

                annotations = aligned_events.loc[
                    aligned_events["stream"].eq(AnnotationStreamID)
                ]
                self.assertEqual(tc.expected_total_num_events, aligned_events.shape[0])
                self.assertEqual(tc.expected_num_annotations, annotations.shape[0])
                self.assertTrue(
                    annotations.loc[annotations["num_kernels"].lt(0)].shape[0] == 0
                )
                pd.testing.assert_series_equal(
                    annotations["first_kernel_start"].astype(int),
                    annotations["ts"].astype(int),
                    check_names=False,
                )

    def test_figure_title_helpers(self) -> None:
        fig_name = "my_figure_name"
        expected_title = "My Figure Name"
        result_title = get_fig_title_from_fig_name(fig_name)
        self.assertEqual(result_title, expected_title)

        fig_name = ""
        expected_title = "Untitled Figure"
        result_title = get_fig_title_from_fig_name(fig_name)
        self.assertEqual(result_title, expected_title)

        title = "My Figure Title"
        expected_fig_name = "my_figure_title"
        result_fig_name = get_fig_name_from_fig_title(title)
        self.assertEqual(result_fig_name, expected_fig_name)

        title = ""
        expected_fig_name = "untitled_figure"
        result_fig_name = get_fig_name_from_fig_title(title)
        self.assertEqual(result_fig_name, expected_fig_name)

    def test_simplify_name(self) -> None:
        class _TCase(NamedTuple):
            long_name: str
            short_name: str

        testCases = [
            _TCase("aten::t", "aten::t"),
            _TCase("__exit__", "__exit__"),
            _TCase("## qat ##", "## qat ##"),
            _TCase("<forward op>", "<forward op>"),
            _TCase("Memcpy HtoD (Pinned -> Device)", "Memcpy HtoD (Pinned -> Device)"),
            _TCase("Memset (Device)", "Memset (Device)"),
            _TCase(
                "autograd::engine::evaluate_function: TBackward0",
                "autograd::TBackward0",
            ),
            _TCase(
                "void permute_pooled_embs_kernel<float>(float const*, long const*, long const*, long const*, float*, long, long, long)",
                "permute_pooled_embs_kernel<float>",
            ),
            _TCase(
                "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x128_16x5_nt_align4>(cutlass_80_tensorop_s1688gemm_128x128_16x5_nt_align4::Params)",
                "cutlass::Kernel<cutlass_80_tensorop_s1688g...",
            ),
            _TCase("<forward op>", "<forward op>"),
            _TCase(
                "void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy<long, float, int>::Policy800, false, long, int>(int*, long const*, int, int, int)",
                "cub::DeviceRadixSortHistogramKernel<cub::D...",
            ),
        ]
        for tc in testCases:
            got = _simplify_name(tc.long_name, max_length=45)
            self.assertEqual(
                tc.short_name,
                got,
                f"name={tc.long_name} expect: {tc.short_name}, got: {got}.",
            )

    @patch(f"{_MODULE}.plot_events_timeline")
    def test_timeline_class(self, mock_plot: Mock) -> None:
        df = self.t.get_trace(0)
        save_path = os.path.join(self.trace_path, "timeline_class.html")
        tl = Timeline(df, self.t.symbol_table, filter_func=GPUKernelFilter())
        tl.setting.plot_format = PlotFormat.File

        tl.prepare()
        tl.plot("timeline class", save_path=save_path)

        self.assertFalse(tl.is_timeline_events(df, tl.setting))
        self.assertTrue(tl.is_timeline_events(tl.timeline_events, tl.setting))
        mock_plot.assert_called_once()
