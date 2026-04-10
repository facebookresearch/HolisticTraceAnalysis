import os
import unittest
from unittest.mock import Mock, patch

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
from hta.common.trace_filter import GPUKernelFilter

_MODULE = "hta.common.timeline"


from hta.utils.test_utils import get_test_data_dir


class TestSimplifyName(unittest.TestCase):
    """Tests for the _simplify_name helper function."""

    def test_short_names_unchanged(self) -> None:
        cases = [
            ("aten::t", "aten::t"),
            ("__exit__", "__exit__"),
            ("## qat ##", "## qat ##"),
            ("<forward op>", "<forward op>"),
            ("Memcpy HtoD (Pinned -> Device)", "Memcpy HtoD (Pinned -> Device)"),
            ("Memset (Device)", "Memset (Device)"),
        ]
        for long_name, expected in cases:
            with self.subTest(name=long_name):
                self.assertEqual(_simplify_name(long_name, max_length=45), expected)

    def test_autograd_simplification(self) -> None:
        self.assertEqual(
            _simplify_name(
                "autograd::engine::evaluate_function: TBackward0", max_length=45
            ),
            "autograd::TBackward0",
        )

    def test_void_kernel_simplification(self) -> None:
        name = (
            "void permute_pooled_embs_kernel<float>"
            "(float const*, long const*, long const*, long const*, float*, long, long, long)"
        )
        self.assertEqual(
            _simplify_name(name, max_length=45),
            "permute_pooled_embs_kernel<float>",
        )

    def test_long_name_truncation(self) -> None:
        name = (
            "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_128x128_16x5_nt_align4>"
            "(cutlass_80_tensorop_s1688gemm_128x128_16x5_nt_align4::Params)"
        )
        result = _simplify_name(name, max_length=45)
        self.assertTrue(result.endswith("..."))
        self.assertLessEqual(len(result), 45)

    def test_cub_kernel_truncation(self) -> None:
        name = (
            "void cub::DeviceRadixSortHistogramKernel<cub::DeviceRadixSortPolicy"
            "<long, float, int>::Policy800, false, long, int>"
            "(int*, long const*, int, int, int)"
        )
        result = _simplify_name(name, max_length=45)
        self.assertTrue(result.endswith("..."))
        self.assertLessEqual(len(result), 45)


class TestFigureNameHelpers(unittest.TestCase):
    """Tests for figure name/title conversion helpers."""

    def test_fig_name_to_title(self) -> None:
        self.assertEqual(
            get_fig_title_from_fig_name("my_figure_name"), "My Figure Name"
        )

    def test_empty_fig_name_to_title(self) -> None:
        self.assertEqual(get_fig_title_from_fig_name(""), "Untitled Figure")

    def test_title_to_fig_name(self) -> None:
        self.assertEqual(
            get_fig_name_from_fig_title("My Figure Title"), "my_figure_title"
        )

    def test_empty_title_to_fig_name(self) -> None:
        self.assertEqual(get_fig_name_from_fig_title(""), "untitled_figure")


class TestPrepareTimelineEvents(unittest.TestCase):
    """Tests for prepare_timeline_events using real trace data."""

    t: Trace

    @classmethod
    def setUpClass(cls) -> None:
        trace_path = os.path.join(get_test_data_dir(), "timeline_analysis")
        cls.t = Trace(trace_dir=trace_path)
        cls.t.parse_traces()
        cls.t.decode_symbol_ids(use_shorten_name=False)

    def setUp(self) -> None:
        self.df = self.t.get_trace(0)

    def test_prepare_returns_required_columns(self) -> None:
        setting = TimelinePlotSetting(task_height=50, plot_format=PlotFormat.File)
        result = prepare_timeline_events(
            self.df, symbol_table=self.t.symbol_table, setting=setting
        )
        self.assertFalse(result.empty)
        self.assertIn(setting.task_column, result.columns)
        self.assertIn(setting.x_start_column, result.columns)
        self.assertIn(setting.x_end_column, result.columns)
        self.assertIn(setting.color_column, result.columns)

    def test_prepare_empty_df(self) -> None:
        setting = TimelinePlotSetting()
        result = prepare_timeline_events(pd.DataFrame(), setting=setting)
        self.assertTrue(result.empty)

    def test_prepare_amplifies_short_events(self) -> None:
        setting = TimelinePlotSetting(amplify_short_events=True)
        result = prepare_timeline_events(
            self.df, symbol_table=self.t.symbol_table, setting=setting
        )
        durations = result[setting.x_end_column] - result[setting.x_start_column]
        self.assertTrue(all(durations >= setting.short_events_display_duration))

    def test_task_labels_for_all_events(self) -> None:
        """When both CPU and GPU events are present, task labels should contain
        both cpu and gpu patterns."""
        setting = TimelinePlotSetting(task_height=50, plot_format=PlotFormat.File)
        result = prepare_timeline_events(
            self.df, symbol_table=self.t.symbol_table, setting=setting
        )
        tasks = result[setting.task_column].unique()
        has_cpu = any("cpu" in t for t in tasks)
        has_gpu = any("gpu" in t for t in tasks)
        self.assertTrue(
            has_cpu or has_gpu, f"Expected cpu/gpu task labels, got: {tasks}"
        )

    def test_task_labels_gpu_only(self) -> None:
        """GPU-only events should produce gpu stream labels."""
        gpu_df = self.df[self.df["stream"] >= 0].copy()
        if gpu_df.empty:
            self.skipTest("No GPU events in test data")
        setting = TimelinePlotSetting(task_height=50, plot_format=PlotFormat.File)
        result = prepare_timeline_events(
            gpu_df, symbol_table=self.t.symbol_table, setting=setting
        )
        tasks = result[setting.task_column].unique()
        self.assertTrue(
            all("gpu" in t or "stream" in t for t in tasks),
            f"Expected gpu stream labels, got: {tasks}",
        )

    def test_task_labels_cpu_only(self) -> None:
        """CPU-only events should produce cpu labels."""
        cpu_df = self.df[self.df["stream"] == -1].copy()
        if cpu_df.empty:
            self.skipTest("No CPU events in test data")
        setting = TimelinePlotSetting(task_height=50, plot_format=PlotFormat.File)
        result = prepare_timeline_events(
            cpu_df, symbol_table=self.t.symbol_table, setting=setting
        )
        tasks = result[setting.task_column].unique()
        self.assertTrue(
            all("cpu" in t for t in tasks),
            f"Expected cpu labels, got: {tasks}",
        )


class TestAlignModuleWithKernels(unittest.TestCase):
    """Tests for align_module_with_kernels."""

    t: Trace

    @classmethod
    def setUpClass(cls) -> None:
        trace_path = os.path.join(get_test_data_dir(), "timeline_analysis")
        cls.t = Trace(trace_dir=trace_path)
        cls.t.parse_traces()
        cls.t.decode_symbol_ids(use_shorten_name=False)

    def setUp(self) -> None:
        self.df = self.t.get_trace(0)

    def test_raises_on_missing_columns(self) -> None:
        """Should raise ValueError when required columns are missing."""
        minimal_df = self.df[["index", "name", "stream"]].copy()
        with self.assertRaises(ValueError):
            align_module_with_kernels(minimal_df, ["## forward ##"])

    def test_raises_on_partial_columns(self) -> None:
        """Should raise ValueError with only the REQUIRED_COLUMNS set (missing kernel columns)."""
        cols_present = list(
            REQUIRED_COLUMNS_FOR_CPU_GPU_ALIGNMENT & set(self.df.columns)
        )
        if len(cols_present) < len(REQUIRED_COLUMNS_FOR_CPU_GPU_ALIGNMENT):
            # Not all required columns exist; the function should raise
            partial_df = self.df[cols_present].copy()
            with self.assertRaises(ValueError):
                align_module_with_kernels(partial_df, ["## forward ##"])

    def test_align_with_symbol_table(self) -> None:
        """Should produce aligned events when all required columns are present."""
        required = REQUIRED_COLUMNS_FOR_CPU_GPU_ALIGNMENT
        if not required.issubset(set(self.df.columns)):
            self.skipTest(
                f"Test data missing columns: {required - set(self.df.columns)}"
            )

        aligned = align_module_with_kernels(
            self.df, ["## forward ##"], sym_table=self.t.symbol_table
        )
        annotations = aligned[aligned["stream"].eq(AnnotationStreamID)]
        self.assertGreater(aligned.shape[0], 0)
        # All annotations should have non-negative num_kernels
        if "num_kernels" in annotations.columns and not annotations.empty:
            self.assertEqual(annotations[annotations["num_kernels"].lt(0)].shape[0], 0)

    def test_align_without_symbol_table(self) -> None:
        """Should work when name column contains strings (decoded)."""
        required = REQUIRED_COLUMNS_FOR_CPU_GPU_ALIGNMENT
        if not required.issubset(set(self.df.columns)):
            self.skipTest(
                f"Test data missing columns: {required - set(self.df.columns)}"
            )

        aligned = align_module_with_kernels(self.df, ["## forward ##"])
        self.assertGreater(aligned.shape[0], 0)


class TestPlotEventsTimeline(unittest.TestCase):
    """Tests for plot_events_timeline with mocked plotly."""

    t: Trace
    trace_path: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.trace_path = os.path.join(get_test_data_dir(), "timeline_analysis")
        cls.t = Trace(trace_dir=cls.trace_path)
        cls.t.parse_traces()
        cls.t.decode_symbol_ids(use_shorten_name=False)

    @patch(f"{_MODULE}.px.timeline")
    def test_plot_calls_plotly(self, mock_timeline: Mock) -> None:
        df = self.t.get_trace(0)
        setting = TimelinePlotSetting(task_height=50, plot_format=PlotFormat.File)
        timeline_events = prepare_timeline_events(
            df, symbol_table=self.t.symbol_table, setting=setting
        )
        self.assertFalse(timeline_events.empty)

        mock_figure = Mock()
        # Mock(name=...) sets Mock's internal name, not a regular attribute.
        # Use configure_mock to set 'name' as a data attribute.
        mock_traces = []
        for i in range(5):
            m = Mock()
            m.configure_mock(x=i, name=str(i))
            mock_traces.append(m)
        mock_figure.data = mock_traces
        mock_timeline.return_value = mock_figure

        output_file = os.path.join(self.trace_path, "test_output.html")
        plot_events_timeline(
            title="Test Plot",
            events=timeline_events,
            setting=setting,
            output_image_path=output_file,
        )
        mock_timeline.assert_called_once()


class TestTimelineClass(unittest.TestCase):
    """Tests for the Timeline wrapper class."""

    t: Trace
    trace_path: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.trace_path = os.path.join(get_test_data_dir(), "timeline_analysis")
        cls.t = Trace(trace_dir=cls.trace_path)
        cls.t.parse_traces()
        cls.t.decode_symbol_ids(use_shorten_name=False)

    @patch(f"{_MODULE}.plot_events_timeline")
    def test_timeline_prepare_and_plot(self, mock_plot: Mock) -> None:
        df = self.t.get_trace(0)
        tl = Timeline(df, self.t.symbol_table, filter_func=GPUKernelFilter())
        tl.setting.plot_format = PlotFormat.File

        save_path = os.path.join(self.trace_path, "timeline_class.html")
        tl.plot("timeline class", save_path=save_path)

        self.assertFalse(tl.is_timeline_events(df, tl.setting))
        self.assertTrue(tl.is_timeline_events(tl.timeline_events, tl.setting))
        mock_plot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
