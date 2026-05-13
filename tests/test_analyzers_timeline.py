# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for hta.analyzers.timeline module."""

import os
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from hta.analyzers.timeline import (
    _simplify_name,
    plot_timeline_gpu_kernels,
    plot_timeline_gpu_kernels_from_trace,
)
from hta.common.trace import Trace
from hta.common.trace_symbol_table import TraceSymbolTable
from hta.utils.test_utils import get_test_data_dir


class TestAnalyzersTimeline(unittest.TestCase):
    """Coverage tests for hta/analyzers/timeline.py."""

    def test_simplify_name_truncates_long_names(self) -> None:
        long_name = "a" * 100
        result = _simplify_name(long_name)
        self.assertEqual(len(result), 45)
        self.assertTrue(result.endswith("..."))

    @patch("hta.analyzers.timeline.px.timeline")
    def test_plot_gpu_kernels_delegates(self, mock_px_timeline: Mock) -> None:
        mock_px_timeline.return_value = Mock()
        sym_table = TraceSymbolTable()
        sym_table.add_symbols(["aten::mm", "kernel"])
        sym_id_map = sym_table.get_sym_id_map()
        df = pd.DataFrame(
            {
                "iteration": [1],
                "name": [sym_id_map["aten::mm"]],
                "cat": [sym_id_map["kernel"]],
                "rank": [0],
                "stream": [7],
                "ts": [1000],
                "dur": [500],
            }
        )
        plot_timeline_gpu_kernels("test", df, sym_table)
        mock_px_timeline.assert_called_once()

    @patch("hta.analyzers.timeline.plot_timeline")
    def test_plot_gpu_kernels_from_trace(self, mock_plot: Mock) -> None:
        trace_path = os.path.join(get_test_data_dir(), "timeline_analysis")
        t = Trace(trace_dir=trace_path)
        t.parse_traces()
        t.decode_symbol_ids(use_shorten_name=False)
        plot_timeline_gpu_kernels_from_trace("test", t)
        mock_plot.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
