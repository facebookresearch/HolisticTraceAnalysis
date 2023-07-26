# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd

from hta.common import execution_trace
from hta.trace_analysis import TraceAnalysis


class TraceAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.execution_trace_dir: str = "tests/data/execution_trace/"
        self.analyzer_t = TraceAnalysis(trace_dir=self.execution_trace_dir)
        self.execution_trace_file: str = (
            self.execution_trace_dir + "benchmark_simple_add_et.json.gz"
        )

    def test_execution_trace_load(self):
        et = execution_trace.load_execution_trace(self.execution_trace_file)
        self.assertIsNotNone(et)
        self.assertEqual(len(et.nodes), 38)

    def _validate_correlated_trace(
        self, trace_df: pd.DataFrame, et: execution_trace.ExecutionGraph
    ):
        """Common checks for correlated traces"""
        et_node_col = trace_df[trace_df.et_node >= 0].et_node

        self.assertEqual(
            et_node_col.count(),
            et_node_col.unique().size,
            msg="ET nodes should be mapped 1:1 only, "
            f"ET node.value_count() = \n{et_node_col.value_counts()}",
        )
        self.assertTrue(et_node_col.count() <= len(et.nodes))

        # Add correlated ET columns
        execution_trace.add_et_column(trace_df, et, "et_node_name")
        execution_trace.add_et_column(trace_df, et, "op_schema")
        execution_trace.add_et_column(trace_df, et, "input_shapes")
        execution_trace.add_et_column(trace_df, et, "input_types")
        execution_trace.add_et_column(trace_df, et, "output_shapes")
        execution_trace.add_et_column(trace_df, et, "output_types")

        # Check if correlated nodes and actual nodes have same name
        self.analyzer_t.t.symbol_table.add_symbols_to_trace_df(trace_df, col="name")

        correlated_rows = trace_df.loc[~trace_df.et_node.isna()]
        compare_names = correlated_rows["name"] == correlated_rows["et_node_name"]

        self.assertTrue(
            compare_names.all(),
            msg="Trace event names and ET node names"
            " are not a perfect match, see series =\n"
            f"{correlated_rows[['name', 'et_node_name']]}",
        )

    def test_correlate_execution_trace_with_overlap(self):
        et = execution_trace.load_execution_trace(self.execution_trace_file)
        self.assertIsNotNone(et)

        # Correlate rank 0
        execution_trace.correlate_execution_trace(self.analyzer_t.t, 0, et)
        trace_df = self.analyzer_t.t.get_trace(0).copy()

        self._validate_correlated_trace(trace_df, et)
