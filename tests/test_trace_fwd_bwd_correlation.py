import unittest
from collections import namedtuple
from pathlib import Path

import pandas as pd
from hta.common.trace import Trace


class TraceFWDBWDLinkTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        test_data_path = Path(__file__).parent.parent.joinpath(
            "tests/data/h100/h100_trace.json"
        )
        self.trace = Trace(
            trace_files={0: str(test_data_path)},
            trace_dir="",
        )
        self.trace.parse_traces()
        self.trace.decode_symbol_ids(use_shorten_name=False)

    def test_fwdbwd_index_column(self):
        self.assertIn(
            "fwdbwd_index",
            self.trace.get_trace(0).columns,
            "fwdbwd_index column not found in trace DataFrame",
        )

    def test_fwdbwd_symbol_and_id(self):
        self.assertIn(
            "fwdbwd",
            self.trace.symbol_table.sym_index,
            "fwdbwd symbol not found in trace symbol table",
        )

    def test_fwdbwd_correlation(self):
        df = self.trace.traces[0]
        fwd_func_name = "fbgemm::split_embedding_codegen_lookup_rowwise_adagrad_function"
        expect_bwd_func_name = "torch::autograd::CppNode<SplitLookupFunction_rowwise_adagrad_Op>"
        for index, row in df[df['s_name'].str.match(pat=r"^"+fwd_func_name+r"$")].iterrows():
            fwdbwd_type, bwd_id = row['fwdbwd'], row['fwdbwd_index']
            self.assertEqual(fwdbwd_type, 0, "fwdbwd type should be 0")
            bwd_func_name = df.loc[bwd_id, 's_name'] if bwd_id in df.index else None
            self.assertEqual(
                bwd_func_name,
                expect_bwd_func_name,
                f"Expected bwd function name to be {bwd_func_name}, but got {expect_bwd_func_name}",
            )
            bwd_type = df.loc[bwd_id, 'fwdbwd'] if bwd_id in df.index else None
            self.assertEqual(
                bwd_type, 1, "bwd type should be 1 for backward function"
            )       
    
if __name__ == "__main__":  # pragma: no cover
    unittest.main()
