# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import re
import tempfile
import unittest
from typing import Dict, Iterable, List, Set

import pandas as pd
from hta.common.trace_symbol_table import TraceSymbolTable
from hta.common.types import GroupingPattern
from hta.utils.test_utils import data_provider


class SymbolDecoder:
    def __init__(self, symbol_table: TraceSymbolTable) -> None:
        self.sym_table: List[str] = symbol_table.get_sym_table()
        self.sym_id_map: Dict[str, int] = symbol_table.get_sym_id_map()

    def __call__(self, idx: int) -> str:
        return self.sym_table[idx]


def check_symbol_table(st: TraceSymbolTable) -> bool:
    sym_id_map = st.get_sym_id_map()
    sym_table = st.get_sym_table()
    two_way_map_consistency = [sym_table[i] == s for s, i in sym_id_map.items()]
    return all(two_way_map_consistency)


class TraceSymbolTableTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.symbols_1: List[str] = ["a", "b", "c", "b1"]
        self.symbols_2: List[str] = ["a", "b", "c", "b2", "d1"]
        self.symbols_3: List[str] = ["a", "b", "f", "b3", "d2"]
        ss: Set[str] = set()
        self.symbols_list: List[Iterable[str]] = [
            self.symbols_1,
            self.symbols_2,
            self.symbols_3,
        ]
        for s in self.symbols_list:
            ss = ss.union(set(s))
        self.symbols: List[str] = sorted(ss)

    def test_add_symbols_single_process(self) -> None:
        st = TraceSymbolTable()
        for symbols in [self.symbols_1, self.symbols_2, self.symbols_3]:
            st.add_symbols(symbols)

        self.assertListEqual(sorted(st.get_sym_table()), self.symbols)
        self.assertTrue(check_symbol_table(st))

    def test_add_symbols_multi_processing(self) -> None:
        st = TraceSymbolTable()
        st.add_symbols_mp(self.symbols_list)

        self.assertListEqual(sorted(st.get_sym_table()), self.symbols)
        self.assertTrue(check_symbol_table(st))

    def test_query_symbols_multi_processes(self) -> None:
        st = TraceSymbolTable()
        st.add_symbols_mp(self.symbols_list)
        decoder = SymbolDecoder(st)
        indices = [i for i, _ in enumerate(st.get_sym_table())]

        np = 4
        with mp.Pool(np) as pool:
            decoded_symbols = pool.map(decoder, indices)
        pool.join()
        pool.close()
        self.assertEqual(len(decoded_symbols), len(self.symbols))

        sym_id_map = st.get_sym_id_map()
        is_consistent = [
            sym_id_map[sym] == idx for (sym, idx) in zip(decoded_symbols, indices)
        ]
        self.assertTrue(all(is_consistent))

    def test_symbol_pattern_match(self):
        st = TraceSymbolTable()
        st.add_symbols_mp(self.symbols_list)
        patterns = ["b"]

        expected_symbols = {"b", "b3", "b2", "b1"}
        self.assertEqual(expected_symbols, set(st.find_matched_symbols(patterns)))

        expected_idxs = {st.sym_index[s] for s in expected_symbols}
        self.assertEqual(expected_idxs, set(st.find_matches(patterns)))

    def test_combine_symbol_tables(self) -> None:
        t1: TraceSymbolTable = TraceSymbolTable()
        t1.add_symbols(self.symbols_1)
        t2: TraceSymbolTable = TraceSymbolTable()
        t2.add_symbols(self.symbols_2)
        t3: TraceSymbolTable = TraceSymbolTable()
        t3.add_symbols(self.symbols_3)
        t_combined = TraceSymbolTable.combine_symbol_tables([t1, t2, t3])
        all_symbols = set(self.symbols_1)
        all_symbols.update(self.symbols_2)
        all_symbols.update(self.symbols_3)
        self.assertSetEqual(all_symbols, set(t_combined.get_sym_id_map().keys()))

    def test_clone_symbol_table(self) -> None:
        t: TraceSymbolTable = TraceSymbolTable()
        t.add_symbols(self.symbols_1)

        t_cloned = TraceSymbolTable.clone(t)
        # the two symbol table should be identical
        self.assertListEqual(t.sym_table, t_cloned.sym_table)
        self.assertDictEqual(t.sym_index, t_cloned.sym_index)

        # changing one table should change the other.
        t_cloned.add_symbols(["clone"])
        self.assertIn("clone", t_cloned.sym_index)
        self.assertNotIn("clone", t.sym_index)

    def test_save_to_load_from_file(self) -> None:
        t: TraceSymbolTable = TraceSymbolTable()
        t.add_symbols(self.symbols_1)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            symbol_table_file = os.path.join(tmp_dirname, "test_symbols.csv")
            t._to_csv_file(symbol_table_file)
            t_loaded = TraceSymbolTable.from_csv_file(symbol_table_file)
            self.assertListEqual(t.sym_table, t_loaded.sym_table)
            self.assertDictEqual(t.sym_index, t_loaded.sym_index)

    def test_get_sym_table_series(self) -> None:
        t: TraceSymbolTable = TraceSymbolTable()
        t.add_symbols(self.symbols_1)
        t.add_symbols(self.symbols_2)
        self.assertListEqual(t.get_sym_table(), t.get_sym_table_series().to_list())
        self.assertDictEqual(t.get_sym_index(), t.get_sym_index_series().to_dict())

    def test_create_from_df(self) -> None:
        df = pd.DataFrame(
            data={
                "name": ["a", "b", "a", "c"],
                "cat": ["kernel", "cpu_op", "kernel", "user_annotation"],
            }
        )
        t = TraceSymbolTable.create_from_df(df)
        self.assertSetEqual(
            set(t.get_sym_table_series().to_list()),
            {"a", "b", "c", "kernel", "cpu_op", "user_annotation"},
        )

    def test_encode_decode_df(self) -> None:
        df = pd.DataFrame(
            data={
                "name": ["a", "b", "a", "c"],
                "cat": ["kernel", "cpu_op", "kernel", "user_annotation"],
            }
        )
        st = TraceSymbolTable.create_from_df(df)

        # encode the df
        df_original = df.copy()
        st.encode_df(df)

        # decode the df
        st.decode_df(df)
        df_decoded = df.copy()

        # Equality check
        self.assertListEqual(
            df_original["name"].to_list(), df_decoded["s_name"].to_list()
        )
        self.assertListEqual(
            df_original["cat"].to_list(), df_decoded["s_cat"].to_list()
        )

    # pyre-ignore[56]
    @data_provider(
        lambda: (
            {
                "symbol_id_map": {"A": 1, "B": 2, "C": 4},
                "expected_sym_table": ["Undefined-0", "A", "B", "Undefined-3", "C"],
                "expected_sym_index": {
                    "Undefined-0": 0,
                    "A": 1,
                    "B": 2,
                    "Undefined-3": 3,
                    "C": 4,
                },
            },
        )
    )
    def test_create_from_symbol_id_map(
        self,
        symbol_id_map: Dict[str, int],
        expected_sym_table: List[str],
        expected_sym_index: Dict[str, int],
    ) -> None:
        tst = TraceSymbolTable.create_from_symbol_id_map(symbol_id_map)
        self.assertListEqual(tst.sym_table, expected_sym_table)
        self.assertDictEqual(tst.sym_index, expected_sym_index)

    # pyre-ignore[56]
    @data_provider(
        lambda: (
            {
                "symbols": ["a", "b", "c", "b1"],
                "symbol_pattern": GroupingPattern(
                    group_name="test_1", pattern=re.compile(r"a|b"), inverse_match=False
                ),
                "expected_symbol_ids": {"a": 0, "b": 1, "b1": 3},
            },
            {
                "symbols": ["a", "b", "c", "b1"],
                "symbol_pattern": GroupingPattern(
                    group_name="test_2", pattern=re.compile(r"a|b"), inverse_match=True
                ),
                "expected_symbol_ids": {"c": 2},
            },
        )
    )
    def test_get_symbol_ids(
        self,
        symbols: List[str],
        symbol_pattern: GroupingPattern,
        expected_symbol_ids: Dict[str, int],
    ) -> None:
        tst = TraceSymbolTable()
        tst.add_symbols(symbols)

        self.assertDictEqual(tst.get_symbol_ids(symbol_pattern), expected_symbol_ids)

    # pyre-ignore[56]
    @data_provider(
        lambda: (
            {
                "symbols": ["a", "b", "c", "b1"],
                "symbol_ids": [0, 1],
                "expected_symbols": {0: "a", 1: "b"},
            },
            {
                "symbols": ["a", "b", "c", "b1"],
                "symbol_ids": [-1, 2, 5],
                "expected_symbols": {2: "c"},
            },
        )
    )
    def test_get_symbol_names(
        self,
        symbols: List[str],
        symbol_ids: List[int],
        expected_symbols: Dict[int, str],
    ) -> None:
        tst = TraceSymbolTable()
        tst.add_symbols(symbols)
        self.assertDictEqual(tst.get_symbol_names(symbol_ids), expected_symbols)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
