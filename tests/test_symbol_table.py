# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import unittest
from typing import List, Set

from hta.common.trace_symbol_table import TraceSymbolTable


class SymbolDecoder:
    def __init__(self, symbol_table: TraceSymbolTable):
        self.sym_table = symbol_table.get_sym_table()
        self.sym_id_map = symbol_table.get_sym_id_map()

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
        self.symbols_list = [self.symbols_1, self.symbols_2, self.symbols_3]
        for s in self.symbols_list:
            ss = ss.union(set(s))
        self.symbols: List[str] = sorted(list(ss))

    def test_add_symbols_single_process(self):
        st = TraceSymbolTable()
        for symbols in [self.symbols_1, self.symbols_2, self.symbols_3]:
            st.add_symbols(symbols)

        self.assertListEqual(sorted(st.get_sym_table()), self.symbols)
        self.assertTrue(check_symbol_table(st))

    def test_add_symbols_multi_processing(self):
        st = TraceSymbolTable()
        st.add_symbols_mp(self.symbols_list)

        self.assertListEqual(sorted(st.get_sym_table()), self.symbols)
        self.assertTrue(check_symbol_table(st))

    def test_query_symbols_multi_processes(self):
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
