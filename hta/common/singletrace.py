from typing import Any, Dict, List

import pandas as pd

from hta.common.trace_symbol_table import TraceSymbolTable

MetaData = Dict[str, Any]


class _SingleTrace:
    """Class representing a single rank in a trace."""

    device: str = "GENERIC"

    def __init__(
        self, meta: MetaData, df: pd.DataFrame, symbol_table: TraceSymbolTable
    ):
        self.meta = meta
        self.df = df
        self.symbol_table = symbol_table

    def get_sym_table(self) -> List[str]:
        """Get the list of symbols from the symbol table.

        Returns:
            List of symbol strings in order of their IDs.
        """
        return self.symbol_table.get_sym_table()


class _XPUSingleTrace(_SingleTrace):
    """Class representing a single XPU rank in a trace."""

    device: str = "INTEL GPU"


Trace = _SingleTrace


def create(device_type: str, meta, df, symbol_table) -> Trace:
    """Factory method to create Trace object based on device type."""

    if device_type == "INTEL GPU":
        return _XPUSingleTrace(meta, df, symbol_table)
    else:
        return _SingleTrace(meta, df, symbol_table)
