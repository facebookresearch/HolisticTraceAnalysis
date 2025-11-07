from typing import Any, Dict

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


class _XPUSingleTrace(_SingleTrace):
    """Class representing a single XPU rank in a trace."""

    device: str = "INTEL GPU"


Trace = _SingleTrace


def create_default(meta=None, df=None, symbol_table=None) -> Trace:
    """Factory method to create default Trace object."""
    return _SingleTrace(meta, df, symbol_table)


def create(device_type: str, meta, df, symbol_table) -> Trace:
    """Factory method to create Trace object based on device type."""

    if device_type == "INTEL GPU":
        return _XPUSingleTrace(meta, df, symbol_table)
    else:
        return _SingleTrace(meta, df, symbol_table)
