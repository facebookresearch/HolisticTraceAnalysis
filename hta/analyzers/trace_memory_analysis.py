import enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly
import plotly.graph_objects as go
import pandas as pd

from hta.common.trace import Trace
from hta.configs.config import logger

colorscheme = plotly.colors.qualitative.Pastel

class MemoryAnalysis:
    """Class for analyzing memory usage patterns in HTA traces"""

    def __init__(self, t: Trace):
        """Initialize with an HTA Trace object

        Args:
            t (Trace): HTA Trace object containing memory events
        """
        self.t = t

    def _process_memory_events(self, rank: Optional[int] = None) -> pd.DataFrame:
        """Process memory events from trace into a DataFrame

        Args:
            rank (Optional[int]): Process events for specific rank. If None, use first rank.

        Returns:
            pd.DataFrame containing memory events with columns:
                - ts: timestamp
                - device_id: Device ID
                - device_type: Device type (1=CPU, 2=CUDA)
                - bytes_delta: Change in bytes
                - total_allocated: Total allocated memory
                - total_reserved: Total reserved memory
                - addr: Memory address
        """
        # Get trace for rank
        if rank is None:
            ranks = sorted(self.t.get_all_traces().keys())
            if not ranks:
                raise ValueError("No ranks found in trace")
            rank = ranks[0]

        trace_df = self.t.get_trace(rank)

        # Filter memory events
        memory_events = trace_df[
            (trace_df["ph"] == "i") &
            (trace_df["name"].apply(lambda x: self.t.symbol_table.get_sym_table()[x] == "[memory]"))
        ]

        # Extract memory data
        def extract_arg(row, arg_name, default=0):
            args = row.get("args", {})
            if isinstance(args, dict):
                return args.get(arg_name, default)
            return default

        events_data = {
            "ts": [],
            "device_id": [],
            "device_type": [],
            "bytes_delta": [],
            "total_allocated": [],
            "total_reserved": [],
            "addr": []
        }

        for _, event in memory_events.iterrows():
            args = event.get("args", {})
            if not isinstance(args, dict):
                continue

            events_data["ts"].append(event["ts"])
            events_data["device_id"].append(args.get("Device Id", 0))
            events_data["device_type"].append(args.get("Device Type", 1))
            events_data["bytes_delta"].append(args.get("Bytes", 0))
            events_data["total_allocated"].append(args.get("Total Allocated", 0))
            events_data["total_reserved"].append(args.get("Total Reserved", 0))
            events_data["addr"].append(args.get("Addr", 0))

        return pd.DataFrame(events_data)

    def get_memory_timeline(self, rank: Optional[int] = None, visualize: bool = True) -> pd.DataFrame:
        """Generate timeline of memory usage

        Args:
            rank (Optional[int]): Analyze specific rank. If None, use first rank.
            visualize (bool): Whether to display the plot. Default=True.

        Returns:
            pd.DataFrame: DataFrame containing memory timeline data
        """
        # Process events
        events_df = self._process_memory_events(rank)

        if events_df.empty:
            logger.warning("No memory events found in trace")
            return pd.DataFrame()

        if visualize:
            # Create plot
            fig = go.Figure()

            # Plot allocated memory
            fig.add_trace(
                go.Scatter(
                    x=events_df["ts"]/1e6,  # Convert to milliseconds
                    y=events_df["total_allocated"]/(1024**3),  # Convert to GB
                    name='Allocated Memory',
                    mode='lines',
                    line=dict(color=colorscheme[0])
                )
            )

            # Plot reserved memory
            fig.add_trace(
                go.Scatter(
                    x=events_df["ts"]/1e6,
                    y=events_df["total_reserved"]/(1024**3),
                    name='Reserved Memory',
                    mode='lines',
                    line=dict(color=colorscheme[1])
                )
            )

            # Update layout
            fig.update_layout(
                title='Memory Usage Timeline',
                xaxis_title='Time (ms)',
                yaxis_title='Memory (GB)',
                hovermode='x unified',
                width=1200,
                height=800
            )

            fig.show()

        return events_df