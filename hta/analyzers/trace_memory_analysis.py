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
            pd.DataFrame containing memory events
        """
        # Get trace for rank
        if rank is None:
            ranks = sorted(self.t.get_all_traces().keys())
            if not ranks:
                raise ValueError("No ranks found in trace")
            rank = ranks[0]

        trace_df = self.t.get_trace(rank)

        # Filter memory events using the column names from the default parser config
        memory_events = trace_df[
            (trace_df["total_allocated"] >= 0) |
            (trace_df["total_reserved"] >= 0)
        ].copy()

        if memory_events.empty:
            logger.warning("No memory events found in trace")
            return pd.DataFrame()

        return memory_events

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
            return pd.DataFrame()

        if visualize:
            # Create plot
            fig = go.Figure()

            # Plot allocated memory
            events_df.sort_values("ts", inplace=True)
            gpu_device = events_df.device_id != -1
            allocated_gb = events_df.loc[gpu_device, "total_allocated"]/(1024**3)
            reserved_gb = events_df.loc[gpu_device, "total_reserved"]/(1024**3)

            fig.add_trace(
                go.Scatter(
                    x=events_df["ts"]/1e6,  # Convert to milliseconds
                    y=allocated_gb,  # Convert to GB
                    name='Allocated Memory',
                    mode='lines',
                    line=dict(color=colorscheme[0])
                )
            )

            # Plot reserved memory
            fig.add_trace(
                go.Scatter(
                    x=events_df["ts"]/1e6,
                    y=reserved_gb,
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