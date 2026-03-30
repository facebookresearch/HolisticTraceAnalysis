import enum
import gzip
import json
import os
from typing import List, Tuple

import numpy as np
import plotly
import plotly.graph_objects as go


colorscheme = plotly.colors.qualitative.Pastel


class Category(enum.Enum):
    INPUT = enum.auto()
    TEMPORARY = enum.auto()
    ACTIVATION = enum.auto()
    GRADIENT = enum.auto()
    AUTOGRAD_DETAIL = enum.auto()
    PARAMETER = enum.auto()
    OPTIMIZER_STATE = enum.auto()


_CATEGORY_TO_COLORS = {
    "Parameter": colorscheme[0],
    "Optimizer state": colorscheme[1],
    "Input": colorscheme[2],
    "Temporary": colorscheme[3],
    "Activation": colorscheme[4],
    "Gradient": colorscheme[5],
    "Autograd": colorscheme[6],
    "Unknown": colorscheme[7],
}

_CATEGORY_TO_INDEX = {c: i for i, c in enumerate(_CATEGORY_TO_COLORS)}


class MemoryAnalysis:
    def __init__(self, path):
        self.path = path

    def _process_raw_events(self) -> Tuple[List[int], List[List[int]]]:
        """
        Loads the raw events and converts them into list of timestamps and list
        of memory used by each category at each timestamp.

        Returns:
            ([timestamps], [memory_for_each_category_in_bytes])
        """
        file_handle = gzip.open(self.path, "r")
        raw_timeline = json.loads(file_handle.read())

        times: List[int] = []
        sizes: List[List[int]] = []

        t_min = -1
        for t, _, numbytes, category in raw_timeline:
            # Save the smallest timestamp to populate pre-existing allocs.
            if t_min == -1 or (t < t_min and t > 0):
                t_min = t

            # Handle timestep
            if len(times) == 0:
                times.append(t)
                sizes.append([0 for _ in _CATEGORY_TO_INDEX])

            elif t != times[-1]:
                times.append(t)
                sizes.append(sizes[-1].copy())

            # Handle memory and categories
            sizes[-1][category] += numbytes

        times = [t_min if t < 0 else t for t in times]
        return times, sizes

    def plot_memory_timeline(self, save_graph=False) -> None:  # pragma: no cover
        r"""
        Generates a plot of memory usage across the following categories: input,
        temporary, activations, gradients, autograd, parameters and optimizer
        state.

        Args:
            save_graph (bool): Set to True to save the generated graph. Creates
            a folder called images in the current working directory and saves
            the graph in it. Default = False

        Returns:
            None

        """
        mem_timeline = self._process_raw_events()
        times, sizes = np.array(mem_timeline[0]), np.array(mem_timeline[1])

        # start timeline at 0
        t_min = min(times)
        times -= t_min

        stacked = np.cumsum(sizes, axis=1) / 1024**3
        fig = go.Figure()

        for category, color in _CATEGORY_TO_COLORS.items():
            idx = _CATEGORY_TO_INDEX[category]
            fig.add_trace(
                go.Scatter(
                    x=times / 1e6,
                    y=stacked[:, idx],
                    mode="lines",
                    fillcolor=color,
                    fill="tonexty",
                    name=category,
                    marker=dict(color=color),
                    showlegend=True,
                )
            )

        fig.update_xaxes(title="Time (ms)")
        fig.update_yaxes(title="Memory (GB)")
        fig.update_layout(
            title_text="Memory timeline", title_x=0.5, width=1200, height=800
        )

        if save_graph:
            if not os.path.exists("images"):
                os.mkdir("images")
            fig.write_image("images/memory_plot.png")
        else:
            fig.show()
