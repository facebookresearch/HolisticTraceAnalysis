# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

import pandas as pd
import plotly.express as px

from hta.utils.utils import KernelType, get_kernel_type, merge_kernel_intervals

# import statement used without the "if TYPE_CHECKING" guard will cause a circular
# dependency with trace_analysis.py causing mypy to fail and should not be removed.
if TYPE_CHECKING:
    from hta.trace import Trace


class CommunicationAnalysis:
    def __init__(self):
        pass

    @classmethod
    def get_comm_comp_overlap(cls, t: "Trace", visualize: bool = True) -> pd.DataFrame:
        """
        Communication analysis implementation. See `get_comm_comp_overlap` in `trace_analysis.py` for details.
        """
        sym_table = t.symbol_table.get_sym_table()

        def get_comm_comp_overlap_value(trace_df: pd.DataFrame) -> float:
            """
            Compute the overlap percentage between communication and computation kernels for one rank.
            """
            gpu_kernels = trace_df[trace_df["stream"].ne(-1)].copy()
            gpu_kernels["kernel_type"] = gpu_kernels[["name"]].apply(
                lambda x: get_kernel_type(sym_table[x["name"]]), axis=1
            )

            # Isolate communication and computation kernels and merge each one of them.
            comp_kernels = merge_kernel_intervals(
                gpu_kernels[gpu_kernels["kernel_type"].eq(KernelType.COMPUTATION.name)].copy()
            )
            comm_kernels = merge_kernel_intervals(
                gpu_kernels[gpu_kernels["kernel_type"].eq(KernelType.COMMUNICATION.name)].copy()
            )

            # When a communication kernel starts and ends, the cumulative status is changed by 1 and -1;
            # when a computation kernel starts and ends, the cumulative status is changed by 2 and -2.
            status_df = (
                pd.concat(
                    [
                        comm_kernels.melt(var_name="status", value_name="time").replace({"ts": 1, "end": -1}),
                        comp_kernels.melt(var_name="status", value_name="time").replace({"ts": 2, "end": -2}),
                    ]
                )
                .sort_values(by="time")
                .reset_index(drop=True)
            )
            status_df["running"] = status_df["status"].cumsum()
            # Time intervals when status is 3 indicate overlapping communication and computation kernels.
            overlap = status_df[status_df["running"].eq(3)]
            shifted_overlap = overlap.merge(status_df.shift(-1).dropna(), left_index=True, right_index=True)
            return (shifted_overlap["time_y"] - shifted_overlap["time_x"]).sum() / (
                comm_kernels["end"] - comm_kernels["ts"]
            ).sum()

        result: Dict[str, List[float]] = defaultdict(list)
        for rank, trace_df in t.traces.items():
            result["rank"].append(rank)
            result["comp_comm_overlap_ratio"].append(get_comm_comp_overlap_value(trace_df))
        result_df = pd.DataFrame(result)
        result_df["comp_comm_overlap_pctg"] = round(100 * result_df["comp_comm_overlap_ratio"], 2)

        if visualize:
            fig = px.bar(
                result_df,
                x="rank",
                y="comp_comm_overlap_ratio",
                title="Computation-Communication Overlap",
                labels={
                    "rank": "Rank",
                    "comp_comm_overlap_ratio": "Computation-Communication Overlap Percentage",
                },
            )

            fig.update_layout(yaxis_tickformat=".2%")
            fig.show()

        return result_df[["rank", "comp_comm_overlap_pctg"]]
