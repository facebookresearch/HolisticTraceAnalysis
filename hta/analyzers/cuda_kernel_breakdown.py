# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, TYPE_CHECKING

import pandas as pd
from hta.utils.utils import (
    classify_kernel,
    CUDA_KERNEL_CATEGORY_NAMES,
    short_kernel_name,
)

# import statement used without the "if TYPE_CHECKING" guard will cause a circular
# dependency with trace_analysis.py causing mypy to fail and should not be removed.
if TYPE_CHECKING:
    from hta.common.trace import Trace

# This configures the threshold under which we consider gaps between
# kernels to be due to realistic delays in launching back-back kernels on the GPU


class CudaKernelBreakdown:
    def __init__(self):
        pass

    @classmethod
    def get_cuda_kernel_stats(
        cls,
        t: "Trace",
        keep_unknown: bool = False,
        extra_kernel_args: str = "grid,input_dims,warps_per_sm,blocks_per_sm",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        r"""
        Summarizes the time spent by each kernel and by kernel type. We have more categories for kernel types

        Args:
            t: Trace file
            keep_unknown (bool): Whether to keep unknown type kernels
            extra_kernel_args (str): Extra kernel arguments to collect from traces

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]
                Returns two dataframes. The first dataframe shows the min, max, mean, standard deviation,
                total time taken by each kernel. The second dataframe shows the details of the kernels arguments
                for each rank.
        """

        s_tab: List[str] = t.symbol_table.sym_table
        ss_tab: List[str] = [short_kernel_name(s) for s in s_tab]

        total_cuda_df = pd.DataFrame()
        for rank in t.traces:
            rank_df = t.get_trace(rank)

            rank_df["s_name"] = rank_df["name"].apply(lambda idx: ss_tab[idx])
            rank_df["s_cat"] = rank_df["cat"].apply(lambda idx: ss_tab[idx])

            demand_index = list(
                set(extra_kernel_args.split(",") + ["s_cat", "s_name", "dur"])
            )
            filter_index = list(
                set(demand_index) & set(rank_df.columns.values.tolist())
            )

            filtered_rank_df = rank_df[filter_index].copy()
            filtered_rank_df["rank"] = rank
            filtered_rank_df["kernel_type"] = filtered_rank_df[["s_name"]].apply(
                lambda x: classify_kernel(x["s_name"]), axis=1
            )

            total_cuda_df = pd.concat(
                [total_cuda_df, filtered_rank_df], ignore_index=True
            )

        if not keep_unknown:
            total_cuda_df = total_cuda_df[
                total_cuda_df["kernel_type"] != CUDA_KERNEL_CATEGORY_NAMES.UNKNOWN
            ]

        kernel_stats_df = total_cuda_df.groupby(by=["s_name", "kernel_type"])[
            "dur"
        ].agg(["sum", "max", "min", "mean", "std", "idxmax", "idxmin"])
        kernel_stats_df["max_rank"] = kernel_stats_df["idxmax"].apply(
            lambda idx: total_cuda_df.loc[idx, "rank"]
        )
        kernel_stats_df["min_rank"] = kernel_stats_df["idxmin"].apply(
            lambda idx: total_cuda_df.loc[idx, "rank"]
        )
        kernel_stats_df.sort_values(by=["sum"], ascending=False, inplace=True)
        kernel_stats_df.reset_index(inplace=True)

        return (kernel_stats_df, total_cuda_df)
