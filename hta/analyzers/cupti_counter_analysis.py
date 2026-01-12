# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import pandas as pd
from hta.common.trace import Trace
from hta.common.trace_call_graph import CallGraph
from hta.configs.config import logger

CUDA_SASS_INSTRUCTION_COUNTER_FLOPS: Dict[str, float] = {
    f"smsp__sass_thread_inst_executed_op_{op}_pred_on.sum": 2 if "fma" in op else 1
    for op in ["ffma", "fmul", "fadd", "hfma", "hmul", "hadd", "dfma", "dmul", "dadd"]
}


class CuptiCounterAnalysis:
    cuda_profiler_cat = "cuda_profiler_range"

    def __init__(self):
        pass

    @classmethod
    def _get_counter_data_with_operators_for_rank(
        cls,
        t: Trace,
        rank: int,
        cg: CallGraph,
    ) -> Optional[pd.DataFrame]:
        sym_table = t.symbol_table.get_sym_table()
        t.decode_symbol_ids(use_shorten_name=False)
        df = t.get_trace(rank)

        # Get valid cuda kernels
        gpu_kernels = (
            df.loc[df.s_cat.eq(cls.cuda_profiler_cat)]
            .copy()
            .sort_values("ts")
            .reset_index(drop=True)
        )
        # Get cuda kernel launches
        kernel_launches = (
            df.loc[
                df.s_cat.str.match("cuda_runtime")
                & df.s_name.str.match("cudaLaunchKernel")
            ].sort_values("ts")
        ).reset_index(drop=True)
        if len(kernel_launches) != len(gpu_kernels):
            logger.error(
                "Number of kernels launches and kernels do not match for"
                f" rank {rank}\n"
                f" kernel launches ({len(kernel_launches)})"
                f" kernels ({len(gpu_kernels)})"
            )
            return None

        gpu_kernels["index_runtime"] = kernel_launches["index"]
        # Add stack columns
        op_stacks = {}
        top_level_ops = {}
        bottom_level_ops = {}
        for idx, _, launch_idx in gpu_kernels[["index", "index_runtime"]].itertuples(
            index=True
        ):
            stack = cg.get_stack_of_node(launch_idx).sort_values("ts")
            ops = stack.loc[stack.s_cat.eq("cpu_op"), "index"].to_list()
            op_stacks[idx] = ops
            top_level_ops[idx] = ops[0]
            bottom_level_ops[idx] = ops[-1]

        gpu_kernels["op_stack"] = pd.Series(op_stacks)
        gpu_kernels["top_level_op"] = pd.Series(top_level_ops)
        gpu_kernels["bottom_level_op"] = pd.Series(bottom_level_ops)
        gpu_kernels["name"] = gpu_kernels["s_name"]
        gpu_kernels["cat"] = gpu_kernels["s_cat"]
        gpu_kernels.drop(columns=["s_cat", "s_name"], inplace=True)

        def stringify_op_stack(ops: List[int]) -> List[str]:
            return [sym_table[df["name"].loc[op]] for op in ops]

        gpu_kernels["op_stack"] = gpu_kernels["op_stack"].apply(stringify_op_stack)
        for col in ["top_level_op", "bottom_level_op"]:
            gpu_kernels[col] = gpu_kernels[col].apply(
                lambda op: sym_table[df["name"].loc[op]] if op >= 0 else ""
            )
        return gpu_kernels

    @classmethod
    def get_counter_data_with_operators(
        cls,
        t: Trace,
        ranks: Optional[List[int]] = None,
    ) -> List[pd.DataFrame]:
        """Correlates the Kernel counter events with pytorch operators using
        the callgraph.
        Args:
            t (Trace): trace object
            ranks (List[int]): List of ranks on which to run the analysis. Default = [0].
        Returns:
            A list of dataframes, one per rank, containing kernel name,
            op_stack (operator stack), top and bottom level op, and columns
            for individual performance counters.

        For more details see `get_counter_data_with_operators` in `trace_analysis.py`,
        or read more here https://github.com/facebookresearch/HolisticTraceAnalysis/issues/29
        """

        if ranks is None or len(ranks) == 0:
            ranks = [0]

        sym_index = t.symbol_table.get_sym_id_map()
        if "cuda_profiler_range" not in sym_index.keys():
            logger.warning(
                "Could not find events of 'cuda_profiler_range' category "
                "Please check if you ran CUPTI profiler mode correctly"
            )
            return []

        cg = CallGraph(t, ranks=ranks)

        result_list = [
            cls._get_counter_data_with_operators_for_rank(t=t, rank=rank, cg=cg)
            for rank in ranks
        ]

        return [k for k in result_list if k is not None]
