# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Callable, List, Optional

import pandas as pd

from hta.analyzers.straggler import extract_iteration_info, find_stragglers_with_late_start_comm_kernels
from hta.configs.config import logger

# import statement used without the "if TYPE_CHECKING" guard will cause a circular
# dependency with trace_analysis.py causing mypy to fail and should not be removed.
if TYPE_CHECKING:
    from hta.common.trace import Trace


class StragglerAnalysis:
    def __init__(self):
        pass

    @classmethod
    def get_profiler_steps(cls, t: "Trace") -> List[int]:
        """
        Profiler steps implementation. Returns the list of profiler steps.
        """
        return sorted([i for i in extract_iteration_info(t)["iter"].unique() if i != -1])

    @classmethod
    def get_potential_stragglers(
        cls,
        t: "Trace",
        profiler_steps: Optional[List[int]] = None,
        num_candidates: int = 2,
        visualize: bool = False,
        straggler_identification_impl: Callable[..., pd.Series] = find_stragglers_with_late_start_comm_kernels,
    ) -> List[int]:
        """
        Straggler analysis implementation. See `get_potential_stragglers` in `trace_analysis.py` for details.
        """
        if num_candidates < 1:
            num_candidates = 1

        available_profiler_steps = cls.get_profiler_steps(t)
        if profiler_steps is None:
            valid_profiler_steps = available_profiler_steps
        else:
            valid_profiler_steps = [i for i in profiler_steps if i in available_profiler_steps]
        if len(valid_profiler_steps) == 0:
            raise ValueError(
                f"invalid value for argument: profiler_steps={profiler_steps}; available profiler steps={available_profiler_steps}"
            )

        ranks = list(t.get_all_traces().keys())
        df_all = pd.concat([t.get_trace(r) for r in ranks], axis=0, keys=ranks, names=["rank", "idx"]).reset_index()

        df_selected_profiler_steps = df_all.loc[df_all["iteration"].isin(valid_profiler_steps)]

        straggler_counts = straggler_identification_impl(
            df_selected_profiler_steps, t.symbol_table, num_candidates, visualize
        )
        stragglers = straggler_counts.sort_values(ascending=False)[:num_candidates].index.tolist()

        if len(stragglers) > 1:
            logger.debug(f"found ranks {stragglers} are potential stragglers.")
        else:
            logger.debug(f"found rank {stragglers} is a potential straggler.")

        return stragglers
