#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pyperf

from hta.common.trace import Trace
from hta.configs.config import logger

_TRACE_DIRS = ["vision_transformer", "inference_single_rank"]
TRACE_DIRS = [f"tests/data/{d}" for d in _TRACE_DIRS]

# For large number of iterations this makes the logs more readable,
# feel free to change this.
logger.setLevel(logging.ERROR)


def load_and_parse_trace(
    loops: int, trace_dir: str, max_ranks: int = 1, use_multiprocessing: bool = False
):
    """Runs trace load and parsing for a single rank. This helps measure
    and optimize the json load / dataframe construction code path.

    We disable multiprocessing load by default so the benchmarking tool
    can measure memory footprint, this is fine since we only load one rank.
    """
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        trace = Trace(trace_dir=trace_dir)
        trace.parse_traces(max_ranks=max_ranks, use_multiprocessing=use_multiprocessing)
    return pyperf.perf_counter() - t0


runner = pyperf.Runner()
for trace_dir in TRACE_DIRS:
    runner.bench_time_func(
        f"parse[{trace_dir}]", load_and_parse_trace, trace_dir, inner_loops=1
    )
