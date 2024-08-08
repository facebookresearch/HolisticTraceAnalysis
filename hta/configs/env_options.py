# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from typing import Optional


""" HTA provides a set of options to modify behavior of the analyzers using environmenent variables.
Note that all env variables are treated as strings, so for flag like options the values will be "1" or "0".
"""

# Disables the rounding out of nanosecond precision traces.
# The rounding was added due to the events overlapping when the precision of events was increased.
HTA_DISABLE_NS_ROUNDING_ENV = "HTA_DISABLE_NS_ROUNDING"

# Disable adding CG depth in hta/common/call_stack.py
HTA_DISABLE_CG_DEPTH_ENV = "HTA_DISABLE_CG_DEPTH"

# -- Critical path analysis --
# Add zero weight launch edges for causality.
CP_LAUNCH_EDGE_ENV = "CRITICAL_PATH_ADD_ZERO_WEIGHT_LAUNCH_EDGE"
# Show zero weight launch edges in overlaid trace.
CP_LAUNCH_EDGE_SHOW_ENV = "CRITICAL_PATH_SHOW_ZERO_WEIGHT_LAUNCH_EDGE"

# Fail when edges have negative weight, rather than correcting them to 0
CP_STRICT_NEG_WEIGHT_CHECK_ENV = "CRITICAL_PATH_STRICT_NEGATIVE_WEIGHT_CHECKS"


def _get_env(name: str) -> Optional[str]:
    """Checks for env or returns None"""
    return os.environ.get(name)


def _check_env_flag(name: str, default: str = "0") -> bool:
    """Checks if env flag is "1" """
    if (value := _get_env(name)) is None:
        value = default
    return value == "1"


def disable_ns_rounding() -> bool:
    return _check_env_flag(HTA_DISABLE_NS_ROUNDING_ENV, "0")


def disable_call_graph_depth() -> bool:
    return _check_env_flag(HTA_DISABLE_CG_DEPTH_ENV, "0")


def critical_path_add_zero_weight_launch_edges() -> bool:
    return _check_env_flag(CP_LAUNCH_EDGE_ENV, "0")


def critical_path_show_zero_weight_launch_edges() -> bool:
    return _check_env_flag(CP_LAUNCH_EDGE_SHOW_ENV, "0")


def critical_path_strict_negative_weight_check() -> bool:
    return _check_env_flag(CP_STRICT_NEG_WEIGHT_CHECK_ENV, "0")


def get_options() -> str:
    def get_env(name: str) -> str:
        return _get_env(name) or "unset"

    return f"""
disable_ns_rounding={disable_ns_rounding()}, HTA_DISABLE_NS_ROUNDING_ENV={get_env(HTA_DISABLE_NS_ROUNDING_ENV)}
disable_call_graph_depth={disable_call_graph_depth()}, HTA_DISABLE_CG_DEPTH_ENV={get_env(HTA_DISABLE_CG_DEPTH_ENV)}
critical_path_add_zero_weight_launch_edges={critical_path_add_zero_weight_launch_edges()}, CP_LAUNCH_EDGE_ENV={get_env(CP_LAUNCH_EDGE_ENV)}
critical_path_show_zero_weight_launch_edges={critical_path_show_zero_weight_launch_edges()}, CP_LAUNCH_EDGE_SHOW_ENV={get_env(CP_LAUNCH_EDGE_SHOW_ENV)}
critical_path_strict_negative_weight_check={critical_path_strict_negative_weight_check()}, CP_STRICT_NEG_WEIGHT_CHECK_ENV={get_env(CP_STRICT_NEG_WEIGHT_CHECK_ENV)}
"""
