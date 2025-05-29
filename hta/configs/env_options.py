# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Optional


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


class HTAEnvOptions:
    """Singleton class that manages HTA environment options.

    This class reads environment variables when initialized and provides
    methods to access and modify the options. Use the instance() method
    to get the singleton instance.
    """

    _instance = None

    def __init__(self):
        """Initialize options from environment variables."""
        # Read environment variables
        self._options: Dict[str, bool] = {}
        self._initialize_options()

    def _initialize_options(self) -> None:
        """Initialize options from environment variables."""
        self._options = {
            HTA_DISABLE_NS_ROUNDING_ENV: self._check_env_flag(
                HTA_DISABLE_NS_ROUNDING_ENV, "0"
            ),
            HTA_DISABLE_CG_DEPTH_ENV: self._check_env_flag(
                HTA_DISABLE_CG_DEPTH_ENV, "0"
            ),
            CP_LAUNCH_EDGE_ENV: self._check_env_flag(CP_LAUNCH_EDGE_ENV, "0"),
            CP_LAUNCH_EDGE_SHOW_ENV: self._check_env_flag(CP_LAUNCH_EDGE_SHOW_ENV, "0"),
            CP_STRICT_NEG_WEIGHT_CHECK_ENV: self._check_env_flag(
                CP_STRICT_NEG_WEIGHT_CHECK_ENV, "0"
            ),
        }

    @classmethod
    def instance(cls) -> "HTAEnvOptions":
        """Get the singleton instance of HTAEnvOptions."""
        if cls._instance is None:
            cls._instance = HTAEnvOptions()
        return cls._instance

    def _get_env(self, name: str) -> Optional[str]:
        """Checks for env or returns None"""
        return os.environ.get(name)

    def _check_env_flag(self, name: str, default: str = "0") -> bool:
        """Checks if env flag is "1" """
        if (value := self._get_env(name)) is None:
            value = default
        return value == "1"

    def disable_ns_rounding(self) -> bool:
        """Check if nanosecond rounding is disabled."""
        return self._options[HTA_DISABLE_NS_ROUNDING_ENV]

    def set_disable_ns_rounding(self, value: bool) -> None:
        """Set whether nanosecond rounding is disabled."""
        self._options[HTA_DISABLE_NS_ROUNDING_ENV] = value

    def disable_call_graph_depth(self) -> bool:
        """Check if call graph depth is disabled."""
        return self._options[HTA_DISABLE_CG_DEPTH_ENV]

    def set_disable_call_graph_depth(self, value: bool) -> None:
        """Set whether call graph depth is disabled."""
        self._options[HTA_DISABLE_CG_DEPTH_ENV] = value

    def critical_path_add_zero_weight_launch_edges(self) -> bool:
        """Check if zero weight launch edges should be added for critical path analysis."""
        return self._options[CP_LAUNCH_EDGE_ENV]

    def set_critical_path_add_zero_weight_launch_edges(self, value: bool) -> None:
        """Set whether zero weight launch edges should be added for critical path analysis."""
        self._options[CP_LAUNCH_EDGE_ENV] = value

    def critical_path_show_zero_weight_launch_edges(self) -> bool:
        """Check if zero weight launch edges should be shown in overlaid trace."""
        return self._options[CP_LAUNCH_EDGE_SHOW_ENV]

    def set_critical_path_show_zero_weight_launch_edges(self, value: bool) -> None:
        """Set whether zero weight launch edges should be shown in overlaid trace."""
        self._options[CP_LAUNCH_EDGE_SHOW_ENV] = value

    def critical_path_strict_negative_weight_check(self) -> bool:
        """Check if strict negative weight checking is enabled for critical path analysis."""
        return self._options[CP_STRICT_NEG_WEIGHT_CHECK_ENV]

    def set_critical_path_strict_negative_weight_check(self, value: bool) -> None:
        """Set whether strict negative weight checking is enabled for critical path analysis."""
        self._options[CP_STRICT_NEG_WEIGHT_CHECK_ENV] = value

    def get_options_str(self) -> str:
        """Get a string representation of all options."""

        def get_env(name: str) -> str:
            return self._get_env(name) or "unset"

        return f"""
disable_ns_rounding={self.disable_ns_rounding()}, HTA_DISABLE_NS_ROUNDING_ENV={get_env(HTA_DISABLE_NS_ROUNDING_ENV)}
disable_call_graph_depth={self.disable_call_graph_depth()}, HTA_DISABLE_CG_DEPTH_ENV={get_env(HTA_DISABLE_CG_DEPTH_ENV)}
critical_path_add_zero_weight_launch_edges={self.critical_path_add_zero_weight_launch_edges()}, CP_LAUNCH_EDGE_ENV={get_env(CP_LAUNCH_EDGE_ENV)}
critical_path_show_zero_weight_launch_edges={self.critical_path_show_zero_weight_launch_edges()}, CP_LAUNCH_EDGE_SHOW_ENV={get_env(CP_LAUNCH_EDGE_SHOW_ENV)}
critical_path_strict_negative_weight_check={self.critical_path_strict_negative_weight_check()}, CP_STRICT_NEG_WEIGHT_CHECK_ENV={get_env(CP_STRICT_NEG_WEIGHT_CHECK_ENV)}
"""


def disable_ns_rounding() -> bool:
    return HTAEnvOptions.instance().disable_ns_rounding()


def disable_call_graph_depth() -> bool:
    return HTAEnvOptions.instance().disable_call_graph_depth()


def critical_path_add_zero_weight_launch_edges() -> bool:
    return HTAEnvOptions.instance().critical_path_add_zero_weight_launch_edges()


def critical_path_show_zero_weight_launch_edges() -> bool:
    return HTAEnvOptions.instance().critical_path_show_zero_weight_launch_edges()


def critical_path_strict_negative_weight_check() -> bool:
    return HTAEnvOptions.instance().critical_path_strict_negative_weight_check()


def get_options() -> str:
    return HTAEnvOptions.instance().get_options_str()
