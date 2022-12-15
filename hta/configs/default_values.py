# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

# Default Paths
DEFAULT_TRACE_DIR = "/tmp/trace"
DEFAULT_CONFIG_FILENAME: str = "trace_analyzer.json"

# Trace related
DF_SYMBOL_COLUMNS: List[str] = ["cat", "name"]

# Runtime configurations
IS_DEBUG_ENABLED: bool = True
MAX_NUM_PROCESSES: int = 32
