# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from hta.memory_analysis import MemoryAnalysis


class MemoryAnalysisTestCase(unittest.TestCase):
    memory_analyzer: MemoryAnalysis

    @classmethod
    def setUpClass(cls) -> None:
        memory_events_path: str = "tests/data/memory_analysis/memory_timeline.raw.gz"
        cls.memory_analyzer = MemoryAnalysis(path=memory_events_path)

    def test_process_raw_events(self):
        times, sizes = self.memory_analyzer._process_raw_events()
        assert len(times) == len(sizes)
        assert len(sizes[0]) == 8
        assert sizes[744] == [
            102440608.0,
            102228128.0,
            19395584.0,
            0.0,
            2618160128.0,
            58417152.0,
            41811968.0,
            25494440.0,
        ]
