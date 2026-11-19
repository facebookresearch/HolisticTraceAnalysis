# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
import unittest
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import patch

from hta.common.trace import Trace
from hta.utils import utils

TRACE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data/vision_transformer"
)


def _die(*_args, **_kwargs):
    os.kill(os.getpid(), signal.SIGKILL)  # what the OOM killer does


class TestParsePool(unittest.TestCase):
    def test_killed_worker_raises_instead_of_hanging(self) -> None:
        t = Trace(trace_dir=TRACE_DIR)
        with patch("hta.common.trace.parse_trace_file", _die):
            with self.assertRaises(BrokenProcessPool):
                t.parse_multiple_ranks([0, 1], use_multiprocessing=True)

    def test_pool_size_follows_memory_for_few_ranks(self) -> None:
        t = Trace(trace_dir=TRACE_DIR)
        sizes = []
        real_get = utils.get_mp_pool_size

        def spy(obj_size, num_objs):
            sizes.append(real_get(obj_size, num_objs))
            return sizes[-1]

        # Room for exactly one parse (0.8 * available / estimate == 1.x).
        one_parse = utils.estimate_parse_memory(t.trace_files[0])
        with patch("hta.common.trace.get_mp_pool_size", spy):
            with patch(
                "hta.utils.utils.get_available_memory",
                return_value=int(1.5 * one_parse),
            ):
                t.parse_multiple_ranks([0, 1, 2], use_multiprocessing=True)
        self.assertEqual(sizes, [1])
        self.assertEqual(len(t.traces), 3)


class TestAvailableMemory(unittest.TestCase):
    def _patch_files(self, files):
        def fake_read_text(path, *a, **k):
            try:
                return files[str(path)]
            except KeyError:
                raise FileNotFoundError(str(path))

        return patch("hta.utils.utils.Path.read_text", fake_read_text)

    def test_cgroup_v2_limit_caps_host_memory(self) -> None:
        files = {
            "/sys/fs/cgroup/memory.max": "1000\n",
            "/sys/fs/cgroup/memory.current": "400\n",
        }
        with self._patch_files(files):
            with patch("hta.utils.utils.psutil.virtual_memory") as vm:
                vm.return_value.available = 10**12
                self.assertEqual(utils.get_available_memory(), 600)

    def test_unlimited_cgroup_falls_back_to_host(self) -> None:
        files = {
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/memory.current": "400\n",
        }
        with self._patch_files(files):
            with patch("hta.utils.utils.psutil.virtual_memory") as vm:
                vm.return_value.available = 12345
                self.assertEqual(utils.get_available_memory(), 12345)

    def test_pool_size_never_below_one(self) -> None:
        self.assertEqual(utils.get_mp_pool_size(10**18, 8), 1)


if __name__ == "__main__":
    unittest.main()
