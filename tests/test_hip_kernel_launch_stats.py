# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for HIP/ROCm support in cuda_kernel_launch_stats.

These tests verify that the `cuda_kernel_launch_stats` method in
`CudaKernelAnalysis` correctly handles HIP runtime events
(hipLaunchKernel, hipLaunchKernelExC, hipMemcpyAsync, hipMemsetAsync)
in addition to the existing CUDA and MTIA events.
"""

import unittest
from unittest.mock import MagicMock, PropertyMock

import pandas as pd

from hta.analyzers.cuda_kernel_analysis import CudaKernelAnalysis


def _create_mock_trace(sym_index: dict, trace_df: pd.DataFrame) -> MagicMock:
    """Helper to create a mock Trace object with the given symbol index and trace DataFrame."""
    mock_trace = MagicMock()
    mock_symbol_table = MagicMock()
    mock_symbol_table.get_sym_id_map.return_value = sym_index
    type(mock_trace).symbol_table = PropertyMock(return_value=mock_symbol_table)
    mock_trace.get_trace.return_value = trace_df
    return mock_trace


class TestHipKernelLaunchStats(unittest.TestCase):
    """Tests for HIP/ROCm support in CudaKernelAnalysis.cuda_kernel_launch_stats."""

    def test_hip_launch_kernel_events_are_captured(self):
        """Verify hipLaunchKernel events are included in the launch stats."""
        # Symbol IDs
        sym_index = {
            "hipLaunchKernel": 100,
        }

        # Create trace DataFrame: CPU runtime event + corresponding GPU kernel
        trace_df = pd.DataFrame(
            {
                "name": [100, 200],
                "stream": [-1, 1],
                "correlation": [1001, 1001],
                "dur": [10, 50],
                "ts": [100, 120],
            }
        )

        mock_trace = _create_mock_trace(sym_index, trace_df)
        result = CudaKernelAnalysis.cuda_kernel_launch_stats(
            mock_trace, ranks=[0], visualize=False
        )
        df = result[0]

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["cpu_duration"], 10)
        self.assertEqual(df.iloc[0]["gpu_duration"], 50)
        # launch_delay = gpu_ts - (cpu_ts + cpu_dur) = 120 - (100 + 10) = 10
        self.assertEqual(df.iloc[0]["launch_delay"], 10)

    def test_hip_launch_kernel_ex_c_events_are_captured(self):
        """Verify hipLaunchKernelExC events are included in the launch stats."""
        sym_index = {
            "hipLaunchKernelExC": 101,
        }

        trace_df = pd.DataFrame(
            {
                "name": [101, 201],
                "stream": [-1, 2],
                "correlation": [2001, 2001],
                "dur": [15, 80],
                "ts": [200, 230],
            }
        )

        mock_trace = _create_mock_trace(sym_index, trace_df)
        result = CudaKernelAnalysis.cuda_kernel_launch_stats(
            mock_trace, ranks=[0], visualize=False
        )
        df = result[0]

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["cpu_duration"], 15)
        self.assertEqual(df.iloc[0]["gpu_duration"], 80)
        # launch_delay = 230 - (200 + 15) = 15
        self.assertEqual(df.iloc[0]["launch_delay"], 15)

    def test_hip_memory_events_included(self):
        """Verify hipMemcpyAsync and hipMemsetAsync are included when include_memory_events=True."""
        sym_index = {
            "hipMemcpyAsync": 102,
            "hipMemsetAsync": 103,
        }

        trace_df = pd.DataFrame(
            {
                "name": [102, 103, 202, 203],
                "stream": [-1, -1, 3, 4],
                "correlation": [3001, 3002, 3001, 3002],
                "dur": [5, 8, 20, 30],
                "ts": [300, 400, 310, 415],
            }
        )

        mock_trace = _create_mock_trace(sym_index, trace_df)
        result = CudaKernelAnalysis.cuda_kernel_launch_stats(
            mock_trace, ranks=[0], include_memory_events=True, visualize=False
        )
        df = result[0]

        self.assertEqual(len(df), 2)
        # hipMemcpyAsync event
        memcpy_row = df[df["correlation"] == 3001].iloc[0]
        self.assertEqual(memcpy_row["cpu_duration"], 5)
        self.assertEqual(memcpy_row["gpu_duration"], 20)
        # launch_delay = 310 - (300 + 5) = 5
        self.assertEqual(memcpy_row["launch_delay"], 5)

        # hipMemsetAsync event
        memset_row = df[df["correlation"] == 3002].iloc[0]
        self.assertEqual(memset_row["cpu_duration"], 8)
        self.assertEqual(memset_row["gpu_duration"], 30)
        # launch_delay = 415 - (400 + 8) = 7
        self.assertEqual(memset_row["launch_delay"], 7)

    def test_hip_memory_events_excluded(self):
        """Verify hipMemcpyAsync/hipMemsetAsync are excluded when include_memory_events=False."""
        sym_index = {
            "hipLaunchKernel": 100,
            "hipMemcpyAsync": 102,
            "hipMemsetAsync": 103,
        }

        trace_df = pd.DataFrame(
            {
                "name": [100, 102, 103, 200, 202, 203],
                "stream": [-1, -1, -1, 1, 3, 4],
                "correlation": [1001, 3001, 3002, 1001, 3001, 3002],
                "dur": [10, 5, 8, 50, 20, 30],
                "ts": [100, 300, 400, 120, 310, 415],
            }
        )

        mock_trace = _create_mock_trace(sym_index, trace_df)
        result = CudaKernelAnalysis.cuda_kernel_launch_stats(
            mock_trace, ranks=[0], include_memory_events=False, visualize=False
        )
        df = result[0]

        # Only hipLaunchKernel should be captured, not the memory events
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["correlation"], 1001)

    def test_mixed_cuda_and_hip_events(self):
        """Verify both CUDA and HIP events are captured together in a mixed trace."""
        sym_index = {
            "cudaLaunchKernel": 10,
            "hipLaunchKernel": 100,
            "cudaMemcpyAsync": 11,
            "hipMemcpyAsync": 102,
        }

        trace_df = pd.DataFrame(
            {
                "name": [10, 100, 11, 102, 210, 2100, 211, 2102],
                "stream": [-1, -1, -1, -1, 1, 2, 3, 4],
                "correlation": [
                    1001,
                    2001,
                    1002,
                    2002,
                    1001,
                    2001,
                    1002,
                    2002,
                ],
                "dur": [10, 12, 5, 6, 50, 60, 20, 25],
                "ts": [100, 200, 300, 400, 120, 220, 310, 410],
            }
        )

        mock_trace = _create_mock_trace(sym_index, trace_df)
        result = CudaKernelAnalysis.cuda_kernel_launch_stats(
            mock_trace, ranks=[0], include_memory_events=True, visualize=False
        )
        df = result[0]

        # All 4 events should be captured
        self.assertEqual(len(df), 4)
        correlations = sorted(df["correlation"].tolist())
        self.assertEqual(correlations, [1001, 1002, 2001, 2002])

    def test_negative_launch_delay_clipped_to_zero(self):
        """Verify that negative launch delays are clipped to 0 for HIP events."""
        sym_index = {
            "hipLaunchKernel": 100,
        }

        # GPU kernel starts before CPU op ends (overlap)
        trace_df = pd.DataFrame(
            {
                "name": [100, 200],
                "stream": [-1, 1],
                "correlation": [1001, 1001],
                "dur": [20, 50],
                "ts": [100, 105],  # gpu starts at 105, cpu ends at 120
            }
        )

        mock_trace = _create_mock_trace(sym_index, trace_df)
        result = CudaKernelAnalysis.cuda_kernel_launch_stats(
            mock_trace, ranks=[0], visualize=False
        )
        df = result[0]

        # launch_delay = 105 - (100 + 20) = -15, should be clipped to 0
        self.assertEqual(df.iloc[0]["launch_delay"], 0)

    def test_hip_only_no_cuda_symbols(self):
        """Verify the function works when only HIP symbols exist (pure ROCm trace)."""
        sym_index = {
            "hipLaunchKernel": 100,
            "hipLaunchKernelExC": 101,
            "hipMemcpyAsync": 102,
            "hipMemsetAsync": 103,
        }

        trace_df = pd.DataFrame(
            {
                "name": [100, 101, 102, 103, 200, 201, 202, 203],
                "stream": [-1, -1, -1, -1, 1, 2, 3, 4],
                "correlation": [
                    1001,
                    1002,
                    1003,
                    1004,
                    1001,
                    1002,
                    1003,
                    1004,
                ],
                "dur": [10, 12, 5, 8, 50, 60, 20, 30],
                "ts": [100, 200, 300, 400, 115, 220, 310, 415],
            }
        )

        mock_trace = _create_mock_trace(sym_index, trace_df)
        result = CudaKernelAnalysis.cuda_kernel_launch_stats(
            mock_trace, ranks=[0], include_memory_events=True, visualize=False
        )
        df = result[0]

        self.assertEqual(len(df), 4)

        # Verify each event
        row_1001 = df[df["correlation"] == 1001].iloc[0]
        self.assertEqual(row_1001["cpu_duration"], 10)
        self.assertEqual(row_1001["gpu_duration"], 50)
        self.assertEqual(row_1001["launch_delay"], 5)  # 115 - 110

        row_1002 = df[df["correlation"] == 1002].iloc[0]
        self.assertEqual(row_1002["cpu_duration"], 12)
        self.assertEqual(row_1002["gpu_duration"], 60)
        self.assertEqual(row_1002["launch_delay"], 8)  # 220 - 212

        row_1003 = df[df["correlation"] == 1003].iloc[0]
        self.assertEqual(row_1003["cpu_duration"], 5)
        self.assertEqual(row_1003["gpu_duration"], 20)
        self.assertEqual(row_1003["launch_delay"], 5)  # 310 - 305

        row_1004 = df[df["correlation"] == 1004].iloc[0]
        self.assertEqual(row_1004["cpu_duration"], 8)
        self.assertEqual(row_1004["gpu_duration"], 30)
        self.assertEqual(row_1004["launch_delay"], 7)  # 415 - 408

    def test_empty_trace_returns_empty_result(self):
        """Verify behavior with a trace that has no matching launch events."""
        sym_index = {
            "hipLaunchKernel": 100,
        }

        # Trace with no events matching hipLaunchKernel
        trace_df = pd.DataFrame(
            {
                "name": [999, 998],
                "stream": [-1, 1],
                "correlation": [1001, 1001],
                "dur": [10, 50],
                "ts": [100, 120],
            }
        )

        mock_trace = _create_mock_trace(sym_index, trace_df)
        result = CudaKernelAnalysis.cuda_kernel_launch_stats(
            mock_trace, ranks=[0], visualize=False
        )
        df = result[0]

        self.assertEqual(len(df), 0)

    def test_multiple_ranks_with_hip_events(self):
        """Verify HIP events are processed correctly for multiple ranks."""
        sym_index = {
            "hipLaunchKernel": 100,
        }

        trace_df_rank0 = pd.DataFrame(
            {
                "name": [100, 200],
                "stream": [-1, 1],
                "correlation": [1001, 1001],
                "dur": [10, 50],
                "ts": [100, 120],
            }
        )

        trace_df_rank1 = pd.DataFrame(
            {
                "name": [100, 200],
                "stream": [-1, 1],
                "correlation": [2001, 2001],
                "dur": [15, 70],
                "ts": [200, 225],
            }
        )

        mock_trace = _create_mock_trace(sym_index, trace_df_rank0)
        mock_trace.get_trace.side_effect = lambda rank: (
            trace_df_rank0 if rank == 0 else trace_df_rank1
        )

        result = CudaKernelAnalysis.cuda_kernel_launch_stats(
            mock_trace, ranks=[0, 1], visualize=False
        )

        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[0].iloc[0]["gpu_duration"], 50)
        self.assertEqual(result[1].iloc[0]["gpu_duration"], 70)


if __name__ == "__main__":
    unittest.main()
