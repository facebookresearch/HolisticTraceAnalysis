# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import pandas as pd
from hta.analyzers.trace_counters import TraceCounters

_MEMCPY = "Memcpy HtoD (Pageable -> Device)"
_MEMSET = "Memset (Device)"
_LAUNCH = "cudaLaunchKernel"
_KERNEL = "gemm"
_NAMES: list[str] = [_MEMCPY, _MEMSET, _LAUNCH, _KERNEL]


def _mock_trace(trace_df: pd.DataFrame) -> MagicMock:
    """A Trace whose rank 0 is trace_df, with names indexing _NAMES."""
    trace = MagicMock()
    trace.get_trace.return_value = trace_df
    trace.symbol_table.get_sym_table.return_value = _NAMES
    trace.symbol_table.get_runtime_launch_events_mask.side_effect = lambda df: df[
        "name"
    ] == _NAMES.index(_LAUNCH)
    return trace


def _shuffled(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=7).reset_index(drop=True)


def _copy(ts: int, dur: int, bandwidth: float, pid: int = 0) -> dict[str, object]:
    return {
        "ts": ts,
        "dur": dur,
        "name": _NAMES.index(_MEMCPY),
        "stream": 7,
        "pid": pid,
        "memory_bw_gbps": bandwidth,
    }


def _entry(
    launch_ts: int, gpu_ts: int, correlation: int, stream: int = 7
) -> list[dict[str, int]]:
    """One queue entry: a launch and the start of its kernel."""
    common = {"dur": 1, "correlation": correlation, "pid": 0}
    return [
        {
            **common,
            "ts": launch_ts,
            "tid": 1,
            "name": _NAMES.index(_LAUNCH),
            "stream": -1,
        },
        {
            **common,
            "ts": gpu_ts,
            "tid": stream,
            "name": _NAMES.index(_KERNEL),
            "stream": stream,
        },
    ]


def _indexed(rows: list[dict[str, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["index"] = df.index
    return df


def _memory_bw(copies: pd.DataFrame) -> pd.DataFrame:
    series = TraceCounters._get_memory_bw_time_series_for_rank(_mock_trace(copies), 0)
    assert series is not None
    return series.reset_index(drop=True)


def _queue_length(trace_df: pd.DataFrame) -> pd.DataFrame:
    series = TraceCounters._get_queue_length_time_series_for_rank(
        _mock_trace(trace_df), 0
    )
    assert series is not None
    return series.reset_index(drop=True)


class TraceCountersTieOrderTest(unittest.TestCase):
    """Points at one timestamp count in a fixed order, whatever the row order."""

    @staticmethod
    def _copies() -> pd.DataFrame:
        # 400 memory kernels, one every 10 us. Every second one is shorter than
        # a microsecond, so its rounded start equals its rounded end.
        rows = [
            {
                "ts": 10 * i,
                "dur": 0 if i % 2 == 0 else 5,
                "name": _NAMES.index(_MEMSET if i % 4 == 3 else _MEMCPY),
                "stream": 7,
                "pid": 0,
                "memory_bw_gbps": float(1 + i % 7),
            }
            for i in range(400)
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _sorted_memory_bw(trace_df: pd.DataFrame) -> pd.DataFrame:
        return _memory_bw(trace_df).sort_values(
            ["name", "ts", "memory_bw_gbps"], kind="stable"
        )

    def test_memory_bw_counts_a_start_before_an_end(self) -> None:
        copies = self._copies()
        series = self._sorted_memory_bw(copies)

        self.assertFalse(
            (series["memory_bw_gbps"] < 0).any(),
            "a bandwidth level is never negative",
        )
        positive = series[series["memory_bw_gbps"] > 0]
        # Copies neither overlap nor touch: each one gives exactly one positive
        # point, its own bandwidth, zero-length copies included.
        self.assertEqual(len(positive), len(copies))
        self.assertEqual(
            sorted(positive["memory_bw_gbps"].tolist()),
            sorted(copies["memory_bw_gbps"].tolist()),
        )

    def test_memory_bw_does_not_depend_on_row_order(self) -> None:
        copies = self._copies()
        forward = self._sorted_memory_bw(copies).reset_index(drop=True)
        shuffled = self._sorted_memory_bw(_shuffled(copies)).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            forward[["ts", "name", "memory_bw_gbps"]],
            shuffled[["ts", "name", "memory_bw_gbps"]],
        )

    @staticmethod
    def _launches() -> pd.DataFrame:
        # 300 kernels, each starting in the microsecond its launch does.
        rows: list[dict[str, int]] = []
        for i in range(300):
            common = {"ts": 10 * i, "dur": 1, "correlation": i, "pid": 0, "tid": 1}
            rows.append({**common, "name": _NAMES.index(_LAUNCH), "stream": -1})
            rows.append({**common, "name": _NAMES.index(_KERNEL), "stream": 7})
        df = pd.DataFrame(rows)
        df["index"] = df.index
        return df

    def test_queue_length_counts_a_launch_before_its_kernel(self) -> None:
        series = _queue_length(self._launches())
        self.assertFalse(
            (series["queue_length"] < 0).any(), "a queue is never negative"
        )
        # Each launch takes the queue to 1 and its kernel back to 0.
        self.assertEqual(series["queue_length"].tolist(), [1, 0] * 300)

    def test_queue_length_does_not_depend_on_row_order(self) -> None:
        launches = self._launches()
        shuffled = _shuffled(launches)
        shuffled["index"] = shuffled.index
        self.assertEqual(
            _queue_length(shuffled)["queue_length"].tolist(),
            _queue_length(launches)["queue_length"].tolist(),
        )


class TraceCountersIntervalTieTest(unittest.TestCase):
    """At one timestamp, ends of intervals of positive length count first,
    then starts, then each zero-length interval, its start just before its
    end. No level adds an interval that ends at a timestamp to one that starts
    there."""

    def test_adjacent_copies_never_add(self) -> None:
        # A [0, 5) at 10 GB/s, then B [5, 9) at 20 GB/s.
        copies = pd.DataFrame([_copy(0, 5, 10.0), _copy(5, 4, 20.0)])
        series = _memory_bw(copies)
        self.assertEqual(series["ts"].tolist(), [0, 5, 5, 9])
        self.assertEqual(
            series["memory_bw_gbps"].tolist(),
            [10.0, 0.0, 20.0, 0.0],
            "A ends before B starts: the level 30 never exists",
        )
        summary = TraceCounters.get_memory_bw_summary(_mock_trace(copies), [0])
        assert summary is not None
        stats = summary.loc[(0, "Memcpy HtoD"), "memory_bw_gbps"]
        self.assertEqual(stats["count"], 2, "one positive point per copy")
        self.assertEqual(stats["max"], 20.0)

    def test_zero_length_copies_at_one_timestamp_keep_their_own_points(self) -> None:
        # C and D are zero-length copies at 3 us, inside E [0, 10).
        copies = pd.DataFrame([_copy(3, 0, 2.0), _copy(0, 10, 4.0), _copy(3, 0, 1.0)])
        series = _memory_bw(copies)
        self.assertEqual(series["ts"].tolist(), [0, 3, 3, 3, 3, 10])
        self.assertEqual(
            series["memory_bw_gbps"].tolist(),
            [4.0, 5.0, 4.0, 6.0, 4.0, 0.0],
            "each zero-length copy adds to E alone, never to the other",
        )

    def test_memory_bw_rows_do_not_depend_on_row_order(self) -> None:
        copies = pd.DataFrame(
            [
                _copy(0, 5, 10.0),
                _copy(5, 4, 20.0),
                _copy(5, 4, 20.0, pid=1),
                _copy(5, 0, 3.0),
                _copy(5, 0, 3.0),
                _copy(0, 9, 7.0),
                _copy(9, 0, 1.0),
                _copy(2, 3, 10.0),
            ]
        )
        forward = _memory_bw(copies)
        for seed in range(5):
            shuffled = _memory_bw(
                copies.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(forward, shuffled)
        self.assertFalse((forward["memory_bw_gbps"] < 0).any())

    def test_a_queue_entry_that_ends_does_not_add_to_one_that_starts(self) -> None:
        # Y is launched at 0 and starts at 5; X is launched at 5 and starts
        # at 9, on the same stream.
        series = _queue_length(_indexed(_entry(0, 5, 1) + _entry(5, 9, 2)))
        self.assertEqual(series["ts"].tolist(), [0, 5, 5, 9])
        self.assertEqual(
            series["queue_length"].tolist(),
            [1, 0, 1, 0],
            "Y leaves the queue as X enters: the length 2 never exists",
        )

    def test_kernels_launched_and_started_in_one_microsecond_keep_their_points(
        self,
    ) -> None:
        series = _queue_length(_indexed(_entry(4, 4, 1) + _entry(4, 4, 2)))
        self.assertEqual(series["queue_length"].tolist(), [1, 0, 1, 0])

    def test_queue_length_rows_do_not_depend_on_row_order(self) -> None:
        rows = (
            _entry(0, 5, 1)
            + _entry(5, 9, 2)
            + _entry(5, 5, 3)
            + _entry(5, 5, 4)
            + _entry(1, 9, 5)
            + _entry(9, 12, 6, stream=8)
            + _entry(2, 9, 7, stream=8)
        )
        forward = _queue_length(_indexed(rows))
        for seed in range(5):
            shuffled = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)
            shuffled = _indexed(shuffled.to_dict("records"))
            pd.testing.assert_frame_equal(forward, _queue_length(shuffled))
        self.assertFalse((forward["queue_length"] < 0).any())
