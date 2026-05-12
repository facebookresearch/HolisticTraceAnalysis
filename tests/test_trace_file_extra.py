# Copyright (c) Meta Platforms, Inc. and affiliates.

import gzip
import json
import os
import tempfile
import unittest

from hta.common.trace_file import (
    create_rank_to_trace_dict,
    create_rank_to_trace_dict_from_dir,
    get_trace_files,
    read_trace,
    update_trace_rank,
    write_trace,
)


def _write_json_with_rank(path: str, rank: int) -> None:
    data = {"distributedInfo": {"rank": rank}, "traceEvents": []}
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            f.write(json.dumps(data).encode())
    else:
        with open(path, "w") as f:
            json.dump(data, f)


class TestCreateRankToTraceDictFromDir(unittest.TestCase):
    def test_path_does_not_exist(self) -> None:
        ok, d = create_rank_to_trace_dict_from_dir("/no/such/dir")
        self.assertFalse(ok)
        self.assertEqual(d, {})

    def test_no_trace_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "readme.txt"), "w").close()
            ok, d = create_rank_to_trace_dict_from_dir(tmp)
            self.assertFalse(ok)

    def test_finds_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_json_with_rank(os.path.join(tmp, "rank0.json"), 0)
            _write_json_with_rank(os.path.join(tmp, "rank1.json.gz"), 1)
            ok, d = create_rank_to_trace_dict_from_dir(tmp)
            self.assertTrue(ok)
            self.assertEqual(set(d.keys()), {0, 1})


class TestCreateRankToTraceDict(unittest.TestCase):
    def test_duplicate_rank_warns_and_keeps_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f1 = os.path.join(tmp, "a.json")
            f2 = os.path.join(tmp, "b.json")
            _write_json_with_rank(f1, 5)
            _write_json_with_rank(f2, 5)
            ok, d = create_rank_to_trace_dict([f1, f2])
            self.assertTrue(ok)
            # Last one wins
            self.assertEqual(d[5], f2)

    def test_no_rank_in_file_defaults_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "x.json")
            with open(f, "w") as fh:
                json.dump({"traceEvents": []}, fh)
            ok, d = create_rank_to_trace_dict([f])
            self.assertTrue(ok)
            self.assertEqual(d.get(0), f)


class TestGetTraceFiles(unittest.TestCase):
    def test_invalid_path_returns_empty(self) -> None:
        self.assertEqual(get_trace_files("/no/such/dir"), {})

    def test_empty_dir_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(get_trace_files(tmp), {})

    def test_finds_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_json_with_rank(os.path.join(tmp, "rank0.json"), 0)
            d = get_trace_files(tmp)
            self.assertEqual(set(d.keys()), {0})


class TestReadAndWriteTrace(unittest.TestCase):
    def test_roundtrip_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.json")
            data = {"distributedInfo": {"rank": 2}, "traceEvents": [1]}
            write_trace(data, path)
            self.assertEqual(read_trace(path), data)

    def test_roundtrip_gz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.json.gz")
            data = {"distributedInfo": {"rank": 3}, "traceEvents": [2]}
            write_trace(data, path)
            self.assertEqual(read_trace(path), data)

    def test_invalid_extension_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\.gz' or 'json'"):
            read_trace("/tmp/x.txt")

    def test_write_creates_missing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "nested", "deep")
            path = os.path.join(new_dir, "trace.json")
            write_trace({"a": 1}, path)
            self.assertTrue(os.path.exists(path))


class TestUpdateTraceRank(unittest.TestCase):
    def test_updates_existing_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.json")
            _write_json_with_rank(path, 0)
            update_trace_rank(path, 7)
            self.assertEqual(read_trace(path)["distributedInfo"]["rank"], 7)

    def test_adds_distributed_info_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.json")
            with open(path, "w") as f:
                json.dump({"traceEvents": []}, f)
            update_trace_rank(path, 4)
            self.assertEqual(read_trace(path)["distributedInfo"]["rank"], 4)
