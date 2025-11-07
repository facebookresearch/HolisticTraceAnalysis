import os.path
import tempfile
import unittest

from hta.common.trace_collection import TraceCollection
from hta.common.trace_file import write_trace
from hta.configs.parser_config import AVAILABLE_ARGS, ParserConfig


class CustomTraceParserTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.rank = 1
        self.trace_data = {
            "distributedInfo": {"rank": self.rank},
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "aten::addmm",
                    "pid": 1850067,
                    "tid": 1850067,
                    "ts": 1689861288827753,
                    "dur": 130,
                    "args": {
                        "External id": 192,
                        "Ev Idx": 191,
                        "Input Dims": [[512], [512, 1408], [1408, 512], [], []],
                        "Input type": ["float", "float", "float", "Scalar", "Scalar"],
                        "Concrete Inputs": ["", "", "", "1", "1"],
                        "Fwd thread id": 0,
                        "Sequence number": 124343,
                    },
                },
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "sm80_kernel",
                    "pid": 1,
                    "tid": 7,
                    "ts": 1689861288841850,
                    "dur": 9,
                    "args": {
                        "External id": 1289495,
                        "queued": 0,
                        "device": 1,
                        "context": 1,
                        "stream": 7,
                        "correlation": 1289495,
                        "registers per thread": 128,
                        "shared memory": 13056,
                        "blocks per SM": 0.72727275,
                        "warps per SM": 2.909091,
                        "grid": [8, 12, 1],
                        "block": [128, 1, 1],
                        "est. achieved occupancy %": 5,
                    },
                },
                {
                    "ph": "X",
                    "cat": "gpu_memcpy",
                    "name": "Memcpy HtoD (Pinned -> Device)",
                    "pid": 1,
                    "tid": 20,
                    "ts": 1689861288825108,
                    "dur": 31,
                    "args": {
                        "External id": 1281474,
                        "device": 1,
                        "context": 1,
                        "stream": 20,
                        "correlation": 1281474,
                        "bytes": 1609728,
                        "memory bandwidth (GB/s)": 50.45694762248065,
                    },
                },
            ],
        }

    def _create_and_load_trace(self, trace_dir: str) -> TraceCollection:
        write_trace(self.trace_data, os.path.join(trace_dir, "trace_0.json.gz"))
        t = TraceCollection(trace_dir=trace_dir)
        t.parse_traces(use_multiprocessing=False)
        return t

    def test_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as t_dir:
            t = self._create_and_load_trace(t_dir)
            cfg = ParserConfig.get_default_cfg()
            df = t.get_trace_df(self.rank)
            self.assertTrue(all(arg.name in df.columns for arg in cfg.get_args()))

    def test_custom_config(self) -> None:
        """
        Note:
            Because ParserConfig.set_default_cfg() modifies a global variable, it normally should be
            only called once before parsing any trace to prevent accidentially removing some attributes
            needed by other analyzers within the same run.
        """
        original_args = ParserConfig.get_default_cfg().get_args().copy()
        args = [
            AVAILABLE_ARGS[key]
            for key in [
                "cuda::stream",
                "correlation::cpu_gpu",
                "data::bandwidth",
                "index::external_id",
                "kernel::queued",
                "cuda_sync::stream",
                "cuda_sync::event",
            ]
        ]
        custom_cfg = cfg = ParserConfig(args)
        ParserConfig.set_default_cfg(custom_cfg)
        current_arg_names = {
            arg.name for arg in ParserConfig.get_default_cfg().get_args()
        }
        removed_args = [arg for arg in original_args if arg not in current_arg_names]

        with tempfile.TemporaryDirectory() as t_dir:
            t = self._create_and_load_trace(t_dir)
            cfg = ParserConfig.get_default_cfg()
            df = t.get_trace_df(self.rank)
            self.assertTrue(all(arg.name in df.columns for arg in cfg.get_args()))
            self.assertFalse(all(arg.name in df.columns for arg in removed_args))
