import os
import unittest
from pathlib import Path

from hta.analyzers.cupti_counter_analysis import CUDA_SASS_INSTRUCTION_COUNTER_FLOPS
from hta.trace_analysis import TraceAnalysis


class CuptiTestCase(unittest.TestCase):
    def setUp(self):
        self.base_data_dir = str(Path(__file__).parent.parent.joinpath("tests/data"))

    def test_get_cupti_counter_data_with_operators(self):
        # regular trace should return empty list since it will not have cuda_profiler_range events
        inference_t = TraceAnalysis(
            trace_dir=os.path.join(self.base_data_dir, "inference_single_rank")
        )
        self.assertEqual(inference_t.get_cupti_counter_data_with_operators(), [])

        cupti_profiler_trace_dir: str = os.path.join(
            self.base_data_dir, "cupti_profiler"
        )
        cupti_profiler_t = TraceAnalysis(trace_dir=cupti_profiler_trace_dir)
        counters_df = cupti_profiler_t.get_cupti_counter_data_with_operators()[0]

        self.assertEqual(len(counters_df), 77)

        # Example trace has CUPTI SASS FLOPS instruction counters
        counter_names = set(CUDA_SASS_INSTRUCTION_COUNTER_FLOPS.keys())
        self.assertEqual(
            set(counters_df.columns.unique()) & counter_names, counter_names
        )

        # Pick 5th kernel that executed.
        test_row = counters_df.sort_values(axis=0, by="ts").iloc[5].to_dict()

        self.assertEqual(test_row["cat"], "cuda_profiler_range")
        self.assertTrue("fft2d_r2c_32x32" in test_row["name"])

        self.assertEqual(
            test_row["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"], 86114304
        )
        self.assertEqual(test_row["top_level_op"], "aten::conv2d")
        # self.assertEqual(test_row["bottom_level_op"], "aten::_convolution")
        self.assertEqual(test_row["bottom_level_op"], "aten::cudnn_convolution")


if __name__ == "__main__":
    unittest.main()
