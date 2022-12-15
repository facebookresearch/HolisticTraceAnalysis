# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# A script to convert pytorch profiler trace file to a Perfetto compatible trace file
# Usage example:
#   python3 convert_to_perfetto.py --input_file /trace/file.json.gz --output_file output.json.gz


import argparse
import gzip
import json
import logging
import time
from typing import Dict


def check_file_names(input_file, output_file):
    assert input_file.endswith(".gz") or input_file.endswith(".json"), "Input file must end with .json or .json.gz"
    assert output_file.endswith(".gz") or output_file.endswith(".json"), "Output file must end with .json.gz"


def _load_file(input_file: str) -> Dict:
    start_time = time.perf_counter()
    if input_file.endswith(".gz"):
        with gzip.open(input_file, "rb") as f1:
            trace_record = json.loads(f1.read())
    elif input_file.endswith(".json"):
        with open(input_file, "r") as f2:
            trace_record = json.loads(f2.read())
    else:
        raise ValueError(f"Input file ({input_file}) must ends with '.gz' or '.json'.")
    end_time = time.perf_counter()
    logging.info(f"Parsed {input_file} in {(end_time - start_time):.2f} seconds.")
    return trace_record


def _to_perfetto(input_data: Dict, input_file: str, output_file: str) -> None:
    trace_extraction_begins = time.perf_counter()
    output_data = {"traceEvents": input_data["traceEvents"]}
    dumped_json = json.dumps(output_data).encode("utf-8")
    trace_extraction_ends = time.perf_counter()
    logging.info(f"Trace events extracted in {(trace_extraction_ends - trace_extraction_begins):.2f} seconds.")

    result = "perfetto_%s" % input_file if output_file is None else output_file
    write_file_begins = time.perf_counter()
    with gzip.open(result, "wb") as f:
        f.write(dumped_json)
    write_file_ends = time.perf_counter()

    logging.info(f"Output gzip file name: {result} written in {(write_file_ends - write_file_begins):.2f} seconds.")
    logging.info("Converted to Perfetto!")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch Profiler trace to a Perfetto compatible trace",
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to trace file")
    parser.add_argument("--output_file", type=str, required=True, help="Name of output trace file")
    args = parser.parse_args()

    input_file, output_file = args.input_file, args.output_file
    check_file_names(input_file, output_file)
    input_data = _load_file(input_file)
    _to_perfetto(input_data, input_file, output_file)


if __name__ == "__main__":
    main()
