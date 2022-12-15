# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import Any, Dict, Tuple

from hta.configs.config import logger


def create_rank_to_trace_dict(trace_path: str) -> Tuple[bool, Dict]:
    """
    Create a rank -> trace_filename map for traces located within the directory <trace_path>

    Args:
        trace_path (str) : the path to the directory where the traces are located.

    Returns:
        (success: bool, rank_trace_map: dict) : a tuple indicating whether the operation succeeds and the path
         to the map file being created.
            success (bool) : True if the file is created successfully; False otherwise.
            map_file_path (dict) : a dict with rank as key and trace_filename as the value
    """

    if not (os.path.exists(trace_path) and os.access(trace_path, os.R_OK)):
        logger.error(f"trace_path {trace_path} doesn't exist or is not readable")
        return False, {}

    file_list = [fn for fn in os.listdir(trace_path) if fn.endswith(".gz") or fn.endswith(".json")]
    if len(file_list) == 0:
        logger.warning(f"No trace file is found in {trace_path}")
        return False, {}

    rank_to_trace_dict: Dict[int, str] = {}
    for file in file_list:
        filename = os.path.join(trace_path, file)
        data = None
        if file.endswith("gz"):
            gzip_file_handle = gzip.open(filename, "rb")
            data = json.loads(gzip_file_handle.read())
        if file.endswith("json"):
            json_file_handle = open(filename, "r")
            data = json.loads(json_file_handle.read())

        if "distributedInfo" not in data:
            logger.warning(
                "If the trace is from an inference run, then add the following snippet key to the json files "
                'to use HTA "distributedInfo": {"rank": 0}. If there are multiple traces files, then each file '
                "should have a unique rank value."
            )
            raise ValueError("The distributedInfo key does not exist in the trace. Trace cannot be processed.")
        if "rank" not in data["distributedInfo"]:
            logger.warning(
                "If the trace is from an inference run, then add the following snippet key to the json files "
                'to use HTA "distributedInfo": {"rank": 0}. If there are multiple traces files, then each file '
                "should have a unique rank value."
            )
            raise ValueError("Rank unavailable in the trace file. Trace cannot be processed.")
        rank = data["distributedInfo"]["rank"]

        assert isinstance(rank, int), "Rank is expected to be an integer"
        assert rank >= 0, "Rank must be a non-negative integer"
        rank_to_trace_dict[rank] = filename
    if len(rank_to_trace_dict) == 0:
        logger.warning("Rank to trace dict has size zero.")
    return True, rank_to_trace_dict


def get_trace_files(trace_path: str) -> Dict[int, str]:
    """
    Get the rank to trace map from traces in the directory <trace_path>.

    Args:
        trace_path (str) : the path to the directory where the traces are located.

    Returns:
        rank_trace_file_map (Dict[int, str]) : a dictionary with rank as key and trace filename as value.
        Only valid rank_trace_file_map entry will be included in the dictionary. A rank_trace_file_map entry
        is valid only when  (1) the rank is a positive integer, and (2) the trace file exists and is readable.
    """
    rank_to_trace_dict: Dict[int, str] = {}
    if not os.path.exists(trace_path):
        logger.warning(f"{trace_path} is not a valid path")
    else:
        ok, rank_to_trace_dict = create_rank_to_trace_dict(trace_path)
        if not ok:
            logger.warning("failed to create rank to trace map")
            return {}

    if len(rank_to_trace_dict) == 0:
        logger.warning("There is no item in the rank to trace file map.")
    else:
        logger.info(f"Rank to trace file map:\n{rank_to_trace_dict}")
    return rank_to_trace_dict


def read_trace(file_path: str) -> Dict[str, Any]:
    """
    Read the trace from a file.

    Args:
        file_path (str): a trace file with ".gz" or "json" postfix.

    Returns:
        trace_data (Dict[str, Any]): the raw trace data
    """
    trace_data: Dict[str, Any] = {}
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rb") as fh:
            trace_data = json.loads(fh.read())
    elif file_path.endswith(".json"):
        with open(file_path, "r") as fh2:
            trace_data = json.loads(fh2.read())
    else:
        raise ValueError(f"trace_file ({file_path}) must end with '.gz' or 'json'")
    return trace_data


def write_trace(trace_data: Dict[str, Any], file_path: str) -> None:
    """
    Write a trace into a file.

    Args:
        trace_data (Dict[str, Any]): the trace data
        file_path (str): a trace file with ".gz" or "json" postfix.

    Returns:
        None
    """
    base_dir = os.path.dirname(file_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if file_path.endswith(".gz"):
        json_str = json.dumps(trace_data)
        json_bytes = json_str.encode("utf-8")
        with gzip.open(file_path, "wb") as fp:
            fp.write(json_bytes)
    else:
        with open(file_path, "w+") as fp:
            fp.write(json.dumps(trace_data, indent=2))


def update_trace_rank(file_path: str, rank: int) -> None:
    """
    Add or update a rank metadata entry in a trace file.

    Args:
        file_path (str): a trace file.
        rank (int): the rank assigned to the trace file.

    Returns:
        None
    """

    def _add_rank_meta(trace_data: Dict[str, Any], rank: int) -> None:
        if "distributedInfo" in trace_data:
            trace_data["distributedInfo"]["rank"] = rank
        else:
            trace_data["distributedInfo"] = {"rank": rank}

    trace_data = read_trace(file_path)
    _add_rank_meta(trace_data, rank)
    write_trace(trace_data, file_path)
