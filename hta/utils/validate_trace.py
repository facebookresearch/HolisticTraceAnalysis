# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

from typing import Any, Dict, List, Tuple

import pandas as pd
from hta.common.trace_file import read_trace
from hta.configs.parser_config import AttributeSpec, ParserConfig, ValueType


def get_argument_spec(level: str = "default") -> List[AttributeSpec]:
    """Get the argument spec for the given level.

    Args:
        level (str): the level, can be "minimal", "default", or "complete".

    Returns:
        A list of AttributeSpec.
    """
    return {
        "minimal": ParserConfig.ARGS_MINIMUM,
        "standard": ParserConfig.ARGS_DEFAULT,
        "complete": ParserConfig.ARGS_COMPLETE,
    }[level]


def get_expected_arguments(attribute_specs: List[AttributeSpec]) -> pd.DataFrame:
    """Extract the expected arguments as a DataFrame from the given list of AttributeSpec.

    Args:
        attribute_specs (List[AttributeSpec]): a list of AttributeSpec.

    Returns:
        A pandas DataFrame with the following columns:
            - arg_keys: the argument keys
            - arg_value_types: the argument value types
            - arg_default_values: the argument default values
            - trace_df_column_name: the trace DataFrame column name
    """
    column_name = []
    arg_key = []
    arg_value_type = []
    arg_default_value = []
    for v in attribute_specs:
        column_name.append(v.name)
        arg_key.append(v.raw_name)
        arg_value_type.append(v.value_type)
        arg_default_value.append(v.default_value)

    df = pd.DataFrame(
        {
            "arg_keys": arg_key,
            "arg_value_types": arg_value_type,
            "arg_default_values": arg_default_value,
            "trace_df_column_name": column_name,
        }
    )
    df.sort_values("trace_df_column_name")
    return df


# pyre-ignore[3]
def _get_argument_value_types(df: pd.DataFrame) -> Dict[str, Tuple[ValueType, Any]]:
    """Extract the argument value types from the given DataFrame.

    Args:
        df (pd.DataFrame): a DataFrame with the following columns:
            - arg_keys: the argument keys
            - arg_value_types: the argument value types
            - arg_default_values: the argument default values
            - trace_df_column_name: the trace DataFrame column name

    Returns:
        A dictionary with the following keys:
            - arg_keys: the argument keys
            - arg_value: (arg_value_types, arg_default_values)
    """
    df["arg_value"] = df.apply(
        lambda row: (row.arg_value_types, row.arg_default_values), axis=1
    )
    return df.set_index("arg_keys")["arg_value"].to_dict()


def _check_args(
    args: Dict[str, Any],
    # pyre-ignore[2]
    arg_type_map: Dict[str, Tuple[ValueType, Any]],
    skipped_arguments: Dict[str, int],
    type_violations: Dict[str, str],
) -> None:
    """Check the arguments against the expected argument types.

    Args:
        args (Dict[str, Any]): a dictionary of arguments.
        arg_type_map (Dict[str, Tuple[ValueType, Any]]): a dictionary of expected argument types.

    Returns:
        A dictionary with the following keys:
            - arg_keys: the argument keys
            - arg_value: (arg_value_types, arg_default_values)
    """
    for k, v in args.items():
        if k not in arg_type_map:
            skipped_arguments[k] += 1
        else:
            # pyre-ignore[23]
            arg_type, arg_value = arg_type_map.get(k)
            if arg_type != ValueType.Object and isinstance(v, type(arg_value)):
                type_violations[
                    k
                ] = f"key={k}: value={v} type={type(v)}; expect type={arg_type.name} default value={arg_value}"


def validate_trace_format(
    trace_file_path: str, level: str = "default"
) -> Tuple[bool, Dict[str, Any]]:
    """Validate validate_trace_format

    Args:
        trace_file_path (str): the path to the trace file to be validated.
        level (str): the level of validation, can be "minimal", "default", or "complete".

    Returns:
        A tuple of (ok: bool, errors: Dict[str, Any])
    """
    ok: bool = False
    errors: Dict[str, Any] = {}

    attribute_specs = get_argument_spec(level)
    df = get_expected_arguments(attribute_specs)
    arg_type_map = _get_argument_value_types(df)

    try:
        trace_data = read_trace(trace_file_path)
    except Exception as ex:
        errors["trace_read_error"] = ex
        return False, errors

    if "traceEvents" not in trace_data:
        errors["trace_data_error"] = "'traceEvents' section not found"
        return False, errors

    events = trace_data["traceEvents"]

    skipped_arguments: Dict[str, int] = defaultdict(int)
    type_violations: Dict[str, str] = defaultdict(str)

    for e in events:
        if "args" in e:
            args = e["args"]
            if isinstance(args, dict):
                _check_args(args, arg_type_map, skipped_arguments, type_violations)
            else:
                errors["args_data_type"] = f"args is not a dict: {args}"

    if len(skipped_arguments) > 0:
        errors["skipped_arguments"] = skipped_arguments

    if len(type_violations) > 0:
        errors["type_violations"] = type_violations

    ok = len(errors) == 0

    return ok, errors
