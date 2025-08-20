# pyre-strict

import copy
import re
from enum import Enum
from typing import Dict, List, Optional, Set, Union

import pandas as pd

from hta.configs.default_values import AttributeSpec, EventArgs, ValueType
from hta.configs.event_args_yaml_parser import (
    parse_event_args_yaml,
    v1_0_0,
    YamlVersion,
)


class ParserBackend(str, Enum):
    """Tracer parser and loader backend
    See https://github.com/facebookresearch/HolisticTraceAnalysis/pull/125
    for details on performance and memory usage.
    """

    JSON = "json"
    IJSON = "ijson"
    IJSON_BATCHED = "ijson_batched"
    IJSON_BATCH_AND_COMPRESS = "ijson_batch_and_compress"


class TraceType(str, Enum):
    """TraceType enumerates the possible trace types"""

    Training = "training"
    TrainingWoProfilerstepAnnot = "training_wo_profilerstep_annot"
    Inference = "inference"


DEFAULT_PARSE_VERSION: YamlVersion = v1_0_0
DEFAULT_VER_EVENT_ARGS: EventArgs = parse_event_args_yaml(DEFAULT_PARSE_VERSION)
AVAILABLE_ARGS: Dict[str, AttributeSpec] = DEFAULT_VER_EVENT_ARGS.AVAILABLE_ARGS


class ParserConfig:
    """TraceParserConfig specifies how to parse a json trace file.

    +args (List[AttributeSpec]): Supports customization on `args` parsing.
        Please see the `AttributeSpec` class for details.
    +trace_memory (bool): Measures the peak memory usage during parsing with `tracemalloc`.
        This is off by default as it adds performance overhead.
    +parser_backend (Optional[ParserBackend]): HTA supports simple 'JSON' as well as iterative 'IJSON'
        backend for loading large traces in a batched/memory efficient manner.
        Default is "JSON" as the ijson backend is under development.
        See supported modes for in the enum ParserBackend.
        Note: Use ParserBackend.IJSON_BATCH_AND_COMPRESS for best results.
        Please see https://github.com/facebookresearch/HolisticTraceAnalysis/pull/125

    This class can be extended to support other customizations.
    """

    ARGS_INPUT_SHAPE: List[AttributeSpec] = DEFAULT_VER_EVENT_ARGS.ARGS_INPUT_SHAPE
    ARGS_BANDWIDTH: List[AttributeSpec] = DEFAULT_VER_EVENT_ARGS.ARGS_BANDWIDTH
    ARGS_SYNC: List[AttributeSpec] = DEFAULT_VER_EVENT_ARGS.ARGS_SYNC
    ARGS_MINIMUM: List[AttributeSpec] = DEFAULT_VER_EVENT_ARGS.ARGS_MINIMUM
    ARGS_COMPLETE: List[AttributeSpec] = DEFAULT_VER_EVENT_ARGS.ARGS_COMPLETE
    ARGS_INFO: List[AttributeSpec] = DEFAULT_VER_EVENT_ARGS.ARGS_INFO
    ARGS_TRITON_KERNELS: List[AttributeSpec] = (
        DEFAULT_VER_EVENT_ARGS.ARGS_TRITON_KERNELS
    )
    ARGS_COMMUNICATION: List[AttributeSpec] = DEFAULT_VER_EVENT_ARGS.ARGS_COMMUNICATION
    ARGS_DEFAULT: List[AttributeSpec] = DEFAULT_VER_EVENT_ARGS.ARGS_DEFAULT
    DEFAULT_MIN_REQUIRED_COLS: List[str] = [
        "ts",
        "dur",
        "name",
        "cat",
        "pid",
        "tid",
    ]

    def __init__(
        self,
        args: Optional[List[AttributeSpec]] = None,
        user_provide_trace_type: Optional[TraceType] = None,
        version: YamlVersion = DEFAULT_PARSE_VERSION,
        parse_all_args: bool = False,
        selected_arg_keys: Optional[List[str]] = None,
        skip_event_types: Optional[Set[str]] = None,
        convert_ts_to_integer: bool = True,
    ) -> None:
        """Initialize the ParserConfig object."""
        # Note remember to update set_default_cfg() when adding new fields.
        self.args: List[AttributeSpec] = (
            args if args else self.get_default_args(version=version)
        )
        self.arg_map: Dict[str, AttributeSpec] = {arg.name: arg for arg in self.args}
        self.parser_backend: Optional[ParserBackend] = None
        self.trace_memory: bool = False
        self.user_provide_trace_type: Optional[TraceType] = user_provide_trace_type

        # Currently, this is only honored in the ijson backends.
        self.skip_event_types: Set[str] = (
            skip_event_types if skip_event_types is not None else {"python_function"}
        )

        self.min_required_cols: List[str] = self.DEFAULT_MIN_REQUIRED_COLS
        self.drop_gpu_user_annotation: bool = True
        self.version: YamlVersion = version
        self.parse_all_args: bool = parse_all_args
        self.selected_arg_keys: Optional[List[str]] = None

        # Json batched backend specific configs
        self.batch_size = 1000

        # config to convert ts to nearest integer
        self.convert_ts_to_integer = convert_ts_to_integer

    def clone(self) -> "ParserConfig":
        return copy.deepcopy(self)

    @classmethod
    def get_default_cfg(cls) -> "ParserConfig":
        # return copy.deepcopy(_DEFAULT_PARSER_CONFIG)
        return _DEFAULT_PARSER_CONFIG.clone()

    @classmethod
    def get_versioned_cfg(cls, version: YamlVersion) -> "ParserConfig":
        return ParserConfig(version=version).clone()

    @classmethod
    def set_default_cfg(cls, cfg: "ParserConfig") -> None:
        _DEFAULT_PARSER_CONFIG.set_args(cfg.get_args())
        _DEFAULT_PARSER_CONFIG.set_min_required_cols(cfg.get_min_required_cols())
        _DEFAULT_PARSER_CONFIG.set_drop_gpu_user_annotation(
            cfg.drop_gpu_user_annotation
        )
        _DEFAULT_PARSER_CONFIG.set_trace_memory(cfg.trace_memory)
        _DEFAULT_PARSER_CONFIG.set_parser_backend(cfg.parser_backend)
        _DEFAULT_PARSER_CONFIG.set_parse_all_args(cfg.parse_all_args)
        _DEFAULT_PARSER_CONFIG.set_args_selector(cfg.selected_arg_keys)
        _DEFAULT_PARSER_CONFIG.set_skip_event_types(cfg.skip_event_types)

    @classmethod
    def get_minimum_args(
        cls, version: YamlVersion = DEFAULT_PARSE_VERSION
    ) -> List[AttributeSpec]:
        return parse_event_args_yaml(version).ARGS_MINIMUM.copy()

    @classmethod
    def get_default_args(
        cls, version: YamlVersion = DEFAULT_PARSE_VERSION
    ) -> List[AttributeSpec]:
        return parse_event_args_yaml(version).ARGS_DEFAULT.copy()

    @classmethod
    def get_info_args(
        cls, version: YamlVersion = DEFAULT_PARSE_VERSION
    ) -> List[AttributeSpec]:
        return parse_event_args_yaml(version).ARGS_INFO.copy()

    def set_args(self, args: List[AttributeSpec]) -> None:
        if args != self.args:
            self.args.clear()
            self.arg_map.clear()
            self.add_args(args)

    def get_args(self) -> List[AttributeSpec]:
        if self.selected_arg_keys is not None:
            return [self.arg_map[arg] for arg in self.selected_arg_keys]
        return self.args

    def set_min_required_cols(self, cols: List[str]) -> None:
        if cols != self.min_required_cols:
            self.min_required_cols.clear()
            self.min_required_cols = cols

    def get_min_required_cols(self) -> List[str]:
        return self.min_required_cols

    def add_args(self, args: List[AttributeSpec]) -> None:
        for arg in args:
            if arg.name not in self.arg_map:
                self.args.append(arg)
                self.arg_map[arg.name] = arg

    def set_trace_memory(self, trace_memory: bool) -> None:
        self.trace_memory = trace_memory

    def set_drop_gpu_user_annotation(self, should_drop: bool) -> None:
        self.drop_gpu_user_annotation = should_drop

    def set_parser_backend(self, parser_backend: ParserBackend) -> None:
        self.parser_backend = parser_backend

    def set_parse_all_args(self, parse_all_args: bool) -> "ParserConfig":
        self.parse_all_args = parse_all_args
        return self

    def set_args_selector(
        self, selected_arg_keys: Optional[List[str]] = None
    ) -> "ParserConfig":
        self.selected_arg_keys = selected_arg_keys
        return self

    def set_skip_event_types(
        self, skip_event_types: Optional[Set[str]] = None
    ) -> "ParserConfig":
        self.skip_event_types = skip_event_types
        return self

    @classmethod
    def enable_communication_args(
        cls,
        cfg: Optional["ParserConfig"] = None,
        version: YamlVersion = DEFAULT_PARSE_VERSION,
    ) -> "ParserConfig":
        cfg = cfg or _DEFAULT_PARSER_CONFIG
        cfg.add_args(parse_event_args_yaml(version).ARGS_COMMUNICATION)
        return cfg

    @staticmethod
    def set_global_parser_config_version(version: YamlVersion) -> None:
        global _DEFAULT_PARSER_CONFIG
        _DEFAULT_PARSER_CONFIG = ParserConfig(version=version)

        global AVAILABLE_ARGS
        AVAILABLE_ARGS = parse_event_args_yaml(version).AVAILABLE_ARGS

        ParserConfig.ARGS_INPUT_SHAPE = parse_event_args_yaml(version).ARGS_INPUT_SHAPE
        ParserConfig.ARGS_BANDWIDTH = parse_event_args_yaml(version).ARGS_BANDWIDTH
        ParserConfig.ARGS_SYNC = parse_event_args_yaml(version).ARGS_SYNC
        ParserConfig.ARGS_MINIMUM = parse_event_args_yaml(version).ARGS_MINIMUM
        ParserConfig.ARGS_COMPLETE = parse_event_args_yaml(version).ARGS_COMPLETE
        ParserConfig.ARGS_INFO = parse_event_args_yaml(version).ARGS_INFO
        ParserConfig.ARGS_COMMUNICATION = parse_event_args_yaml(
            version
        ).ARGS_COMMUNICATION
        ParserConfig.ARGS_DEFAULT = parse_event_args_yaml(version).ARGS_DEFAULT

    @classmethod
    def get_all_available_args(cls) -> Dict[str, AttributeSpec]:
        return AVAILABLE_ARGS

    @classmethod
    def show_available_args(cls) -> None:
        from pprint import pprint

        pprint(cls.get_all_available_args())

    @classmethod
    def transform_arg_name(cls, arg: str) -> str:
        arg = re.sub(r"\([^)]*\)", "", arg)
        arg = re.sub(r"[ \-\/\.%]+", "_", arg.lower())
        arg = re.sub(r"_+", "_", arg).strip("_")
        if arg == "name":
            arg = "arg_name"
        return arg

    @classmethod
    def infer_attribute_specs(
        cls,
        args: pd.Series,
        reference_specs: Optional[Dict[str, AttributeSpec]] = None,
    ) -> Dict[str, AttributeSpec]:
        """Infer the attribute specs from the args series.

        Args:
            args (pd.Series): A series of dict objects from which to infer the attribute specs.StopAsyncIteration
            reference_specs (Optional[Dict[str, AttributeSpec]]): A dictionary of reference attribute specs.
                If provided, the inferred attribute specs will be merged with the reference specs.

        Returns:
            A dictionary of inferred attribute specs, which contains specs of all the attributes in the args series.
            If an arg_name is found in the reference_specs, it will use the AttributeSpec from the reference_specs.
            Otherwise, it will use the inferred AttributeSpec.
        """
        attribute_spec_map = {}
        attribute_set = set()

        # Initialize the attribute_spec_map with the reference_specs
        reference_specs = reference_specs or {}
        for _, spec in reference_specs.items():
            attribute_spec_map[spec.raw_name] = spec

        # Infer the attribute specs from the args
        for d in args.dropna():
            if isinstance(d, dict):
                for arg_name, arg_value in d.items():
                    attribute_set |= {arg_name}
                    if arg_name not in attribute_spec_map or arg_name == "name":
                        attribute_spec_map[arg_name] = cls.make_attribute_spec(
                            arg_name, arg_value
                        )
        return {k: v for k, v in attribute_spec_map.items() if k in attribute_set}

    @classmethod
    def make_attribute_spec(
        cls,
        raw_name: str,
        value: object,
        min_supported_version: YamlVersion = DEFAULT_PARSE_VERSION,
    ) -> AttributeSpec:
        default_value: Union[int, float, str, object]
        if isinstance(value, int):
            value_type = ValueType.Int
            default_value = 0
        elif isinstance(value, float):
            value_type = ValueType.Float
            default_value = 0.0
        elif isinstance(value, str):
            value_type = ValueType.String
            default_value = ""
        else:
            value_type = ValueType.Object
            default_value = None
        return AttributeSpec(
            cls.transform_arg_name(raw_name),
            raw_name,
            value_type,
            default_value,
            min_supported_version,
        )


# Define a global ParserConfig variable for internal use. To access this variable,
# Clients should use ParserConfig.get_default_cfg and ParserConfig.set_default_cfg.
_DEFAULT_PARSER_CONFIG: ParserConfig = ParserConfig()
