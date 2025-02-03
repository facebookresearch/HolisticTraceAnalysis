# pyre-strict

import copy
from enum import Enum
from typing import Dict, List, Optional, Set

from hta.configs.default_values import AttributeSpec, EventArgs
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
    ) -> None:
        self.args: List[AttributeSpec] = (
            args if args else self.get_default_args(version=version)
        )
        self.parser_backend: Optional[ParserBackend] = None
        self.trace_memory: bool = False
        self.user_provide_trace_type: Optional[TraceType] = user_provide_trace_type
        self.min_required_cols: List[str] = self.DEFAULT_MIN_REQUIRED_COLS

    @classmethod
    def get_default_cfg(cls) -> "ParserConfig":
        return copy.deepcopy(_DEFAULT_PARSER_CONFIG)

    @classmethod
    def get_versioned_cfg(cls, version: YamlVersion) -> "ParserConfig":
        return copy.deepcopy(ParserConfig(version=version))

    @classmethod
    def set_default_cfg(cls, cfg: "ParserConfig") -> None:
        _DEFAULT_PARSER_CONFIG.set_args(cfg.get_args())
        _DEFAULT_PARSER_CONFIG.set_min_required_cols(cfg.get_min_required_cols())

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
            self.add_args(args)

    def get_args(self) -> List[AttributeSpec]:
        return self.args

    def set_min_required_cols(self, cols: List[str]) -> None:
        if cols != self.min_required_cols:
            self.min_required_cols.clear()
            self.min_required_cols = cols

    def get_min_required_cols(self) -> List[str]:
        return self.min_required_cols

    def add_args(self, args: List[AttributeSpec]) -> None:
        arg_set: Set[str] = {arg.name for arg in self.args}
        for arg in args:
            if arg.name not in arg_set:
                self.args.append(arg)
                arg_set.add(arg.name)

    def set_parser_backend(self, parser_backend: ParserBackend) -> None:
        self.parser_backend = parser_backend

    @staticmethod
    def enable_communication_args(version: YamlVersion = DEFAULT_PARSE_VERSION) -> None:
        _DEFAULT_PARSER_CONFIG.add_args(
            parse_event_args_yaml(version).ARGS_COMMUNICATION
        )

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

    @staticmethod
    def show_available_args():
        from pprint import pprint

        global AVAILABLE_ARGS
        pprint(AVAILABLE_ARGS)


# Define a global ParserConfig variable for internal use. To access this variable,
# Clients should use ParserConfig.get_default_cfg and ParserConfig.set_default_cfg.
_DEFAULT_PARSER_CONFIG: ParserConfig = ParserConfig()
