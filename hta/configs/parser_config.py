# pyre-strict

import copy
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Set, Union


class ParserBackend(str, Enum):
    """Tracer parser and loader backend
    See https://github.com/facebookresearch/HolisticTraceAnalysis/pull/125
    for details on performance and memory usage.
    """

    JSON = "json"
    IJSON = "ijson"
    IJSON_BATCHED = "ijson_batched"
    IJSON_BATCH_AND_COMPRESS = "ijson_batch_and_compress"


class ValueType(Enum):
    """ValueType enumerates the possible data types for the attribute values."""

    Int = 1
    Float = 2
    String = 3
    Object = 4


class TraceType(str, Enum):
    """TraceType enumerates the possible trace types"""

    Training = "training"
    TrainingWoProfilerstepAnnot = "training_wo_profilerstep_annot"
    Inference = "inference"


class AttributeSpec(NamedTuple):
    """AttributeSpec specifies what an attribute looks like and how to parse it.

    An AttributeSpec instance has the following fields:
    + name: the column name used in the output dataframe.
    + raw_name: the key used in args dict object in the original json trace.
    + value_type: the expected data type for the values of the attribute.
    + default_value: what value will be used for missing attribute values.
    """

    name: str
    raw_name: str
    value_type: ValueType
    default_value: Union[int, float, str, object]


AVAILABLE_ARGS: Dict[str, AttributeSpec] = {
    "index::ev_idx": AttributeSpec("ev_idx", "Ev Idx", ValueType.Int, -1),
    "index::external_id": AttributeSpec(
        "external_id", "External id", ValueType.Int, -1
    ),
    "cpu_op::concrete_inputs": AttributeSpec(
        "concrete_inputs", "Concrete Inputs", ValueType.Int, -1
    ),
    "cpu_op::fwd_thread": AttributeSpec(
        "fwd_thread_id", "Fwd thread id", ValueType.Int, -1
    ),
    "cpu_op::input_dims": AttributeSpec(
        "input_dims", "Input Dims", ValueType.Object, "-1"
    ),
    "cpu_op::input_type": AttributeSpec(
        "input_type", "Input type", ValueType.Object, "-1"
    ),
    "cpu_op::input_strides": AttributeSpec(
        "input_strides",
        "Input Strides",
        ValueType.Object,
        "-1",
    ),
    "cpu_op::sequence_number": AttributeSpec(
        "sequence", "Sequence number", ValueType.Int, -1
    ),
    # kernel_backend, for example triton.
    "cpu_op::kernel_backend": AttributeSpec(
        "kernel_backend", "kernel_backend", ValueType.String, ""
    ),
    "correlation::cbid": AttributeSpec("cbid", "cbid", ValueType.Int, -1),
    "correlation::cpu_gpu": AttributeSpec(
        "correlation", "correlation", ValueType.Int, -1
    ),
    "sm::blocks": AttributeSpec("blocks_per_sm", "blocks per SM", ValueType.Object, "[]"),
    "sm::occupancy": AttributeSpec(
        "est_occupancy", "est. achieved occupancy %", ValueType.Int, -1
    ),
    "sm::warps": AttributeSpec("warps_per_sm", "warps per SM", ValueType.Int, -1),
    "data::bytes": AttributeSpec("bytes", "bytes", ValueType.Int, -1),
    "data::bandwidth": AttributeSpec(
        "memory_bw_gbps", "memory bandwidth (GB/s)", ValueType.Float, 0.0
    ),
    "cuda::context": AttributeSpec("context", "context", ValueType.Int, -1),
    "cuda::device": AttributeSpec("device", "device", ValueType.Int, -1),
    "cuda::stream": AttributeSpec("stream", "stream", ValueType.Int, -1),
    "kernel::queued": AttributeSpec("queued", "queued", ValueType.Int, -1),
    "kernel::shared_memory": AttributeSpec(
        "shared_memory", "shared memory", ValueType.Int, -1
    ),
    "threads::block": AttributeSpec("block", "block", ValueType.Object, "-1"),
    "threads::grid": AttributeSpec("grid", "grid", ValueType.Int, -1),
    "threads::registers": AttributeSpec(
        "registers_per_thread", "registers per thread", ValueType.Int, -1
    ),
    "cuda_sync::stream": AttributeSpec(
        "wait_on_stream", "wait_on_stream", ValueType.Int, -1
    ),
    "cuda_sync::event": AttributeSpec(
        "wait_on_cuda_event_record_corr_id",
        "wait_on_cuda_event_record_corr_id",
        ValueType.Int,
        -1,
    ),
    "info::labels": AttributeSpec("labels", "labels", ValueType.String, ""),
    "info::name": AttributeSpec("name", "name", ValueType.Int, -1),
    "info::op_count": AttributeSpec("op_count", "Op count", ValueType.Int, -1),
    "info::sort_index": AttributeSpec("sort_index", "sort_index", ValueType.Int, -1),
    "nccl::collective_name": AttributeSpec(
        "collective_name", "Collective name", ValueType.String, ""
    ),
    "nccl::in_msg_nelems": AttributeSpec(
        "in_msg_nelems", "In msg nelems", ValueType.Int, 0
    ),
    "nccl::out_msg_nelems": AttributeSpec(
        "out_msg_nelems", "Out msg nelems", ValueType.Int, 0
    ),
    "nccl::group_size": AttributeSpec("group_size", "Group size", ValueType.Int, 0),
    "nccl::dtype": AttributeSpec("msg_dtype", "dtype", ValueType.String, ""),
    "nccl::in_split_size": AttributeSpec(
        "in_split_size", "In split size", ValueType.Object, "[]"
    ),
    "nccl::out_split_size": AttributeSpec(
        "out_split_size", "Out split size", ValueType.Object, "[]"
    ),
    "nccl::process_group_name": AttributeSpec(
        "process_group_name", "Process Group Name", ValueType.String, ""
    ),
    "nccl::process_group_desc": AttributeSpec(
        "process_group_desc", "Process Group Description", ValueType.String, ""
    ),
    "nccl::process_group_ranks": AttributeSpec(
        "process_group_ranks", "Process Group Ranks", ValueType.Object, "[]"
    ),
    "nccl::rank": AttributeSpec("process_rank", "Rank", ValueType.Int, -1),
}


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

    ARGS_INPUT_SHAPE: List[AttributeSpec] = [
        AVAILABLE_ARGS[k]
        for k in ["cpu_op::input_dims", "cpu_op::input_type", "cpu_op::input_strides"]
    ]
    ARGS_BANDWIDTH: List[AttributeSpec] = [
        AVAILABLE_ARGS[k] for k in ["data::bytes", "data::bandwidth"]
    ]
    ARGS_SYNC: List[AttributeSpec] = [
        AVAILABLE_ARGS[k] for k in ["cuda_sync::stream", "cuda_sync::event"]
    ]
    ARGS_MINIMUM: List[AttributeSpec] = [
        AVAILABLE_ARGS[k] for k in ["cuda::stream", "correlation::cpu_gpu"]
    ]
    ARGS_COMPLETE: List[AttributeSpec] = [
        AVAILABLE_ARGS[k] for k in AVAILABLE_ARGS if not k.startswith("info")
    ]
    ARGS_INFO: List[AttributeSpec] = [
        AVAILABLE_ARGS[k] for k in ["info::labels", "info::name", "info::sort_index"]
    ]
    ARGS_COMMUNICATION: List[AttributeSpec] = [
        AVAILABLE_ARGS[k]
        for k in [
            "nccl::collective_name",
            "nccl::in_msg_nelems",
            "nccl::out_msg_nelems",
            "nccl::dtype",
            "nccl::group_size",
            "nccl::rank",
            "nccl::in_split_size",
            "nccl::out_split_size",
        ]
    ]
    ARGS_DEFAULT: List[AttributeSpec] = (
        ARGS_MINIMUM
        + ARGS_BANDWIDTH
        + ARGS_SYNC
        + ARGS_INPUT_SHAPE
        + [AVAILABLE_ARGS["index::external_id"]]
    )

    def __init__(
        self,
        args: Optional[List[AttributeSpec]] = None,
        user_provide_trace_type: Optional[TraceType] = None,
    ) -> None:
        self.args: List[AttributeSpec] = args if args else self.get_default_args()
        self.parser_backend: Optional[ParserBackend] = None
        self.trace_memory: bool = False
        self.user_provide_trace_type: Optional[TraceType] = user_provide_trace_type

    @classmethod
    def get_default_cfg(cls) -> "ParserConfig":
        return copy.deepcopy(_DEFAULT_PARSER_CONFIG)

    @classmethod
    def set_default_cfg(cls, cfg: "ParserConfig") -> None:
        _DEFAULT_PARSER_CONFIG.set_args(cfg.get_args())

    @classmethod
    def get_minimum_args(cls) -> List[AttributeSpec]:
        return cls.ARGS_MINIMUM.copy()

    @classmethod
    def get_default_args(cls) -> List[AttributeSpec]:
        return cls.ARGS_DEFAULT.copy()

    @classmethod
    def get_info_args(cls) -> List[AttributeSpec]:
        return cls.ARGS_INFO.copy()

    def set_args(self, args: List[AttributeSpec]) -> None:
        if args != self.args:
            self.args.clear()
            self.add_args(args)

    def get_args(self) -> List[AttributeSpec]:
        return self.args

    def add_args(self, args: List[AttributeSpec]) -> None:
        arg_set: Set[str] = {arg.name for arg in self.args}
        for arg in args:
            if arg.name not in arg_set:
                self.args.append(arg)
                arg_set.add(arg.name)

    def set_parser_backend(self, parser_backend: ParserBackend) -> None:
        self.parser_backend = parser_backend

    @staticmethod
    def enable_communication_args() -> None:
        _DEFAULT_PARSER_CONFIG.add_args(ParserConfig.ARGS_COMMUNICATION)


# Define a global ParserConfig variable for internal use. To access this variable,
# Clients should use ParserConfig.get_default_cfg and ParserConfig.set_default_cfg.
_DEFAULT_PARSER_CONFIG: ParserConfig = ParserConfig()
