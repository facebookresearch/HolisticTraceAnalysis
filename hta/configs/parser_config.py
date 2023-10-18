from enum import Enum
from typing import List, NamedTuple, Optional, Union


class ValueType(Enum):
    """ValueType enumerates the possible data types for the attribute values."""

    Int = 1
    Float = 2
    String = 3
    Object = 4


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


class ParserConfig:
    """TraceParserConfig specifies how to parse a json trace file.

    The current implementation only supports customization on `args` parsing.
    This class can be extended to support other customizations.
    """

    ARGS_INPUT_SHAPE: List[AttributeSpec] = [
        AttributeSpec("input_dims", "input_dims", ValueType.Object, "-1"),
        AttributeSpec("input_type", "input_type", ValueType.Object, "-1"),
    ]
    ARGS_CUDA_THREADS: List[AttributeSpec] = [
        AttributeSpec("grid", "grid", ValueType.Object, "-1"),
        AttributeSpec("block", "block", ValueType.Object, "-1"),
        AttributeSpec("blocks_per_sm", "blocks_per_sm", ValueType.Int, -1),
        AttributeSpec(
            "registers_per_thread", "registers_per_thread", ValueType.Int, -1
        ),
    ]
    ARGS_BANDWIDTH: List[AttributeSpec] = [
        AttributeSpec("memory_bw_gbps", "memory bandwidth (GB/s)", ValueType.Float, -1),
        AttributeSpec("bytes", "bytes", ValueType.Int, -1),
    ]
    ARGS_CUDA_MINIMUM: List[AttributeSpec] = [
        AttributeSpec("stream", "stream", ValueType.Int, -1),
        AttributeSpec("correlation", "correlation", ValueType.Int, -1),
    ]
    ARGS_SYNC: List[AttributeSpec] = [
        AttributeSpec("wait_on_stream", "wait_on_stream", ValueType.Int, -1),
        AttributeSpec(
            "wait_on_cuda_event_record_corr_id",
            "wait_on_cuda_event_record_corr_id",
            ValueType.Int,
            -1,
        ),
    ]

    def __init__(self, args: Optional[List[AttributeSpec]] = None):
        self.args: List[AttributeSpec] = []
        self.set_args(args if args else self.get_default_args())

    @classmethod
    def get_minimum_args(cls) -> List[AttributeSpec]:
        return cls.ARGS_CUDA_MINIMUM

    @classmethod
    def get_default_args(cls) -> List[AttributeSpec]:
        return (
            cls.ARGS_CUDA_MINIMUM
            + cls.ARGS_INPUT_SHAPE
            + cls.ARGS_SYNC
            + cls.ARGS_BANDWIDTH
        )

    def set_args(self, args: List[AttributeSpec]) -> None:
        self.args.clear()
        self.add_args(args)

    def get_args(self) -> List[AttributeSpec]:
        return self.args

    def add_args(self, args: List[AttributeSpec]) -> None:
        arg_set = {arg.name for arg in self.args}
        for arg in args:
            if arg.name not in arg_set:
                self.args.append(arg)
