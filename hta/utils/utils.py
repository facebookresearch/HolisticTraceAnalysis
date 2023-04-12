# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


class KernelType(Enum):
    COMMUNICATION = 0
    MEMORY = 1
    COMPUTATION = 2


class IdleTimeType(Enum):
    HOST_WAIT = 0
    KERNEL_WAIT = 1
    OTHER = 2


class CUDA_KERNEL_CATEGORY_NAMES(str, Enum):
    FP32_GEMM = "fp32_gemm"
    FP16_GEMM = "fp16_gemm"
    TF32_GEMM_TC = "tf32_gemm_tensor_cores"
    FP16_GEMM_TC = "fp16_gemm_tensor_cores"
    BF16_GEMM_TC = "bf16_gemm_tensor_cores"
    CONVOLUTION = "convolution"
    EMBEDDING_BAGS = "embedding_bags"
    VECTORIZED_FUNCTORS = "vectorized_functors"
    ELEMENTWISE_FUNCTORS = "elementwise_functors"
    TENSOR_TRANSFORMS = "tensor_transforms"
    LAYER_NORM = "layer_normalization"
    SORT_SCAN_REDUCE = "sort_scan_reduce"
    MEMCOPIES_D2D = "memory_copy_kernels"
    MULTI_TENSOR = "multi_tensor_apply"
    OPTIMIZER = "optimizer"
    NCCL_KERNEL = "nccl_communication"
    GLOO_KERNEL = "gloo_communication"
    TRITON = "triton_kernels"
    JAGGED_NESTED = "jagged_nested_tensors"
    INDEXING = "index_processing"
    SOFTMAX = "softmax"
    UNKNOWN = "no_category"
    MULTIPLE = "multiple_categories"


CUDA_KERNEL_NAME_TO_CATEGORY: Dict[str, str] = {
    # FP32_GEMM not running on tensor cores
    "sgemm": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "splitKreduce_kernel<float, float, float, float>": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "gcgemm": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "cgemm": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "gemmk1_kernel": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "gemv2N_kernel": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "gemv2T_kernel": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "gemvNSP_kernel": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "gemmSN_TN_kernel": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "gemmSN_NN_kernel": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    "internal::gemvx::kernel": CUDA_KERNEL_CATEGORY_NAMES.FP32_GEMM,
    # FP16_GEMM not running on tensor cores
    "fp16_sgemm": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM,
    # TF32_GEMM running on tensor cores
    "tensorop_s1688gemm": CUDA_KERNEL_CATEGORY_NAMES.TF32_GEMM_TC,
    # FP16_GEMM running on tensor cores
    "fp16_s884gemm": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    "fp16_s1688gemm": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    "fp16_s16816gemm": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    "tensorop_f16_s884gemm": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    "tensorop_f16_s1688gemm": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    "tensorop_f16_s16816gemm": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    "tensorop_s16816gemm_f16": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    "wmma_tensorop_f16_s161616gemm": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    "xmma_new::gemm": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    "xmma_gemm_f16f16_f16f32": CUDA_KERNEL_CATEGORY_NAMES.FP16_GEMM_TC,
    # BF16_GEMM running on tensor cores
    "bf16_s884gemm": CUDA_KERNEL_CATEGORY_NAMES.BF16_GEMM_TC,
    "bf16_s1688gemm": CUDA_KERNEL_CATEGORY_NAMES.BF16_GEMM_TC,
    "bf16_s16816gemm": CUDA_KERNEL_CATEGORY_NAMES.BF16_GEMM_TC,
    "bf16_s161616gemm": CUDA_KERNEL_CATEGORY_NAMES.BF16_GEMM_TC,
    # Convolution  kernel names
    "convolveNd": CUDA_KERNEL_CATEGORY_NAMES.CONVOLUTION,
    "wgrad_": CUDA_KERNEL_CATEGORY_NAMES.CONVOLUTION,
    "dgrad_": CUDA_KERNEL_CATEGORY_NAMES.CONVOLUTION,
    "convolve_": CUDA_KERNEL_CATEGORY_NAMES.CONVOLUTION,
    "conv_depthwise": CUDA_KERNEL_CATEGORY_NAMES.CONVOLUTION,
    "GemmConvolution": CUDA_KERNEL_CATEGORY_NAMES.CONVOLUTION,
    # Embedding Table Lookups
    "embedding": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "permute_pooled_embs_kernel": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "permute_indices_weights_kernel": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "bucketize_sparse_features": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "EmbeddingBag": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "bounds_check_indices_kernel": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "linearize_cache_indices_kernel": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "lfu_cache_populate": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "lru_cache_insert_kernel": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "lxu_cache_lookup_kernel": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "lru_cache_find_uncached_kernel": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "permute_2D_data": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "permute_2D_lengths": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    "pack_segments_cuda_kernel": CUDA_KERNEL_CATEGORY_NAMES.EMBEDDING_BAGS,
    # Layer Normalization Kernels
    "layer_norm": CUDA_KERNEL_CATEGORY_NAMES.LAYER_NORM,
    "GammaBetaBackwardCUDAKernel": CUDA_KERNEL_CATEGORY_NAMES.LAYER_NORM,
    "LayerNormForwardCUDAKernel": CUDA_KERNEL_CATEGORY_NAMES.LAYER_NORM,
    "cuApplyLayerNorm": CUDA_KERNEL_CATEGORY_NAMES.LAYER_NORM,
    # Vectorized Elementwise Functors
    "vectorized_elementwise_kernel": CUDA_KERNEL_CATEGORY_NAMES.VECTORIZED_FUNCTORS,
    "_bfloat16_to_float_cuda_kernel": CUDA_KERNEL_CATEGORY_NAMES.VECTORIZED_FUNCTORS,
    "_float_to_bfloat16_cuda_kernel": CUDA_KERNEL_CATEGORY_NAMES.VECTORIZED_FUNCTORS,
    # Elementwise Functors
    "at::native::elementwise_kernel": CUDA_KERNEL_CATEGORY_NAMES.ELEMENTWISE_FUNCTORS,
    "::elementwise_kernel_with_index": CUDA_KERNEL_CATEGORY_NAMES.ELEMENTWISE_FUNCTORS,
    "at::native::index_elementwise_kernel": CUDA_KERNEL_CATEGORY_NAMES.ELEMENTWISE_FUNCTORS,
    "_scatter_gather_elementwise_kernel": CUDA_KERNEL_CATEGORY_NAMES.ELEMENTWISE_FUNCTORS,
    "at::native::unrolled_elementwise_kernel": CUDA_KERNEL_CATEGORY_NAMES.ELEMENTWISE_FUNCTORS,
    # Tensor Transforms
    "genericTranspose_kernel": CUDA_KERNEL_CATEGORY_NAMES.TENSOR_TRANSFORMS,
    "tensorTransformGeneric": CUDA_KERNEL_CATEGORY_NAMES.TENSOR_TRANSFORMS,
    "nchwToNhwcKernel": CUDA_KERNEL_CATEGORY_NAMES.TENSOR_TRANSFORMS,
    "nchwToFoldedNhwcKernel": CUDA_KERNEL_CATEGORY_NAMES.TENSOR_TRANSFORMS,
    "transposeBlock_kernel": CUDA_KERNEL_CATEGORY_NAMES.TENSOR_TRANSFORMS,
    "transposeWarp_kernel": CUDA_KERNEL_CATEGORY_NAMES.TENSOR_TRANSFORMS,
    "nchwAddPaddingKernel": CUDA_KERNEL_CATEGORY_NAMES.TENSOR_TRANSFORMS,
    "convertTensor_kernel": CUDA_KERNEL_CATEGORY_NAMES.TENSOR_TRANSFORMS,
    "nhwcToNchwKernel": CUDA_KERNEL_CATEGORY_NAMES.TENSOR_TRANSFORMS,
    # Sort, Scan and Reduction
    "reduce_kernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceRadixSortDownsweepKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceRadixSortOnesweepKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceRadixSortHistogramKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceRadixSortExclusiveSumKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceRadixSortSingleTileKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "MovingAverageMinMax": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceReduceByKeyKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceRadixSortUpsweepKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceScanKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "RadixSortScanBinsKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceScanInitKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "ScanTileState": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "radixSortKVInPlace": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceReduceKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    "DeviceReduceSingleTileKernel": CUDA_KERNEL_CATEGORY_NAMES.SORT_SCAN_REDUCE,
    # Memory Copies Device to Device
    "direct_copy_kernel_cuda": CUDA_KERNEL_CATEGORY_NAMES.MEMCOPIES_D2D,
    "CatArrayBatchedCopy": CUDA_KERNEL_CATEGORY_NAMES.MEMCOPIES_D2D,
    "copy_device_to_device": CUDA_KERNEL_CATEGORY_NAMES.MEMCOPIES_D2D,
    # Multi-Tensor Apply
    "multi_tensor_apply_kernel": CUDA_KERNEL_CATEGORY_NAMES.MULTI_TENSOR,
    # Triton
    "triton": CUDA_KERNEL_CATEGORY_NAMES.TRITON,
    # Operations on Jagged and Nested Tensors
    "jagged_": CUDA_KERNEL_CATEGORY_NAMES.JAGGED_NESTED,
    "nested_tensor": CUDA_KERNEL_CATEGORY_NAMES.JAGGED_NESTED,
    # Index Processing Kernels
    "linearize_index_kernel": CUDA_KERNEL_CATEGORY_NAMES.INDEXING,
    "indexFuncLargeIndex": CUDA_KERNEL_CATEGORY_NAMES.INDEXING,
    "indexSelectLargeIndex": CUDA_KERNEL_CATEGORY_NAMES.INDEXING,
    "indexing_backward_kernel": CUDA_KERNEL_CATEGORY_NAMES.INDEXING,
    "indexSelectSmallIndex": CUDA_KERNEL_CATEGORY_NAMES.INDEXING,
    "indexFuncSmallIndex": CUDA_KERNEL_CATEGORY_NAMES.INDEXING,
    "index_select_scalar_cumsum_kernel": CUDA_KERNEL_CATEGORY_NAMES.INDEXING,
    "_segment_sum_csr_cuda_kernel": CUDA_KERNEL_CATEGORY_NAMES.INDEXING,
    "_index_hash_cuda_kernel": CUDA_KERNEL_CATEGORY_NAMES.INDEXING,
    # Softmax
    "softmax": CUDA_KERNEL_CATEGORY_NAMES.SOFTMAX,
    "SoftMax": CUDA_KERNEL_CATEGORY_NAMES.SOFTMAX,
    # Optimizer
    "adam_cuda_kernel": CUDA_KERNEL_CATEGORY_NAMES.OPTIMIZER,
}


class CommType(Enum):
    ALL_2_ALL = 0
    ALL_REDUCE = 1
    ALL_GATHER = 2
    REDUCE = 3
    REDUCE_SCATTER = 4
    BROADCAST = 5
    UNKNOWN = 6


COMM_KERNEL_NAME_TO_CATEGORY: Dict[str, CommType] = {
    "ncclSendRecvKernel": CommType.ALL_2_ALL,
    "ncclKernel_SendRecvKernel": CommType.ALL_2_ALL,
    "ncclAllReduce": CommType.ALL_REDUCE,
    "ncclKernel_AllReduce": CommType.ALL_REDUCE,
    "ncclAllGather": CommType.ALL_GATHER,
    "ncclKernel_AllGather": CommType.ALL_GATHER,
    "ncclReduceScatter": CommType.REDUCE_SCATTER,
    "ncclKernel_ReduceScatter": CommType.REDUCE_SCATTER,
    "ncclReduce": CommType.REDUCE,
    "ncclKernel_Reduce": CommType.REDUCE,
    "ncclBroadcast": CommType.BROADCAST,
    "ncclKernel_Broadcast": CommType.BROADCAST,
    "gloo:all_to_all": CommType.ALL_2_ALL,
    "gloo:all_reduce": CommType.ALL_REDUCE,
}
COMMS_OPERATOR: str = "nccl:"


class MemcpyType(Enum):
    DTOD = 0
    DTOH = 1
    HTOD = 2
    UNKNOWN = 3


MEMCPY_TYPE_TO_STR: Dict[MemcpyType, str] = {
    MemcpyType.DTOD: "memcpy_dtod",
    MemcpyType.DTOH: "memcpy_dtoh",
    MemcpyType.HTOD: "memcpy_htod",
    MemcpyType.UNKNOWN: "memcpy_type_unknown",
}


def get_memory_usage(o: Any) -> int:
    """Get the memory usage of an object.

    Args:
        o (object): an object

    Returns:
        the memory usage by the object <o>.
    """
    seen: Set[int] = set()

    def get_size(obj: Any) -> int:
        """Get the size of an object recursively."""
        if id(obj) in seen:
            return 0
        size = sys.getsizeof(obj)
        seen.add(id(obj))

        if isinstance(obj, (str, bytes, bytearray)):
            pass
        elif isinstance(obj, dict):
            size += sum([get_size(v) for v in obj.keys()])
            size += sum([get_size(v) for v in obj.values()])
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(get_size(i) for i in obj)
        elif hasattr(obj, "__dict__"):
            size += get_size(vars(obj))
        return size

    return get_size(o)


def normalize_path(path: str) -> str:
    """
    Convert a Linux path to Python path.

    Args:
        path (str) : a path acceptable by the OS.

    Returns:
        A path supported by Python.
    """
    if path.startswith("./"):
        path2 = path[2:]
        if len(path2) > 0:
            normalized_path = str(Path.cwd().joinpath(path2))
        else:
            normalized_path = str(Path.cwd())
    elif path.startswith("~/"):
        path2 = path[2:]
        if len(path2) > 0:
            normalized_path = str(Path.home().joinpath(path2))
        else:
            normalized_path = str(Path.home())
    else:
        normalized_path = path
    return normalized_path


def is_gloo_kernel(name: str) -> bool:
    return "gloo:" in name


def is_nccl_kernel(name: str) -> bool:
    return name.startswith("nccl")


def is_comm_kernel(name: str) -> bool:
    """
    Check if a given GPU kernel is a communication kernel : nccl or gloo ops are supported

    Args:
        name (str): name of the GPU kernel.

    Returns:
        A boolean indicating if the kernel is a communication kernel.
    """
    return is_nccl_kernel(name) or is_gloo_kernel(name)


def is_memory_kernel(name: str) -> bool:
    """
    Check if a given GPU kernel is a memory kernel.

    Args:
        name (str): name of the GPU kernel.

    Returns:
        A boolean indicating if the kernel is an IO kernel.
    """
    return "Memcpy" in name or "Memset" in name


def get_kernel_type(name: str) -> str:
    if is_comm_kernel(name):
        return KernelType.COMMUNICATION.name
    elif is_memory_kernel(name):
        return KernelType.MEMORY.name
    else:
        return KernelType.COMPUTATION.name


def get_memory_kernel_type(name: str) -> str:
    """Memcpy Type is basically a prefix of the kernel name ~ Memcpy DtoH"""
    if name[:6] == "Memset":
        return "Memset"
    if name[:6] != "Memcpy":
        return "Memcpy Unknown"
    prefix_size = 11  # len("Memcpy DtoH")
    return name[:prefix_size]


def merge_kernel_intervals(kernel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all kernel intervals in the given dataframe such that there are no overlapping.
    """
    kernel_df.sort_values(by="ts", inplace=True)
    kernel_df["end"] = kernel_df["ts"] + kernel_df["dur"]
    # Operators within the same group need to be merged together to form a larger interval.
    kernel_df["group"] = (kernel_df["ts"] > kernel_df["end"].shift().cummax()).cumsum()
    kernel_df = (
        kernel_df.groupby("group", as_index=False)
        .agg({"ts": min, "end": max})
        .drop(["group"], axis=1)
        .sort_values(by="ts")
    )
    return kernel_df


def shorten_name(name: str) -> str:
    """Shorten a long operator/kernel name.

    The CPU operator and CUDA kernel name in the trace can be too long to follow.
    This utility removes the functional arguments, template arguments, and return values
    to make the name easy to understand.
    """
    s: str = name.replace("->", "")
    stack: List[str] = []
    for c in s:
        if c == ">":  # match generic template arguments
            while len(stack) and stack[-1] != "<":
                stack.pop()

            if len(stack) > 0 and stack[-1] == "<":
                stack.pop()
        elif c == ")":  # match arguments or comments
            while len(stack) and stack[-1] != "(":
                stack.pop()
            if len(stack) > 0 and stack[-1] == "(":
                stack.pop()
        else:
            stack.append(c)
    return "".join(stack).split(" ")[-1]


def flatten_column_names(df: pd.DataFrame) -> None:
    """Flatten a DataFrame's a multiple index column names to a single string"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).rstrip("_") for col in df.columns]


def is_triton_kernel(name: str) -> bool:
    return "triton" in name


def comm_type_from_kernel_name(name: str) -> CommType:
    """Classifies comms kernels into a category based on simple rules"""

    for comm_kernel_name in COMM_KERNEL_NAME_TO_CATEGORY:
        if comm_kernel_name in name:
            return COMM_KERNEL_NAME_TO_CATEGORY[comm_kernel_name]

    return CommType.UNKNOWN


def classify_kernel(name: str) -> str:
    """Classifies kernels into a category based on simple rules"""

    if is_nccl_kernel(name):
        return CUDA_KERNEL_CATEGORY_NAMES.NCCL_KERNEL

    if is_gloo_kernel(name):
        return CUDA_KERNEL_CATEGORY_NAMES.GLOO_KERNEL

    if is_triton_kernel(name):
        return CUDA_KERNEL_CATEGORY_NAMES.TRITON

    categories = set()
    for kernel_name in CUDA_KERNEL_NAME_TO_CATEGORY:
        if kernel_name in name:
            categories.add(CUDA_KERNEL_NAME_TO_CATEGORY[kernel_name])

    if len(categories) == 0:
        return CUDA_KERNEL_CATEGORY_NAMES.UNKNOWN
    elif len(categories) == 1:
        return list(categories)[0]
    else:
        # In some cases the kernel may be vectorized/elementwise type
        # but on an operation type for which we can clearly label its
        # compute specific.
        non_functors = 0
        ret_category = ""
        for category in categories:
            if (category != CUDA_KERNEL_CATEGORY_NAMES.ELEMENTWISE_FUNCTORS) and (
                category != CUDA_KERNEL_CATEGORY_NAMES.VECTORIZED_FUNCTORS
            ):
                non_functors += 1
                ret_category = category

        # If there is a single category detected that is a non functor
        # we can just return it.
        if non_functors == 1:
            return ret_category

        # In case there are several non-functor categories, it is better
        # to signal this than label incorrectly
        return CUDA_KERNEL_CATEGORY_NAMES.MULTIPLE


def classify_memcpy(name: str) -> str:
    if "Memcpy DtoD" in name:
        return MEMCPY_TYPE_TO_STR[MemcpyType.DTOD]
    elif "Memcpy DtoH" in name:
        return MEMCPY_TYPE_TO_STR[MemcpyType.DTOH]
    elif "Memcpy HtoD" in name:
        return MEMCPY_TYPE_TO_STR[MemcpyType.HTOD]
    else:
        return MEMCPY_TYPE_TO_STR[MemcpyType.UNKNOWN]


def short_kernel_name(name: str) -> str:
    try:
        composed_keywords = ["void", "cutlass", "at::"]

        is_composed = False
        for keyword in composed_keywords:
            if keyword in name:
                is_composed = True
                break

        if not is_composed:
            return name

        if ("void" in name) and ("cutlass" in name):
            idx_start = name.index("<") + 1
            idx_stop = name.index(">")
            return name[idx_start:idx_stop]

        if "embedding" in name:
            idx_start = name.index(" ") + 1
            idx_stop = name.index(">") + 1
            return name[idx_start:idx_stop]

        if "xmma_new" in name:
            idx_start = name.index("<") + 1
            idx_stop = name.index("::Kernel")
            return name[idx_start:idx_stop]

        if "cub::" in name:
            idx_start = name.index(" ") + 1
            idx_stop = name.index("<")
            return name[idx_start:idx_stop]

        kernel_namespaces = [
            "elementwise_kernel",
            "reduce_kernel",
            "anonymous namespace",
        ]

        is_namespace_sequence = False
        for namespace in kernel_namespaces:
            if namespace in name:
                is_namespace_sequence = True
                break

        if is_namespace_sequence:
            tmp_name = name.replace("at::", "")
            tmp_name = tmp_name.replace("native::", "")
            tmp_name = tmp_name.replace("detail::", "")
            tmp_name = tmp_name.replace("binary_internal::", "")
            tmp_name = tmp_name.replace("cuda::", "")
            tmp_name = tmp_name.replace("c10::", "")
            tmp_name = tmp_name.replace("std::", "")
            tmp_name = tmp_name.replace("(anonymous namespace)::", "")
            tmp_name = tmp_name.replace("void ", "")

            idx_start = tmp_name.find("{")
            if idx_start > 0:
                idx_stop = tmp_name.rindex("}")
                if idx_stop != -1:
                    to_remove = tmp_name[idx_start + 1 : idx_stop]
                    tmp_name = tmp_name.replace(to_remove, "")

            return tmp_name

    except Exception:
        pass

    # If the function hasn't returned by now, this kernel is
    # likely a rare encounter so we do not shorten its name

    return name
