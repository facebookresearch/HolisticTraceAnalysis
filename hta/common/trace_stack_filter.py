from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd

from hta.common.trace import TraceSymbolTable
from hta.common.trace_filter import Filter, FirstIterationFilter
from hta.configs.config import logger
from hta.utils.utils import get_symbol_column_names


def find_op_occurrence(
    df: pd.DataFrame, op_name: str, position: int, name_column: str = "s_name"
) -> Tuple[bool, pd.Series]:
    """Locate the event in the trace that has the specified op name and position.
    Args:
        df: a DataFrame with trace data.
        op_name: name of the operator. e.g., "split_embedding_codegen_forward_unweighted_kernel".
        position: the occurrence position of the operator. Use zero or positive values for forward
            counting and negative values for backward counting. For example, position=0 means
            the first occurrence of the operator and position=-1 means the last (latest) occurrence.
        name_column: Optional; name of the data frame column containing the operator name.
            Default: "s_name".
    Returns:
        A boolean value and a Series.
        The boolean value is True if there is a match, otherwise False.
        When there is a match, the Series is the matching event.
    """
    ops = df.loc[df[name_column].eq(op_name)].sort_values("ts")
    pos = position if position >= 0 else len(ops) + position
    if len(ops) > 0 and 0 <= pos < len(ops):
        return True, ops.iloc[pos]
    else:
        return False, pd.Series(dtype=pd.Int64Dtype)


def get_matching_kernels(
    df_ops: pd.DataFrame, df_both: pd.DataFrame, cat_column: str
) -> pd.DataFrame:
    """Get CUDA Kernels launched by operators in df_ops.

    Args:
        df_ops: a DataFrame that contains cpu operators.
        df_both: a DataFrame that contains both cpu operators and CUDA kernels.
        cat_column: the column name for cat in string type, which can be either `cat` or `s_cat`
            depending on how the DataFrame symbol columns are encoded/decoded.

    Returns:
        A DataFrame that contains the CUDA kernels launched by ops in df_ops.
    """
    df_runtimes = df_ops.loc[df_ops[cat_column].eq("cuda_runtime")]
    df_kernels = df_both.loc[df_both["index_correlation"].isin(df_runtimes["index"])]
    return df_kernels


class OperatorFilterMethod(Enum):
    """Types of Operator Filtering Methods

    Under: Select events which occur between the start and end of an operator occurrence,
        i.e.: {e | e in events and e.ts >= op.ts and e.end <= op.end}
    After: Select events which occur after or at the end of the operator occurrence,
        i.e.: { e | e in events and e.ts >= op.end}
    Before: Select events that occur before or at the start of the operator occurrence,
        i.e.: { e | e in events and e.end <= op.ts}
    """

    Under = 0
    After = 1
    Before = 2


class OperatorFilter(Filter):
    """A filter class that supports filtering out events based on operator names.

    Attributes:
        op_name: The name of the operator/user annotation. e.g., "forward" or "## forward ##".
        position: The position where the operator occurs in the trace. 0 for first occurrence, -1
            for last occurrence.
        method: How to filter out the events. See the enumerator class `OperatorFilterMethod`.
        include_gpu_kernels: Whether to include the GPU kernel before the selected events.
        name_column: Optional; name of the data frame column containing the operator name.
            Default: "" (check DataFrame to use the available one).

    Notes:
        + When the op_name is not found in the DataFrame, the filter will return an empty DataFrame.
    """

    def __init__(
        self,
        op_name: str,
        position: int,
        method: OperatorFilterMethod,
        include_gpu_kernels: bool = False,
        name_column: Optional[str] = None,
    ) -> None:
        self.op_name = op_name
        self.position = position
        self.method = method
        self.include_gpu_kernels = include_gpu_kernels
        self.name_column: str = name_column if name_column else ""

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        if self.name_column == "":
            name_column, cat_column = get_symbol_column_names(df)
        else:
            name_column = self.name_column
            cat_column = self.name_column.replace("name", "cat")

        if (
            name_column not in df.columns
            or cat_column not in df.columns
            or df.dtypes[name_column] != "object"
            or df.dtypes[cat_column] != "object"
        ):
            logger.error(f"df has no string column {name_column} of {cat_column}")
            return df

        found, op = find_op_occurrence(
            df, self.op_name, position=self.position, name_column=name_column
        )
        if not found:
            logger.warning(f"op {self.op_name}/{self.position} doesn't exist.")
            return pd.DataFrame()

        if self.method.value == OperatorFilterMethod.Under.value:
            latest_start, earliest_end, stream = (
                max(op["ts"], df["ts"].min()),
                min(op["end"], df["end"].max()),
                op["stream"],
            )
            df_ops = df.loc[
                (df["ts"].ge(latest_start))
                & (df["end"].le(earliest_end))
                & (df["stream"].eq(stream))
            ]
        elif self.method.value == OperatorFilterMethod.After.value:
            latest_start = max(op["end"], df.ts.min())
            stream = op["stream"]
            df_ops = df.loc[df["ts"].ge(latest_start) & df["stream"].eq(stream)]
        elif self.method.value == OperatorFilterMethod.Before.value:
            earliest_end = min(op["ts"], df["end"].max())
            stream = op["stream"]
            df_ops = df.loc[df["end"].le(earliest_end) & df["stream"].eq(stream)]
        else:
            raise NotImplementedError(f"Operator method {self.method} is not supported")

        if self.include_gpu_kernels:
            df_kernels = get_matching_kernels(df_ops, df, cat_column)
            result = pd.concat([df_ops, df_kernels])
            return result
        else:
            return df_ops


class AfterOperatorFilter(OperatorFilter):
    """Filter events that occur after or at the end of a given operator occurrence.

    Let `op` be the n-th operator occurrence in a trace slice `events`, an `AfterOperatorFilter`
        object selects events `{e | e in events and e.end >= op.end}`.
    """

    def __init__(
        self,
        op_name: str,
        position: int,
        include_gpu_kernels: bool = False,
        name_column: Optional[str] = None,
    ) -> None:
        super().__init__(
            op_name,
            position,
            OperatorFilterMethod.After,
            include_gpu_kernels,
            name_column,
        )


class BeforeOperatorFilter(OperatorFilter):
    """Filter events that occur before or at the start of a given operator occurrence.

    Let `op` be the n-th operator occurrence in a trace slice `events`, an `BeforeOperatorFilter`
        object selects events `{e | e in events and e.end <= op.ts}`.
    """

    def __init__(
        self,
        op_name: str,
        position: int,
        include_gpu_kernels: bool = False,
        name_column: Optional[str] = None,
    ) -> None:
        super().__init__(
            op_name,
            position,
            OperatorFilterMethod.Before,
            include_gpu_kernels,
            name_column,
        )


class UnderOperatorFilter(OperatorFilter):
    """Filter events that occur between the start and end of a given operator occurrence.

    Let `op` be the n-th operator occurrence in a trace slice `events`, an `UnderOperatorFilter`
        object selects events `{e | e in events and e.ts >= op.ts and e.end <= op.ts}`.
    """

    def __init__(
        self,
        op_name: str,
        position: int,
        include_gpu_kernels: bool = False,
        name_column: Optional[str] = None,
    ) -> None:
        super().__init__(
            op_name,
            position,
            OperatorFilterMethod.Under,
            include_gpu_kernels,
            name_column,
        )


class CombinedOperatorFilter(Filter):
    """Filter to select events under which:
        1. In the first iteration.
        2. Occur between the first occurrence of the root_op.
        3. Happen before last occurrence of the preceding_op.
        4. Occur after first occurrence of the following_op.

    Attributes:
        root_op_name: name of the root operator/annotation to be included.
            e.g., r"## forward ##"
        preceding_op_name: the name of the last operator preceding the selected events.
            e.g., r"All2All_Pooled_Wait"
        following_op_name: name of the first operator after the selected events.
            e.g., r"## sdd_preprocess_tensors ##"
        include_gpu_kernels: whether to include GPU kernels in the selected events. Default False.
        stack_depths: a list of integers indicating the depth of the selected events on the stack.
            e.g., [1], [2, 3]. Default None (include all CPU events).

    Notes:
        + For simplicity, the filter always uses the first iteration of the trace. To
            consider other iterations, apply an `IteratorIndexFilter` before this filter.
        + This filter assumes all operators (root, preceding, following) existed in the DataFrame.
            If any of them is missing, such filter criteria is ignored.
        + Criteria order: FirstIteration --> UnderOperator --> AfterOperator --> BeforeOperator
    """

    def __init__(
        self,
        root_op_name: str,
        preceding_op_name: str,
        following_op_name: str,
        include_gpu_kernels: bool = False,
        stack_depths: Optional[List[int]] = None,
    ) -> None:
        """Initialize the filter object."""
        self.root_op_name: str = root_op_name
        self.preceding_op_name: str = preceding_op_name
        self.following_op_name: str = following_op_name
        self.include_gpu_kernels: bool = include_gpu_kernels
        self.stack_depths: List[int] = stack_depths if stack_depths else []

    def __call__(
        self, df: pd.DataFrame, symbol_table: Optional[TraceSymbolTable] = None
    ) -> pd.DataFrame:
        name_col, cat_col = get_symbol_column_names(df)
        if (
            not name_col
            or not cat_col
            or name_col not in df.columns
            or cat_col not in df.columns
        ):
            logger.error("Couldn't find symbol name and/or categorical columns.")
            return df

        df_iter = FirstIterationFilter()(df)
        res = df_iter

        if not res.empty and find_op_occurrence(res, self.root_op_name, 0, name_col)[0]:
            f1 = UnderOperatorFilter(self.root_op_name, 0, False, name_col)
            res = f1(res)

        if (
            not res.empty
            and find_op_occurrence(res, self.preceding_op_name, -1, name_col)[0]
        ):
            f2 = AfterOperatorFilter(self.preceding_op_name, -1, False, name_col)
            res = f2(res)

        if (
            not res.empty
            and find_op_occurrence(res, self.following_op_name, 0, name_col)[0]
        ):
            f3 = BeforeOperatorFilter(self.following_op_name, 0, False, name_col)
            res = f3(res)

        if not res.empty and self.include_gpu_kernels:
            cuda_kernels = get_matching_kernels(res, df, cat_col)
            res = pd.concat([cuda_kernels, res], ignore_index=False)

        if (
            "depth" in res.columns
            and self.stack_depths is not None
            and len(self.stack_depths) > 0
        ):
            res = res.loc[res["stream"].gt(0) | res["depth"].isin(self.stack_depths)]

        return res
