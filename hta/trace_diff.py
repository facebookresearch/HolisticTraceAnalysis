# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from enum import Enum
from typing import Dict, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go

from hta.common.trace import Trace
from hta.configs.config import logger
from hta.utils.utils import flatten_column_names, shorten_name


class DeviceType(Enum):
    CPU = 1
    GPU = 2
    ALL = 3


# Define the TraceDir type alias
TraceDir = str


class LabeledTrace:
    """A wrapper class for the Trace class which assigns a label to each trace object.

    Attributes:
        label (str): a label attached to the trace.
        t (Trace): a Trace object that contains the trace data.
        iteration_df (pd.DataFrame): a DataFrame that contains the
    """

    def __init__(
        self,
        label: str = None,
        t: Optional[Trace] = None,
        trace_dir: Optional[str] = None,
    ):
        """Construct a LabeledTrace from either a Trace object or trace files in trace_dir."""
        self.label = label if label else f"t{random.randint(0,10)}"
        if t is not None:
            self.t = t
        elif trace_dir is not None and os.path.isdir(trace_dir):
            self.t = Trace(trace_dir=trace_dir)
        else:
            raise ValueError("either a trace object or a valid trace dir must be provided in LabeledTrace.__init__()")
        self.t.parse_traces()

        self.s_map = pd.Series(self.t.symbol_table.get_sym_id_map())
        self.s_tab = pd.Series(self.t.symbol_table.get_sym_table())
        self.iteration_df = self._extract_iterations()

    def ranks(self) -> List[int]:
        """Get all available ranks."""
        return sorted(self.t.traces.keys())

    def iterations(self) -> List[int]:
        """Get all iterations"""
        return sorted(self.iteration_df["iteration"].values.tolist())

    def _extract_iterations(self) -> pd.DataFrame:
        """Extract the iterations from the symbol map"""
        s_map = pd.Series(self.t.symbol_table.get_sym_id_map())
        iteration_df = s_map[s_map.index.str.startswith("ProfilerStep")].reset_index()
        iteration_df["iteration"] = iteration_df["index"].apply(lambda s: int(s.replace("ProfilerStep#", "")))
        iteration_df.rename(columns={0: "id", "index": "symbol"}, inplace=True)
        return iteration_df

    def extract_ops(
        self,
        rank: Optional[Union[int, List[int]]] = None,
        iteration: Optional[Union[int, List[int]]] = None,
        device_type: DeviceType = DeviceType.ALL,
    ) -> pd.DataFrame:
        """
        Description:
            Extract operators/kernels for a specific ranks, iterations, and device type (CPU, GPU, or both)

        Args:
            rank (Optional[Union[int, List[int]]]): Specify which rank(s) to use for the comparison.
                Default: None (use the first rank of the traces.)
            iteration (Optional[Union[int, List[int]]]): Specify which iterations(s) to use for the comparison.
                Default: None (use the first iteration of the traces.)
            device_type (DeviceType): Specify whether to compare CPU operators or GPU kernels or both.
                device_type = DeviceType.CPU    - compare CPU operators only
                device_type = DeviceType.GPU    - compare GPU kernels only
                device_type = DeviceType.ALL    - Compare both CPU operators and GPU kernels.
                Default: DeviceType.ALL

        Returns:
            pd.DataFrame
                A DataFrame of the trace records filtered by ranks, iterations, and device type.
        """
        # Set up and validate the ranks argument
        _ranks = self.ranks()
        if rank is None:
            if len(_ranks) >= 1:
                ranks = _ranks[:1]
            else:
                raise ValueError(f"Trace {self.label} doesn't have any rank.")
        else:
            if isinstance(rank, int):
                ranks = [rank]
            elif isinstance(rank, list):
                ranks = rank
            else:
                raise ValueError(f"Invalid type for argument `rank` {rank}.")

        for _rank in ranks:
            if _rank not in _ranks:
                raise ValueError(f"Invalid argument rank={rank}.")

        # Set up and validate the iteration arguments
        _iterations = self.iterations()
        if iteration is None:
            if len(_iterations) >= 1:
                iterations = _iterations[:1]
            else:
                raise ValueError(f"Trace {self.label} doesn't have any iteration.")
        else:
            if isinstance(iteration, int):
                iterations = [iteration]
            elif isinstance(iteration, list):
                iterations = iteration
            else:
                raise ValueError(f"Invalid type for argument `iteration` {iteration}.")

        for iter_nbr in iterations:
            if iter_nbr not in _iterations:
                raise ValueError(f"Invalid argument iteration={iter_nbr}.")

        # Filter the trace by ranks
        if len(ranks) == 1:
            df_rank = self.t.get_trace(ranks[0])
        else:
            df_rank = pd.concat(
                [self.t.get_trace(r) for r in ranks],
                axis=0,
                keys=_ranks,
                names=["rank", "idx"],
            ).reset_index()

        # Filter the trace by iterations
        df_iter = df_rank[df_rank["iteration"].isin(iterations)]

        # Filter the trace by device type
        if device_type == DeviceType.CPU:
            df = df_iter[df_iter["stream"].eq(-1)]
        elif device_type == DeviceType.GPU:
            df = df_iter[df_iter["stream"].ne(-1)]
        else:
            df = df_iter

        return df

    def get_ops_summary(self, ops: pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            Get the summary of a set of ops.

        Args:
            ops (pd.DataFrame): a set of operations/kernels

        Returns:
            A summary of the counts and total duration of ops grouped by categories and names.

                    cat 	        name 	            short_name 	    counts 	total_duration 	cat_id 	name_id
            0 	user_annotation 	nccl:all_reduce 	nccl:all_reduce 	1 	67 	            10 	    3
            1 	user_annotation 	ProfilerStep#1010 	ProfilerStep#1010 	1 	122149 	        10 	    17
            2 	cpu_op 	            aten::as_strided 	aten::as_strided 	1 	0 	            12 	    0
        """
        s_tab = self.s_tab
        df = ops[["cat", "name", "dur"]].groupby(["cat", "name"]).aggregate(["count", "sum"])
        df.columns = ["_".join(col).rstrip("_") for col in df.columns.values]
        df.reset_index(inplace=True)
        df.rename(
            columns={
                "name": "name_id",
                "cat": "cat_id",
                "dur_count": "counts",
                "dur_sum": "total_duration",
            },
            inplace=True,
        )
        df["cat"] = df["cat_id"].apply(lambda i: s_tab[i])
        df["name"] = df["name_id"].apply(lambda i: s_tab[i])
        df["short_name"] = df["name"].apply(lambda n: shorten_name(n))
        df = df[
            [
                "cat",
                "name",
                "short_name",
                "counts",
                "total_duration",
                "cat_id",
                "name_id",
            ]
        ]

        return df


def _trace_argument_adapter(t: Union[LabeledTrace, Trace, TraceDir], default_label: str) -> LabeledTrace:
    """A helper function to construct a LabeledTrace from several argument types."""
    lt: LabeledTrace
    if isinstance(t, TraceDir):
        lt = LabeledTrace(label=default_label, trace_dir=t)
    elif isinstance(t, Trace):
        lt = LabeledTrace(label=default_label, t=t)
    elif isinstance(t, LabeledTrace):
        lt = t
    else:
        raise ValueError(f"Invalid argument type for ({t}).")
    return lt


class TraceDiff:
    @classmethod
    def compare_traces(
        cls,
        control: Union[LabeledTrace, TraceDir],
        test: Union[LabeledTrace, TraceDir],
        control_rank: Optional[Union[int, List[int]]] = None,
        test_rank: Optional[Union[int, List[int]]] = None,
        control_iteration: Optional[Union[int, List[int]]] = None,
        test_iteration: Optional[Union[int, List[int]]] = None,
        device_type: DeviceType = DeviceType.ALL,
        use_short_name: bool = False,
    ) -> pd.DataFrame:
        r"""
        Compare the operators/kernels counts and total duration of two traces.

        Args:
            control (Union[LabeledTrace, Trace, TraceDir]): the control trace.
                A string or Trace object that defines the control trace. Possible values can be:
                    1. a str (TraceDir) that points to parent path of the trace files.
                    2. a Trace object that contains the trace records and metadata.
                    3. a LabeledTrace object, which is a wrapper of the Trace object with a label to identify the trace.

            test (Union[LabeledTrace, Trace, TraceDir]): the test trace.
                Similar to the control trace except it defines the test trace.

            control_rank (Optional[Union[int, List[int]]]): Specify which ranks of the control trace to use.
                This argument can be either a single rank (an integer) or multiple ranks (a list of integers).
                Default: use the first rank of the control traces for comparison

            test_rank (Optional[Union[int, List[int]]]):: Specify which rank of the test trace to use. See ``control_rank`` for usage.
                Default: use the first rank of the test traces for comparison

            control_iteration (Optional[Union[int, List[int]]]): Specify which iteration(s) of the control trace to use for the comparison.
                Default: use the first iteration of the control trace.

            test_iteration (Optional[Union[int, List[int]]]): Specify which iteration(s) of the test trace to use for the comparison.
                Default: use the first iteration of the test trace.

            device_type (DeviceType): Specify whether to compare CPU operators or GPU kernels or both.
                + device_type = DeviceType.CPU compares CPU operators only.
                + device_type = DeviceType.GPU compares GPU kernels only.
                + device_type = DeviceType.ALL compares both CPU operators and GPU kernels.
                Default: DeviceType.ALL

            use_short_name (bool): should the comparison use a shorter name?
                The CPU operator and CUDA kernel name in the trace can be too long to comprehend. The use_short_name argument
                removes the functional arguments, template arguments, and return values so that the name is easy to follow.
                Default: False.

        Returns:
            pd.DataFrame
                A DataFrame that summarizes the difference between the two traces with the following columns:

                + name (or short_name): the operator or kernel name. If ``use_short_name = True``, then short name will be used.
                + control_counts: the number of times an op occurs in the control trace.
                + test_counts: the number of times an op occurs in the test trace.
                + control_total_duration: the total duration of the ops in the control trace.
                + test_total_duration: the total duration of the ops in the test trace.
                + diff_counts: the difference in the counts, i.e. test_counts - control_counts.
                + diff_duration: the difference in total duration, i.e. test_total_duration - control_total_duration.
                + counts_change_categories: the type of changes. '=' (no change); '+' (more ops in test_trace); '-' (fewer ops in test_trace).

        Note:
            The unit for all the duration columns is microsecond (us).
        """
        # Setup control_trace and test_trace
        control_trace = _trace_argument_adapter(control, "Control")
        test_trace = _trace_argument_adapter(test, "Test")

        logger.info(f"comparing traces: {control_trace.label} and {test_trace.label}")
        if control_trace.label == test_trace.label:
            test_trace.label = f"{test_trace.label}_control"
            test_trace.label = f"{test_trace.label}_test"
            logger.warn(f"The two traces have the same label. change test_trace's label to {test_trace.label}")
        control_label = control_trace.label
        test_label = test_trace.label

        # Determine which column use to group the operators
        col_name = "short_name" if use_short_name else "name"
        control_trace_summary = (
            control_trace.get_ops_summary(control_trace.extract_ops(control_rank, control_iteration, device_type))
            .groupby(col_name)[["counts", "total_duration"]]
            .sum()
        )

        test_trace_summary = (
            test_trace.get_ops_summary(test_trace.extract_ops(test_rank, test_iteration, device_type))
            .groupby(col_name)[["counts", "total_duration"]]
            .sum()
        )

        comp = pd.concat(
            [control_trace_summary, test_trace_summary],
            axis=1,
            join="outer",
            keys=[control_label, test_label],
        )
        comp.fillna(0, inplace=True)
        flatten_column_names(comp)

        comp["diff_counts"] = comp[f"{test_label}_counts"] - comp[f"{control_label}_counts"]
        comp["diff_duration"] = comp[f"{test_label}_total_duration"] - comp[f"{control_label}_total_duration"]
        comp["counts_change_categories"] = comp["diff_counts"].apply(lambda c: "+" if c > 0 else "-" if c < 0 else "=")

        return comp

    @classmethod
    def ops_diff(
        cls,
        control: Union[LabeledTrace, TraceDir],
        test: Union[LabeledTrace, TraceDir],
        control_rank: Optional[Union[int, List[int]]] = None,
        test_rank: Optional[Union[int, List[int]]] = None,
        control_iteration: Optional[Union[int, List[int]]] = None,
        test_iteration: Optional[Union[int, List[int]]] = None,
        device_type: DeviceType = DeviceType.ALL,
    ) -> Dict[str, List[str]]:
        r"""
        Get the operator difference between two traces.

        Args:
            control (Union[LabeledTrace, Trace, TraceDir]): The control trace.
                A string or Trace object that defines the control trace. A possible value can be:
                    1. a str (TraceDir) that points to parent path of the trace files.
                    2. a Trace object that contains the trace records and metadata.
                    3. a LabeledTrace object, which is a wrapper of the Trace object with a label to identify the trace.

            test (Union[LabeledTrace, Trace, TraceDir]): The test trace.
                Similar to the control trace except it defines the test trace.

            control_rank (Optional[Union[int, List[int]]]): Specify which ranks
                of the control trace to use. This argument can be either a single
                rank (an integer) or multiple ranks (a list of integers).
                Default: use the first rank of the control traces for comparison

            test_rank (Optional[Union[int, List[int]]]):: Specify which rank
                of the test trace to use. See control_rank for usage.
                Default: use the first rank of the test traces for comparison

            control_iteration (Optional[Union[int, List[int]]]): Specify which
                iteration(s) of the control trace to use for the comparison.
                Default: use the first iteration of the control trace.

            test_iteration (Optional[Union[int, List[int]]]): Specify which
                iteration(s) of the test trace to use for the comparison.
                Default: use the first iteration of the test trace.

            device_type (DeviceType): Specify whether to compare CPU operators
                or GPU kernels or both. DeviceType.CPU, DeviceType.GPU, DeviceType.ALL
                compares CPU operators only, GPU kernels only, both CPU operators
                and GPU kernels respectively. Default: DeviceType.ALL

        Returns:
            Dict[str, List[str]]

                The operator/kernel changes are divided into five types:
                    + added: ops which are absent in control_trace but exist in test_trace.
                    + deleted: ops which exist in control_trace but are absent in test_trace.
                    + increased: ops which exist in both traces but occur more times in test_trace.
                    + decreased: ops which exist in both traces but occur fewer times in test_trace.
                    + unchanged: ops which exist in both traces and occur the same number of times.
        """
        control_trace = _trace_argument_adapter(control, "Control")
        test_trace = _trace_argument_adapter(test, "Test")
        df = cls.compare_traces(
            control_trace,
            test_trace,
            control_rank,
            test_rank,
            control_iteration,
            test_iteration,
            device_type,
        )
        col_control = f"{control_trace.label}_counts"
        col_test = f"{test_trace.label}_counts"
        col_diff = "diff_counts"
        return {
            "added": df.loc[df[col_control].eq(0) & df[col_test].gt(0)].index.tolist(),
            "deleted": df.loc[df[col_control].gt(0) & df[col_test].eq(0)].index.tolist(),
            "increased": df.loc[df[col_control].gt(0) & df[col_diff].gt(0)].index.tolist(),
            "decreased": df.loc[df[col_test].gt(0) & df[col_diff].lt(0)].index.tolist(),
            "unchanged": df.loc[df[col_test].gt(0) & df[col_diff].eq(0)].index.tolist(),
        }

    @classmethod
    def visualize_counts_diff(
        cls,
        df: pd.DataFrame,
        show_image: bool = True,
        export_image_path: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the changes in trace ops count using the output from ``compare_traces``.

        Args:
            df (pd.DataFrame): the result obtained from ``TraceDiff.compare_traces``.
            show_image (bool): set to True to display the image. Default = True.
            export_image_path (str): location where the image is saved.

        Returns:
            None
        """
        labels: List[str] = [
            col.replace("_total_duration", "") for col in df.columns if col.endswith("_total_duration")
        ]
        assert len(labels) >= 2

        fig = go.Figure(
            data=[
                go.Bar(name=labels[0], x=df.index, y=df[f"{labels[0]}_counts"]),
                go.Bar(name=labels[1], x=df.index, y=df[f"{labels[1]}_counts"]),
                go.Bar(name="Difference", x=df.index, y=df["diff_counts"]),
            ]
        )
        fig.update_layout(barmode="group", title_text="Ops Count Comparison")
        if show_image:
            fig.show()
        if export_image_path:
            fig.write_image(export_image_path)

    @classmethod
    def visualize_duration_diff(
        cls,
        df: pd.DataFrame,
        show_image: bool = True,
        export_image_path: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the changes in trace ops duration using the output from ``compare_traces``.

        Args:
            df (pd.DataFrame): the result obtained from ``TraceDiff.compare_traces``.
            show_image (bool): set to True to display the image. Default = True.
            export_image_path (str): location where the image is saved.

        Returns:
            None
        """
        labels: List[str] = [
            col.replace("_total_duration", "") for col in df.columns if col.endswith("_total_duration")
        ]
        assert len(labels) >= 2

        fig = go.Figure(
            data=[
                go.Bar(
                    name=labels[0],
                    x=df.index,
                    y=df[f"{labels[0]}_total_duration"],
                ),
                go.Bar(
                    name=labels[1],
                    x=df.index,
                    y=df[f"{labels[1]}_total_duration"],
                ),
                go.Bar(name="Difference", x=df.index, y=df["diff_duration"]),
            ]
        )
        fig.update_layout(
            barmode="group",
            title_text="Ops Duration Comparison",
            yaxis_title="Total Duration (us)",
            autosize=True,
            width=1000,
            height=1000,
        )
        if show_image:
            fig.show()
        if export_image_path:
            fig.write_image(export_image_path)
