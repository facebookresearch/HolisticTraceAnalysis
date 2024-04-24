import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set

import pandas as pd
import plotly.express as px
from hta.common.trace_df import (
    find_events_by_name_patterns_using_decoded_names,
    find_events_by_name_patterns_using_symbol_table,
)
from hta.common.trace_filter import Filter

from hta.common.trace_symbol_table import TraceSymbolTable
from hta.configs.config import logger
from plotly.io import to_html

AnnotationStreamID: int = 0
CPUStreamID: int = -1
DefaultTTL: int = 3600 * 24 * 30


DefaultMustHaveColumns: List[str] = ["name", "ts", "dur"]
DefaultHoverColumns: List[str] = [
    "trace",
    "rank",
    "iteration",
    "stream",
    "cat",
    "index",
]
DefaultTaskLabelColumns: List[str] = ["stream", "rank", "trace", "depth"]
DefaultCallStackHoverColumns: List[str] = [
    "name",
    "cat",
    "index",
    "ts",
    "dur",
    "rank",
    "stream",
    "height",
    "depth",
    "tid",
    "index_correlation",
]

REQUIRED_COLUMNS_FOR_CPU_GPU_ALIGNMENT = {
    "index",
    "stream",
    "ts",
    "dur",
    "name",
    "first_kernel_start",
    "kernel_span",
    "num_kernels",
}


class EventType(Enum):
    GPUEvents: int = 0
    CPUEvents: int = 1
    AllEvents: int = 2


class PlotFormat(Enum):
    HTML: str = "html"  # Return figure as an HTML string, don't show figure.
    Inline: str = "inline"  # Show the figure inline
    File: str = "file"  # Save the figure to a file, don't show figure.


@dataclass
class TimelinePlotSetting:
    auto_detect: bool = True
    # Events contents
    event_type: EventType = EventType.GPUEvents
    multi_ranks: bool = False
    multi_traces: bool = False
    multi_streams: bool = True
    # Essential timeline columns
    x_start_column: str = "calibrated_start_global"
    x_end_column: str = "calibrated_end_global"
    task_column: str = "task"
    color_column: str = "name"
    # Figure size
    # task_width: int = 1600
    task_height: int = 30
    # Amplify short events
    amplify_short_events: bool = True
    # Amplify short events to be `short_events_display_duration` long
    # so that they can be visible on the timeline.
    short_events_display_duration: int = 1000
    # Timestamp representation
    use_global_time: bool = False
    # plot format
    plot_format: PlotFormat = PlotFormat.Inline
    # compress threads
    compress_threads: bool = True


DefaultTimelinePlotSetting: TimelinePlotSetting = TimelinePlotSetting()


def auto_detect_setting(events: pd.DataFrame, setting: TimelinePlotSetting) -> None:
    """Detect the timeline plot setting based on the `events` DataFrame and set it accordingly.

    Args:
        events (pd.DataFrame): The DataFrame containing the events to be plotted.
        setting (TimelinePlotSetting): Settings for creating timeline plot.
    """
    if setting.auto_detect:
        setting.multi_ranks = (
            True if "rank" in events.columns and events["rank"].nunique() > 1 else False
        )
        setting.multi_traces = (
            True
            if "trace" in events.columns and events["trace"].nunique() > 1
            else False
        )
        setting.multi_streams = (
            True
            if "stream" in events.columns and events["stream"].nunique() > 1
            else False
        )
        streams = (
            [-1]
            if "stream" not in events.columns
            else events["stream"].unique().tolist()
        )

        if len(streams) > 1 and -1 in streams:
            setting.event_type = EventType.AllEvents
        elif -1 not in streams:
            setting.event_type = EventType.GPUEvents
        else:
            setting.event_type = EventType.CPUEvents


def _check_timeline_input(df: pd.DataFrame, setting: TimelinePlotSetting) -> bool:
    if df.empty:
        logger.error("The dataframe doesn't contain any event")
        return False

    must_have: Set[str] = {
        setting.x_start_column,
        setting.x_end_column,
        setting.task_column,
        setting.color_column,
    }
    if not must_have.issubset(set(df.columns)):
        logger.error(
            f"The dataframe doesn't contain the required column(s): {must_have-set(df.columns)}"
        )
        return False
    return True


def _simplify_name(name: str, max_length: int = 45) -> str:
    """Simplify the event name to be shorter than `max_length` chars.

    Args:
        name (str): A name to be simplified.
        max_length (int) : the maximum characters to be included in the simplified name.

    Return:
        str: The simplified name.
    """
    re_keep = re.compile(r"(^##.+##$)|(^<.+>$)|(^Mem\w+(\s+\w+)?\s+\(.+\)$)")
    re_name = re.compile(r"^(void\s+)?([\w_:.#\s<>,]+)(\(.+\))?")
    max_length = max(20, max_length)
    if not re_keep.match(name):
        name = re.sub(r"\(anonymous namespace\)::", "anon::", name)
        name = re.sub(r"^void\s+", "", name)
        name = re.sub(r"autograd::engine::evaluate_function: ", "autograd::", name)
        m = re_name.match(name)
        if m:
            name = m.group(2)
    return name if len(name) < max_length else name[: max_length - 3] + "..."


def _set_task_names(events: pd.DataFrame, setting: TimelinePlotSetting) -> pd.Series:
    tasks: pd.Series = pd.Series(dtype="str")

    def _compute_label_col(prefix: str, col: pd.Series) -> pd.Series:
        return pd.Series(
            f"{prefix} " + col.str.pad(width=5, side="right")
            if col.dtype == object
            else f"{prefix} " + col.astype(str).str.zfill(3)
        )

    def _compute_label_row(row: pd.Series) -> str:
        return (
            "cpu level " + str(row["depth"]).zfill(3)
            if row["stream"] == -1 and "depth" in row
            else (
                "cpu events"
                if row["stream"] == -1 and "depth" not in row
                else (
                    "gpu annotate"
                    if row["stream"] == AnnotationStreamID
                    else "gpu stream " + str(row["stream"]).zfill(3)
                )
            )
        )

    if setting.multi_traces:
        labels = _compute_label_col("trace", events["trace"])
        tasks = tasks + " " + labels if not tasks.empty else labels

    if setting.multi_ranks:
        labels = _compute_label_col("rank", events["rank"])
        tasks = tasks + " " + labels if not tasks.empty else labels

    if setting.event_type == EventType.GPUEvents and setting.multi_streams:
        labels = _compute_label_col("gpu stream", events["stream"])
        tasks = tasks + " " + labels if not tasks.empty else labels

    elif setting.event_type == EventType.CPUEvents:
        if "tid" in events.columns and not setting.compress_threads:
            labels = _compute_label_col("tid", events["tid"])
            tasks = tasks + " " + labels if not tasks.empty else labels
        if "depth" in events.columns:
            labels = _compute_label_col("cpu level", events["depth"])
            tasks = tasks + " " + labels if not tasks.empty else labels
    else:
        labels = pd.Series(events.apply(lambda row: _compute_label_row(row), axis=1))
        tasks = tasks + " " + labels if not tasks.empty else labels

    if tasks.empty:
        raise NotImplementedError(
            "The events must have at least one of label columns (`trace`, `rank`, `stream`, `tid`, `depth`)"
        )
    return tasks


def _get_sort_columns(events: pd.DataFrame, setting: TimelinePlotSetting) -> List[str]:
    sort_columns: List[str] = []
    if setting.multi_traces:
        sort_columns.append("trace")
    if setting.multi_ranks:
        sort_columns.append("rank")
    if setting.event_type == EventType.GPUEvents and setting.multi_streams:
        sort_columns.append("stream")
    elif setting.event_type == EventType.CPUEvents:
        if "tid" in events.columns:
            sort_columns.append("tid")
        if "depth" in events.columns:
            sort_columns.append("depth")
    else:
        if "stream" in events.columns:
            sort_columns.append("stream")

    sort_columns.append("ts")
    return [c for c in sort_columns if c in events.columns]


def align_module_with_kernels(
    events_df: pd.DataFrame,
    module_list: List[str],
    sym_table: Optional[TraceSymbolTable] = None,
    include_original_cpu_events: bool = False,
) -> pd.DataFrame:
    """
    Aligns a given list of modules with the kernels in the event dataframe.

    This function creates a set of annotation events which mirror a selected set of CPU events (cpu_ops or modules)
    and aligns each annotation event with the original CPU event's first kernel start time and last kernel end time.
    By aligning these annotation events with the corresponding kernel events, a user can easily track and examine the
    kernels that correspond to the cpu events on a timeline.

    Args:
        events_df: A dataframe of events.
        module_list: A list of module names to align with kernels.
        sym_table (TraceSymbolTable): A symbol table to map symbol IDs to their names.
        include_original_cpu_events (bool): Whether to include original CPU events in the returned Data Frame.
            If False, only return annotate events and CUDA kernels.

    Returns:
        A dataframe of events with the modules aligned with the kernels.
    """
    # next_index = events_df["index"].max() + 1
    required_columns = REQUIRED_COLUMNS_FOR_CPU_GPU_ALIGNMENT

    if not required_columns.issubset(set(events_df.columns)):
        error_msg = f"The input df doesn't contain required columns: {required_columns-set(events_df.columns)}"
        logger.warning(error_msg)
        raise ValueError(error_msg)

    # Select events which match any pattern described by module_list,
    #  on CPU device (stream = -1), and have at least one kernel.
    cpu_events = events_df.loc[
        events_df["stream"].eq(CPUStreamID) & events_df["num_kernels"].ge(1)
    ]
    if sym_table and "name" in events_df.columns and events_df["name"].dtype != object:
        indices_to_annotate = find_events_by_name_patterns_using_symbol_table(
            cpu_events, module_list, sym_table
        )
    else:
        if "name" in events_df.columns and events_df["name"].dtype == object:
            column = "name"
        elif "s_name" in events_df.columns and events_df["s_name"].dtype == object:
            column = "s_name"
        else:
            error_msg = (
                "The event data Frame doesn't contain a string column `name` or `s_name`"
                " or an int `name` column with a symbol table."
            )
            logger.warning(error_msg)
            raise ValueError(error_msg)
        indices_to_annotate = find_events_by_name_patterns_using_decoded_names(
            cpu_events, module_list, column
        )

    original_events = (
        events_df.loc[~events_df["stream"].isin([AnnotationStreamID])]
        if include_original_cpu_events
        else events_df.loc[events_df["stream"].gt(0)]
    )

    annotate_events = cpu_events.loc[indices_to_annotate].copy()

    # Update certain columns of the selected events
    annotate_events["ts"] = annotate_events["first_kernel_start"].astype(int)
    annotate_events["dur"] = annotate_events["kernel_span"].astype(int)
    annotate_events["stream"] = AnnotationStreamID

    # Merge the annotated events with original events
    events_with_annotation = pd.concat(
        [original_events, annotate_events], ignore_index=True
    )

    return events_with_annotation


def prepare_timeline_events(
    df: pd.DataFrame,
    setting: TimelinePlotSetting = DefaultTimelinePlotSetting,
    symbol_table: Optional[TraceSymbolTable] = None,
    hover_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Prepare the input DataFrame for plotting a timeline.

    Args:
        df (pd.DataFrame) : an input DataFrame.
        setting (TimelinePlotSetting): Settings for the timeline plot.
        symbol_table (TraceSymbolTable): A symbol table to map symbol IDs to their names
        hover_columns (List[str]): A list of columns that will be displayed on the hover card.

    Return:
        a DataFrame for selected events for plot_timeline_px
    """
    if df.empty:
        logger.error("The dataframe is empty.")
        return pd.DataFrame()

    hover_columns = hover_columns or DefaultHoverColumns

    required_columns: List[str] = DefaultMustHaveColumns
    if not set(required_columns).issubset(set(df.columns)):
        logger.error(
            f"columns {set(required_columns) - set(df.columns)} are not in the input DataFrame columns."
        )
        return pd.DataFrame()

    selected_cols = list(
        set(hover_columns)
        .union(set(required_columns))
        .union(DefaultTaskLabelColumns)
        .intersection(df.columns)
    )
    events: pd.DataFrame = df[selected_cols].copy()

    if setting.auto_detect:
        auto_detect_setting(df, setting)

    # decode the name column into s_name
    if symbol_table:
        sym_tab = symbol_table.get_sym_table()
        simplified_sym_tab = [_simplify_name(s) for s in sym_tab]
        events["name"] = events["name"].apply(
            lambda i: _simplify_name(simplified_sym_tab[i])
        )
        if "cat" in events.columns:
            events["cat"] = events["cat"].apply(lambda i: simplified_sym_tab[i])

    # shift the timestamp to the beginning time of the dataframe
    start_t = events["ts"].min()
    events["ts"] = events["ts"] - start_t

    events[setting.x_start_column] = events["ts"]

    if setting.amplify_short_events:
        events[setting.x_end_column] = events["ts"] + events["dur"].apply(
            lambda x: (
                setting.short_events_display_duration
                if x < setting.short_events_display_duration
                else x
            )
        )
        assert all(
            (events[setting.x_end_column] - events[setting.x_start_column]).ge(
                setting.short_events_display_duration
            )
        )
    else:
        events[setting.x_end_column] = events["ts"] + events["dur"]

    if setting.use_global_time:
        events[setting.x_start_column] = pd.to_datetime(
            events[setting.x_start_column] + start_t, unit="us"
        )
        events[setting.x_end_column] = pd.to_datetime(
            events[setting.x_end_column] + start_t, unit="us"
        )

    events[setting.task_column] = _set_task_names(events, setting)
    events = events.sort_values(by=_get_sort_columns(events, setting))

    return events


def plot_events_timeline(
    title: str,
    events: pd.DataFrame,
    setting: TimelinePlotSetting = DefaultTimelinePlotSetting,
    hover_columns: Optional[List[str]] = None,
    output_image_path: Optional[str] = None,
) -> Optional[str]:
    """
    Plots the timeline of trace events.

    Args:
        title (str): Title for the timeline plot.
        events (pd.DataFrame): DataFrame containing the events to be plotted.
        setting (TimelinePlotSetting): Settings for the timeline plot.
        hover_columns (List[str], Optional): A list of column names whose values will be
            displayed when the mouse hovers over each event.
        output_image_path (str): Path to a html file to save the plot. If not specified,
            no html file will be saved.

    Return:
        A string containing a html file of the plot.

    Precondition:
        The events DataFrame must have the following columns:
            + calibrated_start_global
            + calibrated_end_global
            + task
            + `<setting.color_column>`
    Raise:
        ValueError: If the events DataFrame does not have the required columns.
    """
    if not _check_timeline_input(events, setting):
        return None

    hover_columns = hover_columns or DefaultHoverColumns

    hover_data = [c for c in hover_columns if c in events.columns]

    unique_tasks = events[setting.task_column].unique()
    sorted_tasks = sorted(unique_tasks)

    # temp fix to show the plot properly
    if len(sorted_tasks) < 20:
        setting.task_height = 80

    fig = px.timeline(
        events,
        x_start=setting.x_start_column,
        x_end=setting.x_end_column,
        y=setting.task_column,
        hover_data=hover_data,
        category_orders={setting.task_column: sorted_tasks},
        color=setting.color_column,
        color_discrete_sequence=px.colors.qualitative.D3,
        height=50 + setting.task_height * len(sorted_tasks),
        title=title,
    )

    fig.layout.xaxis.type = "linear"
    for d in fig.data:
        d.x = events[events["name"].eq(d.name)]["dur"].tolist()

    fig.update_layout(
        xaxis_range=[
            events[setting.x_start_column].min(),
            events[setting.x_end_column].max(),
        ],
        xaxis={"showgrid": True, "rangeslider_visible": True},
        paper_bgcolor="rgba(0,0,0,0)",
    )

    output_path: str = "" if output_image_path is None else output_image_path
    if output_path != "":
        if output_path.endswith(".html"):
            fig.write_html(file=output_path)
        else:
            fig.write_image(output_path)
        logger.info(f"Saved plot as {output_path}.")

    if setting.plot_format.value == PlotFormat.Inline.value:
        fig.show()

    if setting.plot_format.value == PlotFormat.HTML.value:
        return to_html(fig, include_plotlyjs="cdn")

    return None


def get_fig_title_from_fig_name(fig_name: str, sep: str = r"\s+|_+|-+") -> str:
    fig_name = fig_name if fig_name else "untitled_figure"
    return " ".join([s.capitalize() for s in re.split(sep, fig_name) if s])


def get_fig_name_from_fig_title(title: str) -> str:
    title = title if title else "Untitled Figure"
    return "_".join([s.lower() for s in re.split(r"\s+", title) if s])


class Timeline:
    def __init__(
        self,
        timeline_events: pd.DataFrame,
        symbol_table: TraceSymbolTable,
        filter_func: Optional[Filter] = None,
        setting: TimelinePlotSetting = DefaultTimelinePlotSetting,
        hover_columns: Optional[List[str]] = None,
    ) -> None:
        self.timeline_events: pd.DataFrame = timeline_events
        self.symbol_table: TraceSymbolTable = symbol_table
        self.setting: TimelinePlotSetting = setting
        self.hover_columns: List[str] = hover_columns or DefaultHoverColumns
        self.prepare(filter_func)

    @staticmethod
    def is_timeline_events(df: pd.DataFrame, setting: TimelinePlotSetting) -> bool:
        columns = [setting.task_column, setting.x_start_column, setting.x_end_column]
        return set(columns).issubset(set(df.columns))

    def prepare(self, filter_func: Optional[Filter] = None) -> None:
        if self.is_timeline_events(self.timeline_events, self.setting):
            return
        if filter_func:
            self.timeline_events = filter_func(
                self.timeline_events, self.symbol_table
            ).copy()
        self.timeline_events = prepare_timeline_events(
            self.timeline_events, symbol_table=self.symbol_table, setting=self.setting
        )

    def plot(
        self,
        title: str,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot events timeline.

        Args:
            title (str): Title for the timeline plot.
            save_path (Optional[str], optional): File path to save the plot. If none, no plot will be saved.
        """

        plot_events_timeline(
            title,
            self.timeline_events,
            setting=self.setting,
            hover_columns=self.hover_columns,
            output_image_path=save_path,
        )
