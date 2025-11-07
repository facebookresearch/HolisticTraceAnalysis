# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script illustrate the use of symbol tables and data frames for trace analysis.
To run this script, use the following command:

python3 examples/symbol_table_demo.py --trace_dir tests/data/vision_transformer --max_ranks 4
Note: For the above command to work specify the path to HolisticTraceAnalysis folder on line 22.
"""
import argparse
import logging
import os

from typing import Optional

import pandas as pd
import plotly.express as px

from hta.common.trace_collection import TraceCollection

path_to_hta = "~/HolisticTraceAnalysis"
trace_dir: str = path_to_hta + "/tests/data/vision_transformer"
demo_max_ranks: int = 1


def set_pandas_display_options():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.2f}".format)


def demo_statistics(
    trace: TraceCollection, rank: int, k: Optional[int] = None
) -> pd.DataFrame:
    """
    Show the first k items of the kernels by duration in a specific rank's trace.
    <rank>

    Args:
        trace: a TraceCollection instance.
        rank: the rank to be analyzed.
        k: how many items to show in the output; If None, then show all items.

    Returns:
        The resulted dataframe from this analysis.
    """
    df = trace.get_trace(rank)
    sym_id_map = trace.symbol_table.get_sym_id_map()
    sym_table = trace.symbol_table.get_sym_table()

    df_cpu_ops = df[df["cat"] == sym_id_map["Kernel"]]
    total_time = df_cpu_ops["dur"].sum()
    gb = df_cpu_ops.groupby(by="name")["dur"].agg(
        ["sum", "max", "min", "mean", "std", "count"]
    )
    gb["percent"] = gb["sum"] / total_time * 100

    gb.reset_index(inplace=True)
    gb["name"] = gb["name"].apply(lambda x: sym_table[x])
    gb = gb.set_index("name", drop=True)

    if k is None:
        k = len(gb)
    result_df = gb.sort_values(by="percent", ascending=False)
    k = min(k, len(result_df))
    top_k = result_df[:k].copy()
    if k < len(result_df):
        others = result_df[k:]
        other_sum = others["sum"].sum()
        top_k.loc["all_others"] = [
            other_sum,
            others["max"].max(),
            others["min"].min(),
            others["mean"].mean(),
            others["std"].mean(),
            others["count"].mean(),
            other_sum / total_time * 100,
        ]
    return top_k


def demo_visualization(df: pd.DataFrame, title: str, visualize: bool = False) -> None:
    if visualize:
        fig = px.bar(df, x=df.index, y="sum")
        fig.show()
    else:
        df = df[["sum", "count", "percent"]].copy()
        df["Average Duration (ns)"] = df["sum"] / df["count"]
        df["count"] = df["count"].astype("int").copy()
        df = df.rename(
            columns={
                "sum": "Total Duration (ns)",
                "count": "Counts",
                "percent": "% of Total Time",
            }
        )
        logging.info(f"{title}\n{df}\n")


def load_trace(trace_dir, max_ranks) -> TraceCollection:
    trace = TraceCollection(trace_dir=trace_dir)
    trace.parse_traces(max_ranks=max_ranks, use_multiprocessing=True)
    return trace


def run_demo(
    trace_dir: str,
    max_ranks: int,
    preloaded_trace: Optional[TraceCollection] = None,
):
    """_summary_

    Args:
        trace_name (str): name of the trace
        base_trace_dir (str): the base path of the traces
        max_ranks (int): maximum number of ranks to be analyzed
        preloaded_trace (Optional[TraceCollection], optional): a preloaded collection of traces.
            Defaults to None.
    """
    # load the trace
    if preloaded_trace is None:
        demo_trace = load_trace(trace_dir, max_ranks)
    else:
        demo_trace = preloaded_trace

    sym_id_map = demo_trace.symbol_table.get_sym_id_map()
    sym_table = demo_trace.symbol_table.get_sym_table()

    # example for map encode ID for column `name` to original name
    # rank_0_df_name_id = demo_trace.traces[0]["name"]
    # rank_0_df_name = demo_trace.traces[0]["name"].apply(lambda x: sym_table[x])

    num_entries = min(10, len(sym_table))
    logging.info(
        f"\n===Symbol Table===\ntype={type(sym_table)}\nFirst {num_entries} entries:\n"
    )
    for i, sym in enumerate(sym_table[:num_entries]):
        logging.info(f"sym_table[{i}] = {sym}")
    logging.info("===End of Symbol Table")
    logging.info(
        f"===Symbol to ID Map===\ntype={type(sym_id_map)}\nFirst {num_entries} entries:\n"
    )
    count = num_entries
    for k, v in sym_id_map.items():
        logging.info(f"sym_id_map[{k}] = {v}")
        count -= 1
        if count <= 0:
            break
    logging.info("\n===End of Symbol to ID Map\n")

    df = demo_trace.get_trace(0)
    logging.info(f"\n===Data Frame of Rank-0===\ntype={type(df)}\n")
    logging.info(f"\n{df}\n")
    logging.info("\n===End of Data Frame\n")

    logging.info(f"===Data Frame Info===\ntype={type(df)}\n")
    demo_trace.get_trace(0).info()

    logging.info("\n===Kernel Statistics===\n")
    top_k: int = 10
    all_ranks_results = [
        demo_statistics(demo_trace, rank=r, k=top_k) for r in range(max_ranks)
    ]
    for r in range(max_ranks):
        logging.info(f"\nTop {top_k} kernels of rank {r}:\n{all_ranks_results[r]}\n")

    # uncomment this line to show the visualization on a browser
    for r in range(max_ranks):
        demo_visualization(
            all_ranks_results[r], f"\nTop {top_k} kernels for Rank {r}\n"
        )


def trace_info(trace: TraceCollection):
    rank = next(iter(trace.traces))
    df = trace.get_trace(rank)
    logging.info(f"\n===Dataframe of Rank {rank}")
    df.info()

    logging.info("\n===Event Counts")
    logging.info(df.nunique(axis=0))

    symbol_table = trace.symbol_table.get_sym_table()
    logging.info("\n===Event Categories")
    categories = [symbol_table[i] for i in df["cat"].unique()]
    logging.info(f"categories = {categories}")

    logging.info("\n===Event Streams")
    streams = df["stream"].unique()
    logging.info(f"categories = {streams}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace_dir",
        type=str,
        default=trace_dir,
        help="path where the traces are stored",
    )
    ap.add_argument(
        "--max_ranks",
        type=int,
        default=demo_max_ranks,
        help="max number of ranks to be analyzed",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="a flag to turn on debugging",
    )
    args = ap.parse_args()
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)

    set_pandas_display_options()

    _trace = load_trace(args.trace_dir, args.max_ranks)
    trace_info(_trace)
    run_demo(args.trace_dir, args.max_ranks, _trace)


if __name__ == "__main__":
    assert os.path.isdir(
        trace_dir
    ), f"{trace_dir} is not a valid system path. Use the path_to_hta variable to set the right prefix."
    main()
