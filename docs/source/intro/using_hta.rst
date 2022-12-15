Analyzing Traces with HTA
=========================

We recommend using HTA in a Jupyter notebook and provide `example notebooks
<https://github.com/facebookresearch/HolisticTraceAnalysis/tree/main/examples>`_,
for your convenience. To get started, import the hta package in a Jupyter
notebook, create a ``TraceAnalysis`` object and off we go in exactly two lines of
code.

Trace Analysis
--------------

.. code-block:: python

    from hta.trace_analysis import TraceAnalysis
    analyzer = TraceAnalysis(trace_dir = "/trace/folder/path")


Using the features is straightforward. E.g.

.. code-block:: python

  # Temporal breakdown
  temporal_breakdown_df = analyzer.get_temporal_breakdown()

  # Idle time breakdown
  idle_time_df = analyzer.get_idle_time_breakdown()

  # Kernel breakdown
  kernel_breakdown_df = analyzer.get_gpu_kernel_breakdown()

  # Communication computation overlap
  comm_comp_overlap_df = analyzer.get_comm_comp_overlap()

  # Memory bandwidth time series
  memory_bw_series = analyzer.get_memory_bw_time_series()

  # Memory bandwidth summary
  memory_bw_summary = analyzer.get_memory_bw_summary()

  # Queue length time series
  ql_series = analyzer.get_queue_length_time_series()

  # Queue length summary
  ql_summary = analyzer.get_queue_length_summary()

  # CUDA kernel launch statistics
  cuda_kernel_launch_stats = analyzer.get_cuda_kernel_launch_stats()

  # Frequent CUDA kernel sequences
  frequent_patterns_df = analyzer.get_frequent_cuda_kernel_sequences(operator_name="aten::linear",
                                                                    output_dir="/output/trace/path"
                                                                   )

To learn more about the features in detail we refer the reader to the
**Features** section. The features can be tuned by various
arguments that are available to the user. See the `TraceAnalysis
<../api/trace_analysis_api.html>`_ API for the options available.
For a detailed demo, HTA provides the `trace_analysis_demo notebook
<https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/examples/trace_analysis_demo.ipynb>`_
in the examples folder in the repo.

Trace Diff
----------

HTA also provides an API to compare two sets of traces where the first set can
be thought of as the "control group" and the second as the "test group." This
is useful when engineers need to know about CPU ops and GPU kernels which have
been added or removed resulting from a code change. The API also calculates the
counts and the duration for each op and kernel added or removed. Additionally, the
API also provides visualization capability via the `visualize_counts_diff
<../api/trace_diff_api.html#hta.trace_diff.TraceDiff.visualize_counts_diff>`_
and `visualize_duration_diff
<../api/trace_diff_api.html#hta.trace_diff.TraceDiff.visualize_duration_diff>`_
methods. See the `TraceDiff <../api/trace_diff_api.html>`_ API for more
details.

.. code-block:: python

   from hta.trace_diff import TraceDiff

   # compute the diff between two sets of traces
   compare_traces_df = TraceDiff.compare_traces(control_group_trace, test_group_trace)

For a detailed demo HTA provides the `trace_diff_demo notebook
<https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/examples/trace_diff_demo.ipynb>`_
in the examples folder in the repo.

.. tip::
   HTA generates powerful visualizations using the `plotly
   <http://plotly.com/python/>`_ library. Hovering over the images shows
   useful numerics about the graph and the modebar on the top right allows the
   user to zoom, pan, crop and download the graphs.
