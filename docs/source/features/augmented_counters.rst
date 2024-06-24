Augmented Counters
==================

Memory Bandwidth & Queue Length Counters
----------------------------------------

Memory bandwidth counters measure the memory copy bandwidth used while copying
the data from H2D, D2H and D2D by memory copy (memcpy) and memory set (memset)
events. HTA also computes the number of outstanding operations on each CUDA
stream. We refer to this as **queue length**. When the queue length on a stream
is 1024 or larger new events cannot be scheduled on that stream and the CPU
will stall until the events on the GPU stream have processed.

The `generate_trace_with_counters
<../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.generate_trace_with_counters>`_
API outputs a new trace file with the memory bandwidth and queue length
counters. The new trace file contains tracks which indicate the memory
bandwidth used by memcpy/memset operations and tracks for the queue length on
each stream. By default, these counters are generated using the rank 0
trace file and the new file contains the suffix ``_with_counters`` in its name.
Users have the option to generate the counters for multiple ranks by using the
``ranks`` argument in the `generate_trace_with_counters
<../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.generate_trace_with_counters>`_
API.

.. code-block:: python

  analyzer = TraceAnalysis(trace_dir = "traces/")
  analyzer.generate_trace_with_counters()

A screenshot of the generated trace file with augmented counters.

.. image:: ../_static/mem_bandwidth_queue_length.png

HTA also provides a summary of the memory copy bandwidth and queue length
counters as well as the time series of the counters for the profiled portion of
the code using the following API:

#. `get_memory_bw_summary
   <../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_memory_bw_summary>`_

#. `get_queue_length_summary
   <../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_queue_length_summary>`_

#. `get_memory_bw_time_series
   <../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_memory_bw_time_series>`_

#. `get_queue_length_time_series
   <../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_queue_length_time_series>`_

To view the summary and time series use:

.. code-block:: python

  # generate summary
  mem_bw_summary = analyzer.get_memory_bw_summary()
  queue_len_summary = analyzer.get_queue_length_summary()

  # get time series
  mem_bw_series = analyzer.get_memory_bw_time_series()
  queue_len_series = analyzer.get_queue_length_time_series()

The summary contains the count, min, max, mean, standard deviation, 25th, 50th,
and 75th percentile.

.. image:: ../_static/queue_length_summary.png
   :align: center

The time series only contains the points when a value changes. Once a value is
observed the time series stays constant until the next update. The memory
bandwidth and queue length time series functions return a dictionary whose key
is the rank and the value is the time series for that rank. By default, the
time series is computed for rank 0 only.
