Temporal Breakdown
==================

To best utilize the GPUs it is vital to understand where the GPU is spending
time for a given job. Is the GPU spending time on computation, communication,
memory events, or is it idle? The temporal
breakdown feature breaks down the time spent in three categories

#. Idle time - GPU is idle.
#. Compute time - GPU is being used for matrix multiplications or vector operations.
#. Non-compute time - GPU is being used for communication or memory events.


To achieve high training efficiency the code should maximize compute time and
minimize idle time and non-compute time. This is accomplished by implementing
concurrent execution of computation kernels with communication or memory
kernels.

.. note::
    During concurrent execution of computation kernels with communication/memory
    kernels the time spent by communication/memory kernels is accounted for
    under compute time.

The temporal breakdown can be calculated as follows:

.. code-block:: python

   analyzer = TraceAnalysis(trace_dir = "traces/")
   
   time_spent_df = analyzer.get_temporal_breakdown()

The function returns a dataframe containing the temporal breakdown for each rank.
See figure below.

.. image:: ../_static/temporal_breakdown_df.png

When the ``visualize`` argument is set to True, the `get_temporal_breakdown
<../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_temporal_breakdown>`_
function also generates a bar graph representing the breakdown by rank.

.. image:: ../_static/temporal_breakdown_plot.png
