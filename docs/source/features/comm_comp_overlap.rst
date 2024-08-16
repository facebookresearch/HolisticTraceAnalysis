Communication Computation Overlap
=================================

In distributed training a significant amount of time is spent in communication
and synchronization events between GPUs. To achieve high GPU efficiency (i.e.
TFLOPS/GPU) it is vital to keep the GPU oversubscribed with computation
kernels. In other words, the GPU should not be blocked due to unresolved data
dependencies. One way to measure the extent to which computation is blocked by
data dependencies is to calculate the communication computation overlap. Higher
GPU efficiency is observed if communication events overlap computation events.
Lack of communication and computation overlap will lead to the GPU being idle,
thus the efficiency would be low. To sum up, higher communication computation
overlap is desirable. To calculate the overlap percentage for each rank we
measure the following ratio:

  | **(time spent in computation while communicating) / (time spent in communication)**

Communication computation overlap can be calculated as follows:

.. code-block:: python

   analyzer = TraceAnalysis(trace_dir = "traces/")
   overlap_df = analyzer.get_comm_comp_overlap()

The function returns a dataframe containing the overlap percentage
for each rank.

.. image:: ../_static/overlap_df.png
   :scale: 50%
   :align: center

When the ``visualize`` argument is set to True, the `get_comm_comp_overlap
<../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_comm_comp_overlap>`_
function also generates a bar graph representing the overlap by rank.

.. image:: ../_static/overlap_plot.png
