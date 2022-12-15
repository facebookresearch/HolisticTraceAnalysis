CUDA Kernel Launch Statistics
=============================

.. image:: ../_static/cuda_kernel_launch.png

For each event launched on the GPU there is a corresponding scheduling event on
the CPU e.g. CudaLaunchKernel, CudaMemcpyAsync, CudaMemsetAsync. These events
are linked by a common correlation id in the trace. See figure above. This
feature computes the duration of the CPU runtime event, its corresponding GPU
kernel and the launch delay i.e. the difference between GPU kernel starting and
CPU operator ending. The kernel launch info can be generated as follows:

.. code-block:: python

  analyzer = TraceAnalysis(trace_dir="/path/to/trace/dir")
  kernel_info_df = analyzer.get_cuda_kernel_launch_stats()

A screenshot of the generated dataframe is given below.

.. image:: ../_static/cuda_kernel_launch_stats.png
    :align: center

The duration of the CPU op, GPU kernel and the launch delay allows us to find:

#. **Short GPU kernels** - GPU kernels with duration less than the
   corresponding CPU runtime event.

#. **Runtime event outliers** - CPU runtime events with excessive duration.

#. **Launch delay outliers** - GPU kernels which take too long to be scheduled.

HTA generates distribution plots for each of the aforementioned three categories.


**Short GPU kernels**

Usually, the launch time on the CPU side is between 5-20 microseconds. In some
cases the GPU execution time is lower than the launch time itself. The graph
below allows us to find how frequently such instances appear in the code.

.. image:: ../_static/short_gpu_kernels.png


**Runtime event outliers**

The runtime outliers depend on the cutoff used to classify the outliers, hence
the `get_cuda_kernel_launch_stats
<../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_cuda_kernel_launch_stats>`_
API provides the ``runtime_cutoff`` argument to configure the value.

.. image:: ../_static/runtime_outliers.png

**Launch delay outliers**

The launch delay outliers depend on the cutoff used to classify the outliers,
hence the `get_cuda_kernel_launch_stats
<../api/trace_analysis_api.html#hta.trace_analysis.TraceAnalysis.get_cuda_kernel_launch_stats>`_
API provides the ``launch_delay_cutoff`` argument to configure the value.

.. image:: ../_static/launch_delay_outliers.png
