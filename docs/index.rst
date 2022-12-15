.. Holistic Trace Analysis Documentation master file, created by
   sphinx-quickstart on Thu Dec  8 20:32:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Holistic Trace Analysis
=======================
Holistic Trace Analysis (HTA) is an open source performance analysis and
visualization Python library for PyTorch users. HTA takes as input `Kineto
traces <https://github.com/pytorch/kineto>`_ collected by the `PyTorch Profiler
<https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/>`_
and up-levels the performance information contained in the traces.

ML researchers and systems engineers often struggle to computationally scale up
their models because they are not aware of the performance bottlenecks in their
workloads. The resources requested for a job (e.g. GPUs, memory) are often
misaligned with the resources actually required due to lack of visibility
“under the hood”.

The goal of HTA is to help engineers and researchers achieve the best
performance from the hardware stack. For this to happen it is imperative to
understand the resource utilization and bottlenecks for distributed training
and inference workloads.

Features in Holistic Trace Analysis
***********************************

To aid in performance debugging HTA provides the following features

#. `Temporal Breakdown <source/features/temporal_breakdown.html>`_: Breakdown of GPU time in
   terms of time spent in computation, communication, memory events, and idle
   time on a single node and across all ranks.

#. `Idle Time Breakdown <source/features/idle_time_breakdown.html>`_: Breakdown of GPU idle
   time into waiting for the host, waiting for another kernel or attributed to
   an unknown cause.

#. `Kernel Breakdown <source/features/kernel_breakdown.html>`_: Find
   kernels with the longest duration on each rank.

#. `Kernel Duration Distribution <source/features/kernel_breakdown.html#kernel-duration-distribution>`_: Distribution of average time
   taken by longest kernels across different ranks.

#. `Communication Computation Overlap <source/features/comm_comp_overlap.html>`_: Calculate the
   percentage of time when communication overlaps computation.

#. `CUDA Kernel Launch Statistics <source/features/cuda_kernel_launch_stats.html>`_: Distributions
   of GPU kernels with very small duration, large duration, and excessive
   launch time.

#. `Augmented Counters (Memory copy bandwidth, Queue length) <source/features/augmented_counters.html>`_:
   Augmented trace files which provide insights into memory copy bandwidth and
   number of outstanding operations on each CUDA stream.

#. `Frequent CUDA Kernel Patterns <source/features/frequent_cuda_kernels.html>`_: Find the CUDA
   kernels most frequently launched by any given PyTorch or user defined
   operator.

#. `Trace Diff <source/features/trace_diff.html>`_: A trace comparison tool to identify and
   visualize the differences between traces.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   self
   source/intro/perf_debugging
   source/intro/trace_collection
   source/intro/installation
   source/intro/using_hta

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Features

   source/features/temporal_breakdown
   source/features/idle_time_breakdown
   source/features/kernel_breakdown
   source/features/comm_comp_overlap
   source/features/augmented_counters
   source/features/cuda_kernel_launch_stats
   source/features/frequent_cuda_kernels
   source/features/trace_diff

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   source/api/trace_analysis_api
   source/api/trace_diff_api
