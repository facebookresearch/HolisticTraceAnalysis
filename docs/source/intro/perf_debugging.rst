Performance Debugging 101
=========================

To understand the GPU performance in distributed workloads, we consider how the
model operators interact with the GPU devices and how such interactions are
reflected in certain measurable metrics. At a high level, we can break down the
GPU operations in a model execution into three broad categories, henceforth
referred to as kernel types:

#. **Computation (COMP)** - Computation kernels execute compiled routines for
   matrix multiplication and similar numeric calculations. They are responsible
   for all of the number crunching necessary for model execution.

#. **Communication (COMM)** - Communication kernels are routines which are
   responsible for exchanging and synchronizing data between different GPU
   devices in a distributed training job. The NVIDIA Collective Communication
   Library (NCCL) is a widely used communication library and all its kernels
   have the prefix “nccl”. Example NCCL kernels include NCCL_AllGather,
   NCCL_ReduceScatter, NCCL_AllReduce, etc.

#. **Memory (MEM)** - Memory kernels manage the memory allocations and
   deallocations on the GPU devices and data movement between the memory space
   on the host and the GPUs. The memory kernels include Memcpy_H2D, Memcpy_D2H,
   Memcpy_D2D, Memset, etc. Here, H represents the Host and D represents the
   GPU Device. Thus, H2D, D2H, D2D stands for Host to Device, Device to Host
   and Device to Device respectively.

Because a modern GPU device e.g. NVIDIA A100 is a massively parallel
device which is capable of running multiple kernels simultaneously, it is
possible to overlap the computation, communication, and memory kernels to
reduce the model execution time. One common technique to achieve the overlap is
to utilize multiple CUDA streams. A CUDA stream is a sequence of operations
that execute on a GPU device in the order in which they are issued by the host
code. Different CUDA streams can be interleaved and even run concurrently, thus
achieving the effect of kernel overlap.

The performance of multiple GPU training jobs is affected by multiple factors.
Among these factors, how does a model execution create and orchestrate the GPU
kernels plays a critical role. HTA provides insights on how the model execution
interacts with the GPU devices and highlights the opportunities for performance
improvement.

With the features built in HTA, we aim to provide users insights into “what
is happening under the hood in a distributed GPU workloads?” We describe
these features in the upcoming sections.
