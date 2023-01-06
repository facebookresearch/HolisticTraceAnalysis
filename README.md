[![CircleCI](https://circleci.com/gh/facebookresearch/HolisticTraceAnalysis.svg?style=shield)](https://app.circleci.com/pipelines/github/facebookresearch/HolisticTraceAnalysis)
[![codecov](https://codecov.io/github/facebookresearch/holistictraceanalysis/branch/main/graph/badge.svg?token=R44P6M3RJN)](https://codecov.io/github/facebookresearch/holistictraceanalysis)
[![Docs](https://readthedocs.org/projects/hta/badge/?version=latest)](https://hta.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/CONTRIBUTING.md)

# Holistic Trace Analysis

Holistic Trace Analysis (HTA), is a performance analysis tool to identify performance bottlenecks in
distributed training workloads. HTA achieves this by analyzing traces collected through the [PyTorch
Profiler](https://github.com/pytorch/kineto) a.k.a. Kineto.

## Features

HTA provides the following features:

1. __Temporal Breakdown__ - Breakdown of time taken by the GPUs in terms of time spent in
   computation, communication, memory events, and idle time across all ranks.
1. __Kernel Breakdown__ - Finds kernels with the longest duration on each rank.
1. __Kernel Duration Distribution__ - Distribution of average time taken by longest kernels across
   different ranks.
1. __Idle Time Breakdown__ - Breakdown of GPU idle time into waiting for the host, waiting for
   another kernel or attribution to an unknown cause.
1. __Communication Computation Overlap__ - Calculate the percentage of time when communication
   overlaps computation.
1. __Frequent CUDA Kernel Patterns__ - Find the CUDA kernels most frequently launched by any given
   PyTorch or user defined operator.
1. __CUDA Kernel Launch Statistics__ - Distributions of GPU kernels with very small duration, large
   duration, and excessive launch time.
1. __Augmented Counters (Queue length, Memory bandwidth)__ - Augmented trace files which provide
   insights into memory bandwidth utilized and number of outstanding operations on each CUDA stream.
1. __Trace Comparison__ - A trace comparison tool to identify and visualize the differences between
   traces.

## Installation

HTA runs on Linux and Mac with Python >= 3.8.

### Create a Conda environment (optional)

To install Miniconda see [here](https://docs.conda.io/en/latest/miniconda.html).

To create the environment `env_name`
``` bash
conda create -n env_name
```

To activate the environment
``` bash
conda activate env_name
```

To deactivate the environment
``` bash
conda deactivate
```

### Install using PyPI (stable)

```
pip install HolisticTraceAnalysis
```

### Install from source

```
git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git
cd HolisticTraceAnalysis
pip install -r requirements.txt
pip install -e .
```

## Documentation

Learn more about the features and the API from our [documentation](https://hta.readthedocs.io/en/latest/index.html).

## Usage

### Data Preparation
All traces collected from a job must reside in a unique folder.

### Analysis in a Jupyter notebook

Activate the Conda environment and launch a Jupyter notebook.
```
conda activate env_name
jupyter notebook
```

Import HTA, and create a `TraceAnalysis` object
``` python
from hta.trace_analysis import TraceAnalysis
analyzer = TraceAnalysis(trace_dir = "/path/to/folder/containing/the/traces")
```

#### Basic Usage

``` python
# Temporal breakdown
temporal_breakdown_df = analyzer.get_temporal_breakdown()

# Kernel breakdown
kernel_breakdown_df = analyzer.get_gpu_kernel_breakdown()

# Idle time breakdown
idle_time_df = analyzer.get_idle_time_breakdown()

# Communication computation overlap
comm_comp_overlap_df = analyzer.get_comm_comp_overlap()

# Frequent CUDA kernel patterns
frequent_patterns_df = analyzer.get_frequent_cuda_kernel_patterns(operator_name="aten::linear", output_dir="/new/trace/path")

# CUDA kernel launch statistics
cuda_launch_kernel_stats = analyzer.get_cuda_launch_kernel_info()

# Memory bandwidth time series
memory_bw_series = analyzer.get_memory_bw_time_series()

# Memory bandwidth summary
memory_bw_summary = analyzer.get_memory_bw_summary()

# Queue length time series
ql_series = analyzer.get_queue_length_time_series()

# Queue length summary
ql_summary = analyzer.get_queue_length_summary()
```

For a detailed demo run the `trace_analysis_demo` and `trace_diff_demo` notebooks in the examples folder.

#### Advanced Usage

__Logging Level__

Logging level is set through a configuration file in HTA. The default logging level is set in
`hta/configs/logging.config` and can be changed in the `[logger_hta]` section of the file.
If needed, a different logging file can be configured to use by modifying
`hta/configs/trace_analyzer.json`.

#### Repo Map

```
├── examples                       # folder containing demo notebooks
│         ├── ...
├── hta
│         ├── analyzers            # core logic for each analysis
│         │       ├── ...
│         ├── common               # code common to multiple analysis
│         │       ├── ...
│         ├── configs              # config files
│         │       ├── ...
│         ├── trace_analysis.py    # entrypoint for TraceAnalysis API
│         ├── trace_diff.py        # entrypoint for TraceDiff API
│         └── utils                # utility files
│                 └── ...
├── scripts                        # generic tools for traces
│         └── ...
│── tests                          # unittests
│         └── ...
```

## Contributing
We welcome new contributions. If you plan to contribute new features or extensions, please first
open an [issue](https://github.com/facebookresearch/HolisticTraceAnalysis/issues) and discuss the feature with
us. To learn more about how to contribute, see our [contributing guidelines](https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/CONTRIBUTING.md).

Please let us know if you encounter a bug by filing an [issue](https://github.com/facebookresearch/HolisticTraceAnalysis/issues).

## The Team
HTA is currently maintained by: [Anupam Bhatnagar](https://github.com/anupambhatnagar), [Brian Coutinho](https://github.com/briancoutinho),
[Xizhou Feng](https://github.com/fengxizhou), [Yifan Liu](https://github.com/yifanliu112), [Sung-Han Lin](https://github.com/sunghlin) and
[Louis Feng](https://github.com/louisfeng). Past contributors include [Michael Acar](https://github.com/mjacar) and [Yuzhen Huang](https://github.com/Yuzhen11).

## License
Holistic Trace Analysis is licensed under the [MIT License](https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/LICENSE).
