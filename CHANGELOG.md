# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
#### Added
- Added multiple trace filter classes and demos.
- Added enhanced trace call stack graph implementation.
- Added memory timeline view.
- Added support for trace parser customization.
- Added support for H100 traces.
- (Experimental) Support to read PyTorch Execution Trace and correlate it with PyTorch Profiler Trace.
- (Experimental) Added lightweight critical path analysis feature.
- (Experimental) Critical path analysis features: event attribution and `summary()`
- (Experimental) Critical path analysis fixes: fixing async memcpy and adding GPU to CPU event based synchronization.
- Add a workaround for overlapping events when using ns resolution traces (https://github.com/pytorch/pytorch/pull/122425)
- Better handling of CUDA sync events with steam = -1
- (Experimental) Added save and restore feature for critical path graph.

#### Changed
- Change test data path in unittests from relative path to real path to support running test within IDEs.

#### Deprecated
- Deprecated 'call_stack'; use 'trace_call_stack' and 'trace_call_graph' instead.

#### Removed

#### Fixed
- Fixed issue #65 to handle floating point counter values in cupti\_counter\_analysis.
- Fixes bug in Critical path analysis relating to listing out the edges on the critical path.
- Updated critical path analysis with edge attribution.

## [0.2.0] - 2023-05-22
#### Added
- (Experimental) Added CUPTI Counter analyzer feature to parse kernel and operator level counter statistics.

#### Changed
- Improved loading time by parallelizing reads in the `create_rank_to_trace_dict` function.

#### Fixed
- Fix unit of time in `get_gpu_kernel_breakdown` API.
- Optimized multiprocessing strategy to handle OOMs when process pool is too large.

## [0.1.3] - 2023-02-09

#### Changed
- Split requirements.txt into two files: `requirements.txt` and `requirements-dev.txt`. The former
  does not contain `kaleido`, `coverage` as they are required for development purposes only.

#### Removed
- Coverage tests for graph visualization
- Dependency on Matplotlib for Venn diagram generation

#### Fixed
- LICENSE type in setup.py
- Queue length summary handles missing data and negative values
- Use `.get` for key lookup in dictionary
- Typos in README.md


## [0.1.2] - 2023-01-06
- Initial release

## [0.1.1] - 2023-01-06 [YANKED]
- Test release

## [0.1.0] - 2023-01-06 [YANKED]
- Test release
