# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
#### Added
- (Experimental) Support to read PyTorch Execution Trace and correlate it with PyTorch Profiler Trace.

#### Changed

#### Deprecated

#### Removed

#### Fixed

## [0.2.0] - 2023-05-22
#### Added
- (Experimental) Added CUPTI Counter analyzer feature to parse kernel and operator level counter statistics.

#### Changed
- Improved loading time by parallelizing reads in teh `create_rank_to_trace_dict` function.

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
