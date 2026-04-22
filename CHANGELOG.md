# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.1] - 2026-04-21

#### Fixed
- Added `readme = "README.md"` to `pyproject.toml` to fix missing `long_description` on PyPI.
- Updated package description in `pyproject.toml`.
- Bumped `black` from 24.3.0 to >=26.3.1 to fix path traversal vulnerability in `--python-cell-magics` cache filename.

## [0.6.0] - 2026-04-21

#### Added
- Added automated PyPI and GitHub release workflow triggered on `v*` tag push, with test matrix, OIDC trusted publishing, and GitHub Release creation (#339).
- Handle metadata-only traces gracefully instead of crashing with `KeyError` when `dur`/`cat` columns are absent (#337).
- Added `MAX_EVENT_DURATION_US` constant and `max_event_duration_us` to `ParserConfig` to filter corrupted trace events exceeding maximum duration.
- Added NCCL `seq_num` (sequence number) field mapping to event args config for tracking collective operations.
- Added NCCL `collective_name` field parsing to identify collective operation types.
- Added `get_device_type()` method and MTIA trace support including device type inference from trace metadata.
- Added configurable timestamp integer conversion via parser config.
- Added `KERNEL_KERNEL_DELAY_THRESHOLD_US` constant to identify unrecognized dependencies in critical path analysis.
- Added event duration trimming based on tree hierarchy in CallGraph library.
- Added `pre_grad_nodes` to parser config for inference trace support.

#### Changed
- Changed pypi package name from 'HolisticTraceAnalysis' to 'tracesinght' (#339).
- Updated project maintainers (#338/327).
- Moved `pytest` from `requirements.txt` to `requirements-dev.txt` (dev-only dependency) (#339).
- Migrated package metadata from `setup.py` to `pyproject.toml` (#339).
- Dropped support for Python 3.8 and 3.9 (both EOL). Minimum supported version is now Python 3.10.
- Added Python 3.12 to supported versions and CI matrix.
- Updated GitHub Actions to latest versions (checkout@v4, setup-python@v5).
- Fixed GitHub Actions Node.js 16 deprecation warnings by upgrading to Node 20.
- Improved CI workflow: added pip caching, switched to `actions/checkout` submodules support, consolidated pip install steps.
- Aligned pre-commit CI Python versions with main CI workflow (3.10, 3.12).
- Updated pre-commit hooks: pre-commit-hooks v4.6.0, flake8 7.1.0, mypy v1.11.0.
- Updated ReadTheDocs build environment to ubuntu-22.04 with Python 3.12.
- Updated Sphinx documentation dependencies (sphinx>=7.0.0, sphinx_rtd_theme>=2.0.0).
- Updated kernel patterns for correct analysis in zoomer.
- Updated trace parser to handle unknown event names and values between `start_array` and `end_array`.

#### Fixed
- Fixed numerous test issues and flakiness (#333).
- Fixed placeholder author email in setup.py.
- Fixed empty DataFrame crash when `ProfilerStep` markers are missing.
- Fixed typo: `shortern_names` → `shorten_names`.
- Fixed spelling: "Dowloading" → "Downloading" in print statement.
- Fixed spelling: "Crttical" → "Critical" in error message.
- Return `False` instead of raising an error if critical path graph validation fails.

#### Performance
- Optimized GPU kernel annotation association with indexed symbol table lookups.
- Faster trimming of trace events.
- Switched to `to_numeric` in `normalize_gpu_stream_numbers` for improved performance.


## [0.5.0] - 2023-05-27

#### Added
- Added support for AMD GPUs.
- Update pyproject.toml to workaround missing stub packages for yaml.
- Add trace format validator
- Added multiple trace filter classes and demos.
- Added enhanced trace call stack graph implementation.
- Added memory timeline view.
- Added support for trace parser customization.
- Added support for H100 traces.
- Add nccl collective fields to parser config
- Queue length analysis: Add feature to compute time blocked on a stream hitting max queue length.
- Add `kernel_backend` to parser config for Triton / torch.compile() support.
- Add analyses features for GPU user annotation attribution at trace and kernel level.
- Add support to parse all trace event args.

#### New Feature: Critical Path Analysis
- Added lightweight critical path analysis feature.
- Critical path analysis features: event attribution and `summary()`
- Critical path analysis fixes: fixing async memcpy and adding GPU to CPU event based synchronization.
- Added save and restore feature for critical path graph.
- Added save and restore feature for critical path graph.
- Fixes bug in Critical path analysis relating to listing out the edges on the critical path.
- Updated critical path analysis with edge attribution.
- Improvement: allow filtering of flow events in the overlaid trace.

#### Changed
- Change test data path in unittests from relative path to real path to support running test within IDEs.
- Add a workaround for overlapping events when using ns resolution traces (https://github.com/pytorch/pytorch/pull/122425)
- Better handling of CUDA sync events with steam = -1
- Fix ijson metadata parser for some corner cases
- Add an option for ns rounding and cover ijson loading with it.
- Updated Trace() api to specify a list of files and auto figure out ranks.

#### Fixed
- Fixed issue #65 to handle floating point counter values in cupti\_counter\_analysis.

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
