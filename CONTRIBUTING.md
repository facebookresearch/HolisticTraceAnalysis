# Contributing to Holistic Trace Analysis

We want to make contributing to this project as easy and transparent as possible.

## Our Development Process

Minor changes and improvements will be released on an ongoing basis. Larger
changes will be released on a more periodic basis.

## Pull Requests

We actively welcome your pull requests.

1. Clone the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code passes static analysis (see below).
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Environment Setup

```
conda create -n myenv
conda activate myenv
$ cd /path/to/HolisticTraceAnalysis
$ pip install -r requirements.txt
```

## Coding Style

* We follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide.
* Please setup pre-commit before opening up your PR.

### Pre-Commit (Recommended)

We use pre-commit to maintain the coding style. Pre-Commit checks are run via Github Actions on every
commit. To install all the relevant libraries and run the pre-commit tests locally, execute the following
commands:

```
pip install -e .
pip install pre-commit
pre-commit install
```

After the above, your `git commit` command will automatically trigger pre-commit checks.

## Testing
Holistic Trace Analysis is tested on the following python versions: 3.8, 3.9, 3.10, 3.11.

### Unit tests

__To run the entire test suite__

```
python3 -m unittest -v
# OR using pytest
pytest tests
```

__To run a specific test__

Use `python3 -m unittest -v moduleName.fileName.className.testName`. E.g. To run `test_sort_events`
in the `CallStackTestCase` class in `test_call_stack.py`, use the following command:

```
python3 -m unittest -v tests.test_call_stack.CallStackTestCase.test_sort_events
# OR using pytest
pytest tests -k test_sort_events
# OR using pytest with the exact file
pytest tests/test_call_stack.py -k test_sort_events
```

Note, all our tests are written using `unittest` module, so use of pytest is primarily for a better
test runner and support in the CI.

### CircleCI status
The build status on the `main` branch is visible on the repository homepage. CircleCI status
of your branch is visible on the PR page.

## License
By contributing to Holistic Trace Analysis, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this repository.
