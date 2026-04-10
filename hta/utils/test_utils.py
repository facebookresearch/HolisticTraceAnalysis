import copy
import functools
import os
from pathlib import Path
from typing import Any, Callable, Iterable


def get_test_data_dir(*subdirs: str) -> str:
    """Return the path to the test data directory, handling both Buck and open-source layouts.

    In Buck tests, the env var TEST_DATA_PREFIX_PATH is set (e.g. "HolisticTraceAnalysis")
    and data is at "<prefix>/tests/data/". In open-source (pip install), data is relative
    to the caller's source tree at "<repo_root>/tests/data/".

    Args:
        *subdirs: Optional path components appended after "tests/data".
            Example: get_test_data_dir("vision_transformer") returns ".../tests/data/vision_transformer"

    Returns:
        Absolute path to the test data directory (or subdirectory).
    """
    prefix = os.environ.get("TEST_DATA_PREFIX_PATH", "")
    if prefix:
        test_data_dir = os.path.join(prefix, "tests", "data", *subdirs)
    else:
        # Fallback: assume this file is at hta/utils/test_utils.py, so repo root is two levels up.
        repo_root = Path(__file__).resolve().parent.parent.parent
        test_data_dir = str(
            repo_root / "tests" / "data" / Path(*subdirs)
            if subdirs
            else repo_root / "tests" / "data"
        )

    if not os.path.isdir(test_data_dir):
        raise FileNotFoundError(f"Test data directory does not exist: {test_data_dir}")
    return test_data_dir


def data_provider(
    data_function: Callable[[], Iterable[Any]],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    A simple data provider decorator.

    Args:
        data_function: a Callable object that returns an iterable set of test data.

    Note: This utility is created to support porting unit test code between HTA OSS and an HTA branch at Meta internal.

    Usage of the decorator is as follows:

    class TestTransform(unittest.TestCase):
        # pyre-ignore[56]
        def dataset():
            return (
              {'input_1': 'hello', 'expected_result': 'HeLlO'},
              {'input_1': 'world', 'expected_result': 'WoRlD'}
            )

        @data_provider(dataset)
        def test_transform(self, input_1: str,  expected_result: str):
            self.assertEqual(transform(input_1), expected_result)

        # pyre-ignore[56]
        @data_provider(
           lambda: (
              {'input_1': 'hello', 'expected_result': 'HeLlO'},
              {'input_1': 'world', 'expected_result': 'WoRlD'},
            )
         )
        def test_transform2(self, input_1: str,  expected_result: str):
            self.assertEqual(transform(input_1), expected_result)
    """

    # The decorator is dependent on the function providing the data.
    def decorator(test_func: Callable[..., None]) -> Callable[..., None]:
        @functools.wraps(test_func)
        def wrapper(self, *args, **kwargs) -> None:
            for data in data_function():
                if isinstance(data, dict):
                    kwargs_copy = copy.copy(kwargs)
                    kwargs_copy.update(data)
                    test_func(self, *args, **kwargs_copy)
                elif isinstance(data, tuple):
                    test_func(self, *(args + tuple(data)), **kwargs)
                else:
                    test_func(self, data)

        return wrapper

    return decorator
