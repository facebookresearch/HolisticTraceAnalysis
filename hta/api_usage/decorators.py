# pyre-strict
import functools
import logging

from time import perf_counter
from typing import Callable, TypeVar

try:
    from ads_training.p9e.open_hta.meta.api_usage.call_identity import (
        get_caller_identity,
        get_project_name,
    )
    from ads_training.p9e.open_hta.meta.api_usage.log_to_db import log_to_db
except ImportError:
    from hta.api_usage.call_identity import get_caller_identity, get_project_name
    from hta.api_usage.log_to_db import log_to_db

R = TypeVar("R")

logger: logging.Logger = logging.getLogger(__name__)


def log_usage(func: Callable[..., R]) -> Callable[..., R]:
    """A Decorator that logs the usage the decorated function."""

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> R:
        func_name = func.__name__
        module_name = func.__module__
        project_name = get_project_name(module_name)

        t_start = perf_counter()

        result = func(*args, **kwargs)

        t_end = perf_counter()

        caller = get_caller_identity()

        log_to_db(func_name, module_name)

        logger.debug(
            f"{caller.caller_name} executed {project_name}|{module_name}|{func_name} in {t_end - t_start:.4f} seconds"
        )

        return result

    return wrapper
