# pyre-strict

import getpass
import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CallerIdentity:
    """
    A class to store the data for logging.
    """

    caller_name: str = ""
    caller_id: int = -1
    initialized: bool = False


_caller_identity: CallerIdentity = CallerIdentity()


module_to_project_map: Dict[re.Pattern[str], str] = {
    re.compile(r"^hta.*$", flags=re.IGNORECASE): "hta_oss",
}


def get_project_name(
    module_name: str, default_project_name: Optional[str] = None
) -> str:
    """Infer the project name from module."""
    for pattern, project_name in module_to_project_map.items():
        if pattern.match(module_name):
            return project_name
    return default_project_name or ".".join(module_name.split(".")[:2])


def unixname_to_uid(caller_name: str) -> int:
    """
    Args:
        caller_name (str): A string representation of the caller identity.

    Returns:
        call_id: An integer presentation of the caller identity.
    """
    # For the open-source project, we use a fake
    return 9999


def get_caller_identity() -> CallerIdentity:
    """Get the common log data."""
    global _caller_identity

    if not _caller_identity.initialized:
        _caller_identity.caller_name = getpass.getuser()
        _caller_identity.caller_id = unixname_to_uid(_caller_identity.caller_name)
        _caller_identity.initialized = True

    return _caller_identity


def reset_caller_identity() -> None:
    """Reset the cached caller identity."""
    global _caller_identity
    _caller_identity.caller_name = ""
    _caller_identity.caller_id = -1
    _caller_identity.initialized = False
