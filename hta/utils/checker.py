import os
import os.path

from hta.common.types import OperationOutcome


def is_valid_directory(
    path_to_directory: str, must_be_writable: bool = False
) -> OperationOutcome:
    if (
        path_to_directory
        and os.path.isdir(path_to_directory)
        and os.access(path_to_directory, os.R_OK)
    ):
        if not must_be_writable or os.access(path_to_directory, os.W_OK):
            return OperationOutcome(True, "")
        else:
            return OperationOutcome(False, f"Path {path_to_directory} is not writable.")
    elif not path_to_directory:
        return OperationOutcome(
            False, "Variable `path_to_directory` must be a non-empty string"
        )
    elif not os.path.exists(path_to_directory):
        return OperationOutcome(False, f"Path {path_to_directory} does not exist.")
    elif not os.path.isdir(path_to_directory):
        return OperationOutcome(False, f"Path {path_to_directory} is not a directory.")
    else:
        return OperationOutcome(False, f"`{path_to_directory}` is not a valid path.")
