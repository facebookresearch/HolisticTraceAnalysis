# pyre-strict
import unittest
from unittest.mock import MagicMock, patch

from hta.api_usage.decorators import log_usage

_MODULE_NAME = "hta.api_usage.decorators"


@log_usage
def sample_function(x: int, y: int) -> int:
    return x + y


_module_name: str = __name__


class TestLogUsageDecorator(unittest.TestCase):
    @patch(f"{_MODULE_NAME}.perf_counter", side_effect=[1.0, 2.0])
    @patch(f"{_MODULE_NAME}.get_caller_identity")
    @patch(f"{_MODULE_NAME}.get_project_name", return_value="test_project")
    @patch(f"{_MODULE_NAME}.logger")
    @patch(f"{_MODULE_NAME}.log_to_db")
    def test_log_usage(
        self,
        mock_log_to_db: MagicMock,
        mock_logger: MagicMock,
        mock_get_project_name: MagicMock,
        mock_get_caller_identity: MagicMock,
        mock_perf_counter: MagicMock,
    ) -> None:
        # Mock get_caller_identity
        mock_caller = MagicMock()
        mock_caller.caller_name = "test_caller"
        mock_get_caller_identity.return_value = mock_caller

        # Call the decorated function
        result = sample_function(3, 4)

        # Assert the function result
        self.assertEqual(result, 7)

        # Check if the log_to_db was called with correct arguments
        mock_log_to_db.assert_called_once_with("sample_function", _module_name)

        # Check if the logger.debug was called with the correct message
        mock_logger.debug.assert_called_once_with(
            f"test_caller executed test_project|{_module_name}|sample_function in 1.0000 seconds"
        )

        # Check if the helper functions were called
        mock_get_project_name.assert_called_once_with(_module_name)
        mock_get_caller_identity.assert_called_once()
