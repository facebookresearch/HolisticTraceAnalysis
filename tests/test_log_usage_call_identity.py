# pyre-strict
import unittest
from unittest.mock import MagicMock, patch

from hta.api_usage.call_identity import (
    get_caller_identity,
    get_project_name,
    reset_caller_identity,
)


_MODULE_NAME = "hta.api_usage.call_identity"


class TestLogUsageCallIdentity(unittest.TestCase):

    # pyre-ignore[56]
    def test_get_project_name(self) -> None:
        for module_name, expected_project_name in [
            ("hta.common.trace", "hta_oss"),
            ("plotly.express.pie", "plotly.express"),
        ]:
            result = get_project_name(module_name)
            self.assertEqual(
                result,
                expected_project_name,
                f"{module_name} -> {result} (expected: {expected_project_name})",
            )

    @patch(f"{_MODULE_NAME}.getpass.getuser", return_value="test_user")
    def test_get_caller_identity(
        self,
        mock_getuser: MagicMock,
    ) -> None:
        reset_caller_identity()

        caller = get_caller_identity()
        self.assertEqual(caller.caller_name, "test_user")
        mock_getuser.assert_called_once()

        caller = get_caller_identity()
        self.assertEqual(caller.caller_id, 9999)
        self.assertEqual(caller.caller_name, "test_user")
        self.assertEqual(mock_getuser.call_count, 1)
