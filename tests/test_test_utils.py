import unittest

from hta.utils.test_utils import data_provider


class MyTestCase(unittest.TestCase):
    @data_provider(
        lambda: (
            {
                "var1": "Hello, ",
                "var2": "HTA",
                "expected_result": 10,
            },
        )
    )
    def test_data_provider(self, var1: str, var2: str, expected_result: int):
        self.assertEqual(len(var1) + len(var2), expected_result)


if __name__ == "__main__":
    unittest.main()
