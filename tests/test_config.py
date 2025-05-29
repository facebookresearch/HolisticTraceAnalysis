# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from hta.configs.config import HtaConfig
from hta.configs.default_values import DEFAULT_CONFIG_FILENAME
from hta.configs.env_options import (
    CP_LAUNCH_EDGE_ENV,
    get_options,
    HTA_DISABLE_NS_ROUNDING_ENV,
    HTAEnvOptions,
)


class HtaConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_config_path = "/tmp/test_config.json"
        self.test_config = {
            "a": 1,
            "b": ["s", "t"],
            "c": {"c1": 2, "c2": {"c21": 10.0}},
        }
        with open(self.test_config_path, "w+") as fp:
            json.dump(self.test_config, fp)

    def test_get_default_paths(self):
        paths = HtaConfig.get_default_paths()
        self.assertEqual(
            len(paths), 3, f"expect the default file paths to be 3 but got {len(paths)}"
        )
        self.assertTrue(
            all(str(path).endswith(DEFAULT_CONFIG_FILENAME) for path in paths)
        )

    def test_constructor_no_config_file(self):
        config = HtaConfig(load_default_paths=False)
        self.assertDictEqual(config.get_config(), {})

    def test_constructor_one_config_file(self):
        config = HtaConfig(self.test_config_path, load_default_paths=False)
        self.assertEqual(config.get_config(), self.test_config)

    def test_get_config_file_paths(self):
        config = HtaConfig(self.test_config_path, load_default_paths=False)
        paths = config.get_config_file_paths()
        self.assertListEqual(paths, [self.test_config_path])

    def test_get_config_all(self):
        config = HtaConfig(self.test_config_path, load_default_paths=False)
        config_values = config.get_config()
        self.assertDictEqual(config_values, self.test_config)

    def test_get_config_one_level(self):
        config = HtaConfig(self.test_config_path, load_default_paths=False)
        self.assertEqual(config.get_config("a"), self.test_config["a"])
        self.assertListEqual(config.get_config("b"), self.test_config["b"])
        self.assertDictEqual(config.get_config("c"), self.test_config["c"])

    def test_get_config_multiple_levels(self):
        config = HtaConfig(self.test_config_path, load_default_paths=False)
        self.assertDictEqual(config.get_config("c"), self.test_config["c"])
        self.assertEqual(config.get_config("c.c1"), self.test_config["c"]["c1"])
        self.assertEqual(
            config.get_config("c.c2.c21"), self.test_config["c"]["c2"]["c21"]
        )
        self.assertIsNone(config.get_config("d"))
        self.assertIsNone(config.get_config("c.c2.c22"))
        self.assertIsNone(config.get_config("c.c1.c3"))

    def test_get_config_default_values(self):
        config = HtaConfig(self.test_config_path, load_default_paths=False)
        self.assertEqual(config.get_config("c", 10), self.test_config["c"])
        self.assertEqual(config.get_config("d", 10), 10)

    def test_get_env_options(self):
        self.assertNotEqual(get_options(), "")

    def test_get_test_data_path(self):
        data_path = HtaConfig.get_test_data_path("h100")
        self.assertTrue(Path(data_path).exists())


class HTAEnvOptionsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Reset the singleton instance before each test
        HTAEnvOptions._instance = None
        # Save original environment variables
        self.original_env = os.environ.copy()

    def tearDown(self) -> None:
        # Reset the singleton instance after each test
        HTAEnvOptions._instance = None
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_singleton_behavior(self):
        """Test that instance() always returns the same instance."""
        instance1 = HTAEnvOptions.instance()
        instance2 = HTAEnvOptions.instance()
        self.assertIs(instance1, instance2, "instance() should return the same object")

    def test_get_set_options(self):
        """Test getting and setting options."""
        options = HTAEnvOptions.instance()

        # Test default values
        self.assertFalse(options.disable_ns_rounding())
        self.assertFalse(options.disable_call_graph_depth())
        self.assertFalse(options.critical_path_add_zero_weight_launch_edges())
        self.assertFalse(options.critical_path_show_zero_weight_launch_edges())
        self.assertFalse(options.critical_path_strict_negative_weight_check())

        # Test setting values
        options.set_disable_ns_rounding(True)
        self.assertTrue(options.disable_ns_rounding())

        options.set_critical_path_add_zero_weight_launch_edges(True)
        self.assertTrue(options.critical_path_add_zero_weight_launch_edges())

        # Test that other values remain unchanged
        self.assertFalse(options.disable_call_graph_depth())
        self.assertFalse(options.critical_path_show_zero_weight_launch_edges())
        self.assertFalse(options.critical_path_strict_negative_weight_check())

    def test_environment_variable_reading(self):
        """Test that environment variables are correctly read."""
        # Set environment variables
        os.environ[HTA_DISABLE_NS_ROUNDING_ENV] = "1"
        os.environ[CP_LAUNCH_EDGE_ENV] = "1"

        # Create a new instance that should read these environment variables
        HTAEnvOptions._instance = None
        options = HTAEnvOptions.instance()

        # Check that the environment variables were correctly read
        self.assertTrue(options.disable_ns_rounding())
        self.assertTrue(options.critical_path_add_zero_weight_launch_edges())
        self.assertFalse(options.disable_call_graph_depth())  # Default value

    def test_get_options_str(self):
        """Test the get_options_str method."""
        options = HTAEnvOptions.instance()
        options_str = options.get_options_str()

        # Check that the string contains all option names
        self.assertIn("disable_ns_rounding", options_str)
        self.assertIn("disable_call_graph_depth", options_str)
        self.assertIn("critical_path_add_zero_weight_launch_edges", options_str)
        self.assertIn("critical_path_show_zero_weight_launch_edges", options_str)
        self.assertIn("critical_path_strict_negative_weight_check", options_str)

    @patch.dict(os.environ, {HTA_DISABLE_NS_ROUNDING_ENV: "1"})
    def test_legacy_functions(self):
        """Test that legacy functions use the singleton instance."""
        from hta.configs.env_options import (
            disable_call_graph_depth,
            disable_ns_rounding,
        )

        # Reset the singleton to ensure it reads the patched environment
        HTAEnvOptions._instance = None

        # Check that legacy functions return the correct values
        self.assertTrue(disable_ns_rounding())
        self.assertFalse(disable_call_graph_depth())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
