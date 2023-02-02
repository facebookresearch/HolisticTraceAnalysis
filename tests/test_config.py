# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest

from hta.configs.config import HtaConfig
from hta.configs.default_values import DEFAULT_CONFIG_FILENAME


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
        self.assertEqual(len(paths), 3, f"expect the default file paths to be 3 but got {len(paths)}")
        self.assertTrue(all([str(path).endswith(DEFAULT_CONFIG_FILENAME) for path in paths]))

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
        self.assertEqual(config.get_config("c.c2.c21"), self.test_config["c"]["c2"]["c21"])
        self.assertIsNone(config.get_config("d"))
        self.assertIsNone(config.get_config("c.c2.c22"))
        self.assertIsNone(config.get_config("c.c1.c3"))

    def test_get_config_default_values(self):
        config = HtaConfig(self.test_config_path, load_default_paths=False)
        self.assertEqual(config.get_config("c", 10), self.test_config["c"])
        self.assertEqual(config.get_config("d", 10), 10)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
