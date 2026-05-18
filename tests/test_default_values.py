# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from hta.configs.default_values import AttributeSpec, ValueType, YamlVersion


class TestYamlVersion(unittest.TestCase):
    def test_get_version_str(self) -> None:
        v = YamlVersion(1, 2, 3)
        self.assertEqual(v.get_version_str(), "1.2.3")

    def test_from_string_valid(self) -> None:
        v = YamlVersion.from_string("1.0.0")
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 0)
        self.assertEqual(v.patch, 0)

    def test_from_string_multidigit(self) -> None:
        v = YamlVersion.from_string("12.34.56")
        self.assertEqual(v.major, 12)
        self.assertEqual(v.minor, 34)
        self.assertEqual(v.patch, 56)

    def test_from_string_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            YamlVersion.from_string("1.0")

    def test_from_string_non_numeric_raises(self) -> None:
        with self.assertRaises(ValueError):
            YamlVersion.from_string("a.b.c")

    def test_from_string_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            YamlVersion.from_string("")

    def test_roundtrip(self) -> None:
        original = "2.5.10"
        v = YamlVersion.from_string(original)
        self.assertEqual(v.get_version_str(), original)

    def test_named_tuple_fields(self) -> None:
        v = YamlVersion(major=1, minor=2, patch=3)
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 2)
        self.assertEqual(v[2], 3)

    def test_comparison(self) -> None:
        v1 = YamlVersion(1, 0, 0)
        v2 = YamlVersion(1, 0, 1)
        v3 = YamlVersion(2, 0, 0)
        self.assertLess(v1, v2)
        self.assertLess(v2, v3)
        self.assertEqual(v1, YamlVersion(1, 0, 0))


class TestValueType(unittest.TestCase):
    def test_enum_values(self) -> None:
        self.assertEqual(ValueType.Int.value, 1)
        self.assertEqual(ValueType.Float.value, 2)
        self.assertEqual(ValueType.String.value, 3)
        self.assertEqual(ValueType.Object.value, 4)

    def test_all_types_exist(self) -> None:
        expected_names = {"Int", "Float", "String", "Object"}
        actual_names = {vt.name for vt in ValueType}
        self.assertEqual(actual_names, expected_names)


class TestAttributeSpec(unittest.TestCase):
    def test_equality_same_values(self) -> None:
        v = YamlVersion(1, 0, 0)
        a1 = AttributeSpec("col", "raw", ValueType.Int, 0, v)
        a2 = AttributeSpec("col", "raw", ValueType.Int, 0, v)
        self.assertEqual(a1, a2)

    def test_inequality_different_name(self) -> None:
        v = YamlVersion(1, 0, 0)
        a1 = AttributeSpec("col1", "raw", ValueType.Int, 0, v)
        a2 = AttributeSpec("col2", "raw", ValueType.Int, 0, v)
        self.assertNotEqual(a1, a2)

    def test_inequality_different_type(self) -> None:
        v = YamlVersion(1, 0, 0)
        a1 = AttributeSpec("col", "raw", ValueType.Int, 0, v)
        a2 = AttributeSpec("col", "raw", ValueType.Float, 0.0, v)
        self.assertNotEqual(a1, a2)

    def test_equality_ignores_min_version(self) -> None:
        v1 = YamlVersion(1, 0, 0)
        v2 = YamlVersion(2, 0, 0)
        a1 = AttributeSpec("col", "raw", ValueType.Int, 0, v1)
        a2 = AttributeSpec("col", "raw", ValueType.Int, 0, v2)
        self.assertEqual(a1, a2)

    def test_not_equal_to_non_attributespec(self) -> None:
        v = YamlVersion(1, 0, 0)
        a = AttributeSpec("col", "raw", ValueType.Int, 0, v)
        self.assertNotEqual(a, "not_an_attributespec")

    def test_field_access(self) -> None:
        v = YamlVersion(1, 0, 0)
        a = AttributeSpec("my_col", "my_raw", ValueType.String, "default", v)
        self.assertEqual(a.name, "my_col")
        self.assertEqual(a.raw_name, "my_raw")
        self.assertEqual(a.value_type, ValueType.String)
        self.assertEqual(a.default_value, "default")
        self.assertEqual(a.min_supported_version, v)
