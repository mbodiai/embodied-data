# Copyright 2024 Mbodi AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from gymnasium import spaces
import numpy as np
import os
import tempfile


from embdata.sample import Sample


def test_wrap():
    sample = Sample(1)
    assert sample.wrapped == 1, "Wrapped attribute should be the same as the first argument"


def test_wrap_nested():
    obj = {"a": 1, "b": {"c": 2}}
    sample = Sample(obj)
    assert sample.a == 1, "Wrapped attribute should be accessible as an attribute"
    assert sample.b.c == 2, "Nested wrapped attributes should be accessible as attributes"


def test_wrap_dict():
    sample = Sample({"a": 1})
    assert sample.dict() == {"a": 1}
    assert not hasattr(sample, "wrapped"), "Wrapped attribute should not be present for dict initialization"


def test_from_dict():
    d = {"key1": 1, "key2": [2, 3, 4], "key3": {"nested_key": 5}}
    sample = Sample(**d)
    assert sample.key1 == 1
    assert np.array_equal(sample.key2, [2, 3, 4])
    # Assuming key3 is a dict, not a Sample
    assert sample.key3["nested_key"] == 5


def test_from_space():
    space = spaces.Dict(
        {
            "key1": spaces.Discrete(3),
            "key2": spaces.Box(low=0, high=1, shape=(2,)),
            "key3": spaces.Dict({"nested_key": spaces.Discrete(2)}),
        }
    )
    sample = Sample.from_space(space)
    assert isinstance(sample.key1, int | np.int64)
    assert isinstance(sample.key2, np.ndarray)
    assert (sample.key2 >= 0).all() and (sample.key2 <= 1).all()
    assert isinstance(sample.key3["nested_key"], int | np.int64)


def test_to_dict():
    sample = Sample(key1=1, key2=[2, 3, 4], key3={"nested_key": 5})
    d = sample.dump()  # Adjusted to use the .dump() method
    assert d == {"key1": 1, "key2": [2, 3, 4], "key3": {"nested_key": 5}}


def test_serialize_nonstandard_types():
    data = {"array": np.array([1, 2, 3])}
    sample = Sample(**data)
    assert np.array_equal(sample.array, [1, 2, 3]), "Numpy arrays should work fine"


def test_structured_flatten():
    nested_data = {"key1": 1, "key2": [2, 3, 4], "key3": {"nested_key": 5}}
    sample = Sample(**nested_data)
    flat_list = sample.flatten("list")
    assert flat_list == [1, 2, 3, 4, 5], "Flat list should contain all values from the structure"


def test_pack_as_dict():
    # Scenario with asdict=True
    data = {"key1": [1, 2, 3], "key2": ["a", "b", "c"], "key3": [[1, 2], [3, 4], [5, 6]]}
    sample = Sample(**data)
    packed = sample.pack("dicts")

    expected = [
        {"key1": 1, "key2": "a", "key3": [1, 2]},
        {"key1": 2, "key2": "b", "key3": [3, 4]},
        {"key1": 3, "key2": "c", "key3": [5, 6]},
    ]

    assert packed == expected, "Unrolled wrappeds as dict do not match expected values"


def test_pack_as_sample_instances():
    # Scenario with asdict=False
    data = {"key1": [4, 5, 6], "key2": ["d", "e", "f"], "key3": [[7, 8], [9, 10], [11, 12]]}
    sample = Sample(**data)
    pack = sample.pack("samples")

    # Validate each unrolled wrapped
    for idx, wrapped in enumerate(pack):
        assert wrapped.key1 == data["key1"][idx], f"Unrolled wrapped key1 does not match expected value for index {idx}"
        assert wrapped.key2 == data["key2"][idx], f"Unrolled wrapped key2 does not match expected value for index {idx}"
        assert wrapped.key3 == data["key3"][idx], f"Unrolled wrapped key3 does not match expected value for index {idx}"


@pytest.fixture
def sample_instance():
    # Fixture to create a sample instance with a variety of attribute types
    return Sample(
        int_attr=5,
        float_attr=3.14,
        list_attr=[1, 2, 3],
        dict_attr={"nested": 10},
        numpy_attr=np.array([1, 2, 3]),
        list_of_dicts_attr=[{"a": 1}, {"a": 2}],
        list_of_samples_attr=[Sample(x=1), Sample(x=2)],
        sample_attr=Sample(x=10),
    )


@pytest.fixture
def tmp_path():
    try:
        # Fixture to create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    finally:
        if os.path.exists(tmpdir):
            os.rmdir(tmpdir)


def test_space_for_list_attribute(sample_instance: Sample):
    space = sample_instance.space()
    assert isinstance(space.spaces["list_attr"], spaces.Box), "List attribute should correspond to a Box space"


def test_space_for_dict_attribute(sample_instance: Sample):
    space = sample_instance.space()
    assert isinstance(space.spaces["dict_attr"], spaces.Dict), "Dict attribute should correspond to a Dict space"


def test_space(sample_instance: Sample):
    space = sample_instance.space()
    expected_keys = {
        "int_attr",
        "float_attr",
        "list_attr",
        "dict_attr",
        "numpy_attr",
        "list_of_dicts_attr",
        "list_of_samples_attr",
        "sample_attr",
    }
    assert set(space.spaces.keys()) == expected_keys, "Space should include all attributes of the sample instance"


def test_serialize_deserialize():
    sample = Sample(key1=1, key2=[2, 3, 4], key3={"nested_key": 5})
    serialized = sample.model_dump_json()
    deserialized = Sample.model_validate_json(serialized)
    assert sample == deserialized, "Deserialized sample should match the original sample"


def test_unflatten_dict():
    sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    schema = sample.schema()
    flat_dict = sample.flatten(to="dict")
    unflattened_sample = Sample.unflatten(flat_dict, schema)
    assert unflattened_sample == sample


def test_unflatten_list():
    sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    flat_list = sample.flatten(to="list")
    unflattened_sample = Sample.unflatten(flat_list, sample.schema())
    assert unflattened_sample.x == 1
    assert unflattened_sample.y == 2
    assert unflattened_sample.z == {"a": 3, "b": 4}
    assert unflattened_sample.extra_field == 5


def test_unflatten_numeric_only():
    class AnotherSample(Sample):
        a: int
        b: str = "default"

    class DerivedSample(Sample):
        x: int
        y: str = "default"
        z: AnotherSample
        another_number: float

    sample = DerivedSample(x=1, y="hello", z=AnotherSample(**{"a": 3, "b": "world"}), another_number=5)

    flat_list = sample.flatten(to="list")
    unflattened_sample = DerivedSample.unflatten(flat_list, sample.schema())
    assert unflattened_sample.x == 1
    assert hasattr(unflattened_sample, "y")
    assert unflattened_sample.z.a == 3
    assert unflattened_sample.z.b == "world"


def test_unflatten_numpy_array():
    sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    flat_array = sample.flatten(to="np")
    unflattened_sample = Sample.unflatten(flat_array, sample.schema())
    assert unflattened_sample.x == 1
    assert unflattened_sample.y == 2
    assert unflattened_sample.z == {"a": 3, "b": 4}
    assert unflattened_sample.extra_field == 5


# def test_unflatten_torch_tensor():
#     sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
#     flat_tensor = sample.flatten(to="pt")
#     print(sample.schema())
#     unflattened_sample = Sample.unflatten(flat_tensor, sample.schema())
#     assert unflattened_sample.x == 1
#     assert unflattened_sample.y == 2
#     assert unflattened_sample.z == {"a": 3, "b": 4}
#     assert unflattened_sample.extra_field == 5


# Sample unit test for the schema method
def test_schema():
    sample_instance = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    schema = sample_instance.schema(include="descriptions")
    expected_schema = {
        "description": "A base model class for serializing, recording, and manipulating arbitray data.",
        "properties": {
            "x": {"type": "integer", "title": "X"},
            "y": {"type": "integer", "title": "Y"},
            "z": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "title": "A",
                    },
                    "b": {"type": "integer", "title": "B"},
                },
                "title": "Z",
            },
            "extra_field": {"type": "integer", "title": "Extra Field"},
        },
        "title": "Sample",
        "type": "object",
    }
    assert schema == expected_schema


def test_dict():
    sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    assert sample.dict() == {"x": 1, "y": 2, "z": {"a": 3, "b": 4}, "extra_field": 5}


def test_dict_shallow():
    sample = Sample(x=1, y=2, z=Sample({"a": 3, "b": 4}), extra_field=5)
    assert sample.dict(depth="shallow") == {"x": 1, "y": 2, "z": Sample({"a": 3, "b": 4}), "extra_field": 5}


# def test_setdefault():
#     sample = Sample(a=1, b={"c": 2, "d": [3, 4]}, e=Sample(f=5))

#     # Test setting default value for an existing attribute
#     assert sample.setdefault("a", 10) == 1, "Existing attribute should not be modified"

#     # Test setting default value for a nested attribute
#     assert sample.setdefault("b.c", 20) == 2, "Existing nested attribute should not be modified"
#     assert sample.setdefault("b.e", 30) == 30, "New nested attribute should be set with default value"

#     # Test setting default value for a non-existing attribute
#     assert sample.setdefault("x", 40) == 40, "New attribute should be set with default value"

#     # Test setting default value for a nested attribute with wildcard
#     assert sample.setdefault("b.d.*", 50) == [3, 4], "Existing nested attribute with wildcard should not be modified"
#     assert sample.setdefault("b.d.2", 60) == 60, "New nested attribute with wildcard should be set with default value"

#     # Test setting default value for a non-existing nested attribute with wildcard
#     assert sample.setdefault("b.f.*", [70]) == [70], "New nested attribute with wildcard should be set with default value"


def test_flatten_with_to_and_to():
    obj = Sample(a=1, b={"c": 2, "d": [3, 4]}, e=Sample(f=5, g={"h": 6, "i": 7}))
    result = obj.flatten(include=["a", "b.c", "e.g.h"])
    expected = [1, 2, 6]
    assert result == expected, f"Expected {expected}, but got {result}"

    result_dict = obj.flatten(include=["a", "b.c", "e.g.h"], to="dicts")
    expected_dict = [{"a": 1, "b.c": 2, "e.g.h": 6}]
    assert result_dict == expected_dict, f"Expected {expected_dict}, but got {result_dict}"

    result_dict = obj.flatten(include=["a", "c", "h"], to="dicts")
    assert result_dict == [{"a": 1, "c": 2, "h": 6}], f"Expected {expected_dict}, but got {result_dict}"


if __name__ == "__main__":
    pytest.main()
