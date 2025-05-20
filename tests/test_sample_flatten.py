import pytest
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any
from embdata.sample import Sample



@pytest.fixture
def sample_instance():

    return Sample(
        int_value=1,
        float_value=2.0,
        str_value="test",
        list_value=[1, 2, 3],
        dict_value={"a": 11, "b": "two"},
        nested=Sample(value=5),
        np_array=np.array([1, 2, 3]),
        # torch_tensor=np.array([4, 5, 6]),
    )


def test_flatten_default(sample_instance):
    result = sample_instance.flatten()
    assert isinstance(result, list)
    assert result == [1, 2.0, "test", 1, 2, 3, 11, "two", 5, 1, 2, 3]


def test_flatten_to_dict(sample_instance):
    result = sample_instance.flatten(to="dict")
    assert isinstance(result, dict)
    assert "int_value" in result and result["int_value"] == 1
    assert "float_value" in result and result["float_value"] == 2.0


def test_flatten_non_numerical_allow(sample_instance):
    result = sample_instance.flatten(non_numerical="allow")
    assert "test" in result


def test_flatten_non_numerical_forbid(sample_instance):
    with pytest.raises(TypeError):
        sample_instance.flatten(non_numerical="forbid")


def test_flatten_to_numpy(sample_instance):
    result = sample_instance.flatten(to="np")
    assert isinstance(result, np.ndarray)

# @pytest.mark.torch
# def test_flatten_to_torch(sample_instance):
#     import torch
#     result = sample_instance.flatten(to="pt")
#     assert isinstance(result, torch.Tensor)


def test_flatten_with_ignore(sample_instance):
    result = sample_instance.flatten(exclude={"str_value", "dict_value"}, to="list")
    assert "test" not in result
    assert 11 not in result
    assert result == [1, 2.0, 1, 2, 3, 5, 1, 2, 3]


def test_flatten_nested_structures(sample_instance):
    result = sample_instance.flatten()
    assert np.array_equal(result, [1, 2.0, "test", 1, 2, 3, 11, "two", 5, 1, 2, 3])


def test_flatten_numpy_array(sample_instance):
    result = sample_instance.numpy()
    assert np.array_equal(result, [1, 2.0, 1, 2, 3, 11, 5, 1, 2, 3])

# @pytest.mark.torch
# def test_flatten_torch_tensor(sample_instance):
#     import torch
#     result = sample_instance.torch()
#     assert torch.equal(result, torch.tensor([1, 2.0, 1, 2, 3, 11, 5, 1, 2, 3, 4, 5, 6]))


def test_flatten_to_dict(sample_instance):
    result = sample_instance.flatten(to="dict", sep=".")
    assert isinstance(result, dict)
    assert list(result.values()) == [1, 2.0, "test", 1, 2, 3, 11, "two", 5, 1, 2, 3]
    assert list(result.keys()) == [
        "int_value",
        "float_value",
        "str_value",
        "list_value.0",
        "list_value.1",
        "list_value.2",
        "dict_value.a",
        "dict_value.b",
        "nested.value",
        "np_array.0",
        "np_array.1",
        "np_array.2",
        # "torch_tensor.0",
        # "torch_tensor.1",
        # "torch_tensor.2",
    ]


def test_flatten_to_keys(sample_instance):
    nested_structure = Sample(items=[sample_instance, sample_instance, sample_instance])
    result = nested_structure.flatten(include=["a", "b"], to="dicts")
    assert result == [{"a": 11, "b": "two"}, {"a": 11, "b": "two"}, {"a": 11, "b": "two"}]

    deeper_result = nested_structure.flatten(include="items.*.dict_value.a")
    assert deeper_result == [11, 11, 11]


def test_flatten_nested_dicts_and_lists_output_list():
    sample = Sample(
        a=1, b=[{"c": 2, "d": [3, 4]}, {"c": 5, "d": [6, 7]}], e=Sample(f=8, g=[{"h": 9, "i": 10}, {"h": 11, "i": 12}])
    )
    flattened = sample.flatten(to="list")
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert flattened == expected, f"Expected {expected}, but got {flattened}"

    flattened_dict = sample.flatten(to="dict")
    unflattened_sample = Sample.unflatten(flattened, sample.schema())
    assert unflattened_sample == sample, f"Expected {sample}, but got {unflattened_sample}"

    unflattened_sample_dict = Sample.unflatten(flattened_dict, sample.schema())
    assert unflattened_sample_dict == sample, f"Expected {sample}, but got {unflattened_sample_dict}"
    expected_dict = {
        "a": 1,
        "b.0.c": 2,
        "b.0.d.0": 3,
        "b.0.d.1": 4,
        "b.1.c": 5,
        "b.1.d.0": 6,
        "b.1.d.1": 7,
        "e.f": 8,
        "e.g.0.h": 9,
        "e.g.0.i": 10,
        "e.g.1.h": 11,
        "e.g.1.i": 12,
    }
    assert flattened_dict == expected_dict, f"Expected {expected_dict}, but got {flattened_dict}"

    unflattened_sample = Sample.unflatten(flattened, sample.schema())
    assert unflattened_sample == sample, f"Expected {sample}, but got {unflattened_sample}"

    unflattened_sample_dict = Sample.unflatten(flattened_dict, sample.schema())
    assert unflattened_sample_dict == sample, f"Expected {sample}, but got {unflattened_sample_dict}"


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
