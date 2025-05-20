from typing import Any, Type, TypeAlias

import numpy as np
import pytest
from pydantic import ConfigDict, ValidationError
from pydantic_core import PydanticUndefined
from pydantic import BaseModel
from pydantic.fields import Field
from typing_extensions import Annotated, Dict, Unpack

from embdata.coordinate import CoordinateField
from embdata.ndarray import ndarray
from embdata.sample import Sample


class ModelWithArrays(BaseModel):
    any_array: ndarray[Any]
    flexible_array: ndarray = Field(default_factory=lambda: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    int_vector: ndarray[3, int] = Field(default_factory=lambda: np.array([1, 2, 3]))
    float_matrix: ndarray[2, 2, np.float64] = Field(default_factory=lambda: np.array([[1.0, 2.0], [3.0, 4.0]]))
    any_3d_array: ndarray["*", "*", "*", Any] # type: ignore
    any_float_array: ndarray[float] = Field(description="Any float array")
    array: ndarray = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))
    rotation_matrix: ndarray[3, 3, float] = Field(
        default_factory=lambda: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )


@pytest.fixture()
def nested_model() -> Sample:
    class OtherModel(Sample):
        name: str = "OtherModel"
        any_array: ndarray[Any, float] = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))
        # coordinate: ndarray = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))

    class NestedSample(Sample):
        any_array: ndarray[Any] = Field(default_factory=lambda: np.array([1, 2, 3, 4]))
        array: ndarray = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))
        model: ModelWithArrays
        flexible_array: ndarray = Field(default_factory=lambda: np.array([[1, 2], [3, 4]]))
        int_vector: ndarray[3, int]
        float_matrix: ndarray[2, 2, np.float64] = Field(default_factory=lambda: np.array([[1.0, 2.0], [3.0, 4.0]]))
        any_3d_array: ndarray[Any, Any, Any, Any] 
        any_float_array: ndarray[float] = Field(description="Any float array")
        rotation_matrix: ndarray[3, 3, float] = Field(
            default_factory=lambda: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        coordinate: ndarray[Any, float] = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))
        nested: OtherModel
    
    return NestedSample(
        array=np.array([[1.0, 2.0], [3.0, 4.0]]),
        model=ModelWithArrays(
            any_array=np.array([1, 2, 3, 4]),
            flexible_array=np.array([[1, 2], [3, 4]]),
            int_vector=np.array([1, 2, 3]),
            any_3d_array=np.zeros((2, 3, 4)),
            any_float_array=np.array([1.0, 2.0, 3.0]),
        ),
        coordinate=np.array([1.0, 2.0, 3.0]),
        any_array=np.array([1, 2, 3, 4]),
        flexible_array=np.array([[1, 2], [3, 4]]),
        int_vector=np.array([1, 2, 3]),
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
        any_3d_array=np.zeros((2, 3, 4)),
        any_float_array=np.array([1.0, 2.0, 3.0]),
        rotation_matrix=np.eye(3),
        nested=OtherModel(
            name="OtherModel",
            any_array=np.array([1, 2, 3, 4]),
            # flexible_array=np.array([[1, 2], [3, 4]]),
            # int_vector=np.array([1, 2, 3]),
            # any_3d_array=np.zeros((2, 3, 4)),
            # any_float_array=np.array([1.0, 2.0, 3.0]),
        ),
    )


def assert_models_equal(model1: ModelWithArrays, model2: ModelWithArrays):
    assert np.array_equal(model1.any_array, model2.any_array)
    assert np.array_equal(model1.flexible_array, model2.flexible_array)
    assert np.array_equal(model1.int_vector, model2.int_vector)
    assert np.array_equal(model1.float_matrix, model2.float_matrix)
    assert np.array_equal(model1.any_3d_array, model2.any_3d_array)
    assert np.array_equal(model1.any_float_array, model2.any_float_array)


def test_model_with_arrays():
    model = ModelWithArrays(
        any_array=np.array([1, 2, 3, 4]),
        flexible_array=np.array([[1, 2], [3, 4]]),
        int_vector=np.array([1, 2, 3]),
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
        any_3d_array=np.zeros((2, 3, 4)),
        any_float_array=np.array([1.0, 2.0, 3.0]),
        array=np.array([1.0, 2.0, 3.0]),
        rotation_matrix=np.eye(3),
    )

    assert isinstance(model.any_array, np.ndarray)
    assert isinstance(model.flexible_array, np.ndarray)
    assert isinstance(model.int_vector, np.ndarray)
    assert isinstance(model.float_matrix, np.ndarray)
    assert isinstance(model.any_3d_array, np.ndarray)
    assert isinstance(model.any_float_array, np.ndarray)
    assert isinstance(model.array, np.ndarray)
    assert isinstance(model.rotation_matrix, np.ndarray)


def test_serialization_deserialization_nested(nested_model):
    # Test serialization with model_dump
    serialized = nested_model.model_dump()
    assert isinstance(serialized, Dict)

    # Test serialization with model_dump_json
    serialized_json = nested_model.model.model_dump_json()
    assert isinstance(serialized_json, str)
    # Test serialization with model_dump_json
    serialized_json = nested_model.model_dump_json()
    assert isinstance(serialized_json, str)

    # Test deserialization with model_validate
    deserialized = nested_model.model_validate(serialized)
    assert isinstance(deserialized, type(nested_model))

    # Test deserialization with model_validate_json
    deserialized_json = nested_model.model_validate_json(serialized_json)
    assert isinstance(deserialized_json, type(nested_model))

    # Compare original and deserialized models
    assert_models_equal(nested_model, deserialized)
    assert_models_equal(nested_model, deserialized_json)


def test_serialization_deserialization():
    model = ModelWithArrays(
        any_array=np.array([1, 2, 3, 4]),
        flexible_array=np.array([[1, 2], [3, 4]]),
        int_vector=np.array([1, 2, 3]),
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
        any_3d_array=np.zeros((2, 3, 4)),
        any_float_array=np.array([1.0, 2.0, 3.0]),
        array=np.array([1.0, 2.0, 3.0]),
        rotation_matrix=np.eye(3),
    )

    # Test serialization with model_dump
    serialized = model.model_dump()
    assert isinstance(serialized, dict)

    # Test serialization with model_dump_json
    serialized_json = model.model_dump_json()
    assert isinstance(serialized_json, str)

    # Test deserialization with model_validate
    deserialized = ModelWithArrays.model_validate(serialized)
    assert isinstance(deserialized, ModelWithArrays)

    # Test deserialization with model_validate_json
    deserialized_json = ModelWithArrays.model_validate_json(serialized_json)
    assert isinstance(deserialized_json, ModelWithArrays)

    # Compare original and deserialized models
    assert_models_equal(model, deserialized)
    assert_models_equal(model, deserialized_json)


def test_validation_errors():
    with pytest.raises(ValidationError):
        ModelWithArrays(
            any_array=np.array([1, 2, 3, 4]),  # This is fine
            flexible_array=np.array([1, 2, 3, 4]),  # This is fine
            int_vector=np.array([1, 2]),  # Wrong shape
            float_matrix=np.array([[1, 2], [3, 4]]),  # Wrong dtype
            any_3d_array=np.zeros((2, 3)),  # Wrong number of dimensions
            any_float_array=np.array([1, 2, 3]),  # Wrong dtype
        )


def test_edge_cases():
    # Test with empty arrays
    model = ModelWithArrays(
        any_array=np.array([]),
        flexible_array=np.array([[]]),
        int_vector=np.array([0, 0, 0]),
        float_matrix=np.array([[0.0, 0.0], [0.0, 0.0]]),
        any_3d_array=np.array([[[]]]),
        any_float_array=np.array([]),
        array=np.array([]),
    )
    assert model.any_array.size == 0
    assert model.flexible_array.size == 0
    assert np.all(model.int_vector == 0)
    assert np.all(model.float_matrix == 0.0)
    assert model.any_3d_array.size == 0
    assert model.any_float_array.size == 0

    # Test with extreme values
    model = ModelWithArrays(
        any_array=np.array([np.inf, -np.inf, np.nan], dtype=object),
        flexible_array=np.array([[np.finfo(np.float64).max, np.finfo(np.float64).min]]),
        int_vector=np.array([np.iinfo(np.int64).max, 0, np.iinfo(np.int64).min]),
        float_matrix=np.array([[np.inf, -np.inf], [np.nan, 0.0]]),
        any_3d_array=np.array([[[np.inf, -np.inf, np.nan]]]),
        any_float_array=np.array([np.finfo(np.float64).max, np.finfo(np.float64).min]),
    )
    assert np.any(np.isinf(model.any_array.astype(float)))
    assert np.any(np.isnan(model.any_array.astype(float)))
    assert np.all(model.int_vector == np.array([np.iinfo(np.int64).max, 0, np.iinfo(np.int64).min]))
    assert np.isinf(model.float_matrix).any()
    assert np.isnan(model.float_matrix).any()


def test_type_conversion():
    # Test passing lists instead of numpy arrays
    model = ModelWithArrays(
        any_array=[1, 2, 3, 4],
        flexible_array=[[1, 2], [3, 4]],
        int_vector=[1, 2, 3],
        float_matrix=[[1.0, 2.0], [3.0, 4.0]],
        any_3d_array=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        any_float_array=[1.0, 2.0, 3.0],
    )
    assert isinstance(model.any_array, np.ndarray)
    assert isinstance(model.flexible_array, np.ndarray)
    assert isinstance(model.int_vector, np.ndarray)
    assert isinstance(model.float_matrix, np.ndarray)
    assert isinstance(model.any_3d_array, np.ndarray)
    assert isinstance(model.any_float_array, np.ndarray)


def test_wrong_shape():
    class TestModelWithArrays(BaseModel):
        any_array: ndarray[Any]
        flexible_array: ndarray[Any, Any, Any]
        int_vector: ndarray[3, int]
        float_matrix: ndarray[2, 2, np.float64]
        any_3d_array: ndarray[Any, Any, Any, Any]
        any_float_array: ndarray[float]
        array: ndarray = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))

    with pytest.raises((ValidationError, TypeError)):
        model = TestModelWithArrays(
            any_array=[1, 2, 3, 4],
            flexible_array=[[1, 2], [3, 4]],
            int_vector=[1, 2, 3],
            float_matrix=[[1.0, 2.0], [3.0, 4.0]],
            any_3d_array=[[1, 2], [3, 4]],  # This should raise an error as it's 2D, not 3D
            any_float_array=[1.0, 2.0, 3.0],
        )

    # Test that correct shapes pass validation
    model = TestModelWithArrays(
        any_array=[1, 2, 3, 4],
        flexible_array=[[1, 2], [3, 4]],
        int_vector=[1, 2, 3],
        float_matrix=[[1.0, 2.0], [3.0, 4.0]],
        any_3d_array=[[[1, 2], [3, 4]]],  # This is now 3D
        any_float_array=[1.0, 2.0, 3.0],
    )
    assert isinstance(model.any_3d_array, np.ndarray)
    assert model.any_3d_array.ndim == 3


def test_specific_validation_errors():
    with pytest.raises(ValidationError):
        model = ModelWithArrays(
            any_array=[1, 2, 3, 4],
            flexible_array=[[1, 2], [3, 4]],
            int_vector=[1, 2],  # Wrong shape
            float_matrix=[[1.0, 2.0], [3.0, 4.0]],
            any_3d_array=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            any_float_array=[1.0, 2.0, 3.0],
        )

# Helper function to compare custom ndarray with NumPy ndarray
def assert_ndarray_equal(custom: ndarray, numpy_arr: np.ndarray):
    assert custom.shape == numpy_arr.shape, f"Shapes do not match: {custom.shape} vs {numpy_arr.shape}"
    np.testing.assert_allclose(custom, numpy_arr, rtol=1e-5, atol=1e-8)

# Test Addition
def test_addition_same_shape():
    a_data = np.random.rand(10, 5).astype(np.float32)
    b_data = np.random.rand(10, 5).astype(np.float32)
    a = ndarray[10, 5, np.float32](a_data)
    b = ndarray[10, 5, np.float32](b_data)
    result = a + b
    expected = a_data + b_data
    assert_ndarray_equal(result, expected)

def test_addition_broadcast_shape():
    a_data = np.random.rand(10, 5).astype(np.float32)
    b_data = np.random.rand(5).astype(np.float32)  # Broadcastable shape
    a = ndarray[10, 5, np.float32](a_data)
    b = ndarray[5, np.float32](b_data)
    result = a + b
    expected = a_data + b_data
    assert_ndarray_equal(result, expected)

# Test Subtraction
def test_subtraction_same_shape():
    a_data = np.random.rand(8, 4).astype(np.float32)
    b_data = np.random.rand(8, 4).astype(np.float32)
    a = ndarray[8, 4, np.float32](a_data)
    b = ndarray[8, 4, np.float32](b_data)
    result = a - b
    expected = a_data - b_data
    assert_ndarray_equal(result, expected)

def test_subtraction_broadcast_shape():
    a_data = np.random.rand(8, 4).astype(np.float32)
    b_data = np.random.rand(4).astype(np.float32)  # Broadcastable shape
    a = ndarray[8, 4, np.float32](a_data)
    b = ndarray[4, np.float32](b_data)
    result = a - b
    expected = a_data - b_data
    assert_ndarray_equal(result, expected)

# Test Multiplication
def test_multiplication_same_shape():
    a_data = np.random.rand(6, 3).astype(np.float32)
    b_data = np.random.rand(6, 3).astype(np.float32)
    a = ndarray[6, 3, np.float32](a_data)
    b = ndarray[6, 3, np.float32](b_data)
    result = a * b
    expected = a_data * b_data
    assert_ndarray_equal(result, expected)

def test_multiplication_broadcast_shape():
    a_data = np.random.rand(6, 3).astype(np.float32)
    b_data = np.random.rand(3).astype(np.float32)  # Broadcastable shape
    a = ndarray[6, 3, np.float32](a_data)
    b = ndarray[3, np.float32](b_data)
    result = a * b
    expected = a_data * b_data
    assert_ndarray_equal(result, expected)

# Test Division
def test_division_same_shape():
    a_data = np.random.rand(7, 2).astype(np.float32) + 1e-6  # Avoid division by zero
    b_data = np.random.rand(7, 2).astype(np.float32) + 1e-6
    a = ndarray[7, 2, np.float32](a_data)
    b = ndarray[7, 2, np.float32](b_data)
    result = a / b
    expected = a_data / b_data
    assert_ndarray_equal(result, expected)

def test_division_broadcast_shape():
    a_data = np.random.rand(7, 2).astype(np.float32) + 1e-6
    b_data = np.random.rand(2).astype(np.float32) + 1e-6  # Broadcastable shape
    a = ndarray[7, 2, np.float32](a_data)
    b = ndarray[2, np.float32](b_data)
    result = a / b
    expected = a_data / b_data
    assert_ndarray_equal(result, expected)

# Test True Division
def test_truediv_same_shape():
    a_data = np.random.rand(5, 5).astype(np.float32) + 1e-6
    b_data = np.random.rand(5, 5).astype(np.float32) + 1e-6
    a = ndarray[5, 5, np.float32](a_data)
    b = ndarray[5, 5, np.float32](b_data)
    result = a.__truediv__(b)
    expected = a_data / b_data
    assert_ndarray_equal(result, expected)

# Test Floor Division
def test_floordiv_same_shape():
    a_data = np.random.randint(1, 100, size=(4, 4)).astype(np.float32)
    b_data = np.random.randint(1, 100, size=(4, 4)).astype(np.float32)
    a = ndarray[4, 4, np.float32](a_data)
    b = ndarray[4, 4, np.float32](b_data)
    result = a // b
    expected = a_data // b_data
    assert_ndarray_equal(result, expected)

# Test Modulus
def test_modulus_same_shape():
    a_data = np.random.randint(1, 100, size=(3, 3)).astype(np.float32)
    b_data = np.random.randint(1, 100, size=(3, 3)).astype(np.float32)
    a = ndarray[3, 3, np.float32](a_data)
    b = ndarray[3, 3, np.float32](b_data)
    result = a % b
    expected = a_data % b_data
    assert_ndarray_equal(result, expected)

# Test Covariance
def test_cov_single_rowvar_false():
    # data1: 100 observations, 3 variables
    a_data = np.random.rand(100, 3).astype(np.float32)
    a = ndarray[100, 3, np.float32](a_data)
    cov_result = a.cov(rowvar=False)
    expected = np.cov(a_data, rowvar=False)
    assert_ndarray_equal(cov_result, expected)
    assert cov_result.shape == (3, 3)

def test_cov_single_rowvar_true():
    # data2: 3 variables, 100 observations
    b_data = np.random.rand(3, 100).astype(np.float32)
    b = ndarray[3, 100, np.float32](b_data)
    cov_result = b.cov(rowvar=True)
    expected = np.cov(b_data, rowvar=True)
    assert_ndarray_equal(cov_result, expected)
    assert cov_result.shape == (3, 3)

def test_cov_two_inputs_rowvar_true():
    # data1 and data2: 3 variables each, 100 observations
    a_data = np.random.rand(3, 100).astype(np.float32)
    b_data = np.random.rand(3, 100).astype(np.float32)
    a = ndarray[3, 100, np.float32](a_data)
    b = ndarray[3, 100, np.float32](b_data)
    cov_result = a.cov(other=b, rowvar=True)
    expected = np.cov(a_data, b_data, rowvar=True)
    assert_ndarray_equal(cov_result, expected)
    assert cov_result.shape == (6, 6)

def test_cov_two_inputs_rowvar_false():
    # data1 and data2: 100 observations each, 3 variables
    a_data = np.random.rand(100, 3).astype(np.float32)
    b_data = np.random.rand(100, 3).astype(np.float32)
    a = ndarray[100, 3, np.float32](a_data)
    b = ndarray[100, 3, np.float32](b_data)
    cov_result = a.cov(other=b, rowvar=False)
    expected = np.cov(a_data, b_data, rowvar=False)
    assert_ndarray_equal(cov_result, expected)
    assert cov_result.shape == (6, 6)

# Additional Tests: Ensuring Type Preservation
def test_dtype_preservation_addition():
    a_data = np.random.rand(5, 5).astype(np.float32)
    b_data = np.random.rand(5, 5).astype(np.float32)
    a = ndarray[5, 5, np.float32](a_data)
    b = ndarray[5, 5, np.float32](b_data)
    result = a + b
    assert result.dtype == np.float32

def test_dtype_preservation_covariance():
    a_data = np.random.rand(100, 3).astype(np.float64)
    a = ndarray[100, 3, np.float64](a_data)
    cov_result = a.cov(rowvar=False)
    assert cov_result.dtype == np.float32

# Test for Covariance Shape Mismatch (Optional)
def test_cov_shape_mismatch():
    a_data = np.random.rand(100, 3).astype(np.float32)
    b_data = np.random.rand(4, 3).astype(np.float32)  # Different number of variables
    a = ndarray[100, 3, np.float32](a_data)
    b = ndarray[4, 3, np.float32](b_data)
    with pytest.raises(ValueError):
        a.cov(other=b, rowvar=False)


def test_addition_empty_arrays():
    a_data = np.array([], dtype=np.float32).reshape(0, 3)  # Shape: (0, 3)
    b_data = np.array([], dtype=np.float32).reshape(0, 3)  # Shape: (0, 3)
    a = ndarray[0, 3, np.float32](a_data)
    b = ndarray[0, 3, np.float32](b_data)
    result = a + b
    expected = a_data + b_data
    assert_ndarray_equal(result, expected)
    assert result.shape == (0, 3)



if __name__ == "__main__":
    pytest.main(["-vv", __file__])