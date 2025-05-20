import numpy as np
import pytest
from embdata.utils.geometry_utils import pose_to_transformation_matrix, rotation_between_two_points, align_vector_with_reference
from embdata.coordinate import Pose, Pose6D, PlanarPose
from embdata.motion import Motion
from embdata.geometry import Affine2D, Affine3D
from embdata.sense.depth import Depth

def test_pose_to_transformation_matrix():
    # Test with Pose object
    pose = Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=0)
    result = pose_to_transformation_matrix([1, 2, 3, 0, 0, 0])
    expected = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    np.testing.assert_array_almost_equal(result, expected)

    # Test with numpy array
    pose_array = np.array([1, 2, 3, 0, 0, 0])
    result = pose_to_transformation_matrix(pose_array)
    np.testing.assert_array_almost_equal(result, expected)


def test_transfrom2D():
    # Test with Pose object
    pose = np.array([1, 2, 0])
    result = Affine2D.from_pose(pose)
    expected = Affine2D(rotation=np.array([[1, 0], [0, 1]]), translation=np.array([1, 2]))
    assert np.allclose(result.rotation, expected.rotation)


def test_tranform3d():
    # Test with Pose object
    pose = np.array([1, 2, 3, np.pi/2, np.pi/2, np.pi/2])
    result = Affine3D.from_pose(pose)
    expected = Affine3D(rotation=np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), translation=np.array([1, 2, 3]))
    print(f"Result: {result}")
    assert np.allclose(result.rotation, expected.rotation)


def test_align_same_direction():
    vector = np.array([1, 0, 0])
    reference = np.array([1, 0, 0])
    result = align_vector_with_reference(vector, reference)
    assert np.allclose(result, vector), "Vector should not be negated when aligned."

def test_align_opposite_direction():
    vector = np.array([-1, 0, 0])
    reference = np.array([1, 0, 0])
    result = align_vector_with_reference(vector, reference)
    assert np.allclose(result, -vector), "Vector should be negated when opposite to reference."

def test_align_at_90_degrees():
    vector = np.array([0, 1, 0])
    reference = np.array([1, 0, 0])
    result = align_vector_with_reference(vector, reference)
    # Both vector and -vector are at 90 degrees; function should return the original vector
    assert np.allclose(result, vector), "Vector should not be negated when at 90 degrees."

def test_align_with_zero_vector():
    vector = np.array([0, 0, 0])
    reference = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        align_vector_with_reference(vector, reference)

def test_align_with_zero_reference():
    vector = np.array([1, 0, 0])
    reference = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        align_vector_with_reference(vector, reference)

def test_align_arbitrary_vectors():
    vector = np.array([1, 2, 3])
    reference = np.array([4, 5, 6])
    result = align_vector_with_reference(vector, reference)
    # Compute angles
    angle = np.arccos(np.clip(np.dot(result / np.linalg.norm(result), reference / np.linalg.norm(reference)), -1.0, 1.0))
    angle_neg = np.arccos(np.clip(np.dot(-result / np.linalg.norm(result), reference / np.linalg.norm(reference)), -1.0, 1.0))
    assert angle <= angle_neg + 1e-6, "Aligned vector should have smaller or equal angle with reference."

def test_align_negative_better_alignment():
    vector = np.array([1, 1, 0])
    reference = np.array([-1, -1, 0])
    result = align_vector_with_reference(vector, reference)
    assert np.allclose(result, -vector), "Vector should be negated for better alignment."

def test_align_non_normalized_vectors():
    vector = np.array([10, 0, 0])
    reference = np.array([5, 0, 0])
    result = align_vector_with_reference(vector, reference)
    assert np.allclose(result, vector), "Vector should not be negated even if not normalized."

def test_align_large_vectors():
    vector = np.array([1e6, 0, 0])
    reference = np.array([1e6, 0, 0])
    result = align_vector_with_reference(vector, reference)
    assert np.allclose(result, vector), "Function should handle large magnitude vectors."

def test_align_small_vectors():
    vector = np.array([1e-6, 0, 0])
    reference = np.array([1e-6, 0, 0])
    result = align_vector_with_reference(vector, reference)
    assert np.allclose(result, vector), "Function should handle small magnitude vectors."



# def test_canonicalize_world():
#     url = "https://raw.githubusercontent.com/mbodiai/embodied-agents/main/resources/depth_image.png?raw=true"
#     plane_coeffs =  Depth(url).segment()

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
