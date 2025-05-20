import pytest
import numpy as np
from math import isclose
from embdata.coordinate import Coordinate, Pose6D, CoordinateField, PlanarPose, Pose
from embdata.geometry.affine import Affine3D
import sys

@pytest.fixture
def pose():
    return Pose6D()

def test_coordinate_creation():
    coord = Coordinate()
    assert coord is not None


def test_coordinate_fields():
    coord = PlanarPose()
    assert coord.x == 0.0
    assert coord.y == 0.0
    assert coord.theta == 0.0


def test_coordinate_bounds():
    coord = PlanarPose()
    coord.x = 5.0
    coord.y = 10.0
    coord.theta = 1.57
    assert coord.x == 5.0
    assert coord.y == 10.0
    assert isclose(coord.theta, 1.57, abs_tol=1e-6)


def test_pose6d_fields(pose: Pose6D):
    assert pose.x == 0.0
    assert pose.y == 0.0
    assert pose.z == 0.0
    assert pose.roll == 0.0
    assert pose.pitch == 0.0
    assert pose.yaw == 0.0


def test_pose6d_bounds(pose: Pose6D):
    pose.x = 5.0
    pose.y = 10.0
    pose.z = 2.5
    pose.roll = 0.5
    pose.pitch = 0.3
    pose.yaw = 1.57
    assert pose.x == 5.0
    assert pose.y == 10.0
    assert pose.z == 2.5
    assert pytest.approx(pose.roll, abs=1e-6) == 0.5
    assert pytest.approx(pose.pitch, abs=1e-6) == 0.3
    assert pytest.approx(pose.yaw, abs=1e-6) == 1.57


def test_pose6d_to_conversion():
    pose = Pose6D(x=1, y=2, z=3, roll=np.pi / 4, pitch=np.pi / 3, yaw=np.pi / 2)

    # Test unit conversion
    pose_cm = pose.to(unit="cm")
    assert pose_cm.x == 100
    assert pose_cm.y == 200
    assert pose_cm.z == 300

    # Convert to degrees
    pose_deg = pose.to(angular_unit="deg")
    assert np.allclose(pose_deg.roll, 45.0)
    assert np.allclose(pose_deg.pitch, 60.0)
    assert np.allclose(pose_deg.yaw, 90.0)
    assert pose_deg.x == 1.0  # Linear units should remain unchanged
    assert pose_deg.y == 2.0
    assert pose_deg.z == 3.0

    # Test quaternion conversion
    quat = pose.to("quaternion", sequence="xyz")
    expected_quat = np.array([0.70105738, -0.09229596, 0.56098553, 0.43045933])
    assert np.allclose(quat, expected_quat, atol=1e-6)

    # Test rotation matrix conversion
    rot_matrix = pose.to("rotation_matrix")
    expected_matrix = np.array(
        [[0.0, -0.7071, 0.7071],
         [0.5, 0.6124, 0.6124],
         [-0.8660, 0.3535, 0.3535]]
    )
    assert np.allclose(rot_matrix, expected_matrix, atol=1e-3)


def test_planar_pose_to_conversion():
    pose = PlanarPose(x=1, y=2, theta=np.pi / 2)

    # Test unit conversion
    pose_cm = pose.to(unit="cm")
    assert pose_cm.x == 100
    assert pose_cm.y == 200

    # Test angular unit conversion
    pose_deg = pose.to(angular_unit="deg")
    assert isclose(pose_deg.theta, 90, abs_tol=1e-6)


def test_coordinate_conversion():
    coord = Pose6D(x=1.0, y=2.0, yaw=0.5)

    # Test linear unit conversion
    coord_cm = coord.to(unit="cm")
    assert coord_cm.x == 100.0
    assert coord_cm.y == 200.0
    assert isclose(coord_cm.z, 0.0, abs_tol=1e-6)

    # Test angular unit conversion
    coord_deg = coord.to(angular_unit="deg")
    assert isclose(coord_deg.roll, 0.0, abs_tol=1e-6)


def test_coordinate_array_initialization():
    coord = Pose6D(x=1.0, y=2.0, yaw=0.5)

    # Test array initialization
    coord_array = Pose6D([1.0, 2.0, 0.0, 0.0, 0.0, 0.5])
    assert coord == coord_array


def test_transform_multiplication():
    # Define two sample transformations
    rotation1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation1 = np.array([1, 2, 3])

    rotation2 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    translation2 = np.array([4, 5, 6])

    transform1 = Affine3D(rotation=rotation1, translation=translation1)
    transform2 = Affine3D(rotation=rotation2, translation=translation2)

    # Multiply using the class __mul__ method
    result = transform1 @ transform2
    
    # Multiply using matrix representations
    matrix_multiplication_result = np.dot(transform1.matrix(), transform2.matrix())

    # Check if the results are approximately equal
    assert np.allclose(
        result.matrix(), matrix_multiplication_result
    ), "The results do not match!"


def test_transform_multiplication():
    # Define two sample transformations
    rotation1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation1 = np.array([1, 2, 3])

    rotation2 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    translation2 = np.array([4, 5, 6])

    transform1 = Affine3D(rotation=rotation1, translation=translation1)
    transform2 = Affine3D(rotation=rotation2, translation=translation2)

    # Multiply using the class __mul__ method
    result = transform1 @ transform2
    
    # Multiply using matrix representations
    matrix_multiplication_result = np.dot(transform1.matrix(), transform2.matrix())

    # Check if the results are approximately equal
    assert np.allclose(
        result.matrix(), matrix_multiplication_result
    ), "The results do not match!"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))