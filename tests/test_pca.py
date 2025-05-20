import numpy as np
from scipy.spatial.transform import Rotation as R
import pytest
from embdata.geometry.pca import pca, check_handedness, get_intrinsic_orientation

def test_right_handed_frame():
    """Test if the frame is right-handed using both cross product and determinant."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])

    # Check using cross product
    assert check_handedness(v1, v2, v3) == "R"

    # Check using determinant
    matrix = np.stack([v1, v2, v3], axis=0)
    det = np.linalg.det(matrix)
    assert det > 0, f"Determinant is not positive: {det}"


def test_left_handed_frame():
    """Test if the frame is left-handed using both cross product and determinant."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, -1])  # Reversed third vector

    # Check using cross product
    assert check_handedness(v1, v2, v3) == "L"

    # Check using determinant
    matrix = np.stack([v1, v2, v3], axis=0)
    det = np.linalg.det(matrix)
    assert det < 0, f"Determinant should be negative: {det}"

def test_pca():
    """Test PCA to verify alignment of the primary direction with sample distributions."""

    # Test case 1: Points aligned along the X direction
    points_x = np.array([[0, 0, 1], [1, 0, 0.1], [10, 0, 0.01], [100, 0, 0.001]])
    direction_x, _ = pca(points_x, num_components=3)

    expected_x = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    assert np.allclose(direction_x.T, expected_x, atol=1e-2) or \
           np.allclose(direction_x.T, -expected_x, atol=1e-2), \
           f"Failed X direction test. Got: {direction_x}"

    # Test case 2: Points aligned along the Y direction
    points_y = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    direction_y, _ = pca(points_y, num_components=3)

    expected_y = np.array([0, 1, 0])
    assert np.allclose(direction_y[:, 0], expected_y, atol=1e-6) or \
           np.allclose(direction_y[:, 0], -expected_y, atol=1e-6), \
           f"Failed Y direction test. Got: {direction_y[:, 0]}"

    # Test case 3: Points aligned along the Z direction
    points_z = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]])
    direction_z, _ = pca(points_z, num_components=3)

    expected_z = np.array([0, 0, 1])
    assert np.allclose(direction_z[:, 0], expected_z, atol=1e-6) or \
           np.allclose(direction_z[:, 0], -expected_z, atol=1e-6), \
           f"Failed Z direction test. Got: {direction_z[:, 0]}"

    # Test case 4: X direction with small diagonal component
    points_x_diag = np.array([[0, 0, 0], [1, 0.01, 0.01], [2, 0.02, 0.02], [3, 0.03, 0.03]])
    direction_x_diag, _ = pca(points_x_diag, num_components=3)

    expected_x_diag = np.array([1, 0, 0])
    assert np.allclose(direction_x_diag[:, 0], expected_x_diag, atol=1e-1) or \
           np.allclose(direction_x_diag[:, 0], -expected_x_diag, atol=1e-1), \
           f"Failed X direction with diagonal test. Got: {direction_x_diag[:, 0]}"

    # Test case 5: Y direction with small diagonal component
    points_y_diag = np.array([[0, 0, 0], [0.01, 1, 0.01], [0.02, 2, 0.02], [0.03, 3, 0.03]])
    direction_y_diag, _ = pca(points_y_diag, num_components=3)

    expected_y_diag = np.array([0, 1, 0])
    assert np.allclose(direction_y_diag[:, 0], expected_y_diag, atol=1e-1) or \
           np.allclose(direction_y_diag[:, 0], -expected_y_diag, atol=1e-1), \
           f"Failed Y direction with diagonal test. Got: {direction_y_diag[:, 0]}"

    # Test case 6: Z direction with small diagonal component
    points_z_diag = np.array([[0, 0, 0], [0.01, 0.01, 1], [0.02, 0.02, 2], [0.03, 0.03, 3]])
    direction_z_diag, _ = pca(points_z_diag, num_components=3)

    expected_z_diag = np.array([0, 0, 1])
    assert np.allclose(direction_z_diag[:, 0], expected_z_diag, atol=1e-1) or \
           np.allclose(direction_z_diag[:, 0], -expected_z_diag, atol=1e-1), \
           f"Failed Z direction with diagonal test. Got: {direction_z_diag[:, 0]}"

    
def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Calculate the rotation matrix from roll, pitch, and yaw angles."""
    r = R.from_euler('ZYX', [yaw, pitch, roll])
    return r.as_matrix()

def rpy_from_rotation_matrix(rotation_matrix: np.ndarray) -> tuple:
    """Extract roll, pitch, and yaw from a rotation matrix."""
    r = R.from_matrix(rotation_matrix)
    yaw, pitch, roll = r.as_euler('ZYX')
    return roll, pitch, yaw

def normalize_angle(angle: float) -> float:
    """Normalize an angle to be within the range [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def normalize_rpy(rpy: tuple) -> tuple:
    """Normalize a tuple of RPY angles."""
    return tuple(normalize_angle(angle) for angle in rpy)

# Additional edge cases with +90, -90, +180, -180 degrees
@pytest.mark.parametrize("pca_rpy, arbitrary_rotation, expected_rpy", [
    # Edge Case 1: 90-degree pitch
    ((0, np.pi / 2, 0), R.from_euler('z', np.radians(30)).as_matrix(), (0, np.pi / 2, np.radians(30))),
    
    # Edge Case 2: -90-degree pitch
    ((0, -np.pi / 2, 0), R.from_euler('z', np.radians(45)).as_matrix(), (0, -np.pi / 2, np.radians(45))),
    
    # Edge Case 3: 180-degree yaw
    ((0, 0, np.pi), R.from_euler('z', np.radians(30)).as_matrix(), (0, 0, np.pi + np.radians(30))),
    
    # Edge Case 4: -180-degree yaw (should wrap around to positive equivalent if needed)
    ((0, 0, -np.pi), R.from_euler('z', np.radians(60)).as_matrix(), (0, 0, -np.pi + np.radians(60))),
    
    # Edge Case 5: 180-degree roll
    ((np.pi, 0, 0), R.from_euler('z', np.radians(30)).as_matrix(), (np.pi, 0, np.radians(30))),
    
    # Edge Case 6: -180-degree roll
    ((-np.pi, 0, 0), R.from_euler('z', np.radians(45)).as_matrix(), (-np.pi, 0, np.radians(45))),
    
    # Edge Case 7: Combined 90-degree pitch and 180-degree yaw
    ((0, np.pi / 2, np.pi), R.from_euler('z', np.radians(30)).as_matrix(), (0, np.pi / 2, np.pi + np.radians(30))),
    
    # Edge Case 8: Combined -90-degree pitch and -180-degree yaw
    ((0, -np.pi / 2, -np.pi), R.from_euler('z', np.radians(60)).as_matrix(), (0, -np.pi / 2, -np.pi + np.radians(60))),
])
def test_pca_transformation_edge_cases(pca_rpy, arbitrary_rotation, expected_rpy):
    # Generate the initial rotation matrix from PCA RPY values
    initial_rotation_matrix = rotation_matrix_from_rpy(*pca_rpy)

    # Apply the arbitrary rotation to the initial rotation matrix
    transformed_matrix = arbitrary_rotation @ initial_rotation_matrix

    # Extract RPY from the transformed rotation matrix and normalize
    result_rpy = normalize_rpy(rpy_from_rotation_matrix(transformed_matrix))
    expected_rpy = normalize_rpy(expected_rpy)

    # Tolerance for angle comparison
    tolerance = 1e-5

    # Assert each component of RPY with a tolerance
    np.testing.assert_allclose(result_rpy, expected_rpy, atol=tolerance)


# def generate_cylinder(radius, height, axis, num_points=1000):
#     """Generate a point cloud for a cylinder.

#     Args:
#         radius (float): Radius of the cylinder.
#         height (float): Height of the cylinder.
#         axis (str): Axis along which the cylinder is aligned ('x', 'y', or 'z').
#         num_points (int): Number of points to generate.

#     Returns:
#         np.ndarray: Array of shape (num_points, 3) containing the point cloud.
#     """
#     # Generate angles and heights
#     theta = np.random.uniform(0, 2 * np.pi, num_points)
#     h = np.random.uniform(-height / 2, height / 2, num_points)
#     r = np.full(num_points, radius)

#     # Convert to Cartesian coordinates
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#     z = h

#     # Stack into point cloud
#     if axis == 'x':
#         points = np.column_stack((z, y, x))
#     elif axis == 'y':
#         points = np.column_stack((x, z, y))
#     elif axis == 'z':
#         points = np.column_stack((x, y, z))
#     else:
#         raise ValueError("Axis must be 'x', 'y', or 'z'.")

#     return points

# def test_determine_pca_orientation_cylinder():

#     # Parameters for the cylinder
#     radius = 1.0
#     height = 5.0
#     num_points = 1000

#     # Axes to test
#     axes = ['x', 'y', 'z']
#     plane_normal = np.array([0, 0, 1])  # Assuming plane normal along z-axis

#     for axis in axes:
#         # Generate cylinder point cloud
#         points = generate_cylinder(radius, height, axis, num_points)

#         # Perform PCA
#         principal_directions, eigenvalues = pca(points, num_components=3)

#         # Determine PCA orientation
#         R_object = determine_pca_orientation(eigenvalues, principal_directions, plane_normal)

#         # Extract primary axis from R_object
#         v_z = R_object[:, 2]  # The z-axis in the object's coordinate frame

#         # Expected primary axis
#         if axis == 'x':
#             expected_v_z = np.array([1, 0, 0])
#         elif axis == 'y':
#             expected_v_z = np.array([0, 1, 0])
#         elif axis == 'z':
#             expected_v_z = np.array([0, 0, 1])

#         # Check if the primary axis aligns with the expected axis
#         # Since direction can be flipped, check the absolute value of dot product
#         alignment = np.abs(np.dot(v_z, expected_v_z))
#         assert np.isclose(alignment, 1.0, atol=1e-2), f"Alignment failed for axis {axis}"

#         # Print eigenvalues for inspection
#         print(f"Eigenvalues for cylinder aligned along {axis}-axis: {eigenvalues}")

#         # Check eigenvalue ratios
#         lambda1, lambda2, lambda3 = eigenvalues
#         ratio1 = lambda1 / lambda2
#         ratio2 = lambda2 / lambda3
#         print(f"Eigenvalue ratios for axis {axis}: ratio1={ratio1}, ratio2={ratio2}")

#         # For a cylinder, the largest eigenvalue should be significantly larger than the other two
#         assert ratio1 > ratio2, f"Eigenvalue ratio test failed for axis {axis}"


if __name__ == "__main__":
    pytest.main(['-vv', __file__])