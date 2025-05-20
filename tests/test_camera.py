import pytest
import numpy as np
from embdata.sense.camera import Camera, Distortion, Intrinsics, Extrinsics
from embdata.coordinate import PixelCoords
from importlib.resources import files

DEPTH_IMAGE_PATH = files("embdata") / "resources/depth_image.png"

# Define fixtures
@pytest.fixture
def test_camera_init():
    """Fixture to initialize the Camera object."""
    return Camera(
        intrinsic=Intrinsics(fx=911.0, fy=911.0, cx=653.0, cy=371.0),
        distortion=Distortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        extrinsic=Extrinsics(),
        depth_scale=0.001,
    )

@pytest.fixture
def pixel_coords():
    """Fixture for PixelCoords instance."""
    return PixelCoords(u=640, v=480)

@pytest.fixture
def uv_coords():
    """Fixture for 1D numpy array for UV coordinates."""
    return np.array([640, 480])

@pytest.fixture
def uv_coords_2d():
    """Fixture for 2D numpy array for multiple UV coordinates."""
    return np.array([[480, 640], [480, 650], [490, 640], [470, 630]])

@pytest.fixture
def non_zero_depth_image():
    """Fixture for depth image with non-zero depths."""
    depth_image = np.zeros((960, 1280), dtype=np.float32)
    
    # Set non-zero depths in the image
    depth_image[480, 640] = 5.0  # Depth in millimeters
    depth_image[480, 650] = 5.0
    depth_image[490, 640] = 5.0
    depth_image[470, 630] = 5.0

    return depth_image

# Test functions
def test_project_valid_point(test_camera_init):
    """Test valid 3D to 2D projection."""
    camera: Camera = test_camera_init
    xyz = np.array([0.0, 0.0, 5.0])  # Point directly in front of the camera
    uv = camera.project(xyz)
    expected_uv = np.array([653.0, 371.0])  # Should map to the center pixel

    np.testing.assert_array_almost_equal(uv, expected_uv, decimal=5)

def test_deproject_valid_uv_pixelcoords(test_camera_init, pixel_coords):
    """Test deprojection with PixelCoords."""
    camera = test_camera_init
    uv = pixel_coords
    depth_image = np.zeros((960, 1280), dtype=np.float32)
    depth_image[480, 640] = 5.0  # Depth in meters

    xyz = camera.deproject(uv, depth_image)
    # Adjust expected z value to 0.005 (5 mm converted to meters)
    expected_xyz = np.array([0.0, 0.0, 0.005])

    np.testing.assert_allclose(xyz, expected_xyz, rtol=1e-3, atol=1e-3)

def test_deproject_valid_uv_ndarray_1d(test_camera_init, uv_coords, non_zero_depth_image):
    """Test deprojection with 1D UV coordinates as numpy array."""
    camera = test_camera_init
    uv = uv_coords
    depth_image = non_zero_depth_image

    xyz = camera.deproject(uv, depth_image)
    expected_xyz = np.array([0.0, 0.0, 0.005])  # depth_scale * 5.0 mm = 0.005 m

    np.testing.assert_allclose(xyz, expected_xyz, rtol=1e-3, atol=1e-3)

def test_deproject_valid_uv_ndarray_2d(test_camera_init, uv_coords_2d, non_zero_depth_image):
    """Test deprojection with 2D UV coordinates as numpy array."""
    camera = test_camera_init
    uv = uv_coords_2d
    depth_image = non_zero_depth_image

    xyz = camera.deproject(uv, depth_image)
    print(xyz)

    expected_xyz = []
    depth_value = 5.0  # Depth in millimeters
    for coord in uv:
        u, v = coord
        x = (u - camera.intrinsic.cx) * (depth_value * camera.depth_scale) / camera.intrinsic.fx
        y = (v - camera.intrinsic.cy) * (depth_value * camera.depth_scale) / camera.intrinsic.fy
        z = (depth_value * camera.depth_scale)
        expected_xyz.append([x, y, z])
    expected_xyz = np.array(expected_xyz)

    np.testing.assert_allclose(xyz, expected_xyz, rtol=1e-3, atol=1e-3)

def test_deproject_invalid_uv_type(test_camera_init):
    """Test deprojection with invalid UV type."""
    camera = test_camera_init
    uv = [640, 480]  # Invalid type (list instead of ndarray or PixelCoords)
    depth_image = np.zeros((960, 1280), dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        camera.deproject(uv, depth_image)
    assert "Invalid shape for uv coordinates" in str(excinfo.value)

def test_deproject_invalid_uv_shape(test_camera_init):
    """Test deprojection with invalid UV shape (3D array)."""
    camera = test_camera_init
    uv = np.zeros((2, 2, 2))  # Invalid shape (3D array)
    depth_image = np.zeros((960, 1280), dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        camera.deproject(uv, depth_image)
    assert "Invalid shape for uv coordinates" in str(excinfo.value)

def test_deproject_zero_depth(test_camera_init, uv_coords):
    """Test deprojection with zero depth."""
    camera = test_camera_init
    uv = uv_coords
    depth_image = np.zeros((960, 1280), dtype=np.float32)  # Zero depth at all pixels

    xyz = camera.deproject(uv, depth_image)
    expected_xyz = np.array([0.0, 0.0, 0.0])  # Should result in zero vector

    np.testing.assert_array_almost_equal(xyz, expected_xyz, decimal=5)

def test_deproject_negative_depth(test_camera_init, uv_coords_2d):
    """Test deprojection with negative depth value."""
    camera = test_camera_init
    uv = uv_coords_2d
    depth_image = np.zeros((960, 1280), dtype=np.float32)
    depth_image[480, 640] = -5.0  # Negative depth (invalid in practice)
    depth_image[480, 650] = 5.0   # Valid depth

    xyz = camera.deproject(uv, depth_image)

    # Should only include the point with positive depth
    expected_xyz = np.array([[(650 - 653.0) * 0.005 / 911.0, (480 - 371.0) * 0.005 / 911.0, 0.005]])
    np.testing.assert_array_almost_equal(xyz, expected_xyz, decimal=5)

def test_deproject_invalid_uv_ndarray_dimensions(test_camera_init):
    """Test deprojection with invalid UV array dimensions."""
    camera = test_camera_init
    uv = np.array([640, 480, 500])  # Invalid shape (length 3)
    depth_image = np.zeros((960, 1280), dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        camera.deproject(uv, depth_image)

    # Check for expected error message
    assert "Invalid shape for uv coordinates" in str(excinfo.value)
