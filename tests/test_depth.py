from importlib.resources import files
from pathlib import Path
import numpy as np
from PIL import Image
from embdata.coordinate import Pose
from embdata.sense.depth import Depth, Plane
from embdata.sense.camera import Camera, Intrinsics, Distortion, Extrinsics
from embdata.sense.image import Image as MBImage
from embdata.utils.geometry_utils import rotation_between_two_points as align_plane_normal_to_axis
import open3d as o3d
import logging
import importlib.util
import pytest
import os
import matplotlib.pyplot as plt
import trimesh

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Check if packages are installed
HAS_OPEN3D = importlib.util.find_spec("open3d") is not None
HAS_TRIMESH = importlib.util.find_spec("trimesh") is not None

camera = Camera(
    intrinsic=Intrinsics(fx=911.0, fy=911.0, cx=653.0, cy=371.0),
    distortion=Distortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
    extrinsic=Extrinsics(),
    depth_scale=0.001,
)


@pytest.fixture
def image_path():
    path = files("embdata") / "resources"
    return path / "color_image.png"


@pytest.fixture
def depth_path():
    path: Path = files("embdata") / "resources"
    return path / "depth_image.png"


def test_depth_initialization():
    depth = Depth(mode="I", points=None, array=None, camera=camera)
    assert depth.mode == "I"
    assert depth.points is None
    assert depth.array is None


def test_depth_from_pil():
    pil_image = Image.new("RGB", (100, 100))
    depth = Depth.from_pil(pil_image, mode="I", camera=camera)
    assert depth.mode == "I"
    assert depth.points is None
    assert depth.array is not None
    assert isinstance(depth.array, np.ndarray)
    assert depth.array.dtype == np.uint16


def test_depth_cluster_points():
    depth = Depth(camera=camera)
    depth.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    labels = depth.cluster_points(n_clusters=2)
    assert len(labels) == 3
    assert set(labels) == {0, 1}


@pytest.mark.skipif(not HAS_OPEN3D, reason="open3d is not installed")
def test_depth_segment_plane(image_path, depth_path):
    
    image_path = MBImage(path=image_path, encoding="png", mode="RGB")
    
    depth = Depth(path=depth_path,
        encoding="png",
        mode="I",
        size=(1280, 720),
        camera=camera,
        unit="mm",
        rgb=image_path,
    )
    
    plane = depth.segment_plane()
    assert plane.coefficients is not None


def test_depth_show(depth_path):
    depth = Depth(path=depth_path, size=(100, 100), camera=camera)
    # depth.array = np.zeros((100, 100, 3), dtype=np.uint8)
    # print("depth.array:", depth.array)
    depth.show()  # Just checking if the function runs without errors


def test_depth_segment_cylinder():
    depth = Depth(camera=camera)
    depth.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inlier_points, inlier_indices = depth.segment_cylinder()

    # Ensure that inlier_points and inlier_indices have the correct shapes
    assert len(inlier_points.shape) == 2
    assert inlier_points.shape[1] == 3  # Each point should have 3 coordinates

    assert len(inlier_indices.shape) == 1
    assert inlier_indices.shape[0] == inlier_points.shape[0]  # Number of inliers should matchtests\test_depth.py


def test_rgb(depth_path, image_path):
    depth = Depth(
        path=depth_path,
        mode="I",
        encoding="png",
        size=(1280, 720),
        camera=camera,
        rgb=MBImage(path=image_path, mode="RGB", encoding="png"),
    )

    print("depth.rgb:", depth.rgb.mode)
    assert depth.rgb is not None
    assert depth.rgb.path is not None
    assert depth.rgb.mode == "RGB"
    assert depth.rgb.encoding == "png"


def test_load_from_path(depth_path):
    depth = Depth(path=depth_path, mode="I", encoding="png", camera=camera)
    assert depth.mode == "I"
    assert depth.points is None
    assert depth.array is not None
    assert isinstance(depth.array, np.ndarray)


def test_depth_pil_computed_field():
    depth_array = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    depth = Depth(array=depth_array)
    pil_image = depth.pil

    assert pil_image.size == (100, 100)


def test_depth_rgb_computed_field():
    depth_array = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    depth = Depth(array=depth_array)
    rgb_image = depth.rgb

    assert rgb_image.mode == "RGB"
    assert isinstance(rgb_image.array, np.ndarray)
    assert rgb_image.array.shape == (100, 100, 3)  # Check for RGB shape


def test_depth_base64_computed_field():
    depth_array = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    depth = Depth(array=depth_array)
    base64_str = depth.base64

    assert isinstance(base64_str, str)


def test_depth_url_computed_field():
    depth_array = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    depth = Depth(array=depth_array)
    url_str = depth.url

    assert isinstance(url_str, str)
    assert url_str.startswith("data:image/png;base64,")


@pytest.mark.network
def test_fastapi(depth_path):
    from fastapi import FastAPI
    from httpx import Client
    from embdata.utils.network_utils import get_open_port
    from time import sleep
    import uvicorn

    app = FastAPI()

    @app.post("/test")
    async def test(d: dict) -> dict:
        depth = Depth(**d)
        return depth.model_dump(mode="json")

    port = get_open_port()

    import threading

    thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "localhost", "port": port}, daemon=True)
    thread.start()
    sleep(5)
    client = Client()
    depth = Depth(path=str(depth_path))
    response = client.post(f"http://localhost:{port}/test", json=depth.model_dump(mode="json"))
    assert response.status_code == 200
    resp_depth = Depth.model_validate(response.json())
    assert np.allclose(depth.array, resp_depth.array)
    thread.join(timeout=10)  # Add a timeout to prevent hanging
    print("Test completed")


test_intrinsics = Intrinsics(
    fx=615.0, fy=615.0, cx=320.0, cy=240.0
)
# Assume depth image values are in mm, so scale is 1000 to get meters
test_camera = Camera(intrinsic=test_intrinsics, depth_scale=1000.0)


# Add the new test function
def test_to_pointcloud_backend_consistency(image_path, depth_path):
    """
    Tests if the 'open3d' and 'numpy' backends for to_pointcloud
    produce numerically close point coordinates.
    """
    # Skip test if open3d is not installed
    pytest.importorskip("open3d")
    import open3d as o3d # Import locally after skip check

    logging.info("Starting test_to_pointcloud_backend_consistency...")

    # --- Setup ---
    logging.info(f"Using image: {image_path}, depth: {depth_path}")
    img_pil = Image.open(image_path)
    width, height = img_pil.size
    logging.info(f"Image dimensions: width={width}, height={height}")

    depth_instance = Depth(
        path=depth_path,
        rgb=MBImage(path=image_path),
        camera=test_camera, # Use the camera with realistic intrinsics
        size=(width, height), # Ensure size is set correctly
        unit="mm" # Explicitly state unit, matching depth_scale assumption
    )

    # Ensure array and rgb are loaded
    assert depth_instance.array is not None, "Depth array failed to load"
    assert depth_instance.rgb is not None and depth_instance.rgb.array is not None, "RGB image failed to load"
    assert depth_instance.camera is not None and depth_instance.camera.intrinsic is not None, "Camera intrinsics missing"
    logging.info("Depth and RGB data loaded successfully.")

    # --- Generate Point Clouds ---
    logging.info("Generating point cloud with Open3D backend...")
    pcd_o3d = depth_instance.to_pointcloud(backend="open3d")
    points_o3d = np.asarray(pcd_o3d.points) # Convert O3D points to NumPy array
    logging.info(f"Open3D backend generated {points_o3d.shape[0]} points. Shape: {points_o3d.shape}")
    if points_o3d.shape[0] > 0:
        logging.info(f"First 5 Open3D points:\n{points_o3d[:5]}")
    else:
        logging.warning("Open3D backend generated 0 points.")


    logging.info("Generating point cloud with NumPy backend...")
    result_np = depth_instance.to_pointcloud(backend="numpy")
    assert isinstance(result_np, tuple) and len(result_np) == 2, "NumPy backend should return a tuple (points, colors)"
    points_np, colors_np = result_np
    logging.info(f"NumPy backend generated {points_np.shape[0]} points. Shape: {points_np.shape}")
    if points_np.shape[0] > 0:
        logging.info(f"First 5 NumPy points:\n{points_np[:5]}")
        # Optionally log colors too
        # logging.info(f"First 5 NumPy colors:\n{colors_np[:5]}")
    else:
        logging.warning("NumPy backend generated 0 points.")


    # --- Comparison ---
    # 1. Check if the number of points is the same (basic sanity check)
    num_points_o3d = points_o3d.shape[0]
    num_points_np = points_np.shape[0]
    logging.info(f"Comparing point counts: Open3D={num_points_o3d}, NumPy={num_points_np}")
    # Allow for minor differences due to potential edge case filtering variations
    assert abs(num_points_o3d - num_points_np) < 10, f"Number of points differs significantly between backends ({num_points_o3d} vs {num_points_np})"

    # 2. Check if the point coordinates are numerically close
    if num_points_o3d == 0 and num_points_np == 0:
        logging.info("Both backends produced 0 points. Skipping coordinate comparison.")
        comparison_result = True # Treat as passing if both are empty
    elif num_points_o3d == num_points_np:
        logging.info("Point counts match exactly. Comparing all coordinates...")
        comparison_result = np.allclose(points_o3d, points_np, atol=1e-5)
        assert comparison_result, "Point coordinates differ between Open3D and NumPy backends"
        logging.info("Point coordinates match closely.")
    else:
        logging.warning("Number of points differs slightly. Comparing common subset...")
        min_points = min(num_points_o3d, num_points_np)
        comparison_result = np.allclose(points_o3d[:min_points], points_np[:min_points], atol=1e-5)
        assert comparison_result, \
            "Point coordinates differ between Open3D and NumPy backends (comparing common subset)"
        logging.info("Point coordinates match closely for the common subset of points.")

    logging.info(f"Backend consistency check passed: {comparison_result}")

@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh is not installed")
@pytest.mark.skipif(not HAS_OPEN3D, reason="open3d is not installed")
def test_segment_plane_trimesh_backend(image_path, depth_path):
    """
    Tests the segment_plane method with the 'trimesh' backend.
    """
    logging.info("Starting test_segment_plane_trimesh_backend...")

    # --- Setup ---
    img_pil = Image.open(image_path)
    width, height = img_pil.size

    depth_instance = Depth(
        path=depth_path,
        rgb=MBImage(path=image_path),
        camera=test_camera, # Use the camera with realistic intrinsics
        size=(width, height),
        unit="mm" # Explicitly state unit consistent with test_camera depth_scale
    )

    # Ensure data is loaded
    assert depth_instance.array is not None, "Depth array failed to load"
    assert depth_instance.rgb is not None and depth_instance.rgb.array is not None, "RGB image failed to load"
    assert depth_instance.camera is not None, "Camera is missing"
    logging.info("Depth, RGB, and Camera loaded successfully.")

    # --- Call segment_plane with trimesh backend ---
    logging.info("Calling segment_plane with backend='trimesh'...")
    plane_result = depth_instance.segment_plane(
        plane_backend="trimesh",
        camera=test_camera # Pass camera explicitly if needed, though it uses self.camera
    )

    # --- Assertions ---
    assert plane_result is not None, "segment_plane('trimesh') returned None, expected a Plane object."
    logging.info(f"segment_plane returned: {type(plane_result)}")

    assert isinstance(plane_result, Plane), \
        f"Expected result to be an instance of Plane, but got {type(plane_result)}"

    # Check coefficients
    assert hasattr(plane_result, 'coefficients'), "Plane object missing 'coefficients' attribute."
    assert isinstance(plane_result.coefficients, np.ndarray), \
        f"Expected coefficients to be ndarray, got {type(plane_result.coefficients)}"
    assert len(plane_result.coefficients) == 4, \
        f"Expected coefficients to have 4 elements, got {len(plane_result.coefficients)}"
    logging.info(f"Plane coefficients (a,b,c,d): {plane_result}")

    # Access coefficients directly from the Plane object
    assert isinstance(plane_result.a, float), "Coefficient 'a' is not a float"
    assert isinstance(plane_result.b, float), "Coefficient 'b' is not a float"
    assert isinstance(plane_result.c, float), "Coefficient 'c' is not a float"
    assert isinstance(plane_result.d, float), "Coefficient 'd' is not a float"

    # Check normal vector magnitude (should be close to 1)
    normal_magnitude = np.sqrt(plane_result.a**2 + plane_result.b**2 + plane_result.c**2)
    assert np.isclose(normal_magnitude, 1.0, atol=1e-6), \
        f"Plane normal vector [a, b, c] magnitude is not close to 1 ({normal_magnitude})"
    logging.info(f"Plane normal magnitude: {normal_magnitude}")

    # Check inliers (should be None for trimesh)
    assert hasattr(plane_result, 'inliers'), "Plane object missing 'inliers' attribute."
    assert plane_result.inliers is None, \
        f"Expected inliers to be None for trimesh backend, but got {type(plane_result.inliers)}"
    logging.info("Inliers attribute is None as expected for trimesh.")

    logging.info("test_segment_plane_trimesh_backend passed.")


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-s"]) # Run all tests
