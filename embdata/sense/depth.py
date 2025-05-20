# Copyright 2024 mbodi ai
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
"""Wrap any common image representation in an Image class to convert to any other common format.

The following image representations are supported:
- NumPy array
- PIL Image
- Base64 encoded string
- File path
- URL
- Bytes object

The image can be resized to and from any size, compressed, and converted to and from any supported format:

```python
image = Image("path/to/image.png", size=new_size_tuple).save("path/to/new/image.jpg")
image.save("path/to/new/image.jpg", quality=5)

TODO: Implement Lazy attribute loading for the image data.
"""

import logging
from functools import cached_property, reduce, wraps
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, SupportsBytes, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
from PIL.Image import Image as PILImage
from pydantic import (
    AnyUrl,
    Base64Str,
    FilePath,
    PrivateAttr,
    computed_field,
    model_serializer,
    model_validator,
)

from embdata.coordinate import Plane
from embdata.ndarray import ndarray
from embdata.sample import Sample
from embdata.sense.camera import Camera
from embdata.sense.image import Image
from embdata.units import LinearUnit
from embdata.utils.image_utils import dispatch_arg
from embdata.utils.import_utils import smart_import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SupportsImage = Union[np.ndarray, PILImage, Base64Str, AnyUrl, FilePath]  # noqa: UP007

DepthImageLike = ndarray[1, Any, Any, np.uint16] | ndarray[Any, Any, np.uint16]

PointCloudLike = ndarray[Any, 3, np.float32]

# Define constants for depth range (in meters)
MIN_DEPTH = 0.01 # Minimum valid depth in meters
MAX_DEPTH = 3.0  # Maximum valid depth in meters (adjust as needed)

# Define constants for plane fitting
MIN_POINTS = 3 # Minimum points required for plane fitting
MAX_TRIALS = 1000 # Maximum trials for Open3D RANSAC
THRESHOLD = 0.01 # Distance threshold for Open3D RANSAC

class Depth(Sample):
    """A class for representing depth images and points."""

    DEFAULT_MODE: ClassVar[str] = "I"
    SOURCE_TYPES: ClassVar[List[str]] = ["path", "array", "base64", "bytes", "url", "pil", "points"]
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] = DEFAULT_MODE
    points: ndarray[Any, 3, np.float32] | None = None
    encoding: Literal["png"] = "png"
    size: Tuple[int, int] | None = None
    camera: Camera | None = None
    path: str | Path | FilePath | None = None

    _url: AnyUrl | None = PrivateAttr(default=None)
    _array: ndarray[Any, Any, np.uint16] | None = PrivateAttr(default=None)
    _rgb: Image | None = PrivateAttr(default=None)

    def __eq__(self, other: object) -> bool:
        """Check if two depth images are equal."""
        if not isinstance(other, Depth):
            return False
        return np.allclose(self.array, other.array)

    @computed_field(return_type=ndarray[Any, Any, 3, np.uint16])
    @property
    def array(self) -> ndarray[Any, Any, 3, np.uint16]:
        """The raw depth image represented as a NumPy array."""
        return self._array if self._array is not None else None

    @array.setter
    def array(self, value: ndarray[Any, Any, np.uint16] | None) -> ndarray[Any, Any, 3, np.uint16]:
        self._array = value
        if self._array is not None:
            self.size = (self._array.shape[1], self._array.shape[0])
        return self._array if value is not None else None

    @computed_field(return_type=Image)
    @property
    def rgb(self) -> Image:
        """Convert the depth image to an RGB image."""
        if self._rgb is None:
            if self.array is None:
                msg = "The depth array must be set to convert to an RGB image."
                raise ValueError(msg)
            normalized_array = cv2.normalize(self.array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            rgb_image = Image(normalized_array, mode="RGB")
            self._rgb = rgb_image
        return self._rgb

    @rgb.setter
    def rgb(self, value: Image) -> Image:
        if not isinstance(value, Image):
            msg = "rgb must be an instance of Image"
            raise TypeError(msg)
        if self.array is not None and value.size != (self.array.shape[1], self.array.shape[0]):
            msg = "The size of the RGB image must match the size of the depth image."
            raise ValueError(msg)
        self._rgb = value
        return self._rgb

    @classmethod
    def supports_pointcloud(cls, arg: Any) -> bool:
        """Check if the argument is a point cloud."""
        return isinstance(arg, np.ndarray) and arg.ndim == 2 and arg.shape[1] == 3

    @computed_field(return_type=ndarray[Any, 3, np.uint16])
    @cached_property
    def pil(self) -> PILImage:
        """The PIL image object."""
        if self.array is None:
            msg = "The array must be set to convert to a PIL image."
            raise ValueError(msg)
        return Image(self.array, mode="I", encoding="png").pil

    @computed_field(return_type=Base64Str)
    @cached_property
    def base64(self) -> Base64Str:
        """The base64 encoded string of the image."""
        if self.array is None:
            msg = "The array must be set to convert to a base64 string."
            raise ValueError(msg)
        return Image(self.array, mode="I", encoding="png").base64

    @computed_field
    @cached_property
    def url(self) -> str:
        """The URL of the image."""
        if self._url is not None:
            return self._url
        return f"data:image/{self.encoding};base64,{self.base64}"

    def __init__(  # noqa
        self,
        arg: SupportsImage | DepthImageLike | None = None,
        points: DepthImageLike | None = None,
        path: str | Path | FilePath | None = None,
        array: np.ndarray | None = None,
        base64: Base64Str | None = None,
        rgb: Image | None = None,
        camera: Camera | None = None,
        encoding: str = "png",
        size: Tuple[int, ...] | None = None,
        bytes: SupportsBytes | None = None,  # noqa
        unit: LinearUnit | None = None,
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = "I",
    ):
        """Initializes a Depth representation. Unlike the Image class, an empty array is used as the default image.

        Args:
            arg (SupportsImage, optional): The primary image source.
            points (DepthImageLike, optional): The point cloud data.
            path (str | Path | FilePath, optional): The path to the image file.
            array (np.ndarray, optional): The raw image data.
            base64 (Base64Str, optional): The base64 encoded image data.
            rgb (Image, optional): The RGB image.
            encoding (str, optional): The image encoding.
            size (Tuple[int, ...], optional): The image size.
            bytes (SupportsBytes, optional): The raw image bytes.
            unit (LinearUnit, optional): The linear unit of the image.
            mode (Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"], optional): The image mode.
        """
        kwargs = {}
        kwargs["path"] = path
        kwargs["array"] = array
        kwargs["base64"] = base64
        kwargs["bytes"] = bytes
        kwargs["encoding"] = encoding
        kwargs["size"] = size
        kwargs["mode"] = mode
        kwargs["points"] = points
        kwargs["camera"] = camera
        kwargs["unit"] = unit
        if not self.supports_pointcloud(arg):
            points = kwargs.pop("points", None)

            kwargs["mode"] = "I"
            kwargs["encoding"] = "png"
            num_keys = 0

            for (
                k,
                v,
            ) in kwargs.items():
                if v is not None and k in self.SOURCE_TYPES:
                    num_keys += 1
            if num_keys > 1:
                msg = (
                    "Only one of the following arguments can be provided: path, array, base64, bytes, url, pil, points."
                )
                raise ValueError(msg)

            if rgb or (isinstance(arg, Image) and arg.mode == "RGB"):

                rgb = arg if isinstance(arg, Image) else rgb
            elif isinstance(arg, Image | PILImage):
                if arg.mode == "RGB":
                    msg = "The RGB image must be provided as the 'rgb' argument."
                    raise ValueError(msg)
                points = kwargs.pop("points", None)
                rgb = Image(arg, **kwargs)
                kwargs["points"] = points

        super().__init__(**kwargs)
        # Set self.array directly if array is provided as the source
        if array is not None:
            self._array = array
        elif any(kwargs.get(k) is not None for k in self.SOURCE_TYPES):
            # Otherwise, initialize from another source type if provided
            self._array = Image(arg, **kwargs).array
        else:
            self._array = None

        if rgb is not None:
            self.rgb = rgb

        self._url = kwargs.get("url")

    @model_validator(mode="before")
    @classmethod
    def ensure_pil(cls, values: Dict[str, DepthImageLike] | DepthImageLike) -> Dict[str, Any]:
        """Ensure the image is represented as a PIL Image object."""
        sources = ["array", "base64", "path", "url", "bytes"]
        if not isinstance(values, dict):
            values = dispatch_arg(values, encoding="png", mode="I")
        url = values.get("url")
        if values.get("pil") is None:
            arg = reduce(lambda x, y: x if x is not None else y, [values.get(key) for key in sources])
            arg = arg if arg is not None else np.zeros((224, 224), dtype=np.uint16)
            values.update(dispatch_arg(arg, **values))
            if url is not None:
                values["url"] = url
        return {key: value for key, value in values.items() if key is not None}

    @model_serializer(when_used="json")
    def omit_array(self) -> dict:
        """Omit the array when serializing the object."""
        out = {
            "encoding": self.encoding,
            "size": self.size,
            "camera": self.camera,
        }
        if self._url is not None and self.base64 not in self.url:
            out["url"] = self.url
        elif self.base64 is not None:
            out["base64"] = self.base64
        return out

    @classmethod
    def from_pil(cls, pil: PILImage, **kwargs) -> "Depth":
        """Create an image from a PIL image."""
        array = np.array(pil.convert("I"), dtype=np.uint16)
        kwargs.update({"encoding": "png", "size": pil.size, "mode": cls.DEFAULT_MODE})
        return cls(array=array, **kwargs)

    def segment_plane(
        self,
        plane_backend: Literal["open3d", "trimesh"] = "open3d",
        threshold: float = THRESHOLD,
        min_samples: int = MIN_POINTS,
        max_trials: int = MAX_TRIALS,
        camera: Camera | None = None,
    ) -> Plane | None:
        """Segments a plane from the point cloud using the specified backend.

        Args:
            plane_backend (Literal["open3d", "trimesh"], optional): The library to use for plane fitting.
                'open3d': Uses RANSAC via Open3D (returns Plane with inliers).
                'trimesh': Uses SVD via Trimesh (returns Plane, inliers will be None).
                Defaults to "open3d".
            threshold (float, optional): Distance threshold for Open3D RANSAC. Defaults to 0.01.
            min_samples (int, optional): Minimum samples for Open3D RANSAC / minimum points required. Defaults to 3.
            max_trials (int, optional): Maximum trials for Open3D RANSAC. Defaults to 1000.
            camera (Camera, optional): Camera parameters required for point cloud generation. Defaults to self.camera.

        Returns:
            Optional[Plane]: Contains the plane model and potentially inliers (Open3D only),
                             or None if no plane could be fitted or requirements not met.

        Raises:
            ValueError: If camera is missing or an invalid backend is specified.
            ImportError: If the selected backend library (Open3D or Trimesh) is not installed.
        """
        camera = camera or self.camera
        if camera is None:
            msg = "Camera must be provided for plane segmentation."
            raise ValueError(msg)

        plane_model_coeffs: np.ndarray | None = None
        inliers: list | None = None

        if plane_backend == "open3d":
            try:
                import open3d as o3d
            except ImportError as e:
                msg = "segment_plane with 'open3d' backend requires the 'open3d' package."
                raise ImportError(msg) from e

            # Generate Open3D point cloud directly
            pcd = self.to_pointcloud(camera=camera, backend="open3d")
            if not isinstance(pcd, o3d.geometry.PointCloud):
                    # Handle unexpected return type from to_pointcloud
                    msg = f"Expected Open3D PointCloud, got {type(pcd)}"
                    raise TypeError(msg)

            num_points = len(pcd.points)
            if num_points < min_samples:
                msg = f"Open3D - Point cloud has only {num_points} points (need {min_samples})."
                raise ValueError(msg)

            # Perform Open3D RANSAC
            plane_model_coeffs, inliers_indices = pcd.segment_plane(
                distance_threshold=threshold,
                ransac_n=min_samples,
                num_iterations=max_trials,
            )
            inliers = list(inliers_indices)
            if not inliers:
                msg = "Open3D RANSAC plane segmentation found no inliers."
                raise ValueError(msg)


        elif plane_backend == "trimesh":
            try:
                import trimesh
            except ImportError as e:
                msg = "segment_plane with 'trimesh' backend requires the 'trimesh' package."
                raise ImportError(msg) from e

            # Generate NumPy point cloud directly
            result = self.to_pointcloud(camera=camera, backend="numpy")
            if not (isinstance(result, tuple) and len(result) == 2):
                    msg = f"Expected NumPy tuple (points, colors), got {type(result)}"
                    raise TypeError(msg)
            points, _ = result # We only need the points array

            num_points = points.shape[0]
            if num_points < MIN_POINTS: # Trimesh plane_fit implicitly needs at least 3 points
                msg = f"Trimesh - Point cloud has only {num_points} points (need >= {MIN_POINTS})."
                raise ValueError(msg)

            # Perform Trimesh plane fitting
            centroid, normal = trimesh.points.plane_fit(points)
            normal = normal / np.linalg.norm(normal) # Ensure unit vector
            d = -np.dot(normal, centroid)
            plane_model_coeffs = np.concatenate((normal, [d]))
            inliers = None # Trimesh doesn't provide threshold-based inliers

        if plane_backend not in ["open3d", "trimesh"]:
            msg = f"Unsupported plane_backend: '{plane_backend}'. Choose 'open3d' or 'trimesh'."
            raise ValueError(msg) # Or maybe return None? Raising seems better for invalid input.


        # --- Create and Return Plane Object ---
        if plane_model_coeffs is not None:
            try:

                return Plane(coefficients=plane_model_coeffs, inliers=inliers)
            except Exception as e:
                msg = f"Error creating Plane object: {e}"
                raise ValueError(msg) from e
        else:
            # Should generally not be reached if inner logic returns None on failure
            msg = f"Plane fitting failed for backend '{plane_backend}'."
            raise ValueError(msg)

    def to_pointcloud(
        self,
        camera: Camera | None = None,
        backend: Literal["open3d", "numpy"] = "open3d",
    ) -> o3d.geometry.PointCloud | Tuple[ndarray[Any, 3, np.float32], ndarray[Any, 3, np.uint8]]:
        """Convert the depth image to a point cloud using the specified backend.

        Args:
            camera (Camera | None, optional): Camera parameters. If None, uses self.camera. Defaults to None.
            backend (Literal["open3d", "numpy"], optional): The library to use for conversion.
                'open3d': Uses the Open3D library (returns o3d.geometry.PointCloud).
                'numpy': Uses manual NumPy calculation (returns tuple of points and colors as np.ndarrays).
                Defaults to "open3d".

        Returns:
            Union[o3d.geometry.PointCloud, Tuple[ndarray[Any, 3, np.float32], ndarray[Any, 3, np.uint8]]]:
            The generated point cloud. Type depends on the backend.

        Raises:
            ValueError: If camera intrinsics are missing, required data (depth/rgb) is missing,
                        or an invalid backend is specified.
            ImportError: If the 'open3d' backend is selected but Open3D is not installed.
        """
        camera = camera or self.camera
        if camera is None or camera.intrinsic is None:
            msg = "Camera with intrinsic parameters must be provided."
            raise ValueError(msg)
        if self.array is None:
            msg = "Depth array is not set."
            raise ValueError(msg)
        if self.rgb is None or self.rgb.array is None:
            msg = "RGB image is not set."
            raise ValueError(msg)

        # --- Consistently Calculate Depth in Meters (float32) ---
        depth_array_float = self.array.astype(np.float32)
        scale_to_meters = 1.0

        if self.unit == "m":
            scale_to_meters = 1.0
        elif self.unit == "mm":
            scale_to_meters = 0.001
        else:
            # Unit not specified, rely on camera.depth_scale
            # Assume depth_scale converts raw values to meters
            effective_depth_scale = camera.depth_scale if camera.depth_scale is not None else 1000.0
            if effective_depth_scale != 0: # Avoid division by zero
                 scale_to_meters = 1.0 / effective_depth_scale
            else:
                 # Handle case where depth_scale is zero if necessary, maybe default to mm?
                 logger.warning("camera.depth_scale is zero, assuming depth unit is mm.")
                 scale_to_meters = 0.001

        depth_m = depth_array_float * scale_to_meters
        # --- End of Depth Calculation ---


        width, height = self.size
        fx, fy = camera.intrinsic.fx, camera.intrinsic.fy
        cx, cy = camera.intrinsic.cx, camera.intrinsic.cy

        if backend == "open3d":
            try:
                import open3d as o3d
            except ImportError as e:
                msg = "Open3D backend requires the 'open3d' package to be installed."
                raise ImportError(msg) from e

            # --- Prepare Data for Open3D ---
            # Convert depth in meters back to uint16 millimeters
            # Clamp values before conversion to avoid overflow/underflow issues with uint16
            depth_mm_float = depth_m * 1000.0
            # Ensure values are within valid range for uint16 and make sense as mm
            depth_mm_float_clamped = np.clip(depth_mm_float, 0, 65535)
            depth_for_o3d = depth_mm_float_clamped.astype(np.uint16)

            # Set Open3D scale factor (expecting mm input)
            o3d_depth_scale = 1000.0
            # --- End of Open3D Data Prep ---

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(self.rgb.array),
                o3d.geometry.Image(depth_for_o3d), # Use uint16 mm depth
                depth_scale=o3d_depth_scale,       # Tell O3D the scale to get meters
                depth_trunc=MAX_DEPTH,             # Truncate based on meters
                convert_rgb_to_intensity=False,
            )

            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width,
                height,
                fx,
                fy,
                cx,
                cy,
            )

            return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

        if backend == "numpy":
            # Create pixel grid
            u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

            # Flatten arrays
            z_values = depth_m.flatten() # Use depth in meters directly
            u_flat = u_coords.flatten()
            v_flat = v_coords.flatten()
            colors_flat = self.rgb.array.reshape(-1, 3)

            # Filter out invalid depth points (using meter values)
            valid_mask = (z_values > MIN_DEPTH) & (z_values < MAX_DEPTH)
            z_valid = z_values[valid_mask]
            u_valid = u_flat[valid_mask]
            v_valid = v_flat[valid_mask]
            colors_valid = colors_flat[valid_mask]

            # Back-project to 3D space (Camera Coordinates)
            # Ensure fx and fy are not zero
            if fx == 0 or fy == 0:
                msg = "Camera intrinsics fx and fy must be non-zero for deprojection."
                raise ValueError(msg)
            x_valid = (u_valid - cx) * z_valid / fx
            y_valid = (v_valid - cy) * z_valid / fy

            # Stack points: shape (N, 3)
            points = np.vstack((x_valid, y_valid, z_valid)).T.astype(np.float32)
            # Ensure colors are uint8: shape (N, 3)
            colors = colors_valid.astype(np.uint8)

            # Return as tuple of ndarrays
            return points, colors

        msg = f"Unsupported backend: '{backend}'. Choose 'open3d' or 'numpy'."
        raise ValueError(msg)

    def colormap(self, **kwargs) -> Image:
        """Postprocess the depth array and convert it to an RGB image with a colormap applied."""
        if self.array is None and self.rgb is None:
            msg = "The depth array or RGB image must be set to convert to a colormap image."
            raise ValueError(msg)
        plt = smart_import("matplotlib.pyplot", mode="lazy")

        if plt.__name__ == "matplotlib.pyplot":

            # Normalize the depth array to [0, 255] range and convert to uint8
            depth_normalized = cv2.normalize(self.array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            return Image(depth_colored, mode="RGB")

        # Fallback to creating an Image using the depth array if matplotlib is unavailable
        return Image(self.array, mode="I", encoding="png")

    def show(self) -> None:
        Image(self.colormap()).show()

    @wraps(Image.save, assigned=("__doc__"))
    def save(self, *args, **kwargs) -> None:
        """Save the image to a file."""
        self.colormap().save(*args, **kwargs)

    def dump(self, *_args, as_field: str | None = None, **_kwargs) -> dict | Any:
        """Return a dict or a field of the image."""
        if as_field is not None:
            return getattr(self, as_field)
        out = {
            "size": self.size,
            "mode": self.mode,
            "encoding": self.encoding,
        }
        if self.path is not None:
            out["path"] = self.path
        if self.base64 is not None:
            out["base64"] = self.base64
        if self.url not in self.base64 and len(self.url) < 120:
            out["url"] = self.url
        return out

    def segment_cylinder(
        self,
        min_samples=3,
        threshold: float = 0.01,
        max_trials: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment the largest cylinder using RANSAC.

        Args:
            min_samples (int): The minimum number of data points to fit a model.
            threshold (float): The maximum distance for a point to be considered as an inlier.
            max_trials (int): The maximum number of iterations for RANSAC.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The inlier points and their indices.
        """
        sklearn = smart_import("sklearn", mode="eager")  # noqa
        from sklearn.linear_model import RANSACRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=2)
        ransac = make_pipeline(
            poly,
            RANSACRegressor(min_samples=min_samples, residual_threshold=threshold, max_trials=max_trials),
        )

        X = self.points[:, :2]
        y = self.points[:, 2]

        ransac.fit(X, y)

        inlier_mask = ransac.named_steps["ransacregressor"].inlier_mask_
        inlier_points = self.points[inlier_mask]
        inlier_indices = np.where(inlier_mask)[0]

        return inlier_points, inlier_indices

    def cluster_points(self, n_clusters: int = 3) -> List[int]:
        """Cluster the points using KMeans.

        Args:
            n_clusters (int): The number of clusters to form.

        Returns:
            List[int]: The cluster labels for each point.
        """
        smart_import("sklearn.cluster")
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters)
        return kmeans.fit_predict(self.points.T)
