from typing import Any

import numpy as np
from pydantic import ConfigDict, Field, computed_field, model_serializer

from embdata.coordinate import Coordinate, PixelCoords
from embdata.geometry import Affine3D as Extrinsics
from embdata.ndarray import ndarray
from embdata.sample import Sample


class Intrinsics(Coordinate):
    """Transformation from 3D to 2D coordinates for a pinhole camera model."""

    fx: float = 0.0
    """Focal length in x direction."""
    fy: float = 0.0
    """Focal length in y direction."""
    cx: float = 0.0
    """Optical center in x direction."""
    cy: float = 0.0
    """Optical center in y direction."""

    @model_serializer(when_used="always")
    def omit_matrix(self) -> None:
        """Deserialize the 3x3 matrix to the intrinsic parameters."""
        return self.dump(exclude={"matrix"})

    @computed_field
    def matrix(self) -> ndarray[3, 3, float]:
        """Convert the intrinsic parameters to a 3x3 matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
        )


class Distortion(Coordinate):
    """Model for Camera Distortion Parameters."""

    k1: float = 0.0
    """Radial distortion coefficient k1."""
    k2: float = 0.0
    """Radial distortion coefficient k2."""
    p1: float = 0.0
    """Tangential distortion coefficient p1."""
    p2: float = 0.0
    """Tangential distortion coefficient p2."""
    k3: float = 0.0
    """Radial distortion coefficient k3."""


class Camera(Sample):
    """Model for Camera Parameters."""
    model_config = ConfigDict(extra="forbid")
    intrinsic: Intrinsics = Field(default_factory=Intrinsics, description="Intrinsic parameters of the camera")
    distortion: Distortion = Field(default_factory=Distortion, description="Distortion parameters of the camera")
    extrinsic: Extrinsics = Field(default_factory=Extrinsics, description="Extrinsic parameters of the camera")
    depth_scale: float = 1.0

    def __eq__(self, other: "Camera") -> bool:
        return (
            self.intrinsic == other.intrinsic
            and self.distortion == other.distortion
            and self.extrinsic == other.extrinsic
            and self.depth_scale == other.depth_scale
        )

    def project(self, xyz: ndarray[3, float]) -> ndarray[2, float]:
        """Project 3D world coordinates to 2D pixel coordinates."""
        # Transform the point from world to camera frame using the extrinsic parameters
        xyz_camera = self.extrinsic.transform_points(xyz).squeeze()
        x_pixel = (self.intrinsic.fx * xyz_camera[0] / xyz_camera[2]) + self.intrinsic.cx
        y_pixel = (self.intrinsic.fy * xyz_camera[1] / xyz_camera[2]) + self.intrinsic.cy
        return np.array([x_pixel, y_pixel])

    def deproject(self,
                  uv: ndarray[2, int] | ndarray[Any, 2, int],
                  depth_image: ndarray[Any, Any]) -> ndarray[3, float] | ndarray[Any, 3, float]:
        """Deproject 2D pixel coordinates from the depth image to 3D world coordinates.

        Args:
            uv (ndarray[2, float]): The (u, v) coordinates of the pixel.
            depth_image (np.ndarray): The depth image.
            depth_scale (float): The depth scale factor for the RealSense camera.

        Returns:
            ndarray[3, float]: The (x, y, z) coordinates in 3D space.

        Example:
            >>> estimator = ArucoMarkerBasedObjectPoseEstimation(color_image, depth_image, intrinsic_matrix)
            >>> uv = np.array([320, 240])
            >>> point_3d = estimator.deproject(uv, depth_image, 0.001)
            >>> print(point_3d)
        """
            # Check for invalid shapes upfront
        if isinstance(uv, np.ndarray) and (uv.ndim == 1 and uv.shape[0] != 2):  # noqa: PLR2004
            msg = "Invalid shape for uv coordinates. Expected shape (2,)"
            raise ValueError(msg)
        if isinstance(uv, np.ndarray) and (uv.ndim == 2 and uv.shape[1] != 2):  # noqa: PLR2004
            msg = "Invalid shape for uv coordinates. Expected shape (n, 2)"
            raise ValueError(msg)

        if isinstance(uv, PixelCoords) or (isinstance(uv, np.ndarray) and uv.ndim == 1):
            u, v = uv
            depth = depth_image[int(v), int(u)] * self.depth_scale
            x = (u - self.intrinsic.cx) * depth / self.intrinsic.fx
            y = (v - self.intrinsic.cy) * depth / self.intrinsic.fy
            z = depth
            return np.array([x, y, z])

        if isinstance(uv, np.ndarray) and uv.ndim == 2:  # noqa: PLR2004
            u = uv[:, 1]  # Take u (x) coordinates
            v = uv[:, 0]  # Take v (y) coordinates

            # Get the depth values for all u, v points
            depths = depth_image[v, u] * self.depth_scale

            # Filter out points with zero depth (invalid depth)
            valid_mask = depths > 0
            u, v, depths = u[valid_mask], v[valid_mask], depths[valid_mask]

            # Compute x, y, z coordinates
            x = (u - self.intrinsic.cx) * depths / self.intrinsic.fx
            y = (v - self.intrinsic.cy) * depths / self.intrinsic.fy
            z = depths

            return np.stack((x, y, z), axis=-1)
        msg = "Invalid shape for uv coordinates. Expected shape (2,) or (n, 2)."
        raise ValueError(msg)

