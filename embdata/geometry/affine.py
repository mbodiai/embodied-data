from typing import Any, ClassVar, Self, Sequence, overload

import numpy as np
from numpy import float64
from pydantic import Field
from typing_extensions import override

from embdata.array import array
from embdata.coordinate import Coordinate, Pose
from embdata.geometry.utils import rotation_matrix_to_rpy, rpy_to_rotation_matrix
from embdata.ndarray import Float, ndarray
from embdata.utils.counting import Batch, sz


class Affine2D(Coordinate[sz[3],sz[3]]):
    """Represents a general 2D transformation including rotation and translation."""

    rotation: array[sz[2], sz[2], Float] = Field(
        default_factory=lambda: np.eye(2),
        description="Rotation matrix (2x2) representing orientation.",
    )
    translation: array[sz[2], Float] = Field(
        default_factory=lambda: np.zeros(2),
        description="Translation vector (2x1) representing position.",
    )

    def matrix(self) -> array[sz[3], sz[3], Float]:
        """Convert the transformation to a 3x3 homogeneous transformation matrix."""
        matrix = np.eye(3)
        matrix[:2, :2] = self.rotation
        matrix[:2, 2] = self.translation
        return matrix

    @overload
    def __matmul__(self, other: "Affine2D") -> "Affine2D": ...
    @overload
    def __matmul__(self, other: ndarray[sz[2], Float] | tuple[Float, Float]) -> array[sz[2], Float]:
        """Apply the transformation to a 2D point."""

    @overload
    def __matmul__(self, other: array[Batch, sz[2], Float]) -> array[Batch, sz[2], Float]:...
    def __matmul__(
        self, other: "Affine2D|array[sz[2], Float] | array[Batch,sz[2],Float] | tuple[Float,Float]",
    ) -> "Affine2D|array[Batch, sz[2], Float] | array[sz[2], Float]":
        if isinstance(other, Affine2D):
            rotation = np.dot(self.rotation, other.rotation)
            translation = np.dot(self.rotation, other.translation) + self.translation
            return Affine2D(rotation=rotation, translation=translation)
        return self.transform_points(other)

    def inverse(self) -> "Affine2D":
        """Compute the inverse of the transformation."""
        inverse_rotation = self.rotation.T
        inverse_translation = -np.dot(inverse_rotation, self.translation)
        return Affine2D(rotation=inverse_rotation, translation=inverse_translation)

    @overload
    def transform_points(self, points: array[Batch, sz[2], Float]) -> array[Batch, sz[2], Float]: ...
    @overload
    def transform_points(self, points: array[sz[2], Float] | tuple[Float, Float]) -> array[sz[2], Float]: ...
    def transform_points(
        self,
        points: array[Batch, sz[2], Float] | array[sz[2], Float] | tuple[Float, Float],
    ) -> Any:
        """Transform a single point or a list of points using this transformation.

        Args:
            points: A single point (shape: [2]) or a list of points (shape: [N, 2])

        Returns:
            Transformed point(s) with the same shape as the input.
        """
        points_arr = np.asanyarray(points)
        if points_arr.ndim == 1: # Single point
            # Reshape to (1, 2) for homogeneous multiplication, then squeeze back to (2,)
            homogeneous_point = np.hstack([
                points_arr.reshape(1, -1),
                np.ones((1, 1), dtype=points_arr.dtype),
            ])
            return (self.matrix() @ homogeneous_point.T).T[0, :2]
        # Batch of points
        homogeneous_points = np.hstack([
            points_arr,
            np.ones((points_arr.shape[0], 1), dtype=points_arr.dtype),
        ])
        return (self.matrix() @ homogeneous_points.T).T[:, :2]

    @classmethod
    def from_pose(cls, pose: array[sz[3], Float]) -> "Affine2D":
        """Create a 2D transformation from a planar pose."""
        x, y, theta = pose
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        translation = np.array([x, y])
        return cls(rotation=rotation, translation=translation)


class Affine3D(Coordinate[sz[4],sz[4]]):
    """Represents a general 3D transformation including rotation and translation."""

    rotation: array[sz[3], sz[3], Float] = Field(
        default_factory=lambda: np.eye(3, dtype=np.float64),
        description="Rotation matrix (3x3) representing orientation.",
    )
    translation: array[sz[3], Float]= Field(
        default_factory=lambda: np.zeros(3, dtype=np.float64),
        description="Translation vector (3x1) representing position.",
    )
    shape: ClassVar[tuple[int, int]] = (4, 4)

    def matrix(self) -> array[sz[4], sz[4], Float]:
        """Convert the transformation to a 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Affine3D):
            return False
        return np.allclose(self.rotation, other.rotation) and np.allclose(self.translation, other.translation)

    @overload
    def __matmul__(self, other: "Affine3D") -> "Affine3D": ...
    @overload
    def __matmul__(
        self, other: "array[sz[3], Float] | array[Batch, sz[3], Float] | Self",
    ) -> "array[Batch, sz[3], Float]": ...
    def __matmul__(
        self, other: "array[sz[3], Float] | array[Batch, sz[3], Float] | Self|Affine3D",
    ) -> Any:
        """Apply the transformation to a 3D point."""
        if isinstance(other, Affine3D):
            rotation: array[sz[3], sz[3], Float] = np.dot(self.rotation, other.rotation)
            translation: array[sz[3], Float] = np.dot(self.rotation, other.translation) + self.translation
            return Affine3D(rotation=rotation, translation=translation)
        return self.transform_points(other)

    def inverse(self) -> "Affine3D":
        """Compute the inverse of the transformation."""
        inverse_rotation: array[sz[3], sz[3], Float] = self.rotation.T
        # For a rigid motion (R, t), the inverse translation is -Rᵀ·t.
        inverse_translation: array[sz[3], Float] = -np.dot(inverse_rotation, self.translation)
        return Affine3D(rotation=inverse_rotation, translation=inverse_translation)

    @classmethod
    @overload
    def from_pose(cls, x:float=0.0,y:float=0.0,z:float=0.0,roll:float=0.0,pitch:float=0.0,yaw:float=0.0) -> "Affine3D": ...
    @overload
    @classmethod
    def from_pose(cls, pose: "array[sz[6],Float]|Pose") -> "Affine3D": ...
    @classmethod
    def from_pose(cls,*args,**kwargs) -> "Affine3D":
        """Create affine transform from pose."""
        pose = Pose(*args[0]) if len(args) == 1 and isinstance(args[0], Pose | array) else Pose(*args, **kwargs)
        return cls(rotation=pose.rotation, translation=pose.translation)

    @overload
    def transform_points(self, points: array[Batch, sz[3], Float]) -> array[Batch, sz[3], Float]: ...
    @overload
    def transform_points(self, points: array[sz[3], Float]) -> array[sz[3], Float]: ...
    def transform_points(self, points: array[Batch, sz[3], Float] | array[sz[3], Float]) -> Any:
        """Transform a single point or a list of points using this transformation.

        Args:
            points: A single point (shape: [3]) or a list of points (shape: [N, 3])

        Returns:
            Transformed point(s) with the same shape as the input.
        """
        if points.ndim == 1:
            return (self.matrix() @ np.hstack([points.reshape((1, -1)), np.ones((1, 1))]).T)[:3, 0]
        shape = points.shape
        N = shape[0] if isinstance(shape,Sequence) else 1
        return (self.matrix() @ np.hstack([points, np.ones((N, 1))]).T).T[:, :3]

    def pose(self) -> Pose:
        """Convert the rotation matrix to roll, pitch, yaw angles using the specified sequence."""
        return Pose(*self.translation, *rotation_matrix_to_rpy(self.rotation), reference_frame=self.reference_frame())

    @classmethod
    def from_array(cls, arr: array[sz[4],sz[4], Float]) -> "Affine3D":
        return cls(rotation=arr[:3,:3].astype(float64),translation=arr[:3,3].astype(float64))
    @classmethod
    def from_rpy(cls, rpy: array[sz[3], Float]) -> "Affine3D":
        """Create a 3D transformation from roll, pitch, and yaw angles."""
        rotation: array[sz[3], sz[3], Float] = rpy_to_rotation_matrix(rpy)
        return cls(rotation=rotation)


    def quaternion(self) -> array[sz[4], Float]:
        """Return the orientation component as a quaternion (x, y, z, w)."""
        from scipy.spatial.transform import Rotation
        rot_mat = np.asarray(self.rotation, dtype=float64)
        q = Rotation.from_matrix(rot_mat).as_quat(canonical=True)
        return np.asarray(q) / np.linalg.norm(q)

    @classmethod
    def from_quat_pose(
        cls,
        quat: array[sz[4], Float] | Any,
        translation: array[sz[3], Float] | Any | None = None,
    ) -> "Affine3D":
        """Create an ``Affine3D`` transform from a quaternion and optional translation.

        Args:
            quat: Quaternion in (x, y, z, w) format.
            translation: Optional translation vector. Defaults to zeros if ``None``.
        """
        from scipy.spatial.transform import Rotation
        rotation: array[sz[3], sz[3], Float] = Rotation.from_quat(
            np.asarray(quat) / np.linalg.norm(quat),
        ).as_matrix()
        return cls(rotation=rotation, translation=translation or np.zeros(3))
