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

def demo() -> None:
    """Demonstrate the functionality of the affine transformations module.

    This function serves as a comprehensive test of the functionality described in
    the transforms.md documentation, covering:

    1. Basic creation and usage of Affine2D and Affine3D
    2. Creating transformations from various sources
    3. Transformation operations (composition, inverse)
    4. Point transformation
    5. Converting between representations
    6. Integration with the coordinate system
    7. Practical examples
    """
    import math

    from embdata.coordinate import Pose
    from embdata.utils.safe_print import safe_print

    safe_print("\n=== Affine Transformations Demo ===\n")

    # 1. Basic Usage - Affine2D
    safe_print("1. Basic Usage - Affine2D:")
    # Create an identity transformation
    transform_2d = Affine2D()
    safe_print(f"  Identity Affine2D rotation:\n{transform_2d.rotation}")
    safe_print(f"  Identity Affine2D translation: {transform_2d.translation}")

    # Create with specific rotation and translation
    rotation_2d = np.array([[0, -1], [1, 0]])  # 90-degree rotation
    translation_2d = np.array([1, 2])
    transform_2d = Affine2D(rotation=rotation_2d, translation=translation_2d)
    safe_print(f"  Custom Affine2D rotation:\n{transform_2d.rotation}")
    safe_print(f"  Custom Affine2D translation: {transform_2d.translation}")

    # Get as homogeneous transformation matrix
    matrix_2d = transform_2d.matrix()
    safe_print(f"  Homogeneous matrix (3×3):\n{matrix_2d}")

    # 2. Basic Usage - Affine3D
    safe_print("\n2. Basic Usage - Affine3D:")
    # Create an identity transformation
    transform_3d = Affine3D()
    safe_print(f"  Identity Affine3D rotation:\n{transform_3d.rotation}")
    safe_print(f"  Identity Affine3D translation: {transform_3d.translation}")

    # Create with specific rotation and translation
    rotation_3d = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # Rotation around Z-axis
    translation_3d = np.array([1, 2, 3])
    transform_3d = Affine3D(rotation=rotation_3d, translation=translation_3d)
    safe_print(f"  Custom Affine3D rotation:\n{transform_3d.rotation}")
    safe_print(f"  Custom Affine3D translation: {transform_3d.translation}")

    # Get as homogeneous transformation matrix
    matrix_3d = transform_3d.matrix()
    safe_print(f"  Homogeneous matrix (4×4):\n{matrix_3d}")

    # 3. Creating Transformations - From Pose
    safe_print("\n3. Creating Transformations - From Pose:")
    # From individual components
    transform_from_components = Affine3D.from_pose(x=1, y=2, z=3, roll=0.1, pitch=0.2, yaw=0.3)
    safe_print(f"  From components - rotation:\n{transform_from_components.rotation}")
    safe_print(f"  From components - translation: {transform_from_components.translation}")

    # From a Pose object
    pose = Pose(1, 2, 3, 0.1, 0.2, 0.3)
    transform_from_pose = Affine3D.from_pose(pose)
    safe_print(f"  From Pose object - rotation:\n{transform_from_pose.rotation}")
    safe_print(f"  From Pose object - translation: {transform_from_pose.translation}")

    # 4. Creating Transformations - From Rotation Representations
    safe_print("\n4. Creating Transformations - From Rotation Representations:")
    # From roll, pitch, yaw angles
    rpy = np.array([0.1, 0.2, 0.3])  # roll, pitch, yaw in radians
    transform_from_rpy = Affine3D.from_rpy(rpy)
    safe_print(f"  From RPY - rotation:\n{transform_from_rpy.rotation}")

    # From quaternion
    quat = np.array([0.1, 0.2, 0.3, 0.9])  # Not normalized
    transform_from_quat = Affine3D.from_quat_pose(quat, translation=[1, 2, 3])
    safe_print(f"  From quaternion - rotation:\n{transform_from_quat.rotation}")
    safe_print(f"  From quaternion - translation: {transform_from_quat.translation}")

    # From homogeneous transformation matrix
    matrix = np.eye(4)  # 4×4 identity matrix
    matrix[:3, 3] = [1, 2, 3]  # Set translation
    transform_from_matrix = Affine3D.from_array(matrix)
    safe_print(f"  From matrix - rotation:\n{transform_from_matrix.rotation}")
    safe_print(f"  From matrix - translation: {transform_from_matrix.translation}")

    # 5. Transformation Operations - Composition
    safe_print("\n5. Transformation Operations - Composition:")
    # Define two transformations
    transform1 = Affine3D(
        rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        translation=np.array([1, 2, 3]),
    )

    transform2 = Affine3D(
        rotation=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        translation=np.array([4, 5, 6]),
    )

    # Compose transformations
    composed = transform1 @ transform2
    safe_print(f"  Transform1 rotation:\n{transform1.rotation}")
    safe_print(f"  Transform1 translation: {transform1.translation}")
    safe_print(f"  Transform2 rotation:\n{transform2.rotation}")
    safe_print(f"  Transform2 translation: {transform2.translation}")
    safe_print(f"  Composed rotation:\n{composed.rotation}")
    safe_print(f"  Composed translation: {composed.translation}")

    # 6. Transformation Operations - Inverse
    safe_print("\n6. Transformation Operations - Inverse:")
    # Compute the inverse of a transformation
    transform = Affine3D.from_pose(x=1, y=2, z=3, roll=0.1, pitch=0.2, yaw=0.3)
    inverse = transform.inverse()
    safe_print(f"  Original rotation:\n{transform.rotation}")
    safe_print(f"  Original translation: {transform.translation}")
    safe_print(f"  Inverse rotation:\n{inverse.rotation}")
    safe_print(f"  Inverse translation: {inverse.translation}")

    # Verify: transform @ inverse ≈ identity
    identity = transform @ inverse
    safe_print(f"  Transform @ Inverse rotation:\n{identity.rotation}")
    safe_print(f"  Transform @ Inverse translation: {identity.translation}")
    is_identity = np.allclose(identity.rotation, np.eye(3), atol=1e-6) and np.allclose(identity.translation, np.zeros(3), atol=1e-6)
    safe_print(f"  Is identity: {is_identity}")

    # 7. Transforming Points
    safe_print("\n7. Transforming Points:")
    # Create a transformation
    transform = Affine3D.from_pose(x=1, y=2, z=3, roll=0.1, pitch=0.2, yaw=0.3)

    # Transform a single point
    point = np.array([1, 2, 3])
    transformed_point1 = transform.transform_points(point)
    transformed_point2 = transform @ point
    safe_print(f"  Original point: {point}")
    safe_print(f"  Transformed with transform_points(): {transformed_point1}")
    safe_print(f"  Transformed with @ operator: {transformed_point2}")
    safe_print(f"  Results match: {np.allclose(transformed_point1, transformed_point2)}")

    # Transform multiple points at once (batch transformation)
    points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    transformed_points = transform.transform_points(points)
    safe_print(f"  Original points:\n{points}")
    safe_print(f"  Transformed points:\n{transformed_points}")

    # 8. Converting Between Representations
    safe_print("\n8. Converting Between Representations:")
    # Create a transformation
    transform = Affine3D.from_pose(x=1, y=2, z=3, roll=0.1, pitch=0.2, yaw=0.3)

    # Get the rotation and translation components directly
    safe_print(f"  Transform rotation:\n{transform.rotation}")
    safe_print(f"  Transform translation: {transform.translation}")

    # Direct extraction of roll, pitch, yaw from rotation matrix
    from embdata.geometry.utils import rotation_matrix_to_rpy
    rpy = rotation_matrix_to_rpy(transform.rotation)
    safe_print(f"  Extracted roll, pitch, yaw: {rpy}")

    # Get quaternion representation
    quat = transform.quaternion()
    safe_print(f"  Quaternion (x, y, z, w): {quat}")

    # Skip the reference frame section that depends on Coordinate.info()

    # 9. Practical Example - Camera Extrinsics
    safe_print("\n9. Practical Example - Camera Extrinsics:")
    # Camera is at position [2, 1, 0.5] looking along the X-axis
    camera_to_world = Affine3D.from_pose(x=2, y=1, z=0.5, roll=0, pitch=0, yaw=math.pi/2)

    # Transform a point from camera coordinates to world coordinates
    point_camera = np.array([0.5, 0, 0])  # Point 0.5m in front of camera
    point_world = camera_to_world @ point_camera
    safe_print(f"  Camera to world transform:\n{camera_to_world.matrix()}")
    safe_print(f"  Point in camera frame: {point_camera}")
    safe_print(f"  Point in world frame: {point_world}")

    # Transform from world to camera
    world_to_camera = camera_to_world.inverse()
    point_in_camera_frame = world_to_camera @ point_world
    safe_print(f"  World to camera transform:\n{world_to_camera.matrix()}")
    safe_print(f"  Point back in camera frame: {point_in_camera_frame}")
    safe_print(f"  Matches original: {np.allclose(point_camera, point_in_camera_frame)}")

    # 10. Practical Example - Robot Kinematics
    safe_print("\n10. Practical Example - Robot Kinematics:")
    # Base to end-effector transform
    base_to_end = Affine3D.from_pose(x=0.5, y=0, z=0.8, roll=0, pitch=math.pi/2, yaw=0)

    # Transform a point in end-effector frame to base frame
    point_end = np.array([0.1, 0, 0])  # 10cm in front of end-effector
    point_base = base_to_end @ point_end
    safe_print(f"  Base to end-effector transform:\n{base_to_end.matrix()}")
    safe_print(f"  Point in end-effector frame: {point_end}")
    safe_print(f"  Point in base frame: {point_base}")

    # Define a transformation from world to base
    world_to_base = Affine3D.from_pose(x=1, y=1, z=0, roll=0, pitch=0, yaw=math.pi/4)

    # Combine transformations to get end-effector in world frame
    world_to_end = world_to_base @ base_to_end
    safe_print(f"  World to base transform:\n{world_to_base.matrix()}")
    safe_print(f"  World to end-effector transform:\n{world_to_end.matrix()}")

    # Transform point from end-effector to world frame
    point_world = world_to_end @ point_end
    safe_print(f"  Point in world frame: {point_world}")

    safe_print("\n=== Demo Complete ===\n")

if __name__ == "__main__":
    demo()
