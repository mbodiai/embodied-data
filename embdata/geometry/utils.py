# Copyright (c) 2024 Mbodi AI
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from annotated_types import Len
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from typing_extensions import Annotated, overload

from embdata.array import array
from embdata.coordinate import Pose
from embdata.ndarray import ndarray

sz = Literal
Float = np.float64|np.floating|float
if TYPE_CHECKING:
    from embdata.coordinate import Point

@overload
def rpy_to_rotation_matrix(
    roll: Float,pitch: Float,yaw: Float,
) -> array[sz[3], sz[3], Float]:
    ...
@overload
def rpy_to_rotation_matrix(
    rpy: array[sz[3], Float],
) -> array[sz[3], sz[3], Float]:...

def rpy_to_rotation_matrix(*args:Any,**kwargs:Any)-> array[sz[3], sz[3], Float]:
    """Convert roll-pitch-yaw (stored in that order) to a rotation matrix.

    Args:
        rpy: array[sz[3], Float],
        sequence: Literal["zyx", "xyz","ZXY","XYZ"] = "xyz"

    Note:
            ZYX (Intrinsic ) is the typical roll, pitch, yaw sequence.

    Returns:
        array[sz[3], sz[3], Float]: The rotation matrix.


    """
    if not args and not kwargs:
        msg = "No arguments provided"
        raise ValueError(msg)
    if len(args) > 3:
        msg = f"Too many arguments provided for  rpy: {list(args)}"
        raise ValueError(msg)
    rpy = kwargs.get("rpy", args)
    if len(rpy) == 1:
        rpy = rpy[0]
    if len(rpy) == 3:
        return R.from_euler("xyz", rpy, degrees=False).as_matrix()
    msg = f"Invalid number of arguments provided for  rpy: {list(rpy)}"
    raise ValueError(msg)


def rotation_matrix_to_rpy(
    matrix: array[sz[3], sz[3], Float],
) -> array[sz[3], Float]:
    """Inverse of :pyfunc:`rpy_to_rotation_matrix`. Returns *(roll, pitch, yaw)*. Relies on scipy for canonicalization."""
    # R.from_matrix(matrix).as_euler("xyz", degrees=False) returns angles for X, then Y, then Z rotation
    # These correspond to roll, pitch, yaw respectively for the "xyz" sequence.
    roll, pitch, yaw = R.from_matrix(matrix).as_euler("xyz", degrees=False)
    return ndarray[sz[3],Float]([roll,pitch,yaw])


def bound_angle_half_pi(angle: float) -> float:
    """Bound an angle to be within the range [-π/2, π/2].

    Args:
        angle (float): The angle in radians to normalize.

    Returns:
        float: The angle bounded within [-π/2, π/2].
    """
    # Normalize angle to range [-π, π]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi

    # Bound angle to range [-π/2, π/2]
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi

    return angle

# Rotation matrices and transformations
def rotation_matrix(deg: float) -> array[sz[2], sz[2], Float]:
    """Generate a 2x2 rotation matrix for a given angle in degrees."""
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    return ndarray[sz[2],sz[2],Float]([[c, -s], [s, c]])


def rotation_to_transformation_matrix(R: array[sz[3], sz[3], Float]) -> array[sz[4], sz[4], Float]:
    """Convert a rotation matrix to a transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    return T


def pose_to_transformation_matrix(
    pose: Pose | array[sz[6], Float],
) -> array[sz[4], sz[4], Float]:
    """Convert a pose (position and rotation) to a transformation matrix."""
    from embdata.geometry import Affine3D
    return Affine3D.from_pose(pose).matrix()
def transformation_matrix_to_pose(T: array[Any,sz[4],sz[4],Float]) -> tuple[array[Any,sz[3],Float],array[Any,sz[3],sz[3],Float]]:
    """Extract position and rotation matrix from a transformation matrix."""
    position: array[Any,sz[3],Float] = T[:3, 3]
    rotation: array[Any,sz[3],sz[3],Float] = T[:3, :3]
    return position, rotation


def transformation_matrix_to_position(T: array[Any,sz[4],sz[4],Float]) -> array[Any,sz[3],Float]:
    """Extract position from a transformation matrix."""
    return T[:3, 3]


def transformation_matrix_to_rotation(T: array[Any,sz[4],sz[4],Float]) -> array[Any,sz[3],sz[3],Float]:
    """Extract rotation matrix from a transformation matrix."""
    return T[:3, :3]


def axis_angle_to_euler(axis_angle_vector: array[Any,sz[3],Float], sequence: str = "xyz") -> array[Any,sz[3],Float]:
    """Convert axis-angle to Euler angles using scipy.

    Args:
        axis_angle_vector (np.ndarray): Axis-angle vector (3 elements).
        sequence (str): The order of rotations:
            'xyz', 'zyx' for extrinsic rotations.
            'XYZ', 'ZYX' for intrinsic rotations.

    Returns:
        np.ndarray: Euler angles (roll, pitch, yaw).
    """
    # Create a rotation object from the axis-angle vector
    rotation = R.from_rotvec(axis_angle_vector)

    return rotation.as_euler(seq=sequence, degrees=False)


def euler_to_axis_angle(euler_angles: array[Any,sz[3],Float], sequence: str = "xyz") -> array[Any, sz[3], Float]:
    """Convert Euler angles (roll, pitch, yaw) to axis-angle using scipy.

    Args:
        euler_angles (np.ndarray): Euler angles (roll, pitch, yaw).
        sequence (str): The order of rotations, e.g., 'xyz', 'zyx', etc.

    Returns:
        np.ndarray: Axis-angle vector (3 elements).
    """
    # Create a rotation object from the Euler angles
    rotation = R.from_euler(sequence, euler_angles, degrees=False)

    # Convert to axis-angle (rotation vector)
    axis_angle_vector: array[Any,sz[3],Float] = rotation.as_rotvec()

    return axis_angle_vector


def rpy_to_vector(roll: float, pitch: float, yaw: float):
    """Convert roll, pitch, yaw angles to a direction vector.

    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

    Returns:
        np.ndarray: 3D direction vector.
    """
    # Calculate the direction vector
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)

    return np.array([x, y, z])


def rotation_matrix_to_angular_velocity(R: array[Any,sz[3],sz[3],Float]) -> array[Any,sz[3],Float]:
    """Convert a rotation matrix to an angular velocity vector."""
    el: array[Any,sz[3],Float] = np.array([[R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]], [R[1, 0] - R[0, 1]]])
    norm_el: Float = np.linalg.norm(el)

    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R) - 1) / norm_el * el
    elif R[0, 0] > 0 and R[1, 1] > 0 and R[2, 2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.pi / 2 * np.array([[R[0, 0] + 1], [R[1, 1] + 1], [R[2, 2] + 1]])

    return w.flatten()



def skew_symmetric_matrix(vector: array[Any,sz[3],Float]) -> array[Any,sz[3],sz[3],Float]:
    """Generate a skew-symmetric matrix from a vector."""
    return np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])


def rodrigues_rotation(axis: array[Any,sz[3],Float], angle_rad: Float) -> array[Any,sz[3],sz[3],Float]:
    """Compute the rotation matrix from an angular velocity vector."""
    axis_norm = np.linalg.norm(axis)
    if abs(axis_norm - 1) > 1e-6:
        msg = "Norm of axis should be 1.0"
        raise ValueError(msg)

    axis: array[Any,sz[3],Float] = axis / axis_norm
    angle_rad: Float = angle_rad * axis_norm
    axis_skew: array[Any,sz[3],sz[3],Float] = skew_symmetric_matrix(axis)

    return np.eye(3) + axis_skew * np.sin(angle_rad) + axis_skew @ axis_skew * (1 - np.cos(angle_rad))


def unit_vector(vector: array[Any,sz[3],Float]) -> array[Any,sz[3],Float]:
    """Return the unit vector of the input vector."""
    return vector / np.linalg.norm(vector)


def rotation_from_z(p: "array[Any,sz[3],Float]|Point") -> array[sz[3],sz[3],Float]:
    """Generate a rotation matrix that aligns one point with the z-axis."""
    return rotation_between_two_points(np.asarray(p,dtype=np.float64),np.array([0,0,1],dtype=np.float64))


def rotation_between_two_points(a: "array[Any,sz[3],Float]|Point", b: "array[Any,sz[3],Float]|Point") -> array[sz[3],sz[3],Float]:
    rotation,error,sensitivity = R.align_vectors(np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64))
    if error > 1e-4:
        msg = f"Error between vectors is too large: {error}"
        raise ValueError(msg)
    if not all(np.asarray(sensitivity).flatten() < 1e-4):
        warnings.warn(f"Sensitivity between vectors is too large: {sensitivity}",stacklevel=2)
    return rotation.as_matrix()





def soft_squash(x: array[Any,sz[1],Float], x_min: float = -1, x_max: float = 1, margin: float = 0.1) -> array[Any,sz[1],Float]:
    """Softly squash the values of an array within a specified range with margins."""

    def threshold_function(z, margin=0.0):
        return margin * (np.exp(2 / margin * z) - 1) / (np.exp(2 / margin * z) + 1)

    x_copy = np.copy(x)
    upper_idx = np.where(x_copy > (x_max - margin))
    x_copy[upper_idx] = threshold_function(x_copy[upper_idx] - (x_max - margin), margin=margin) + (x_max - margin)

    lower_idx = np.where(x_copy < (x_min + margin))
    x_copy[lower_idx] = threshold_function(x_copy[lower_idx] - (x_min + margin), margin=margin) + (x_min + margin)

    return x_copy


def soft_squash_multidim(x: array[Any,sz[1],Float], x_min: array[Any,sz[1],Float], x_max: array[Any,sz[1],Float], margin: float = 0.1) -> array[Any,sz[1],Float]:
    """Apply soft squashing to a multi-dimensional array."""
    x_squashed = np.copy(x)
    dim = x.shape[1]
    for d_idx in range(dim):
        x_squashed[:, d_idx] = soft_squash(x[:, d_idx], x_min[d_idx], x_max[d_idx], margin)
    return x_squashed


def squared_exponential_kernel(X1: array[Any,sz[1],Float], X2: array[Any,sz[1],Float], hyp: dict) -> array[Any,sz[1],Float]:
    """Compute the squared exponential (SE) kernel between two sets of points."""
    return hyp["g"] * np.exp(-cdist(X1, X2, "sqeuclidean") / (2 * hyp["l"] ** 2))


def leveraged_squared_exponential_kernel(
    X1: array[Any,sz[1],Float],
    X2: array[Any,sz[1],Float],
    L1: array[Any,sz[1],Float],
    L2: array[Any,sz[1],Float],
    hyp: dict,
) -> array[Any,sz[1],Float]:
    """Compute the leveraged SE kernel between two sets of points."""
    K = hyp["g"] * np.exp(-cdist(X1, X2, "sqeuclidean") / (2 * hyp["l"] ** 2))
    L = np.cos(np.pi / 2.0 * cdist(L1, L2, "cityblock"))
    return np.multiply(K, L)



def is_point_in_polygon(point:"array[Any,sz[3],Float]|Point", polygon: "Polygon") -> bool:
    """Check if a point is inside a given polygon."""
    from shapely.geometry import Point

    point_geom = Point(point) if isinstance(point, np.ndarray) else point
    return polygon.contains(point_geom)


def is_point_feasible(point: "array[Any,sz[3],Float]|Point", obstacles: "list[Polygon]") -> bool:
    """Check if a point is feasible (not inside any obstacles)."""
    return not any(is_point_in_polygon(point, obs) for obs in obstacles)



def is_line_connectable(p1: "array[Any,sz[3],Float]|Point", p2: "array[Any,sz[3],Float]|Point", obstacles: "list[Polygon]") -> bool:
    """Check if a line between two points is connectable (does not intersect any obstacles)."""
    from shapely.geometry import LineString

    line = LineString([p1, p2])
    return not any(line.intersects(obs) for obs in obstacles)


def interpolate_constant_velocity_trajectory(
    traj_anchor: array[Any,sz[1],Float],
    velocity: float = 1.0,
    hz: int = 100,
    order: int|Float = np.inf,
) -> tuple[array[Any,sz[1],Float],array[Any,sz[1],sz[1],Float]]:
    """Interpolate a trajectory to achieve constant velocity."""
    num_points = traj_anchor.shape[0]
    dims = traj_anchor.shape[1]

    distances = np.zeros(num_points)
    for i in range(1, num_points):
        distances[i] = np.linalg.norm(traj_anchor[i] - traj_anchor[i - 1], ord=order)

    times_anchor = np.cumsum(distances / velocity)
    interp_len = int(times_anchor[-1] * hz)
    times_interp = np.linspace(0, times_anchor[-1], interp_len)
    traj_interp = np.zeros((interp_len, dims))

    for d in range(dims):
        traj_interp[:, d] = np.interp(times_interp, times_anchor, traj_anchor[:, d])

    return times_interp, traj_interp


def depth_image_to_pointcloud(depth_img: array[Any,sz[2],Float], cam_matrix: array[Any,sz[2],sz[2],Float]) -> array[Any,sz[2],sz[3],Float]:
    """Convert a scaled depth image to a point cloud."""
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]

    height, width = depth_img.shape
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)

    z = depth_img
    x = (indices[..., 1] - cx) * z / fx
    y = (indices[..., 0] - cy) * z / fy

    return np.stack([z, -x, -y], axis=-1)

Z_AXIS = np.array([0,0,1],dtype=np.float64)

def compute_view_params(
    camera_pos: Pose | array[Any,sz[3],Float],
    target_pos: Pose | array[Any,sz[3],Float],
    up_vector: array[Any,sz[3],Float] = Z_AXIS,
) -> tuple[array[Any,sz[1],Float],array[Any,sz[1],Float],array[Any,sz[1],Float],array[Any,sz[3],Float]]:
    """Compute view parameters (azimuth, distance, elevation, lookat) for a camera."""
    camera_pos_array: array[sz[3],Float] = camera_pos.position.numpy() if isinstance(camera_pos, Pose) else np.array(camera_pos)
    target_pos_array: array[sz[3],Float] = target_pos.position.numpy() if isinstance(target_pos, Pose) else np.array(target_pos)

    cam_to_target = target_pos_array - camera_pos_array
    distance = np.linalg.norm(cam_to_target)

    azimuth = np.rad2deg(np.arctan2(cam_to_target[1], cam_to_target[0]))
    elevation = np.rad2deg(np.arcsin(cam_to_target[2] / distance))
    lookat = target_pos_array

    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    np.cross(zaxis, xaxis)

    return azimuth, distance, elevation, lookat


def sample_points_in_3d(
    n_sample: int,
    x_range: Annotated[list[Float],Len(2)],
    y_range: Annotated[list[Float],Len(2)],
    z_range: Annotated[list[Float],Len(2)],
    min_dist: float,
    xy_margin: float = 0.0,
) -> array[Any,sz[1],sz[3],Float]:
    """Sample points in 3D space ensuring a minimum distance between them."""
    xyzs = np.zeros((n_sample, 3))
    iter_tick = 0

    for i in range(n_sample):
        while True:
            x_rand = np.random.uniform(x_range[0] + xy_margin, x_range[1] - xy_margin)
            y_rand = np.random.uniform(y_range[0] + xy_margin, y_range[1] - xy_margin)
            z_rand = np.random.uniform(z_range[0], z_range[1])
            xyz = np.array([x_rand, y_rand, z_rand])

            if i == 0 or cdist(xyz.reshape(1, -1), xyzs[:i]).min() > min_dist:
                break

            iter_tick += 1
            if iter_tick > 1000:
                break

        xyzs[i] = xyz

    return xyzs


def quintic_trajectory(
    start_pos: array[Any,sz[6],Float],
    start_vel: array[Any,sz[6],Float],
    start_acc: array[Any,sz[6],Float],
    end_pos: array[Any,sz[6],Float],
    end_vel: array[Any,sz[6],Float],
    end_acc: array[Any,sz[6],Float],
    duration: float,
    num_points: int,
    max_velocity: float,
    max_acceleration: float,
) -> tuple[array[Any,sz[1],sz[6],Float],array[Any,sz[1],sz[6],Float],array[Any,sz[1],sz[6],Float],array[Any,sz[1],sz[6],Float]] :
    """Generate a quintic trajectory with velocity and acceleration constraints."""
    t: array[Any,sz[1],Float] = np.linspace(0, duration, num_points)
    joint_coeffs: list[array[Any,sz[6],Float]] = []

    for i in range(6):
        A: array[Any,sz[6],sz[6],Float] = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [duration**5, duration**4, duration**3, duration**2, duration, 1],
                [5 * duration**4, 4 * duration**3, 3 * duration**2, 2 * duration, 1, 0],
                [20 * duration**3, 12 * duration**2, 6 * duration, 2, 0, 0],
            ],
        )
        b: array[Any,sz[6],Float] = np.array([start_pos[i], start_vel[i], start_acc[i], end_pos[i], end_vel[i], end_acc[i]])
        x: array[Any,sz[6],Float] = np.linalg.solve(A, b)
        joint_coeffs.append(x)

    positions: array[Any,sz[1],sz[6],Float] = np.zeros((num_points, 6))
    velocities: array[Any,sz[1],sz[6],Float] = np.zeros((num_points, 6))
    accelerations: array[Any,sz[1],sz[6],Float] = np.zeros((num_points, 6))
    jerks: array[Any,sz[1],sz[6],Float] = np.zeros((num_points, 6))

    for i in range(num_points):
        for j in range(6):
            positions[i, j] = np.polyval(joint_coeffs[j], t[i])
            velocities[i, j] = np.polyval(np.polyder(joint_coeffs[j]), t[i])
            accelerations[i, j] = np.polyval(np.polyder(np.polyder(joint_coeffs[j])), t[i])
            jerks[i, j] = np.polyval(np.polyder(np.polyder(np.polyder(joint_coeffs[j]))), t[i])

    velocities = np.clip(velocities, -max_velocity, max_velocity)
    accelerations = np.clip(accelerations, -max_acceleration, max_acceleration)

    return positions, velocities, accelerations, jerks


def passthrough_filter(pcd: array[Any,sz[3],Float], axis: int, interval: list[Float,Float]) -> array[Any,sz[1],sz[3],Float]:
    """Filter a point cloud along a specified axis within a given interval."""
    mask = (pcd[:, axis] > interval[0]) & (pcd[:, axis] < interval[1])
    return pcd[mask]


def remove_duplicates(pointcloud: array[Any,sz[3],Float], threshold: float = 0.05) -> array[Any,sz[1],sz[3],Float]:
    """Remove duplicate points from a point cloud within a given threshold."""
    filtered_pointcloud = []
    for point in pointcloud:
        if all(np.linalg.norm(point - existing_point) > threshold for existing_point in filtered_pointcloud):
            filtered_pointcloud.append(point)
    return np.array(filtered_pointcloud)


def remove_duplicates_with_reference(
    pointcloud: array[Any,sz[3],Float],
    reference_point: array[Any,sz[3],Float],
    threshold: float = 0.05,
) -> array[Any,sz[1],sz[3],Float]:
    """Remove duplicate points close to a specific reference point within a given threshold."""
    return np.array([point for point in pointcloud if np.linalg.norm(point - reference_point) < threshold])


def downsample_pointcloud(pointcloud: array[Any,sz[3],Float], grid_size: float) -> array[Any,sz[1],sz[3],Float]:
    """Downsample a point cloud based on a specified grid size."""
    min_vals = pointcloud.min(axis=0)
    grid_pointcloud = np.floor((pointcloud - min_vals) / grid_size).astype(int)
    unique_pointcloud = {
        tuple(pos): original_pos for pos, original_pos in zip(grid_pointcloud, pointcloud, strict=False)
    }
    return np.array(list(unique_pointcloud.values()))


def align_vector_with_reference(vector: array[Any,sz[3],Float], reference: array[Any,sz[3],Float]) -> array[Any,sz[3],Float]:
    """Aligns the 'vector' with the 'reference' vector such that the angle between them is minimized.

    If the negative of 'vector' results in a smaller angle with 'reference', it returns the negated vector.

    Parameters:
    vector (np.ndarray): The vector to be aligned.
    reference (np.ndarray): The reference vector.

    Returns:
    np.ndarray: The possibly negated 'vector' that is most aligned with 'reference'.
    """
    return rotation_between_two_points(vector,reference)
