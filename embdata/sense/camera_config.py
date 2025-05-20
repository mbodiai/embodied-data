
import open3d as o3d

from embdata.sense.camera import Camera, Distortion, Extrinsics, Intrinsics
from embdata.utils.custom_logger import get_logger

o3dg = o3d.geometry
o3du = o3d.utility

logger = get_logger(__name__)

# --- Camera Configs ---
CAMERA_D435_1 = Camera(
    intrinsic=Intrinsics(fx=911.541, fy=911.777, cx=642.86, cy=374.433),
    distortion=Distortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
    extrinsic=Extrinsics(),
    depth_scale=0.001,
)

CAMERA_D435_2 = Camera(
    intrinsic=Intrinsics(fx=911.023, fy=911.503, cx=653.95, cy=371.601),
    distortion=Distortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
    extrinsic=Extrinsics(),
    depth_scale=0.001,
)

CAMERA_D455_0 = Camera(
    intrinsic=Intrinsics(fx=641.6, fy=640.684, cx=653.528, cy=362.808),
    distortion=Distortion(
        k1=-0.0559837, k2=0.0681921, p1=-0.000532545, p2=0.000916953, k3=-0.021557,
    ),
    extrinsic=Extrinsics(),
    depth_scale=0.001,
)

CAMERA_D455_1 = Camera(
    intrinsic=Intrinsics(fx=643.355, fy=642.532, cx=640.161, cy=373.284),
    distortion=Distortion(
        k1=-0.0556881, k2=0.0669827, p1=0.000216677, p2=0.000234341, k3=-0.0213151,
    ),
    extrinsic=Extrinsics(),
    depth_scale=0.001,
)

CAMERA_D455_2 = Camera(
    intrinsic=Intrinsics(fx=640.963, fy=640.295, cx=652.577, cy=372.942),
    distortion=Distortion(
        k1=-0.0557521, k2=0.0651432, p1=1.86459e-05, p2=0.000416844, k3=-0.0210544,
    ),
    extrinsic=Extrinsics(),
    depth_scale=0.001,
)

# # --- ArUco Configs ---
# # ARUCO_1 = ArucoParams(
# #     marker_size=0.15,
# #     world_pose=Pose(x=-0.645, y=0.756, z=0.0, roll=0.0, pitch=0.0, yaw=90.0),
# # )

# ARUCO_3 = ArucoParams(
#     marker_size=0.15,
#     world_pose=Pose(x=-0.21, y=0.0, z=0.015, roll=0.0, pitch=0.0, yaw=90.0),
# )

# # ARUCO_3 = ArucoParams(
# #     marker_size=0.20,
# #     world_pose=Pose(x=-0.968, y=0.021, z=-0.16, roll=0.0, pitch=0.0, yaw=90.0),
# # )


class DefaultCameras:
    CAM_0: Camera = CAMERA_D455_0
    CAM_1: Camera = CAMERA_D455_1
    CAM_2: Camera = CAMERA_D455_2



