from importlib.resources import files
from typing import Any, cast

import numpy as np
from typing_extensions import Annotated

from embdata.coordinate import BBox2D, Corner2D, PixelCoords, Pose
from embdata.geometry.utils import axis_angle_to_euler
from embdata.sample import Sample
from embdata.sense.camera import Camera
from embdata.sense.depth import Depth
from embdata.sense.image import Image
from embdata.sense.world_object import WorldObject
from embdata.units import LinearUnit


class ArucoParams(Sample):
    marker_size: Annotated[float, LinearUnit]
    world_pose: Pose


class Aruco(Sample):
    image: Image
    camera: Camera
    params: ArucoParams
    depth: Depth

    def detect(self,
               image: Image | None = None,
               params: ArucoParams | None = None,
               camera: Camera | None = None,
               depth: Depth | None = None) -> WorldObject:
        """Detect ArUco markers in the given frame and return a WorldObject.

        https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

        Returns:
            WorldObject: Detected ArUco marker as a WorldObject with pose and bounding box.

        Example:
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> intrinsic_matrix = np.eye(3)
            >>> distortion_coeffs = np.zeros(5)
            >>> marker_size = 0.1
            >>> detect_markers(frame, intrinsic_matrix, distortion_coeffs, marker_size)
            (WorldObject)
        """
        from cv2 import COLOR_RGB2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, cornerSubPix, cvtColor
        from cv2.aruco import (
            DICT_6X6_250,
            DetectorParameters,
            detectMarkers,
            estimatePoseSingleMarkers,
            getPredefinedDictionary,
        )
        aruco_dict = getPredefinedDictionary(DICT_6X6_250)
        parameters = DetectorParameters()

        current_image = image or self.image
        current_params = params or self.params
        current_camera = camera or self.camera
        current_depth = depth or self.depth
        gray = cvtColor(current_image.numpy(), COLOR_RGB2GRAY)
        corners, _, _ = detectMarkers(gray, dictionary=aruco_dict, parameters=parameters)

        detected_corners: list[Corner2D] = []
        centroids: list[PixelCoords] = []

        if corners:
            criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            for corner in corners:
                _ = cornerSubPix(image=gray, corners=corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
                detected_corners.append(
                    Corner2D(
                        top_left=PixelCoords(u=int(corner[0][0][0]), v=int(corner[0][0][1])),
                        top_right=PixelCoords(u=int(corner[0][1][0]), v=int(corner[0][1][1])),
                        bottom_right=PixelCoords(u=int(corner[0][2][0]), v=int(corner[0][2][1])),
                        bottom_left=PixelCoords(u=int(corner[0][3][0]), v=int(corner[0][3][1])),
                    ),
                )
                centroid = np.mean(corner[0], axis=0)
                centroids.append(PixelCoords(u=int(centroid[0]), v=int(centroid[1])))

        if not corners:
            msg = "No ArUco markers detected"
            raise ValueError(msg)

        rvecs, tvecs, _ = estimatePoseSingleMarkers(
            corners=cast(list[np.ndarray[Any, Any]],corners),
            markerLength=current_params.marker_size,
            cameraMatrix=np.array(current_camera.intrinsic.matrix),
            distCoeffs=np.array(current_camera.distortion.numpy()),
        )

        roll, pitch, yaw = axis_angle_to_euler(rvecs[0][0], sequence="xyz")

        return WorldObject(
            name="aruco",
            bbox_2d=BBox2D(
                x1=detected_corners[0].top_left.u,
                y1=detected_corners[0].top_left.v,
                x2=detected_corners[0].bottom_right.u,
                y2=detected_corners[0].bottom_right.v,
            ),
            pixel_coords=centroids[0],
            pose=Pose(*current_camera.deproject(centroids[0], current_depth.array),
                      roll=roll,
                      pitch=pitch,
                      yaw=yaw),
        )


if __name__ == "__main__":

    from embdata.sense.camera_config import CAMERA_D435_1
    from embdata.utils.safe_print import safe_print

    RGB_FILE = str(files("embdata")/"resources"/"color_image.png")
    DEPTH_FILE = str(files("embdata")/"resources"/"depth_image.png")
    CAMERA = CAMERA_D435_1

    aruco = Aruco(
        image=Image(path=RGB_FILE, encoding="png", mode="RGB"),
        camera=CAMERA,
        depth=Depth(path=DEPTH_FILE, encoding="png", mode="I"),
        params=ArucoParams(marker_size=0.1, world_pose=Pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0)),
    )
    aruco: WorldObject = aruco.detect()
    safe_print(aruco)
